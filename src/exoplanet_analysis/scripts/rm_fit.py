# -*- coding: utf-8 -*-
"""
    Description: Fit the Rossiter-McLaughlin (RM) effect to radial velocities
    taken around a planetary transit, using MCMC. This reuses the same fitting
    infrastructure as the orbit-only RV fit; the RM anomaly (Ohta et al. 2005)
    is added to the RV model.

    @author: Eder Martioli, Shweta Dalal, Alexandre Teissier

    Simple usage example:

    rm_fit --rvdata="data/WASP-108_ghost_*.rdb" --planet_priors=WASP-108_rm.pars --rv_units=kmps --nsteps=1500 --walkers=32 --burnin=500 -vp
"""

__version__ = "1.0"

import sys, os
from optparse import OptionParser

import numpy as np
from exoplanet_analysis import fitlib, priorslib, rvutils


def main() :
    """Main entry point for the RM fitting command-line tool."""
    parser = OptionParser()
    parser.add_option("-r", "--rvdata", dest="rvdata", help="Input RV data file pattern (e.g. 'data/*.rdb')", type='string', default="")
    parser.add_option("-l", "--planet_priors", dest="planet_priors", help="Planet prior parameters file (.pars or system .json) including RM parameters", type='string', default="")
    parser.add_option("-u", "--rv_units", dest="rv_units", help="Units of the input RVs and vsini: 'kmps' (default) or 'mps'", type='string', default="kmps")
    parser.add_option("-n", "--nsteps", dest="nsteps", help="Number of MCMC steps", type='int', default=1500)
    parser.add_option("-w", "--walkers", dest="walkers", help="Number of MCMC walkers", type='int', default=32)
    parser.add_option("-b", "--burnin", dest="burnin", help="Number of MCMC burn-in samples", type='int', default=500)
    parser.add_option("-o", "--output", dest="output", help="Output posterior .pars file", type='string', default="")
    parser.add_option("-s", "--samples", dest="samples", help="Output HDF5 samples file", type='string', default="rm_mcmc_samples.h5")
    parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
    parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

    try:
        options, args = parser.parse_args(sys.argv[1:])
    except SystemExit as e :
        if e.code == 0 or e.code is None :
            raise
        print("Error: check usage with rm_fit -h")
        sys.exit(1)

    if options.rvdata == "" or options.planet_priors == "" :
        print("Error: both --rvdata and --planet_priors are required. Check usage with rm_fit -h")
        sys.exit(1)

    # velocity unit conversion: RVs and vsini must be in the same units. The
    # reader converts km/s -> m/s by conv_factor=1000; keep km/s with 1.
    conv_factor = 1.0 if options.rv_units.lower() in ("kmps", "km/s", "kms") else 1000.0

    import glob
    inputrvdata = sorted(glob.glob(options.rvdata))
    if len(inputrvdata) == 0 :
        print("Error: no input RV data files found using pattern:", options.rvdata)
        sys.exit(1)

    if options.verbose :
        print("Loading {} RV dataset(s)...".format(len(inputrvdata)))

    bjds, rvs, rverrs, rvdatalabels = [], [], [], []
    for f in inputrvdata :
        bjd, rv, rverr = rvutils.read_rv_time_series(f, conv_factor=conv_factor)
        keep = np.isfinite(rv) & np.isfinite(rverr)
        bjds.append(bjd[keep]); rvs.append(rv[keep]); rverrs.append(rverr[keep])
        rvdatalabels.append(os.path.basename(f))
        if options.verbose :
            print("  {}: {} points".format(os.path.basename(f), int(np.sum(keep))))

    # Read RM priors and initialize the RV calibration (per-dataset zero points)
    priors = fitlib.read_rm_priors(options.planet_priors, len(rvs), verbose=options.verbose)
    posterior = fitlib.guess_rvcalib(priors, bjds, rvs, prior_type="Normal", plot=False)

    if options.verbose :
        print("Free parameters:", posterior["labels"])
        print("Running RM MCMC fit...")

    posterior = fitlib.fitRMWithMCMC(bjds, rvs, rverrs, posterior,
                                     nwalkers=options.walkers, niter=options.nsteps,
                                     burnin=options.burnin, samples_filename=options.samples,
                                     rvdatalabels=rvdatalabels, verbose=options.verbose,
                                     plot=options.plot)

    # Save the posterior (RM parameters are included automatically)
    output = options.output
    if output == "" :
        output = priorslib.derive_filename(options.planet_priors, "_rm_posterior.pars")
    fitlib.priorslib.save_posterior(output, posterior["planet_params"],
                                    posterior["planet_theta_fit"] if "planet_theta_fit" in posterior else [],
                                    posterior["planet_theta_labels"] if "planet_theta_labels" in posterior else [],
                                    posterior["planet_theta_err"] if "planet_theta_err" in posterior else [])
    if options.verbose :
        print("Saved RM posterior to:", output)


if __name__ == "__main__" :
    main()
