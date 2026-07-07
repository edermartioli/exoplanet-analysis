"""
    Created on May 18 2022
    
    Description: This routine fits the orbit of a planetary system to the RV data using MCMC
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    # for TOI-1736
    rv_fit --rvdata=/Volumes/Samsung_T5/Science/TOI-1736/TOI-1736_sophie_000.rdb --planet_priors=/Volumes/Samsung_T5/Science/TOI-1736/TOI-1736.pars -v

    rv_fit --rvdata=/Volumes/Samsung_T5/Science/TOI-1736/TOI-1736_sophie_000.rdb --planet_priors=/Volumes/Samsung_T5/Science/TOI-1736/TOI-1736.pars -v
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import sys, os

from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from exoplanet_analysis import fitlib, priorslib
import glob
from exoplanet_analysis import rvutils
from exoplanet_analysis import tess

from astropy.io import ascii

from exoplanet_analysis.config import priors_dir


def main() :

    """Main.
    """
    parser = OptionParser()
    parser.add_option("-o", "--object", dest="object", help='Object ID',type='string',default="")
    parser.add_option("-r", "--rvdata", dest="rvdata", help='Input radial velocity data file',type='string',default="")
    parser.add_option("-l", "--planet_priors", dest="planet_priors", help='Planet prior parameters file',type='string',default="")
    parser.add_option("-g", "--gp_priors", dest="gp_priors_file", help='QP GP prior parameters file',type='string',default="")
    parser.add_option("-n", "--nsteps", dest="nsteps", help="Number of MCMC steps",type='int',default=3000)
    parser.add_option("-w", "--walkers", dest="walkers", help="Number of MCMC walkers",type='int',default=50)
    parser.add_option("-i", "--burnin", dest="burnin", help="Number of MCMC burn-in samples",type='int',default=1000)
    parser.add_option("-j", action="store_true", dest="fit_rv_jitter", help="Fit RV jitter",default=False)
    parser.add_option("-a", action="store_true", dest="fit_gp_activity", help="Run GP activity analysis",default=False)
    parser.add_option("-m", action="store_true", dest="run_gp_mcmc", help="Run MCMC to fit GP parameters",default=False)
    parser.add_option("-p", action="store_true", dest="plot", help="plot",default=False)
    parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

    try:
        options,args = parser.parse_args(sys.argv[1:])
    except SystemExit as e :
        # allow clean exits from optparse (e.g. --help)
        if e.code == 0 or e.code is None :
            raise
        print("Error: check usage with rv_fit -h "); sys.exit(1);

    if options.verbose:
        print('Object ID: ', options.object)
        print('Pattern for input radial velocity data file: ', options.rvdata)
        print('Planet prior parameters file: ', options.planet_priors)
        print('QP GP prior parameters file: ', options.gp_priors_file)
        print('Number of MCMC steps: ', options.nsteps)
        print('Number of MCMC walkers: ', options.walkers)
        print('Number of MCMC burn-in samples: ', options.burnin)


    if options.verbose:
        print("Creating list of RV data files...")
    inputrvdata = sorted(glob.glob(options.rvdata))

    bjds, rvs, rverrs = [], [], []
    rvdatalabels = []
    for i in range(len(inputrvdata)) :
        if options.verbose:
            print("Loading Radial Velocity data from file: ", inputrvdata[i])
        # Load RVs
        bjd, rv, rverr = rvutils.read_rv_time_series(inputrvdata[i], conv_factor=1)
        keep = (np.isfinite(rv)) & (np.isfinite(rverr))
        #keep &= bjd > 2459580
        bjds.append(bjd[keep])
        rvs.append(rv[keep])
        rverrs.append(rverr[keep])
        #rvdatalabels.append("SOPHIE data")
        basename = os.path.basename(inputrvdata[i])
        rvdatalabels.append(basename)

    n_rvdatasets = len(rvs)

    if n_rvdatasets == 0 :
        print("Error: no input RV data file found using pattern:",options.rvdata)
        exit()

    priors = fitlib.read_rv_priors(options.planet_priors, n_rvdatasets, verbose=options.verbose)

    # Fit RV calibration parameters for initial guess
    posterior = fitlib.guess_rvcalib(priors, bjds, rvs, prior_type="Normal", plot=False)

    #if options.plot :
    #    fitlib.plot_rv_timeseries_new(posterior["planet_params"], posterior["rvcalib_params"], bjds, rvs, rverrs, planet_index=i, phasefold=False, t0=t0)

    # OLS fit involving all priors
    posterior = fitlib.fit_RVs_ols(bjds, rvs, rverrs, posterior, fix_eccentricity=False, rvcalib_post_type="Normal", calib_unc=0.01, verbose=False, plot=True)

    # fit jitter
    if options.fit_rv_jitter :
        rverrs, jitter, jitter_err = rvutils.fit_RV_jitter(posterior, bjds, rvs, rverrs, rvdatalabels=rvdatalabels)
        # OLS fit again
        posterior = fitlib.fit_RVs_ols(bjds, rvs, rverrs, posterior, fix_eccentricity=False, rvcalib_post_type="Normal", calib_unc=0.01, verbose=False, plot=True)

    if options.plot :
        fitlib.plot_rv_global_timeseries(posterior["planet_params"], posterior["rvcalib_params"], bjds, rvs, rverrs, samples=None, labels=None, nsamples=100, plot_residuals=True, rvdatalabels=rvdatalabels)

        n_planets = int(posterior["planet_params"]["n_planets"])

        for planet_index in range(n_planets) :
            fitlib.plot_rv_perplanet_timeseries(posterior["planet_params"], posterior["rvcalib_params"], bjds, rvs, rverrs, planet_index=planet_index, samples=None, labels=None, nsamples=100, plot_residuals=True, rvdatalabels=rvdatalabels, phase_plot=True)

    nwalkers = options.walkers
    niter = options.nsteps
    burnin = options.burnin
    amp = 1e-4

    samples_filename = priorslib.derive_filename(options.planet_priors, "_mcmc_samples.h5")

    posterior = fitlib.fitRVsWithMCMC(bjds, rvs, rverrs, posterior, amp=amp, nwalkers=nwalkers, niter=niter, burnin=burnin, verbose=True, plot=True, samples_filename=samples_filename, appendsamples=False, rvdatalabels=rvdatalabels)


if __name__ == "__main__" :
    main()
