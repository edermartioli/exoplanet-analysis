"""
    Created on Mar 10 2023
    
    Description: This routine fits the orbit(s) of a planetary system to the RV data using MCMC and detrend RV data with activity indices
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    # for TOI-1736
    rv_fit_detrend --rvdata=/Volumes/Samsung_T5/Science/TOI-1736/RVDATA/TOI-1736_sophie_drsrvs+ccftool.rdb --planet_priors=/Volumes/Samsung_T5/Science/TOI-1736/RV_ONLY_ANALYSIS/TOI-1736_rv_only.pars -vjp

    # TOI-2141
    rv_fit_detrend --rvdata=/Volumes/Samsung_T5/Science/TOI-2141/TOI-2141_sophie_results.rdb --planet_priors=/Volumes/Samsung_T5/Science/TOI-2141/DRS_data_analysis/TOI-2141_rv_only.pars -vjpmd
    rv_fit_detrend --rvdata=/Volumes/Samsung_T5/Science/TOI-2141/DRS_data_analysis/TOI-2141_sophiedrs_data.rdb --planet_priors=/Volumes/Samsung_T5/Science/TOI-2141/DRS_data_analysis/TOI-2141_rv_only.pars -vjpm
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import sys, os

from optparse import OptionParser

import numpy as np
from exoplanet_analysis import fitlib, rvutils
from exoplanet_analysis import priorslib
import glob

from exoplanet_analysis.config import priors_dir


def main() :

    """Main.
    """
    parser = OptionParser()
    parser.add_option("-o", "--object", dest="object", help='Object ID',type='string',default="")
    parser.add_option("-r", "--rvdata", dest="rvdata", help='Input radial velocity data file',type='string',default="")
    parser.add_option("-l", "--planet_priors", dest="planet_priors", help='Planet prior parameters file',type='string',default="")
    parser.add_option("-n", "--nsteps", dest="nsteps", help="Number of MCMC steps",type='int',default=3000)
    parser.add_option("-w", "--walkers", dest="walkers", help="Number of MCMC walkers",type='int',default=50)
    parser.add_option("-i", "--burnin", dest="burnin", help="Number of MCMC burn-in samples",type='int',default=1000)
    parser.add_option("-d", action="store_true", dest="detrend_rv_data", help="Detrend RV data with activity indices",default=False)
    parser.add_option("-f", action="store_true", dest="ols_fit", help="Perform OLS fit prior to MCMC", default=False)
    parser.add_option("-m", action="store_true", dest="run_mcmc", help="Run MCMC",default=False)
    parser.add_option("-j", action="store_true", dest="fit_rv_jitter", help="Fit RV jitter",default=False)
    parser.add_option("-a", action="store_true", dest="fit_gp_activity", help="Run GP activity analysis",default=False)
    parser.add_option("-g", action="store_true", dest="run_gp_mcmc", help="Run MCMC to fit GP parameters",default=False)
    parser.add_option("-p", action="store_true", dest="plot", help="plot",default=False)
    parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

    try:
        options,args = parser.parse_args(sys.argv[1:])
    except SystemExit as e :
        # allow clean exits from optparse (e.g. --help)
        if e.code == 0 or e.code is None :
            raise
        print("Error: check usage with rv_fit.py -h "); sys.exit(1);

    if options.verbose:
        print('Object ID: ', options.object)
        print('Pattern for input radial velocity data file: ', options.rvdata)
        print('Planet prior parameters file: ', options.planet_priors)
        print('Number of MCMC steps: ', options.nsteps)
        print('Number of MCMC walkers: ', options.walkers)
        print('Number of MCMC burn-in samples: ', options.burnin)

    if options.verbose:
        print("Creating list of RV data files...")
    inputrvdata = sorted(glob.glob(options.rvdata))

    # load rv data into a dictionary container
    rvdata = rvutils.load_rvdata_from_rdbfiles(inputrvdata, factor=1., verbose=options.verbose)
    bjds, rvs, rverrs = rvdata['bjds'], rvdata['rvs'], rvdata['rverrs']

    fwhms, sig_fwhms = rvdata['fwhms'], rvdata['sig_fwhms']
    biss, sig_biss = rvdata['biss'], rvdata['sig_biss']
    sindexs, sig_sindexs = rvdata['sindexs'], rvdata['sig_sindexs']
    has, sig_has = rvdata['has'], rvdata['sig_has']

    rvdatalabels = rvdata['rvdatalabels']
    #rvdatalabels = ["SOPHIE data"]
    rvdatalabels = ["Blue (2023)","Blue (2024)","Red (2023)","Red (2024)"]

    n_rvdatasets = len(rvs)

    if n_rvdatasets == 0 :
        print("Error: no input RV data file found using pattern:",options.rvdata)
        exit()

    # load priors from file
    priors = fitlib.read_rv_priors(options.planet_priors, n_rvdatasets, verbose=options.verbose)

    # Fit RV calibration parameters for initial guess
    #posterior = fitlib.guess_rvcalib(priors, bjds, rvs, prior_type="Normal", plot=False)
    posterior = fitlib.guess_rvcalib(priors, bjds, rvs, prior_type="Normal", plot=False)

    # OLS fit involving all priors
    if options.ols_fit :
        posterior = fitlib.fit_RVs_ols(bjds, rvs, rverrs, posterior, fix_eccentricity=False, rvcalib_post_type="Normal", calib_unc=0.01, verbose=False, plot=True)

    print(posterior["rvcalib_params"])

    if options.plot :
        fitlib.plot_rv_global_timeseries(posterior["planet_params"], posterior["rvcalib_params"], bjds, rvs, rverrs, number_of_free_params=len(posterior["theta"]), samples=None, labels=None, nsamples=100, plot_residuals=True, rvdatalabels=rvdatalabels)

    if options.detrend_rv_data :
        plot_detrends = True
        n_sigma_clip = 4

        # detrend RVs with activity indices
        bjds, rvs, rverrs = rvutils.detrend_rvs_with_activity_indices(posterior, bjds, rvs, rverrs,
                                                                    biss, sig_biss,
                                                                    fwhms, sig_fwhms,
                                                                    sindexs, sig_sindexs,
                                                                    has, sig_has,
                                                                    n_sigma_clip=n_sigma_clip, plot=plot_detrends)

    # fit jitter
    if options.fit_rv_jitter :
        rverrs, jitter, jitter_err = rvutils.fit_RV_jitter(posterior, bjds, rvs, rverrs, rvdatalabels=rvdatalabels)

    if options.ols_fit :
        # OLS fit involving all priors
        posterior = fitlib.fit_RVs_ols(bjds, rvs, rverrs, posterior, fix_eccentricity=False, rvcalib_post_type="Normal", calib_unc=0.01, verbose=False, plot=True)

    if options.plot :
        fitlib.plot_rv_global_timeseries(posterior["planet_params"], posterior["rvcalib_params"], bjds, rvs, rverrs, number_of_free_params=len(posterior["theta"]), samples=None, labels=None, nsamples=100, plot_residuals=True, rvdatalabels=rvdatalabels)

        n_planets = int(posterior["planet_params"]["n_planets"])

        for planet_index in range(n_planets) :
            fitlib.plot_rv_perplanet_timeseries(posterior["planet_params"], posterior["rvcalib_params"], bjds, rvs, rverrs, planet_index=planet_index, number_of_free_params=len(posterior["theta"]), samples=None, labels=None, nsamples=100, plot_residuals=True, rvdatalabels=rvdatalabels, phase_plot=True, bindata=True, binsize=0.05)

    """
    # fit jitter in MCMC
    if options.fit_rv_jitter :
        print("Fitting jitter ... ")
        _, jitter, jitter_err = rvutils.fit_RV_jitter(posterior, bjds, rvs, rverrs, rvdatalabels=rvdatalabels)
        posterior = fitlib.update_rv_jitter(posterior, jitter, jitter_err, prior_type="Uniform")
    """
    if options.run_mcmc :

        nwalkers = options.walkers
        niter = options.nsteps
        burnin = options.burnin
        amp = 1e-4

        samples_filename = priorslib.derive_filename(options.planet_priors, "_mcmc_samples.h5")

        posterior = fitlib.fitRVsWithMCMC(bjds, rvs, rverrs, posterior, amp=amp, nwalkers=nwalkers, niter=niter, burnin=burnin, verbose=True, plot=True, samples_filename=samples_filename, appendsamples=False, rvdatalabels=rvdatalabels, bin_plot_data=True)


if __name__ == "__main__" :
    main()
