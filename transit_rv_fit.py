"""
    Created on Feb 17 2022
    
    Description: This routine fits planetary transits and RV data simultaneously using MCMC
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    # for TOI-1452
    python transit_rv_fit.py --object="TIC 420112589" --rvdata=/Users/eder/Science/TOI-1452/RVs/lbl_TOI-1452_GL699_drift.rdb --gp_priors=priors/gp-priors.pars  -vpam
    
    #for TOI-1695
    python transit_rv_fit.py --object="TIC 422756130" --rvdata=/Users/eder/Science/TOI-1695/lbl_TOI-1695_GL15A_drift.rdb --gp_priors=priors/gp-priors.pars -vpam
    
    #for TOI-1759
    python transit_rv_fit.py --object="TOI-1759" --rvdata=/Users/eder/Science/TOI-1759/PaperRVData/lbl_TOI-1759_GL846_drift.rdb --gp_priors=priors/gp-priors.pars -vpa

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
import fitlib, priorslib
import glob
import rvutils
import tess

from astropy.io import ascii

ExoplanetAnalysis_dir = os.path.dirname(__file__)
priors_dir = os.path.join(ExoplanetAnalysis_dir, 'priors/')

parser = OptionParser()
parser.add_option("-o", "--object", dest="object", help='Object ID',type='string',default="")
parser.add_option("-r", "--rvdata", dest="rvdata", help='Input radial velocity data file',type='string',default="")
parser.add_option("-b", "--blongdata", dest="blongdata", help='Input B-longitudinal data file',type='string',default="")
parser.add_option("-l", "--planet_priors", dest="planet_priors_pattern", help='Pattern to get planet prior parameters files',type='string',default="")
parser.add_option("-g", "--gp_priors", dest="gp_priors_file", help='QP GP prior parameters file',type='string',default="")
parser.add_option("-c", "--calib_order", dest="calib_order", help='Order of calibration polynomial',type='string',default="1")
parser.add_option("-n", "--nsteps", dest="nsteps", help="Number of MCMC steps",type='int',default=1000)
parser.add_option("-w", "--walkers", dest="walkers", help="Number of MCMC walkers",type='int',default=32)
parser.add_option("-i", "--burnin", dest="burnin", help="Number of MCMC burn-in samples",type='int',default=300)
parser.add_option("-z", "--phot_binsize", dest="phot_binsize", help="Bin size of photometric data [days]",type='float',default=0.1)
parser.add_option("-O", action="store_true", dest="force_tess_pl_priors", help="Force using TESS planet priors",default=False)
parser.add_option("-a", action="store_true", dest="fit_gp_activity", help="Run GP activity analysis",default=False)
parser.add_option("-m", action="store_true", dest="run_gp_mcmc", help="Run MCMC to fit GP parameters",default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot",default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with transit_rv_fit.py -h "); sys.exit(1);

if options.verbose:
    print('Object ID: ', options.object)
    print('Pattern for input radial velocity data file: ', options.rvdata)
    print('Input B-longitudinal data file: ', options.blongdata)
    print('Pattern for input planets prior files: ', options.planet_priors_pattern)
    print('QP GP prior parameters file: ', options.gp_priors_file)
    print('Order of calibration polynomial: ', options.calib_order)
    print('Number of MCMC steps: ', options.nsteps)
    print('Number of MCMC walkers: ', options.walkers)
    print('Number of MCMC burn-in samples: ', options.burnin)
    print('Bin size of photometric data [days]: ', options.phot_binsize)

# Download TESS DVT products and return a list of input data files
dvt_filenames = tess.retrieve_tess_data_files(options.object, products_wanted_keys = ["DVT"], verbose=options.verbose)

if options.verbose:
    print("Loading TESS lightcurves ...")
# Load TESS data
tesslc = tess.load_dvt_files(options.object, priors_dir=priors_dir, save_priors=options.force_tess_pl_priors, hasrvdata=True, plot=options.plot, verbose=options.verbose)

if options.verbose:
    print("Creating list of RV data files...")
inputrvdata = sorted(glob.glob(options.rvdata))

bjds, rvs, rverrs = [], [], []
for i in range(len(inputrvdata)) :
    if options.verbose:
        print("Loading Radial Velocity data from file: ", inputrvdata[i])
    # Load RVs
    bjd, rv, rverr = rvutils.read_rv_time_series(inputrvdata[i])

    keep = (np.isfinite(rv)) & (np.isfinite(rverr))

    bjds.append(bjd[keep])
    rvs.append(rv[keep])
    rverrs.append(rverr[keep])

if options.blongdata != "" :
    if options.verbose:
        print("Loading B-long data ...")
    blong_data = ascii.read(options.blongdata)
    keep = (np.isfinite(blong_data["col2"])) & (np.isfinite(blong_data["col3"]))
    bbjds, blong, blongerr = blong_data["col1"][keep], blong_data["col2"][keep], blong_data["col3"][keep]
else :
    bbjds = blong = blongerr = np.array([])

if len(rvs) == 0 :
    print("Error: no input RV data file found using pattern:",options.rvdata)
    exit()
    
for i in range(len(tesslc["PLANETS"])) :
    
    planet = tesslc["PLANETS"][i]
    
    # select data within certain ranges
    times, fluxes, fluxerrs = planet["times"], planet["fluxes"], planet["fluxerrs"]

    #Load priors information:
    if options.planet_priors_pattern != "" :
        planet_priors_files = sorted(glob.glob(options.planet_priors_pattern))
    else :
        planet_priors_files = sorted(glob.glob(planet["prior_file"]))

    calib_polyorder = int(options.calib_order)

    # read priors from input files
    priors = fitlib.read_priors(planet_priors_files, len(times), calib_polyorder=calib_polyorder, n_rvdatasets=len(rvs), verbose=False)

    # Fit calibration parameters for initial guess
    priors = fitlib.guess_calib(priors, times, fluxes, prior_type="Normal")

    # Fit RV calibration parameters for initial guess
    posterior = fitlib.guess_rvcalib(priors, bjds, rvs, prior_type="Normal")

    if options.plot :
        # plot light curves and models in priors
        fitlib.plot_mosaic_of_lightcurves(times, fluxes, fluxerrs, posterior)

    t0 = posterior["planet_params"][i]['tc_{0:03d}'.format(i)] + 0.5 * posterior["planet_params"][i]['per_{0:03d}'.format(i)]

    if options.plot :
        fitlib.plot_rv_timeseries(posterior["planet_params"], posterior["rvcalib_params"], bjds, rvs, rverrs, planet_index=i, phasefold=False, t0=t0)

    # OLS fit involving all priors
    posterior = fitlib.fitTransits_and_RVs_ols(times, fluxes, fluxerrs, bjds, rvs, rverrs, posterior, fix_eccentricity=True, calib_post_type="Normal", rvcalib_post_type="Normal", calib_unc=0.01, verbose=False, plot=False)

    # OLS fit involving all priors
    posterior = fitlib.fitTransits_and_RVs_ols(times, fluxes, fluxerrs, bjds, rvs, rverrs, posterior, fix_eccentricity=True, calib_post_type="FIXED", rvcalib_post_type="Normal", calib_unc=0.01, verbose=False, plot=options.plot)

    nwalkers = options.walkers
    niter = options.nsteps
    burnin = options.burnin
    amp = 1e-5

    reduction_type = 'LBL'
    if 'ccf' in inputrvdata[0] :
        reduction_type = 'CCF'
        
    for i in range(len(inputrvdata)) :
        print("Dataset: {} -> RMS of RVs: {:.2f} m/s Median errors: {:.2f} m/s -> file: {}".format(i,np.nanstd(rvs[i]),np.nanmedian(rverrs[i]),inputrvdata[i]) )

    samples_filename = planet["prior_file"].replace(".pars","_mcmc_samples_{}.h5".format(reduction_type))

    posterior = fitlib.fitTransitsAndRVsWithMCMC(times, fluxes, fluxerrs, bjds, rvs, rverrs, posterior, amp=amp, nwalkers=nwalkers, niter=niter, burnin=burnin, verbose=True, plot=True, samples_filename=samples_filename, appendsamples=False, rvlabel="SPIRou {} RVs".format(reduction_type))

    fit_period = posterior["planet_params"][0]['per_000']
    period_range = [2,100]
    
    rvutils.periodgram_rv_check_signal(bjds[0], rvs[0], rverrs[0], period=[fit_period], period_labels=["fit orbit"],period_range=period_range, nbins=3, phase_plot=options.plot)

    #############################
    # Start activity GP analysis
    #############################
    
    # If option below is turned on it gets very slow
    joint_gp_activity_and_orbit = False
    Blong_and_RV_constrained = False
    
    if options.fit_gp_activity :

        gp_priors = priorslib.read_priors(options.gp_priors_file)
        gp_priors["gp_priorsfile"] = options.gp_priors_file

        rv_t, rv_res, rv_err = fitlib.get_rv_residuals(posterior, bjds, rvs, rverrs)
    
        gp_rv, gp_blong, gp_phot, rv_gp_feed, blong_gp_feed, phot_gp_feed, gp_priors = fitlib.set_star_rotation_gp (posterior, gp_priors, tesslc, rv_t, rv_res, rv_err, bbjds, blong, blongerr, binsize=options.phot_binsize, gp_ls_fit=True, run_gp_mcmc=options.run_gp_mcmc, amp=amp, nwalkers=nwalkers, niter=niter, burnin=burnin, remove_transits=True, plot=options.plot, verbose=True, spec_constrained=Blong_and_RV_constrained)

        # reduce gp activity model in phot time series
        red_fluxes, red_fluxerrs = fitlib.reduce_gp(gp_phot, times, fluxes, fluxerrs, phot_gp_feed)

        # reduce gp activity model in RV time series
        red_rvs, red_rverrs = fitlib.reduce_gp(gp_rv, bjds, rvs, rverrs, rv_gp_feed)

        # save RVs - GP to file
        for j in range(len(bjds)) :
            gpremovedRVsoutput = inputrvdata[0].replace(".rdb","_GPremoved.rdb")
            print("Saving GP-removed RVs to file: ",gpremovedRVsoutput)
            rvutils.save_rv_time_series(gpremovedRVsoutput, bjds[j], red_rvs[j], red_rverrs[j], time_in_rjd=True, rv_in_mps=False)
    
        rvutils.periodgram_rv_check_signal(bjds[0], red_rvs[0], red_rverrs[0], period=[fit_period], period_labels=["fit orbit"],period_range=period_range, nbins=3, phase_plot=options.plot)
    
        if joint_gp_activity_and_orbit :
            # OLS fit involving all priors
            posterior = fitlib.fitTransits_and_RVs_ols(times, red_fluxes, red_fluxerrs, bjds, red_rvs, red_rverrs, posterior, fix_eccentricity=True, flare_post_type="FIXED", calib_post_type="FIXED", rvcalib_post_type="FIXED", calib_unc=0.01, verbose=False, plot=True, rvlabel="SPIRou {} RVs".format(reduction_type))
            
            samples_filename = planet["prior_file"].replace(".pars","_mcmc_samples_{}+gp.h5".format(reduction_type))
            
            posterior = fitlib.fitTransitsAndRVsWithMCMCAndGP(times, fluxes, fluxerrs, bjds, rvs, rverrs, bbjds, blong, blongerr, posterior, gp_priors, tesslc, phot_binsize=options.phot_binsize, amp=amp, nwalkers=nwalkers, niter=niter, burnin=burnin, samples_filename=samples_filename, verbose=True, plot=True, rvlabel="SPIRou {} RVs".format(reduction_type), gp_spec_constrained=Blong_and_RV_constrained)

        else :
            # OLS fit involving all priors
            posterior = fitlib.fitTransits_and_RVs_ols(times, red_fluxes, red_fluxerrs, bjds, red_rvs, red_rverrs, posterior, fix_eccentricity=True, flare_post_type="FIXED", calib_post_type="FIXED", rvcalib_post_type="Normal", calib_unc=0.01, verbose=False, plot=True, rvlabel="SPIRou {} RVs".format(reduction_type))

            for j in range(len(inputrvdata)) :
                print("Dataset: {} -> RMS of RVs-GP: {:.2f} m/s Median errors: {:.2f} m/s -> file: {}".format(i,np.nanstd(red_rvs[j]),np.nanmedian(red_rverrs[j]),inputrvdata[j]) )

            samples_filename = planet["prior_file"].replace(".pars","_mcmc_samples_{}_notjointgp.h5".format(reduction_type))

            posterior = fitlib.fitTransitsAndRVsWithMCMC(times, red_fluxes, red_fluxerrs, bjds, red_rvs, red_rverrs, posterior, amp=amp, nwalkers=nwalkers, niter=niter, burnin=burnin, verbose=True, plot=True, samples_filename=samples_filename, appendsamples=False, rvlabel="SPIRou {} RVs".format(reduction_type), addnparams=0)

            # plot photometry gp posterior
            fitlib.plot_gp_photometry(tesslc, posterior, gp_phot, phot_gp_feed["y"], options.phot_binsize)

            # plot RV gp posterior
            fitlib.plot_gp_rv(posterior, gp_rv, rv_gp_feed, bjds, rvs, rverrs, default_legend="{} RV".format(reduction_type))

            #t0 = posterior["planet_params"][i]['tc_{0:03d}'.format(i)] + 0.5 * posterior["planet_params"][i]['per_{0:03d}'.format(i)]
            t0 = posterior["planet_params"][i]['tc_{0:03d}'.format(i)]
            
            # RV phase plot:
            fitlib.plot_rv_timeseries(posterior["planet_params"], posterior["rvcalib_params"], bjds, red_rvs, red_rverrs, samples=None, labels=posterior["labels"], planet_index=0, plot_residuals=True, phasefold=True, plot_bin_data=True, rvlabel="SPIRou {} RVs".format(reduction_type),t0=t0)

