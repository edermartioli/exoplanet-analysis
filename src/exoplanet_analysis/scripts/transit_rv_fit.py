"""
    Created on Feb 17 2022
    
    Description: This routine fits planetary transits and RV data simultaneously using MCMC
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    # for TOI-1452
    transit_rv_fit --object="TIC 420112589" --rvdata=/Users/eder/Science/TOI-1452/RVs/lbl_TOI-1452_GL699_drift.rdb --gp_priors=priors/gp-priors.pars  -vpam
    
    #for TOI-1695
    transit_rv_fit --object="TIC 422756130" --rvdata=/Users/eder/Science/TOI-1695/lbl_TOI-1695_GL15A_drift.rdb --gp_priors=priors/gp-priors.pars -vpam
    
    #for TOI-1759
    transit_rv_fit --object="TOI-1759" --rvdata=/Users/eder/Science/TOI-1759/PaperRVData/lbl_TOI-1759_GL846_drift.rdb --gp_priors=priors/gp-priors.pars -vpa

    #for TOI-1736
    transit_rv_fit --object="TOI-1736" --lcdata=/Volumes/Samsung_T5/Science/TOI-1736/TESS/*lc.fits --rvdata=/Volumes/Samsung_T5/Science/TOI-1736/RVDATA/TOI-1736_sophie_drsrvs+ccftool.rdb --planet_priors=/Volumes/Samsung_T5/Science/TOI-1736/RV+TRANSITS_ANALYSIS/TOI-1736.pars --nsteps=10000 --walkers=32 --burnin=3000 -vpfjm

    # for TOI-2141
    transit_rv_fit --object="TOI-2141" --lcdata=/Volumes/Samsung_T5/Science/TOI-2141/TESS/*lc.fits --rvdata=/Volumes/Samsung_T5/Science/TOI-2141/RVDATA/TOI-2141_sophie_drsresults.rdb --planet_priors=/Volumes/Samsung_T5/Science/TOI-2141/RV+TRANSITS_ANALYSIS/TOI-2141.pars --nsteps=10000 --walkers=32 --burnin=3000 -vjmpfe
    
    #for TOI-1718
    transit_rv_fit --object="TIC257241363" --rvdata=/Volumes/Samsung_T5/Science/TOI-1718/TOI-1718_sophie.rdb --lcdata=/Volumes/Samsung_T5/Science/TOI-1718/TESS/*lc.fits --planet_priors=/Volumes/Samsung_T5/Science/TOI-1718/TOI-1718.pars --burnin=2000 --nsteps=5000 -vp


    #for TOI-3568
    transit_rv_fit --object="TIC 160390955" --rvdata=/Volumes/Samsung_T5/Science/TOI-3568/MAROON-X/*.rdb --lcdata=/Volumes/Samsung_T5/Science/TOI-3568/TESS/*lc.fits --planet_priors=/Volumes/Samsung_T5/Science/TOI-3568/TOI-3568.pars -vp
    
    transit_rv_fit --object="TIC 160390955" --rvdata=/Volumes/Samsung_T5/Science/TOI-3568/RVs/*.rdb --lcdata=/Volumes/Samsung_T5/Science/TOI-3568/TESS/*lc.fits --planet_priors=/Volumes/Samsung_T5/Science/TOI-3568/TOI-3568.pars -vp
    
    # for TOI-4643
    
    transit_rv_fit --object="TOI-4643" --rvdata=/Users/eder/Science/TOI-4643/RVs/*ALL*.rdb --lcdata=/Users/eder/Science/TOI-4643/TESS/*/*lc.fits --planet_priors=/Users/eder/Science/TOI-4643/RV+TRANSITS_ANALYSIS/TOI-4643.pars -vp

    
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
    parser.add_option("-t", "--lcdata", dest="lcdata", help='Pattern for input light curve data',type='string',default="")
    parser.add_option("-o", "--object", dest="object", help='Object ID',type='string',default="")
    parser.add_option("-r", "--rvdata", dest="rvdata", help='Input radial velocity data file',type='string',default="")
    parser.add_option("-b", "--blongdata", dest="blongdata", help='Input B-longitudinal data file',type='string',default="")
    parser.add_option("-l", "--planet_priors", dest="planet_priors_file", help='Planet prior parameters file',type='string',default="")
    parser.add_option("-g", "--gp_priors", dest="gp_priors_file", help='QP GP prior parameters file',type='string',default="")
    parser.add_option("-c", "--calib_order", dest="calib_order", help='Order of calibration polynomial',type='string',default="1")
    parser.add_option("-n", "--nsteps", dest="nsteps", help="Number of MCMC steps",type='int',default=1000)
    parser.add_option("-w", "--walkers", dest="walkers", help="Number of MCMC walkers",type='int',default=32)
    parser.add_option("-i", "--burnin", dest="burnin", help="Number of MCMC burn-in samples",type='int',default=300)
    parser.add_option("-z", "--phot_binsize", dest="phot_binsize", help="Bin size of photometric data [days]",type='float',default=0.1)
    parser.add_option("-d", action="store_true", dest="detrend_rv_data", help="Detrend RV data with activity indices",default=False)
    parser.add_option("-j", action="store_true", dest="fit_rv_jitter", help="Fit RV jitter",default=False)
    parser.add_option("-O", action="store_true", dest="force_tess_pl_priors", help="Force using TESS planet priors",default=False)
    parser.add_option("-a", action="store_true", dest="fit_gp_activity", help="Run GP activity analysis",default=False)
    parser.add_option("-m", action="store_true", dest="run_mcmc", help="Run MCMC to fit parameters",default=False)
    parser.add_option("-G", action="store_true", dest="run_gp_mcmc", help="Run MCMC to fit GP parameters",default=False)
    parser.add_option("-f", action="store_true", dest="ols_fit", help="Perform OLS fit prior to MCMC", default=False)
    parser.add_option("-e", action="store_true", dest="mode", help="Best fit parameters obtained by the mode instead of median", default=False)
    parser.add_option("-p", action="store_true", dest="plot", help="plot",default=False)
    parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

    try:
        options,args = parser.parse_args(sys.argv[1:])
    except SystemExit as e :
        # allow clean exits from optparse (e.g. --help)
        if e.code == 0 or e.code is None :
            raise
        print("Error: check usage with transit_rv_fit -h "); sys.exit(1);

    if options.verbose:
        print('Object ID: ', options.object)
        print('Pattern for input light curve data: ', options.lcdata)
        print('Pattern for input radial velocity data file: ', options.rvdata)
        print('Input B-longitudinal data file: ', options.blongdata)
        print('Pattern for input planets prior files: ', options.planet_priors_file)
        print('QP GP prior parameters file: ', options.gp_priors_file)
        print('Order of calibration polynomial: ', options.calib_order)
        print('Number of MCMC steps: ', options.nsteps)
        print('Number of MCMC walkers: ', options.walkers)
        print('Number of MCMC burn-in samples: ', options.burnin)
        print('Bin size of photometric data [days]: ', options.phot_binsize)

    fix_rv_calib = True
    mcmc_phot_calib_priortype = "FIXED"

    plot_detrends = False
    save_clean_rvs = False

    min_npoints_per_bin = 30
    min_npoints_within_transit = 300
    transit_window_size = 2
    tess_times_in_bjd=True
    timelabel='TBJD'
    if tess_times_in_bjd :
        timelabel = 'BJD'
    #min_npoints_per_bin = 3
    #min_npoints_within_transit = 10
    #transit_window_size = 7

    # Download TESS DVT products and return a list of input data files
    dvt_filenames = tess.retrieve_tess_data_files(options.object, products_wanted_keys = ["DVT"], verbose=options.verbose)

    if options.verbose:
        print("Loading TESS lightcurves ...")

    # Load TESS data
    if options.lcdata != "" :
    
        inputlcdata = sorted(glob.glob(options.lcdata))
        
        tesslc = tess.load_lc(inputlcdata, object_name=options.object, transit_window_size=transit_window_size, min_npoints_within_transit=min_npoints_within_transit, binbymedian=False, binsize=options.phot_binsize, min_npoints_per_bin=min_npoints_per_bin,convert_times_to_bjd=tess_times_in_bjd, plot=options.plot, verbose=options.verbose)

    else :
        star_density, impact_parameter = False, False
        tesslc = tess.load_dvt_files(options.object, priors_dir=priors_dir, save_priors=options.force_tess_pl_priors, hasrvdata=True, use_star_density=star_density, use_impact_parameter=impact_parameter, plot=options.plot, verbose=options.verbose)
    
    #rvutils.periodogram(tesslc['time'], tesslc['nflux'], tesslc['nfluxerr'], period=0., nyquist_factor=20, probabilities = [0.01, 0.001], plot=True)

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
    rvdatalabels = ["Blue (2023)","Blue (2024)","Red (2023)","Red (2024)"]
    #rvdatalabels = ["SOPHIE data"]
    #rvdatalabels = ["SPIRou data"]
    #rvdatalabels = ["MAROON-X Red","MAROON-X Blue","SPIRou"]
    
    if options.blongdata != "" :
        if options.verbose:
            print("Loading B-long data ...")
        blong_data = ascii.read(options.blongdata)
        keep = (np.isfinite(blong_data["col2"])) & (np.isfinite(blong_data["col3"]))
        bbjds, blong, blongerr = blong_data["col1"][keep], blong_data["col2"][keep], blong_data["col3"][keep]
    else :
        bbjds, blong, blongerr = None, None, None

    if len(rvs) == 0 :
        print("Error: no input RV data file found using pattern:",options.rvdata)
        exit()
    
    calib_polyorder = int(options.calib_order)

    for i in range(len(tesslc["PLANETS"])) :
    
        planet = tesslc["PLANETS"][i]
    
        planet = tess.redefine_tessranges(planet, tesslc, options.planet_priors_file, transit_window_size=transit_window_size, verbose=True)
    
        # select data within certain ranges
        times, fluxes, fluxerrs = planet["times"], planet["fluxes"], planet["fluxerrs"]

        #Load priors information:
        priors = fitlib.read_transit_rv_priors(options.planet_priors_file, len(inputrvdata), len(times), calib_polyorder=calib_polyorder, verbose=options.verbose)

        # Set rv calibration prior type to Normal, but reset to "FIXED" if a trend or per target rvsys exists
        rv_prior_type = "Normal"
        if fix_rv_calib :
            for key in priors["labels"] :
                if "trend" in key or "rvsys" in key :
                    rv_prior_type = "FIXED"

        # Fit photometry calibration parameters and RV calibration parameters for initial guess
        posterior = fitlib.guess_calib_transit_rv(priors, times, fluxes, bjds, rvs, prior_type="FIXED", remove_transits=True, rv_prior_type=rv_prior_type, plot=False)
        
        # get number of planets in the system
        n_planets = int(posterior["planet_priors"]["n_planets"]['object'].value)
    
        # set rv prior to 'FIXED' if rvsys parameter is free.
        for planet_index in range(n_planets) :
            if posterior["planet_priors"]['rvsys_{0:03d}'.format(planet_index)]["type"] != 'FIXED' :
                rv_prior_type = 'FIXED'
            
        if options.plot :
            # plot light curves and models in priors
            fitlib.plot_mosaic_of_lightcurves(times, fluxes, fluxerrs, posterior)

        t0 = posterior["planet_params"]['tc_{0:03d}'.format(i)] + 0.5 * posterior["planet_params"]['per_{0:03d}'.format(i)]

        if options.plot :

            fitlib.plot_rv_global_timeseries(posterior["planet_params"], posterior["rvcalib_params"], bjds, rvs, rverrs, number_of_free_params=len(posterior["theta"]), samples=None, labels=None, nsamples=100, plot_residuals=True, rvdatalabels=rvdatalabels)
            n_planets = int(posterior["planet_params"]["n_planets"])
            for planet_index in range(n_planets) :
                fitlib.plot_rv_perplanet_timeseries(posterior["planet_params"], posterior["rvcalib_params"], bjds, rvs, rverrs, number_of_free_params=len(posterior["theta"]), planet_index=planet_index, samples=None, labels=None, nsamples=100, plot_residuals=True, rvdatalabels=rvdatalabels, bindata=True, phase_plot=True)


        if options.ols_fit :
            # OLS fit involving all priors
            posterior = fitlib.fitTransits_and_RVs_ols(times, fluxes, fluxerrs, bjds, rvs, rverrs, posterior, fix_eccentricity=True, calib_post_type="FIXED", rvcalib_post_type=rv_prior_type, calib_unc=0.01, verbose=False, plot=False)

        if options.detrend_rv_data :
            n_sigma_clip = 4
            #n_sigma_clip = 0
            # detrend RVs with activity indices
            bjds, rvs, rverrs = rvutils.detrend_rvs_with_activity_indices(posterior, bjds, rvs, rverrs,
                                                                  biss, sig_biss,
                                                                  fwhms, sig_fwhms,
                                                                  sindexs, sig_sindexs,
                                                                  has, sig_has,
                                                                  n_sigma_clip=n_sigma_clip, plot=plot_detrends)

        if options.fit_rv_jitter :
            # fit RV jitter
            print("Fitting RV jitter ... ")
            rverrs, jitter, jitter_err = rvutils.fit_RV_jitter(posterior, bjds, rvs, rverrs, rvdatalabels=rvdatalabels)
        
        if save_clean_rvs :
            for ii in range(len(inputrvdata)) :
                output = inputrvdata[ii].replace(".rdb","_clean.rdb")
                print("Saving output clean RVs to file:",output)
                rvutils.rv_and_models_time_series(posterior, bjds[ii], rvs[ii], rverrs[ii], dataset_index=ii, output=output)
                #rvutils.save_rv_time_series(output, bjds[ii], rvs[ii], rverrs[ii])
                
        if options.ols_fit :
            # OLS fit involving all priors
            posterior = fitlib.fitTransits_and_RVs_ols(times, fluxes, fluxerrs, bjds, rvs, rverrs, posterior, fix_eccentricity=True, calib_post_type=mcmc_phot_calib_priortype, rvcalib_post_type=rv_prior_type, calib_unc=0.01, verbose=False, plot=options.plot)

        print("Systemic RV: ", posterior["rvcalib_params"])
 
        if options.plot :
            fitlib.plot_rv_global_timeseries(posterior["planet_params"], posterior["rvcalib_params"], bjds, rvs, rverrs, number_of_free_params=len(posterior["theta"]), samples=None, labels=None, nsamples=100, plot_residuals=True, rvdatalabels=rvdatalabels)
            n_planets = int(posterior["planet_params"]["n_planets"])
            for planet_index in range(n_planets) :
                fitlib.plot_rv_perplanet_timeseries(posterior["planet_params"], posterior["rvcalib_params"], bjds, rvs, rverrs, number_of_free_params=len(posterior["theta"]), planet_index=planet_index, samples=None, labels=None, nsamples=100, plot_residuals=True, rvdatalabels=rvdatalabels, phase_plot=True, bindata=True, binsize=0.05)
    
        for i in range(len(inputrvdata)) :
            print("Dataset: {} -> RMS of RVs: {:.2f} m/s Median errors: {:.2f} m/s -> file: {}".format(i,np.nanstd(rvs[i]),np.nanmedian(rverrs[i]),inputrvdata[i]) )
  
        if options.run_mcmc :
            nwalkers = options.walkers
            niter = options.nsteps
            burnin = options.burnin
            amp = 1e-7
        
            samples_filename = priorslib.derive_filename(options.planet_priors_file, "_mcmc_samples.h5")
        
            posterior = fitlib.fitTransitsAndRVsWithMCMC(times, fluxes, fluxerrs, bjds, rvs, rverrs, posterior, amp=amp, nwalkers=nwalkers, niter=niter, burnin=burnin, verbose=True, plot=True, samples_filename=samples_filename, appendsamples=False, best_fit_from_mode=options.mode, rvlabel="SOPHIE RVs", timelabel=timelabel, plot_rv_bins=True, rvdatalabels=rvdatalabels)


        period_range,fit_periods = [1,100], []
        for j in range(n_planets) :
            planet_period_id = "{0}_{1:03d}".format('per', j)
            planet_period = posterior["planet_params"][planet_period_id]
            fit_periods.append(planet_period)
        
        # plot individual periodograms
        for j in range(n_planets) :
            rv_t, rv_res, rv_err = fitlib.get_rv_residuals(posterior, bjds, rvs, rverrs, selected_planet_index=j)
            if n_planets > 1 :
                rvutils.periodogram(rv_t, rv_res, rv_err, period=fit_periods[n_planets-j-1], nyquist_factor=20, probabilities = [0.001,0.0001,0.00001,0.000001], plot=options.plot)
            else :
                rvutils.periodogram(bjds[0], rvs[0], rverrs[0], period=fit_periods[n_planets-j-1], nyquist_factor=20, probabilities = [0.01,1e-10,1e-14], plot=options.plot, title=options.object)
    
        rvutils.periodgram_rv_check_signal(bjds[0], rvs[0], rverrs[0], period=fit_periods, period_labels=["fit orbit"],period_range=period_range, nbins=3, phase_plot=False)

        #############################
        # Start activity GP analysis
        #############################
    
        # If option below is turned on it gets very slow
        joint_gp_activity_and_orbit = True
        Blong_and_RV_constrained = False
    
        if options.fit_gp_activity :

            gp_priors = priorslib.read_priors(options.gp_priors_file)
            gp_priors["gp_priorsfile"] = options.gp_priors_file

            gp_rv, gp_blong, gp_phot, rv_gp_feed, blong_gp_feed, phot_gp_feed, gp_priors = fitlib.set_star_rotation_gp (posterior, gp_priors, tesslc, rv_t, rv_res, rv_err, bbjds, blong, blongerr, binsize=options.phot_binsize, gp_ls_fit=True, run_gp_mcmc=options.run_gp_mcmc, amp=amp, nwalkers=nwalkers, niter=niter, burnin=burnin, remove_transits=True, plot=True, verbose=True, spec_constrained=Blong_and_RV_constrained)

            # reduce gp activity model in phot time series
            red_fluxes, red_fluxerrs = fitlib.reduce_gp(gp_phot, times, fluxes, fluxerrs, phot_gp_feed)

            # reduce gp activity model in RV time series
            red_rvs, red_rverrs = fitlib.reduce_gp(gp_rv, bjds, rvs, rverrs, rv_gp_feed)

            # save RVs - GP to file
            for j in range(len(bjds)) :
                gpremovedRVsoutput = inputrvdata[0].replace(".rdb","_GPremoved.rdb")
                print("Saving GP-removed RVs to file: ",gpremovedRVsoutput)
                rvutils.save_rv_time_series(gpremovedRVsoutput, bjds[j], red_rvs[j], red_rverrs[j], time_in_rjd=True, rv_in_mps=False)
    
            rvutils.periodgram_rv_check_signal(bjds[0], red_rvs[0], red_rverrs[0], period=fit_periods, period_labels=["fit orbit"],period_range=period_range, nbins=3, phase_plot=options.plot)
    
            if joint_gp_activity_and_orbit :
                # OLS fit involving all priors
                posterior = fitlib.fitTransits_and_RVs_ols(times, red_fluxes, red_fluxerrs, bjds, red_rvs, red_rverrs, posterior, fix_eccentricity=True, flare_post_type="FIXED", calib_post_type="FIXED", rvcalib_post_type="FIXED", calib_unc=0.01, verbose=False, plot=True, rvlabel="SOPHIE RVs")
            
                samples_filename = priorslib.derive_filename(planet["prior_file"], "_mcmc_samples_gp.h5")
            
                posterior = fitlib.fitTransitsAndRVsWithMCMCAndGP(times, fluxes, fluxerrs, bjds, rvs, rverrs, bbjds, blong, blongerr, posterior, gp_priors, tesslc, phot_binsize=options.phot_binsize, amp=amp, nwalkers=nwalkers, niter=niter, burnin=burnin, samples_filename=samples_filename, verbose=True, plot=True, rvlabel="SOPHIE RVs", gp_spec_constrained=Blong_and_RV_constrained)

            else :
                # OLS fit involving all priors
                posterior = fitlib.fitTransits_and_RVs_ols(times, red_fluxes, red_fluxerrs, bjds, red_rvs, red_rverrs, posterior, fix_eccentricity=True, flare_post_type="FIXED", calib_post_type="FIXED", rvcalib_post_type="Normal", calib_unc=0.01, verbose=False, plot=True, rvlabel="SOPHIE RVs")

                for j in range(len(inputrvdata)) :
                    print("Dataset: {} -> RMS of RVs-GP: {:.2f} m/s Median errors: {:.2f} m/s -> file: {}".format(i,np.nanstd(red_rvs[j]),np.nanmedian(red_rverrs[j]),inputrvdata[j]) )

                samples_filename = priorslib.derive_filename(planet["prior_file"], "_mcmc_samples_notjointgp.h5")

                posterior = fitlib.fitTransitsAndRVsWithMCMC(times, red_fluxes, red_fluxerrs, bjds, red_rvs, red_rverrs, posterior, amp=amp, nwalkers=nwalkers, niter=niter, burnin=burnin, verbose=True, plot=True, samples_filename=samples_filename, appendsamples=False, rvlabel="SOPHIE RVs", addnparams=0)

                # plot photometry gp posterior
                fitlib.plot_gp_photometry(tesslc, posterior, gp_phot, phot_gp_feed["y"], options.phot_binsize)

                # plot RV gp posterior
                fitlib.plot_gp_rv(posterior, gp_rv, rv_gp_feed, bjds, rvs, rverrs, default_legend="SOPHIE RV")

                #t0 = posterior["planet_params"][i]['tc_{0:03d}'.format(i)] + 0.5 * posterior["planet_params"][i]['per_{0:03d}'.format(i)]
                t0 = posterior["planet_params"][i]['tc_{0:03d}'.format(i)]
            
                # RV phase plot:
                fitlib.plot_rv_timeseries(posterior["planet_params"], posterior["rvcalib_params"], bjds, red_rvs, red_rverrs, samples=None, labels=posterior["labels"], planet_index=0, plot_residuals=True, phasefold=True, plot_bin_data=True, rvlabel="SOPHIE RVs",t0=t0)


if __name__ == "__main__" :
    main()
