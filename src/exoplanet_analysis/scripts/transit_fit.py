"""
    Created on Feb 14 2022
    
    Description: This routine fits planetary transits data using MCMC
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    transit_fit --object="HATS-24" -vp
    
    transit_fit --object="TIC287256467" --lcdata=/Volumes/Samsung_T5/Science/TOI-2141/TESS/*lc.fits -vp --nsteps=5000 --burnin=1000 -e
    
    transit_fit --object="TOI-1736" --lcdata=/Volumes/Samsung_T5/Science/TOI-1736/TESS/*lc.fits -vp -bd

    transit_fit --object="TOI-1736" --lcdata=/Volumes/Samsung_T5/Science/TOI-1736/TESS/*lc.fits -n 100 -u 30 -vpa --gp_priors=/Volumes/Samsung_T5/Science/TOI-1736/TOI-1736_gpphot.pars
    
    transit_fit --object="TOI-1736" --lcdata=/Volumes/Samsung_T5/Science/TOI-1736/TESS/*lc.fits -n 100 -u 30 -vp --output_binned_lc=/Volumes/Samsung_T5/Science/TOI-1736/TOI-1736_tess_phot.txt


    transit_fit --object="TOI-1736" --lcdata=/Volumes/Samsung_T5/Science/TOI-1736/TESS/*lc.fits -n 100 -u 30  --gp_priors=/Volumes/Samsung_T5/Science/TOI-1736/TRANSIT_ONLY_ANALYSIS/TOI-1736_gpphot.pars --output_binned_lc=/Volumes/Samsung_T5/Science/TOI-1736/TRANSIT_ONLY_ANALYSIS/TOI-1736_tess_phot.txt --planet_priors=/Volumes/Samsung_T5/Science/TOI-1736/TRANSIT_ONLY_ANALYSIS/TOI-1736_transits_only.pars -vpaq
    
    transit_fit --object="TOI-2141" --lcdata=/Volumes/Samsung_T5/Science/TOI-2141/TESS/*lc.fits -n 100 -u 30 --gp_priors=/Volumes/Samsung_T5/Science/TOI-2141/TRANSIT_ONLY_ANALYSIS/TOI-2141_gpphot.pars --output_binned_lc=/Volumes/Samsung_T5/Science/TOI-2141/TRANSIT_ONLY_ANALYSIS/TOI-2141_tess_phot.txt --planet_priors=/Volumes/Samsung_T5/Science/TOI-2141/TRANSIT_ONLY_ANALYSIS/TOI-2141_transits_only.pars -vpaq

    transit_fit --object="TOI-3568" --lcdata=/Volumes/Samsung_T5/Science/TOI-3568/TESS/*lc.fits -n 100 -u 30 -vpa --gp_priors=/Volumes/Samsung_T5/Science/TOI-3568/TRANSIT_ONLY_ANALYSIS/TOI-3568_gpphot.pars --output_binned_lc=/Volumes/Samsung_T5/Science/TOI-3568/TRANSIT_ONLY_ANALYSIS/TOI-3568_tess_phot.txt --planet_priors=/Volumes/Samsung_T5/Science/TOI-3568/TRANSIT_ONLY_ANALYSIS/TOI-3568_transit_only.pars --morelcdata=/Volumes/Samsung_T5/Science/TOI-3568/FFI_EXTRACTION/TOI-3568_ffi_lc.fits --binsize=0.03 --calib_order=2

    """
    

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os, sys

from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from exoplanet_analysis import fitlib, priorslib
import glob

from copy import deepcopy
from exoplanet_analysis import exoplanetlib, gp_lib
from exoplanet_analysis import timeseries_lib as ts
from exoplanet_analysis import tess
from exoplanet_analysis import rvutils

from exoplanet_analysis.config import priors_dir
default_gp_priors_phot = os.path.join(priors_dir, 'default_gp_priors_phot.pars')

def run_gp_analysis(time, flux, fluxerr, gp_priors_file="", run_gls=False,  run_mcmc=True, walkers=32, nsteps=1000, burnin=300, amp=1e-4, best_fit_from_mode=True, useQPKernel=True, plot_distributions=False, verbose=False, plot=False) :

    """Run gp analysis.

    Parameters
    ----------
    time
        Array of times [BJD].
    flux
        Array of fluxes.
    fluxerr
        Array of flux uncertainties.
    gp_priors_file : str, optional (default: "")
    run_gls : bool, optional (default: False)
    run_mcmc : bool, optional (default: True)
    walkers : int, optional (default: 32)
    nsteps : int, optional (default: 1000)
    burnin : int, optional (default: 300)
        Number of MCMC burn-in steps discarded from the chain.
    amp : float, optional (default: 1e-4)
        Amplitude of the initial ball of walkers around the starting point.
    best_fit_from_mode : bool, optional (default: True)
        Use the mode instead of the median of the posterior distributions.
    useQPKernel : bool, optional (default: True)
    plot_distributions : bool, optional (default: False)
    verbose : bool, optional (default: False)
        Print progress information.
    plot : bool, optional (default: False)
        Show diagnostic plots.
    """
    t0 = np.nanmin(time)
    if verbose :
        print("T0 = {:.6f} BTJD".format(t0))

    if gp_priors_file == "" :
        gp_priors_file = default_gp_priors_phot
    gp_posterior = priorslib.derive_filename(gp_priors_file, "_posterior.pars")
    output_pairsplot = priorslib.derive_filename(gp_priors_file, "_posterior_pairsplot.png")

    # Load gp parameters priors
    gp_priors = priorslib.read_priors(gp_priors_file)
    gp_params = priorslib.read_phot_starrot_gp_params(gp_priors)
    param_lim, param_fixed = {}, {}
    # print out gp priors
    print("----------------")
    print("Input GP parameters:")
    for key in gp_params.keys() :
        if ("_err" not in key) and ("_pdf" not in key) :
            param_fixed[key] = False
            pdf_key = "{0}_pdf".format(key)
            if gp_params[pdf_key] == "FIXED" :
                print("{0} = {1} ({2})".format(key, gp_params[key], gp_params[pdf_key]))
                param_lim[key] = (gp_params[key],gp_params[key])
                param_fixed[key] = True
            elif gp_params[pdf_key] == "Uniform" or gp_params[pdf_key] == "Jeffreys":
                error_key = "{0}_err".format(key)
                min = gp_params[error_key][0]
                max = gp_params[error_key][1]
                param_lim[key] = (min,max)
                print("{0} <= {1} ({2}) <= {3} ({4})".format(min, key, gp_params[key], max, gp_params[pdf_key]))
            elif gp_params[pdf_key] == "Normal" or gp_params[pdf_key] == "Normal_positive":
                error_key = "{0}_err".format(key)
                error = gp_params[error_key][1]
                param_lim[key] = (gp_params[key]-5*error,gp_params[key]+5*error)
                print("{0} = {1} +/- {2} ({3})".format(key, gp_params[key], error, gp_params[pdf_key]))
    print("----------------")

    ##########################################
    ### ANALYSIS of input data
    ##########################################
    ylabel = r"Flux"
    if run_gls :
        gls = rvutils.periodogram(time, flux, fluxerr, nyquist_factor=10, probabilities = [0.00001], plot=plot)
    
        best_period = gls['period']
        if verbose :
            print("GLS periodogram highest peak at P={:.3f} d".format(best_period))
            
        if param_fixed['period'] :
            best_period = gp_params["period"]
    else :
        best_period = gp_params["period"]

    amplitude = gp_params["amplitude"]
    decaytime = gp_params["decaytime"]
    smoothfactor = gp_params["smoothfactor"]

    fit_mean = True
    if param_fixed["mean"] :
        fit_mean = False

    fit_white_noise = True
    if param_fixed["white_noise"] :
        fit_white_noise = False

    fix_period = param_fixed["period"]
    period_lim = param_lim["period"]

    fix_amplitude = param_fixed["amplitude"]
    amplitude_lim = param_lim["amplitude"]

    fix_decaytime = param_fixed["decaytime"]
    decaytime_lim = param_lim["decaytime"]

    fix_smoothfactor = param_fixed["smoothfactor"]
    smoothfactor_lim = param_lim["smoothfactor"]

    if useQPKernel :
        # Run GP on B-long data with a QP kernel
        gp = gp_lib.star_rotation_gp(time, flux, fluxerr, period=best_period, period_lim=period_lim, fix_period=fix_period, amplitude=amplitude, amplitude_lim=amplitude_lim, fix_amplitude=fix_amplitude, decaytime=decaytime, decaytime_lim=decaytime_lim, fix_decaytime=fix_decaytime, smoothfactor=smoothfactor, smoothfactor_lim=smoothfactor_lim, fix_smoothfactor=fix_smoothfactor, fixpars_before_fit=True, fit_mean=fit_mean, fit_white_noise=fit_white_noise, period_label=r"Prot [d]", amplitude_label=r"$\alpha$", decaytime_label=r"$l$ [d]", smoothfactor_label=r"$\beta$", mean_label=r"$\mu$", white_noise_label=r"$\sigma$", output_pairsplot=output_pairsplot, run_mcmc=run_mcmc, amp=amp, nwalkers=walkers, niter=nsteps, burnin=burnin, x_label="BJD", y_label=ylabel, output=gp_posterior, best_fit_from_mode = best_fit_from_mode, plot_distributions = plot_distributions, plot=plot, verbose=verbose)

        gp_params = gp_lib.get_star_rotation_gp_params(gp)
        best_period = gp_params["period"]

        #if plot :
        #    phase_plot(time, flux, fluxerr, gp, best_period, ylabel=ylabel, t0=t0, alpha=1)
    else :
        #gp = gp_lib.ConstantKernel_gp(time, flux, fluxerr, amplitude=amplitude, amplitude_lim=amplitude_lim, fix_amplitude=fix_amplitude, fixpars_before_fit=True, fit_mean=fit_mean, fit_white_noise=fit_white_noise, amplitude_label=r"$\alpha$", mean_label=r"$\mu$", white_noise_label=r"$\sigma$", x_label="BJD", y_label=ylabel, plot=plot, verbose=verbose)
        
        gp = gp_lib.ExpSquaredKernel_gp(time, flux, fluxerr, run_optimization=True, amplitude=amplitude, amplitude_lim=amplitude_lim, fix_amplitude=fix_amplitude, decaytime=decaytime, decaytime_lim=decaytime_lim, fix_decaytime=fix_decaytime, fixpars_before_fit=True, fit_mean=fit_mean, fit_white_noise=fit_white_noise, amplitude_label=r"$\alpha$", decaytime_label=r"$l$ [d]", mean_label=r"$\mu$", white_noise_label=r"$\sigma$", x_label="BJD", y_label=ylabel, plot=plot, verbose=verbose)
        
        #gp_params = gp_lib.get_ConstantKernel_gp_params(gp)

        gp_params = gp_lib.get_ExpSquaredKernel_gp_params(gp)
        
    gp_feed = {}
    gp_feed["t"], gp_feed["y"], gp_feed["yerr"] = time, flux, fluxerr

    return gp, gp_feed


def reduce_transits_lc(posterior, time, flux, fluxerr) :

    """
    Description: function to remove transits from TESS light curve data given posteriors
    """
    planet_params = posterior["planet_params"]

    # First remove transits from all planets:
    redflux, redfluxerr = deepcopy(flux), deepcopy(fluxerr)
    
    transit_models = np.full_like(time, 1.0)
    ti,tf =  time[0], time[-1]
    #highsamptime = np.arange(ti, tf, 0.0005)
    #highsamp_transit_model = np.full_like(highsamptime, 1.0)

    for j in range(int(posterior["n_planets"])) :
        planet_transit_id = "{0}_{1:03d}".format('transit', j)
        if planet_params[planet_transit_id] :
            transit_models *= exoplanetlib.batman_transit_model(time, planet_params, planet_index=j)
            #highsamp_transit_model *= exoplanetlib.batman_transit_model(highsamptime, planet_params, planet_index=j)

    # Remove transits before running gp
    redflux /= transit_models
    redfluxerr /= transit_models

    return time, redflux, redfluxerr, transit_models


def save_time_series(output, time, flux, fluxerr) :
    
    """Save time series.

    Parameters
    ----------
    output
        Output file path.
    time
        Array of times [BJD].
    flux
        Array of fluxes.
    fluxerr
        Array of flux uncertainties.
    """
    outfile = open(output,"w+")
    #outfile.write("btjd\tflux\tfluxerr\n")
    #outfile.write("---\t----\t-----\n")
    
    for i in range(len(time)) :
        outfile.write("{0:.10f}\t{1:.5f}\t{2:.5f}\n".format(time[i], flux[i], fluxerr[i]))

    outfile.close()


def plot_timeseries_with_gp(bin_time, bin_flux, bin_fluxerr, posterior, gp_phot, phot_gp_feed, tesslc) :

    """Plot timeseries with gp.

    Parameters
    ----------
    bin_time
    bin_flux
    bin_fluxerr
    posterior
        Posterior dictionary as returned by the fitting routines.
    gp_phot
    phot_gp_feed
    tesslc
    """
    planet_params = posterior["planet_params"]
    bin_transit_models = np.full_like(bin_time, 1.0)
    for j in range(int(posterior["n_planets"])) :
        planet_transit_id = "{0}_{1:03d}".format('transit', j)
        if planet_params[planet_transit_id] :
            bin_transit_models *= exoplanetlib.batman_transit_model(bin_time, planet_params, planet_index=j)

    plt.plot(bin_time, bin_flux)
    plt.show()

    gp_phot.compute(phot_gp_feed["t"], phot_gp_feed["yerr"])
    yout, yerrout = deepcopy(tesslc['nflux']), deepcopy(tesslc['nfluxerr'])
    pred_mean, pred_var = gp_phot.predict(phot_gp_feed['y'], tesslc['time'], return_var=True)
    
    plt.plot(tesslc['time'], yout, '.')
    plt.plot(phot_gp_feed["t"], pred_mean, '-')
    plt.show()
    
    #pred_mean -= 1
    pred_std = np.sqrt(pred_var)
    yout /= pred_mean
    yerrout /= pred_mean

    color = "darkred"
    meanflux, meanrms = 0, 0
    
    for i in range(len(tesslc['dataset_ranges'])) :
    
        trange = tesslc['dataset_ranges'][i]
        ti,tf = trange[0], trange[1]
    
        time, flux = tesslc['time'], tesslc['nflux']
        
        keep = (time>=ti) & (time<=tf)
        
        plt.plot(time[keep], pred_mean[keep]*transit_model[keep], "-", lw=2, color=color, zorder=3)
        plt.fill_between(time[keep], (pred_mean*transit_model+pred_std)[keep], (pred_mean*transit_model-pred_std)[keep], color=color, alpha=0.3, edgecolor="none", zorder=3)

        keep_bin = (bin_time>=ti) & (bin_time<=tf)

        plt.errorbar(bin_time[keep_bin], bin_flux[keep_bin]*bin_transit_models[keep_bin], yerr=bin_fluxerr[keep_bin]*bin_transit_models[keep_bin], fmt='.', color='k', alpha=0.75, zorder=2)
        
        plt.plot(time[keep], flux[keep],'.',color='grey', alpha=0.1, zorder=1)
        
        residuals = (flux[keep] - pred_mean[keep]*transit_model[keep])
        rms = np.nanstd(residuals)
        number_of_free_parameters = len(posterior['theta']) + 5 # transit + GP parameters
        chi2 = np.nansum((residuals/tesslc['nfluxerr'][keep])**2) / (len(residuals) - number_of_free_parameters)

        print("Range {}/{} Ti:{:.8f} Tf:{:.8f} rms:{:.8f} chi-sqr:{:.3f}".format(i+1,len(tesslc['dataset_ranges']),ti,tf,rms,chi2))
        
        meanflux += np.nanmedian(bin_flux[keep_bin]) / len(tesslc['dataset_ranges'])
        meanrms += rms / len(tesslc['dataset_ranges'])

    residuals = (flux - pred_mean*transit_model)
    rms = np.nanstd(residuals)
    number_of_free_parameters = len(posterior['theta']) + 5 # transit + GP parameters
    chi2 = np.nansum((residuals/tesslc['nfluxerr'])**2) / (len(residuals) - number_of_free_parameters)
    print("Global Ti:{:.8f} Tf:{:.8f} rms:{:.8f} chi-sqr:{:.3f}".format(tesslc['time'][0],tesslc['time'][-1],rms,chi2))

    nsig = 4
    
    for planet in tesslc["PLANETS"] :
        obstcs, obstcs_y = planet["selectedtcs"], np.full_like(planet["selectedtcs"],meanflux-2.5*meanrms)
        tcs, tcs_y = planet["tcs"], np.full_like(planet["tcs"],meanflux-nsig*meanrms)
        p = plt.plot(tcs, tcs_y, '^', alpha=0.8, markersize=12, color=color, zorder=1)
        current_color = p[0].get_color()
        #current_color = color
        #plt.plot(obstcs, obstcs_y, '^', markersize=12, color=current_color, zorder=3)
        #plt.vlines(tcs, minflux, maxflux, color="k", ls=":", lw=0.85, zorder=2)
    
    plt.xlabel("time [BTJD]", fontsize=26)
    plt.ylabel("rel. flux", fontsize=26)
    #plt.legend(fontsize=18)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.show()


def main() :

    """Main.
    """
    parser = OptionParser()
    parser.add_option("-i", "--lcdata", dest="lcdata", help='Pattern for input light curve data',type='string',default="")
    parser.add_option("-f", "--morelcdata", dest="morelcdata", help='Input an additional light curve data',type='string',default="")
    parser.add_option("-g", "--gp_priors", dest="gp_priors", help='GP prior parameters file',type='string',default="")
    parser.add_option("-o", "--object", dest="object", help='Object ID',type='string',default="")
    parser.add_option("-t", "--sector", dest="sector", help='Select TESS sector',type='int',default=0)
    parser.add_option("-c", "--calib_order", dest="calib_order", help='Order of calibration polynomial',type='string',default="1")
    parser.add_option("-r", "--planet_priors", dest="planet_priors", help='Planet priors file name',type='string',default="")
    parser.add_option("-n", "--nsteps", dest="nsteps", help="Number of MCMC steps",type='int',default=1000)
    parser.add_option("-w", "--walkers", dest="walkers", help="Number of MCMC walkers",type='int',default=32)
    parser.add_option("-u", "--burnin", dest="burnin", help="Number of MCMC burn-in samples",type='int',default=300)
    parser.add_option("-s", "--samples_filename", dest="samples_filename", help='MCMC samples filename',type='string',default="")
    parser.add_option("-z", "--binsize", dest="binsize", help="Light curve binsize [d]",type='float',default=0.1)
    parser.add_option("-1", "--output_binned_lc", dest="output_binned_lc", help='Output binned reduced lightcurve',type='string',default="")
    parser.add_option("-2", "--output_residuals_lc", dest="output_residuals_lc", help='Output residuals lightcurve',type='string',default="")
    parser.add_option("-b", action="store_true", dest="impact_parameter", help="Use impact parameter instead of inclination", default=False)
    parser.add_option("-d", action="store_true", dest="star_density", help="Use star density instead of a/Rs", default=False)
    parser.add_option("-l", action="store_true", dest="ols_fit", help="Perform OLS fit prior to MCMC", default=False)
    parser.add_option("-a", action="store_true", dest="fit_gp_activity", help="Run GP activity analysis",default=False)
    parser.add_option("-q", action="store_true", dest="use_qp_kernel", help="Use quasi-periodic GP kernel",default=False)
    parser.add_option("-e", action="store_true", dest="mode", help="Best fit parameters obtained by the mode instead of median", default=False)
    parser.add_option("-p", action="store_true", dest="plot", help="verbose",default=False)
    parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

    try:
        options,args = parser.parse_args(sys.argv[1:])
    except SystemExit as e :
        # allow clean exits from optparse (e.g. --help)
        if e.code == 0 or e.code is None :
            raise
        print("Error: check usage with transit_fit -h "); sys.exit(1);

    if options.verbose:
        if options.lcdata != '' :
            print('Pattern for input light curve data: ', options.lcdata)
        print('Object ID: ', options.object)
        if options.sector :
            print('TESS sector selected: ', options.sector)
        print('Planet prior parameters file: ', options.planet_priors)
        print('GP prior parameters file: ', options.gp_priors)
        print('Order of calibration polynomial: ', options.calib_order)
        print('Number of MCMC steps: ', options.nsteps)
        print('Number of MCMC walkers: ', options.walkers)
        print('Number of MCMC burn-in samples: ', options.burnin)
        print('MCMC samples filenames: ', options.samples_filename)
        print('Light curve binsize [d]: ', options.binsize)
        print('Output binned reduced lightcurve: ', options.output_binned_lc)

    #min_npoints_per_bin = 30
    #min_npoints_within_transit = 300
    #transit_window_size = 3

    min_npoints_per_bin = 3
    min_npoints_within_transit = 10
    transit_window_size = 2

    #min_npoints_per_bin = 3
    #min_npoints_within_transit = 10
    #transit_window_size = 7

    amp=1e-6

    if options.verbose:
        print("Loading TESS lightcurves ...")
    
    # Load TESS data
    if options.lcdata != "" :
        inputlcdata = sorted(glob.glob(options.lcdata))
        # load TESS data from input lc files
        tesslc = tess.load_lc(inputlcdata, more_lc_data=options.morelcdata, object_name=options.object, transit_window_size=transit_window_size, min_npoints_within_transit=min_npoints_within_transit, binbymedian=False, binsize=options.binsize, min_npoints_per_bin=min_npoints_per_bin, plot=options.plot, verbose=options.verbose, convert_times_to_bjd=True)
    else :
        # Download TESS DVT products and return a list of input data files
        dvt_filenames = tess.retrieve_tess_data_files(options.object, sector=options.sector, products_wanted_keys = ["DVT"], verbose=options.verbose)

        # load TESS data from dvt files
        tesslc = tess.load_dvt_files(options.object, priors_dir=priors_dir, save_priors=True, use_star_density=options.star_density, use_impact_parameter=options.impact_parameter, plot=options.plot, verbose=options.verbose)

    posterior = None
    planet_index=0
        
    for planet in tesslc["PLANETS"] :
        #Load priors information:
    
        if options.planet_priors != "" :
            tesslc["PRIOR_FILE"] = options.planet_priors
            planet = tess.redefine_tessranges(planet, tesslc, tesslc["PRIOR_FILE"], planet_index=planet_index, transit_window_size=transit_window_size, verbose=True)
        
        planet_priors_file = tesslc["PRIOR_FILE"]

        # select data within certain ranges
        times, fluxes, fluxerrs = planet["times"], planet["fluxes"], planet["fluxerrs"]

        calib_polyorder = int(options.calib_order)

        # read priors from input files
        #priors = fitlib.read_priors(planet_priors_file, len(times), calib_polyorder=calib_polyorder, verbose=False)
        priors = fitlib.read_transit_rv_priors(planet_priors_file, 0, len(times), planet_index=planet_index, calib_polyorder=calib_polyorder,verbose=False)
    
        # Fit calibration parameters for initial guess
        if options.ols_fit :
            posterior = fitlib.guess_calib(priors, times, fluxes, prior_type="Normal")
        else :
            posterior = fitlib.guess_calib(priors, times, fluxes, prior_type="FIXED")

        if options.plot :
            # plot light curves and models in priors
            fitlib.plot_mosaic_of_lightcurves(times, fluxes, fluxerrs, posterior)

        if options.ols_fit :
            # OLS fit involving all priors
            posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, posterior, calib_post_type="Normal", calib_unc=0.01, verbose=False, plot=False)

        if options.fit_gp_activity :
    
            # Remove transits from light curve
            reduc_time, reduc_flux, reduc_fluxerr, transit_model = reduce_transits_lc(posterior, tesslc['time'], tesslc['nflux'], tesslc['nfluxerr'])

            bin_time, bin_flux, bin_fluxerr  = fitlib.bin_data(reduc_time, reduc_flux, reduc_fluxerr, median=False, binsize=options.binsize, min_npoints=min_npoints_per_bin)

            # save binned lightcurve
            if options.output_binned_lc != "" :
                save_time_series(options.output_binned_lc, bin_time, bin_flux, bin_fluxerr)
            
            # GLS periodogram
            #ts.periodogram(bin_time, bin_flux, bin_fluxerr, period=0, nyquist_factor=20, probabilities = [0.01, 0.001], y_label="power", check_period=31, npeaks=1, phaseplot=False, plot=True, plot_frequencies=False)
            # Bayesian GLS periodogram
            #per, powr = ts.bgls(bin_time, bin_flux, bin_fluxerr, plow=16, phigh=46, ofac=1000)
            #ts.plot_bgls(per, powr, period=0, npeaks=1, y_label='power', phaseplot=False)

            gp_phot, phot_gp_feed = run_gp_analysis(bin_time, bin_flux, bin_fluxerr, gp_priors_file=options.gp_priors, walkers=32, nsteps=3000, burnin=1000, amp=1e-6, run_gls=False, run_mcmc=True, useQPKernel=options.use_qp_kernel, plot_distributions=False, verbose=True, plot=True)

            if options.plot :
                plot_timeseries_with_gp(bin_time, bin_flux, bin_fluxerr, posterior, gp_phot, phot_gp_feed, tesslc)
    
            # detrend fluxes using GP solution
            for i in range(len(times)) :
                fluxes[i], fluxerrs[i] = fitlib.reduce_gp(gp_phot, times[i], fluxes[i], fluxerrs[i], phot_gp_feed, subtract=False)

            if options.output_residuals_lc != "" :
                tesslc['nflux'], tesslc['nfluxerr'] = fitlib.reduce_gp(gp_phot, tesslc['time'], tesslc['nflux'], tesslc['nfluxerr'], phot_gp_feed, subtract=False)

            if options.ols_fit :
                # OLS fit again after GP detrending
                posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, posterior, calib_post_type="Normal", calib_unc=0.01, verbose=False, plot=False)
            
        if options.ols_fit :
            # OLS fit involving all priors
            posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, posterior, calib_post_type="FIXED", verbose=False, plot=False)
            if options.plot :
                fitlib.plot_mosaic_of_lightcurves(times, fluxes, fluxerrs, posterior)
            
        # Make sure the number of walkers is sufficient, and if not assing a new value
        if options.walkers < 2*len(posterior["theta"]):
            print("WARNING: insufficient number of MCMC walkers, resetting nwalkers={}".format(2*len(posterior["theta"])))
            options.walkers = 2*len(posterior["theta"])

        if options.samples_filename == "" :
            options.samples_filename = priorslib.derive_filename(planet_priors_file, "_mcmc_samples.h5")
    
        # Final fit with MCMC
        posterior = fitlib.fitTransitsWithMCMC(times, fluxes, fluxerrs, posterior, amp=amp, nwalkers=options.walkers, niter=options.nsteps, burnin=options.burnin, verbose=True, plot=True, samples_filename=options.samples_filename, best_fit_from_mode=options.mode, appendsamples=False, plot_individual_transits=False)

        planet_index += 1
    
        if options.output_residuals_lc != "" :
            ## save residuals time series
            time, residuals, errors, transit_model = reduce_transits_lc(posterior, tesslc['time'], tesslc['nflux'], tesslc['nfluxerr'])
            
            plt.errorbar(time, residuals, yerr=errors, fmt='.', color='k', alpha=0.8, zorder=2)
            plt.plot(time, transit_model, '-', color='g', alpha=0.5, zorder=3)
            plt.xlabel("time [BTJD]", fontsize=26)
            plt.ylabel("rel. flux", fontsize=26)
            #plt.legend(fontsize=18)
            plt.xticks(fontsize=26)
            plt.yticks(fontsize=26)
            plt.show()
        
            save_time_series(options.output_residuals_lc, time, residuals, errors)


if __name__ == "__main__" :
    main()
