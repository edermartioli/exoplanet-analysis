"""
    Created on April 18 2022
    
    Description: This routine fits planetary transits data using MCMC and gp
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    python transit_fit_gp.py --input=TOI-3568_tess_lc.fits --epoch_time=1712.5551018 --fold_period=4.4199503 --transit_duration=2.232 --planet_priors=priors/TOI-3568.pars --calib_order=2 --nsteps=1000 --burnin=300 --gp_priors=priors/TOI-3568_gp_priors_phot.pars -vpmal

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
import fitlib, priorslib
import glob

import tess
import modelslib
import gp_lib

from copy import deepcopy
import astropy.io.fits as fits
from astropy.timeseries import LombScargle
from PyAstronomy.pyasl import foldAt

ExoplanetAnalysis_dir = os.path.dirname(__file__)
priors_dir = os.path.join(ExoplanetAnalysis_dir, 'priors/')
default_gp_priors_phot = os.path.join(priors_dir, 'default_gp_priors_phot.pars')

def reduce_lightcurve(time, flux, fluxerr, posterior, planet_index=0) :
    
    reduced_flux, reduced_fluxerr = deepcopy(flux), deepcopy(fluxerr)
    
    transit_model = np.full_like(time, 1.0)
    transit_model *= modelslib.batman_transit_model(time, posterior["planet_params"][planet_index], planet_index=planet_index)
    
    reduced_flux /= transit_model
    reduced_fluxerr /= transit_model

    return reduced_flux, reduced_fluxerr


def plot_light_curve(time, flux, fluxerr, posterior, planet_index=0, phase_plot=False) :

    tmin, tmax = np.min(time), np.max(time)
    highsamptime = np.arange(tmin, tmax, 0.0005)
    print(highsamptime, time)

    highsamp_transit_model = np.full_like(highsamptime, 1.0)
    highsamp_transit_model *= modelslib.batman_transit_model(highsamptime, posterior["planet_params"][planet_index], planet_index=planet_index)
    
    #transit_model = np.full_like(time, 1.0)
    #transit_model *= modelslib.batman_transit_model(time, posterior["planet_params"][planet_index], planet_index=planet_index)
    #plt.plot(time, transit_model, '+', color='darkblue', lw=1, zorder=2, label="Transit model")
    
    plt.errorbar(time, flux, yerr=fluxerr, fmt='.', alpha=0.3, color='tab:blue', lw=.7, zorder=0, label=r"TESS data")

    plt.plot(highsamptime, highsamp_transit_model, '-', color='darkred', lw=3, zorder=1, label="Transit model")
    plt.legend(fontsize=14)
    plt.xlabel(r"BTJD")
    plt.ylabel(r"Relative flux")
    plt.show()
    
    
    if phase_plot :
        #color = "#ff7f0e"
        
        t0 = posterior["planet_params"][0]["tc_000"]
        period = posterior["planet_params"][0]["per_000"]
        phases = foldAt(time, period, T0=t0+0.5*period)
        mphases = foldAt(highsamptime, period, T0=t0+0.5*period)
        sortIndi = np.argsort(phases)
        msortIndi = np.argsort(mphases)
        
        plt.errorbar(phases[sortIndi], flux[sortIndi], yerr=fluxerr[sortIndi], fmt='.', alpha=0.3, color='tab:blue', lw=.7, label=r"TESS data", zorder=0.5)

        #rbin_phases, rbin_flux, rbin_fluxerr = bin_data(red_phases, red_phased_flux, red_phased_fluxerr, median=False, binsize=binsize/period)
        #plt.errorbar(rbin_phases, rbin_flux, yerr=rbin_fluxerr, fmt=".k", alpha=0.5, label="TESS data binned by {0:.2f} d".format(binsize), zorder=1)

        plt.plot(mphases[msortIndi], highsamp_transit_model[msortIndi], '-', color="darkred", lw=3, zorder=1, label=r"Transit model")
    
        plt.ylabel(r"Relative flux", fontsize=16)
        plt.xlabel("phase (P={0:.3f} d)".format(period), fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()



def periodogram(x, y, yerr, period=0., nyquist_factor=20, probabilities = [0.01, 0.001], y_label="y", check_period=0, check_period_label="", npeaks=1, phaseplot=False, plot=False) :
    """
        Description: calculate GLS periodogram
        """
    
    ls = LombScargle(x, y, yerr)

    frequency, power = ls.autopower(nyquist_factor=nyquist_factor)

    fap = ls.false_alarm_level(probabilities)
    
    if period == 0 :
        sorted = np.argsort(power)
        #best_frequency = frequency[np.argmax(power)]
        best_frequencies = frequency[sorted][-npeaks:]
        best_powers = power[sorted][-npeaks:]
        period = 1./best_frequencies
    else :
        best_frequencies = [1./period]
        best_powers = [np.nanmax(power)]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    if npeaks > 10 :
        print("ERROR: npeaks must be up to 10, exiting ... ")
        exit()

    periods = 1/frequency

    if plot :
        plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=16)    # fontsize of the tick labels

        plt.plot(periods, power, color="k", zorder=1)
        for i in range(len(best_frequencies)) :
            best_frequency, best_power = best_frequencies[i], best_powers[i]
            plt.vlines(1/best_frequency, np.min(power), best_power, ls="--", color=colors[i], label="Max power at P={0:.1f} d".format(1/best_frequency), zorder=2)
            #plt.hlines(best_power, np.min(periods), 1/best_frequency, ls="--", color=colors[i], zorder=2)

        if check_period :
            plt.vlines(check_period, np.min(power), np.max(power), color="red",ls="--", label="{} at P={:.1f} d".format(check_period_label,check_period))

        for i in range(len(fap)) :
            plt.text(np.min(periods),fap[i]*1.01,r"FAP={0:.3f}%".format(100*probabilities[i]),horizontalalignment='left', fontsize=26)
            plt.hlines([fap[i]], np.min(periods), np.max(periods),ls=":", lw=0.5)

        plt.xscale('log')
        plt.xlabel("Period [d]", fontsize=26)
        plt.ylabel("Power", fontsize=26)
        plt.legend(fontsize=26)
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.show()
    else :
        for i in range(len(best_frequencies)) :
            best_frequency, best_power = best_frequencies[i], best_powers[i]

    phases = foldAt(x, 1/best_frequency, T0=x[0])
    sortIndi = np.argsort(phases)

    if plot and phaseplot:
        plt.errorbar(phases[sortIndi],y[sortIndi],yerr=yerr[sortIndi],fmt='o', color="k")
        plt.ylabel(r"{}".format(y_label), fontsize=26)
        plt.xlabel("phase (P={0:.1f} d)".format(1/best_frequency), fontsize=26)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=20)
        plt.show()

    loc = {}
    if npeaks == 1 :
        loc['best_frequency'] = best_frequency
        loc['period'] = 1 / best_frequency
    else :
        loc['best_frequency'] = best_frequencies
        loc['period'] = 1 / best_frequencies
    loc['power'] = power
    loc['frequency'] = frequency
    loc['phases'] = phases
    loc['fap'] = fap
    loc['probabilities'] = probabilities
    loc['nyquist_factor'] = nyquist_factor

    return loc


def phase_plot(x, y, yerr, gp, fold_period, ylabel="y", t0=0, alpha=0.7, timesampling=0.001) :
    
    if t0 == 0:
        t0 = np.nanmin(x)

    phases, epochs = foldAt(x, fold_period, T0=t0, getEpoch=True)
    sortIndi = np.argsort(phases)
    min_epoch, max_epoch = int(np.nanmin(epochs)), int(np.nanmax(epochs))


    ti, tf = np.min(x), np.max(x)
    time = np.arange(ti, tf, timesampling)
    mphases = foldAt(time, fold_period, T0=t0)
    msortIndi = np.argsort(mphases)
    pred_mean, pred_var = gp.predict(y, time, return_var=True)
    pred_std = np.sqrt(pred_var)

    color = "#ff7f0e"
    plt.plot(mphases[msortIndi], pred_mean[msortIndi], "-", color=color, lw=2, alpha=0.5, label="GP model")
    #plt.fill_between(mphases[msortIndi], pred_mean[msortIndi]+pred_std[msortIndi], pred_mean[msortIndi]-pred_std[msortIndi], color=color, alpha=0.3, edgecolor="none")
    
    for ep in range(min_epoch, max_epoch+1) :
        inepoch = epochs[sortIndi] == ep
        if len(phases[sortIndi][inepoch]) :
            plt.errorbar(phases[sortIndi][inepoch],y[sortIndi][inepoch],yerr=yerr[sortIndi][inepoch], fmt='o', alpha=alpha, label="Cycle {}".format(ep))

    plt.ylabel(r"{}".format(ylabel), fontsize=16)
    plt.xlabel("phase (P={0:.3f} d)".format(fold_period), fontsize=16)
    plt.legend()
    plt.show()


def run_gp_analysis(time, flux, fluxerr, gp_priors_file="", run_gls=False,  run_mcmc=True, walkers=32, nsteps=1000, burnin=300, amp=1e-4, best_fit_from_mode=True, plot_distributions=False, verbose=False, plot=False) :

    t0 = np.nanmin(time)
    if verbose :
        print("T0 = {:.6f} BTJD".format(t0))

    if gp_priors_file == "" :
        gp_priors_file = default_gp_priors_phot
    gp_posterior = gp_priors_file.replace(".pars", "_posterior.pars")
    output_pairsplot = gp_priors_file.replace(".pars", "_posterior_pairsplot.png")

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
        gls = periodogram(time, flux, fluxerr, nyquist_factor=0.02, probabilities = [0.00001], npeaks=1, y_label="Relative flux", plot=plot, phaseplot=plot)
    
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

    # Run GP on B-long data with a QP kernel
    gp = gp_lib.star_rotation_gp(time, flux, fluxerr, period=best_period, period_lim=period_lim, fix_period=fix_period, amplitude=amplitude, amplitude_lim=amplitude_lim, fix_amplitude=fix_amplitude, decaytime=decaytime, decaytime_lim=decaytime_lim, fix_decaytime=fix_decaytime, smoothfactor=smoothfactor, smoothfactor_lim=smoothfactor_lim, fix_smoothfactor=fix_smoothfactor, fixpars_before_fit=True, fit_mean=fit_mean, fit_white_noise=fit_white_noise, period_label=r"Prot [d]", amplitude_label=r"$\alpha$", decaytime_label=r"$l$ [d]", smoothfactor_label=r"$\beta$", mean_label=r"$\mu$", white_noise_label=r"$\sigma$", output_pairsplot=output_pairsplot, run_mcmc=run_mcmc, amp=amp, nwalkers=walkers, niter=nsteps, burnin=burnin, x_label="BJD", y_label=ylabel, output=gp_posterior, best_fit_from_mode = best_fit_from_mode, plot_distributions = plot_distributions, plot=plot, verbose=verbose)

    gp_params = gp_lib.get_star_rotation_gp_params(gp)
    best_period = gp_params["period"]

    if plot :
        phase_plot(time, flux, fluxerr, gp, best_period, ylabel=ylabel, t0=t0, alpha=1)

    gp_feed = {}
    gp_feed["t"], gp_feed["y"], gp_feed["yerr"] = time, flux, fluxerr

    return gp, gp_feed



parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help='Input light curve file',type='string',default="")
parser.add_option("-r", "--planet_priors", dest="planet_priors", help='Planet prior parameters file',type='string',default="")
parser.add_option("-g", "--gp_priors", dest="gp_priors", help='GP prior parameters file',type='string',default="")
parser.add_option("-e", "--epoch_time", dest="epoch_time", help='Epoch time (BJD)',type='float',default=0)
parser.add_option("-f", "--fold_period", dest="fold_period", help='Fold period (d)',type='float',default=0)
parser.add_option("-u", "--transit_duration", dest="transit_duration", help='Transit duration (hr)',type='float',default=0)
parser.add_option("-c", "--calib_order", dest="calib_order", help='Order of calibration polynomial',type='string',default="1")
parser.add_option("-n", "--nsteps", dest="nsteps", help="Number of MCMC steps",type='int',default=1000)
parser.add_option("-w", "--walkers", dest="walkers", help="Number of MCMC walkers",type='int',default=32)
parser.add_option("-b", "--burnin", dest="burnin", help="Number of MCMC burn-in samples",type='int',default=300)
parser.add_option("-s", "--samples_filename", dest="samples_filename", help='MCMC samples filename',type='string',default="")
parser.add_option("-l", action="store_true", dest="ols_fit", help="Perform OLS fit prior to MCMC", default=False)
parser.add_option("-m", action="store_true", dest="run_mcmc", help="Run MCMC fit", default=False)
parser.add_option("-a", action="store_true", dest="run_gp", help="Run GP analysis", default=False)
parser.add_option("-d", action="store_true", dest="mode", help="Best fit parameters obtained by the mode instead of median", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="verbose",default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with transit_fit_gp.py -h "); sys.exit(1);

if options.verbose:
    print('Input light curve file: ', options.input)
    print('Planet prior parameters file: ', options.planet_priors)
    print('Order of calibration polynomial: ', options.calib_order)
    print('Number of MCMC steps: ', options.nsteps)
    print('Number of MCMC walkers: ', options.walkers)
    print('Number of MCMC burn-in samples: ', options.burnin)
    print('MCMC samples filenames: ', options.samples_filename)
    print('Fold period (d): ', options.fold_period)
    print('Epoch time (BJD): ', options.epoch_time)
    print('Transit duration (hr): ', options.transit_duration)

if options.verbose:
    print("Loading TESS lightcurves ...")
# Load TESS data
hdu = fits.open(options.input)
time = np.array(hdu['LIGHTCURVE'].data["TIME"], dtype=float)
flux = np.array(hdu['LIGHTCURVE'].data["FLUX"], dtype=float)
fluxerr = np.array(hdu['LIGHTCURVE'].data["FLUX_ERR"], dtype=float)

tmin, tmax = np.min(time), np.max(time)

tepoch, tperiod, tduration = options.epoch_time, options.fold_period, options.transit_duration

tcs = tess.calculate_tcs_within_range(tmin, tmax, tepoch, tperiod)
if options.verbose :
    print("tmin={} tmax={} tepoch={} tperiod={} d tduration= {} hr".format(tmin, tmax, tepoch, tperiod, tduration))
    print("N_transits={} 1st_transit={} last_transit={} BTJD".format(len(tcs),tcs[0],tcs[-1]))
    
tdur_days = tduration/24.
transit_window_size = 4.
min_npoints_within_transit = 10
# initialize mask with all False
times, fluxes, fluxerrs = [], [], []
print("tcs=",tcs)
for tc in tcs :
    in_transit = (time > tc - tdur_days * transit_window_size) & (time < tc + tdur_days * transit_window_size)
    if len(time[in_transit]) >= min_npoints_within_transit :
        times.append(time[in_transit])
        fluxes.append(flux[in_transit])
        fluxerrs.append(fluxerr[in_transit])

#Load priors information:
planet_priors_files = sorted(glob.glob(options.planet_priors))

calib_polyorder = int(options.calib_order)

# read priors from input files
priors = fitlib.read_priors(planet_priors_files, len(times), calib_polyorder=calib_polyorder, verbose=False)

# Fit calibration parameters for initial guess
posterior = fitlib.guess_calib(priors, times, fluxes, prior_type="Normal")

if options.plot :
    # plot global light curve
    plot_light_curve(time, flux, fluxerr, posterior)
    # plot light curves and models in priors
    fitlib.plot_mosaic_of_lightcurves(times, fluxes, fluxerrs, posterior)

if options.ols_fit :
    # OLS fit involving all priors
    posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, posterior, calib_post_type="Normal", calib_unc=0.01, verbose=False, plot=False)
    # OLS fit involving all priors
    posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, posterior, calib_post_type="FIXED", verbose=False, plot=False)

    if options.plot :
        fitlib.plot_mosaic_of_lightcurves(times, fluxes, fluxerrs, posterior)

if options.run_mcmc :
    # Make sure the number of walkers is sufficient, and if not assing a new value
    if options.walkers < 2*len(posterior["theta"]):
        print("WARNING: insufficient number of MCMC walkers, resetting nwalkers={}".format(2*len(posterior["theta"])))
        options.walkers = 2*len(posterior["theta"])

    if options.samples_filename == "" :
        options.samples_filename = planet_priors_files[0].replace(".pars","_mcmc_samples.h5")

    posterior = fitlib.fitTransitsWithMCMC(times, fluxes, fluxerrs, posterior, amp=1e-5, nwalkers=options.walkers, niter=options.nsteps, burnin=options.burnin, verbose=True, plot=True, samples_filename=options.samples_filename, appendsamples=False, plot_individual_transits=True)

if options.plot :
    # plot global light curve
    plot_light_curve(time, flux, fluxerr, posterior)
    # plot light curves and models in priors
    fitlib.plot_mosaic_of_lightcurves(times, fluxes, fluxerrs, posterior)

reduced_flux, reduced_fluxerr = reduce_lightcurve(time, flux, fluxerr, posterior)

if options.run_gp :

    gp_phot, phot_gp_feed = run_gp_analysis(time, reduced_flux, reduced_fluxerr, gp_priors_file=options.gp_priors, walkers=32, nsteps=500, burnin=100, plot_distributions=False, verbose=True, plot=True)

    reduced_flux, reduced_fluxerr = fitlib.reduce_gp(gp_phot, time, flux, fluxerr, phot_gp_feed, subtract=False)

    if options.plot :
        # plot global light curve
        plot_light_curve(time, reduced_flux, reduced_fluxerr, posterior, phase_plot=True)
        

