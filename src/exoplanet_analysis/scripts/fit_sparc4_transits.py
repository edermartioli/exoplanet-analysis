"""
    Created on Aug 12 2024
    
    Description: This routine fits multiband photometry SPARC4 data of planetary transits using MCMC
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    python -W"ignore" fit_sparc4_transits.py --object="CoRoT-2" --input1="/Users/eder/Science/Transits-OPD_2024A/CoRot-2/20240620/20240620_s4c1_CoRot-2_S_lc.fits" --input2="/Users/eder/Science/Transits-OPD_2024A/CoRot-2/20240620/20240620_s4c2_CoRot-2_S_lc.fits" --input3="/Users/eder/Science/Transits-OPD_2024A/CoRot-2/20240620/20240620_s4c3_CoRot-2_S_lc.fits" --input4="/Users/eder/Science/Transits-OPD_2024A/CoRot-2/20240620/20240620_s4c4_CoRot-2_S_lc.fits" --instrument_priors="/Users/eder/Science/Transits-OPD_2024A/CoRot-2/20240620/corot-2_instrument.pars" --planet_priors=/Users/eder/Science/Transits-OPD_2024A/CoRot-2/20240620/COROT-2.pars --binsize=0.005 --calib_order=3 --targets="0|3|3|4" --comps="1,2,4,5,6,7|0,1,2,5|2,5,6,8|2,3,6,8" --niter=2 --nsteps=10000 --burnin=3000 -yemp
    
    
    python -W"ignore" fit_sparc4_transits.py --object="WASP-78" --input1="//Users/eder/Science/Transits-OPD_2024A/WASP-78/20231107_s4c1_WASP-78b_POLAR_L2_S+N_lc.fits" --input2="/Users/eder/Science/Transits-OPD_2024A/WASP-78/20231107_s4c2_WASP-78b_POLAR_L2_S+N_lc.fits" --input3="/Users/eder/Science/Transits-OPD_2024A/WASP-78/20231107_s4c3_WASP-78b_POLAR_L2_S+N_lc.fits" --input4="/Users/eder/Science/Transits-OPD_2024A/WASP-78/20231107_s4c4_WASP-78b_POLAR_L2_S+N_lc.fits" --catalog="CATALOG_PHOT_AP012,CATALOG_PHOT_AP012,CATALOG_PHOT_AP012,CATALOG_PHOT_AP012" --planet_priors=/Users/eder/Science/Transits-OPD_2024A/WASP-78/WASP-78.pars --binsize=0.005 --calib_order=3 --targets="0|2|2|2" --comps="1,2,3,4,5|0,1,3,4|0,1,3,4,5|0,1,3,4,5" --niter=2 --nsteps=1000 --burnin=300 -yempj
    

    python -W"ignore" fit_sparc4_transits.py --object="WASP-4" --input1="/Users/eder/Science/Transits-OPD_2024A/WASP-4/20240910/20240910_s4c1_wasp4_S_lc.fits" --input2="/Users/eder/Science/Transits-OPD_2024A/WASP-4/20240910/20240910_s4c2_wasp4_S_lc.fits" --input3="/Users/eder/Science/Transits-OPD_2024A/WASP-4/20240910/20240910_s4c3_wasp4_S_lc.fits" --input4="/Users/eder/Science/Transits-OPD_2024A/WASP-4/20240910/20240910_s4c4_wasp4_S_lc.fits" --planet_priors=/Users/eder/Science/Transits-OPD_2024A/WASP-4/20240910/WASP-4.pars --binsize=0.01 --calib_order=1 --targets="1|1|1|1" --comps="2,3|2,3|2,3|2,3" --niter=2 --nsteps=1000 --burnin=300 -yemp

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
import glob

from exoplanet_analysis import fitlib, priorslib

from copy import deepcopy

from scipy import optimize
from exoplanet_analysis import exoplanetlib
from scipy import stats
from scipy import interpolate

from exoplanet_analysis.config import priors_dir

from astropy.io import fits
from uncertainties import ufloat,umath

from astropy.table import Table

from exoplanet_analysis import rvutils, gp_lib
from exoplanet_analysis import timeseries_lib as tslib

from exoplanet_analysis import tess

def get_opddata(tsfile, target=0, comps=[], extname="BEST_APERTURES", t0=0, normalize_fratio=False) :

    """Get opddata.

    Parameters
    ----------
    tsfile
    target : int, optional (default: 0)
    comps : list, optional (default: [])
    extname : str, optional (default: "BEST_APERTURES")
    t0 : int, optional (default: 0)
    normalize_fratio : bool, optional (default: False)
    """
    times, fluxes, fluxerrs = [], [], []
    
    hdul = fits.open(tsfile)

    tbl = hdul[extname].data
    
    targettbl = tbl[tbl['SRCINDEX']==target]
    bjd = targettbl['TIME'] - t0
    
    target_uflux = []
    for i in range(len(targettbl['MAG'])) :
        umag = ufloat(targettbl['MAG'][i],targettbl['EMAG'][i])
        tuflux = 10 ** (- 0.4 * umag)
        target_uflux.append(tuflux)
    
    # read comparisons flux data
    for j in range(len(comps)) :
    
        comptbl = tbl[tbl['SRCINDEX']==comps[j]]
        
        comp_mag = comptbl["MAG"]
        comp_magerr = comptbl["EMAG"]
        
        fratio, fratioerr = np.array([]),np.array([])
        
        for i in range(len(comp_mag)) :
            umag = ufloat(comp_mag[i],comp_magerr[i])
            uflux = 10 ** (- 0.4 * umag)
            
            fr = target_uflux[i] / uflux
            
            fratio = np.append(fratio,fr.nominal_value)
            fratioerr = np.append(fratioerr,fr.std_dev)
            
        mfratio = 1.0
        if normalize_fratio :
            mfratio = np.nanmedian(fratio)
        fluxes.append(fratio/mfratio)
        fluxerrs.append(fratioerr/mfratio)
        times.append(bjd)
        
    return times, fluxes, fluxerrs
    
    
def get_wppos(tsfile, target=0, comps=[], extname="BEST_APERTURES", verbose=False) :

    """Get wppos.

    Parameters
    ----------
    tsfile
    target : int, optional (default: 0)
    comps : list, optional (default: [])
    extname : str, optional (default: "BEST_APERTURES")
    verbose : bool, optional (default: False)
        Print progress information.
    """
    wppositions = []
    
    hdul = fits.open(tsfile)
    tbl = hdul[extname].data

    targettbl = tbl[tbl['SRCINDEX']==target]
    
    try :
        # read comparisons flux data
        for j in range(len(comps)) :
            comptbl = tbl[tbl['SRCINDEX'] == comps[j]]
            comp_wppos = comptbl['WPPOS']
            wppositions.append(comp_wppos)
    except :
        if verbose :
            print("WARNING: could not find 'WPPOS' column, exiting ... ")
        pass
        
    return wppositions


def clean_data(times, fluxes, fluxerrs, posterior, n_sigma_clip=5, plot=False, verbose=False) :

    """Clean data.

    Parameters
    ----------
    times
        List of time arrays [BJD], one per dataset.
    fluxes
        List of flux arrays, one per dataset.
    fluxerrs
        List of flux uncertainty arrays, one per dataset.
    posterior
        Posterior dictionary as returned by the fitting routines.
    n_sigma_clip : int, optional (default: 5)
    plot : bool, optional (default: False)
        Show diagnostic plots.
    verbose : bool, optional (default: False)
        Print progress information.
    """
    planet_params = posterior["planet_params"]
    instrum_params = posterior["instrum_params"]
    instrument_indexes = posterior["instrument_indexes"]
    
    out_fluxes, out_fluxerrs = [], []

    for i in range(len(times)) :
    
        transit_models = np.full_like(times[i], 1.0)

        instrum_index = 0
        if instrum_params is not None and instrument_indexes is not None:
            instrum_index = instrument_indexes[i]
            
        for j in range(int(posterior["n_planets"])) :
            planet_transit_id = "{0}_{1:03d}".format('transit', j)
            if planet_params[planet_transit_id] :
                transit_models *= exoplanetlib.batman_transit_model(times[i], planet_params, planet_index=j, instrum_params=instrum_params, instrum_index=instrum_index)
                #plt.plot(times[i], fluxes[i], 'r.')
                #plt.plot(times[i], transit_models, 'g-', lw=2)
                #plt.show()

        keep = np.isfinite(fluxes[i]) *  np.isfinite(fluxerrs[i])
        
        flux_without_transit = fluxes[i] / transit_models

        median = np.nanmedian(flux_without_transit)
        mad = stats.median_abs_deviation(flux_without_transit[keep], scale="normal")

        if verbose :
            print("Comparison {} -> sigma = {:.10f}".format(i, mad))
        
        keep &= np.abs(flux_without_transit - median) < n_sigma_clip * mad
        keep &= fluxerrs[i] < 3 * mad
        
        if plot :
            plt.plot(times[i], flux_without_transit, 'g.')
            plt.plot(times[i][~keep], flux_without_transit[~keep], 'ro', alpha=0.3)
            plt.plot(times[i], np.full_like(times[i], median), 'b-')
            plt.plot(times[i], np.full_like(times[i], median + n_sigma_clip * mad), 'b:')
            plt.plot(times[i], np.full_like(times[i], median - n_sigma_clip * mad), 'b:')
            plt.show()

        out_flux, out_fluxerr = np.full_like(times[i],np.nan),np.full_like(times[i],np.nan)
        out_flux[keep] = fluxes[i][keep]
        out_fluxerr[keep] = fluxerrs[i][keep]

        out_fluxes.append(out_flux)
        out_fluxerrs.append(out_fluxerr)

    return times, out_fluxes, out_fluxerrs



def diff_light_curve(times, fluxes, fluxerrs, posterior, nsig=100, model_time_sampling=0.001, offset_nsig=10, combine_by_median=False, binsize = 0.005, output="", instrum_index=0, use_calibs=True, plot_comps=False, plot=False, verbose=False):
    """
        Detrended differential light curve
    """
    
    font = {'size': 16}
    matplotlib.rc('font', **font)
    
    planet_params = posterior["planet_params"]
    instrum_params = posterior["instrum_params"]
    instrument_indexes = posterior["instrument_indexes"]
    theta, labels = posterior["theta"], posterior["labels"]
    
    if verbose:
        # print out best fit parameters and errors
        print("----------------")
        print("Planet parameters:")
        for key in planet_params.keys() :
            if not key.endswith("_err") and not key.endswith("_pdf") :
                print(key, "=", planet_params[key])
        print("----------------")

    nstars = len(times)
    
    time = times[0]
    fluxes = np.array(fluxes)
    fluxerrs = np.array(fluxerrs)
    
    mintime, maxtime = time[0], time[-1]
        
    # Definir o array de tempos do modelo.
    t = np.arange(mintime, maxtime, model_time_sampling)
    transit_models = np.full_like(t, 1.0)
    obs_transit_models = np.full_like(time, 1.0)
    

    for j in range(int(posterior["n_planets"])) :
        planet_transit_id = "{0}_{1:03d}".format('transit', j)
        if planet_params[planet_transit_id] :
            transit_models *= exoplanetlib.batman_transit_model(t, planet_params, planet_index=j)
            obs_transit_models *= exoplanetlib.batman_transit_model(time, planet_params, planet_index=j, instrum_params=instrum_params, instrum_index=instrum_index)
            
    if plot_comps :
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=False, gridspec_kw={'hspace': 0, 'height_ratios': [2, 1]})
        axs0 = axs[0]
    else :
        fig, axs0 = plt.subplots(1, 1, figsize=(12, 6), sharex=True, sharey=False, gridspec_kw={'hspace': 0})
            
    if plot :
        axs0.plot(t, (transit_models-1.0)*100, "g-", lw=3, zorder=2, label="Transit model")

    lcs = fluxes
    elcs = fluxerrs
    
    if  use_calibs :
        calibs = []
        for i in range(nstars):
            calib = exoplanetlib.calib_model(nstars, i, posterior['calib_params'], times[i])
            calibs.append(calib)
        calibs = np.array(calibs)
    
        lcs /= calibs
        elcs /= calibs

    target_median_fluxes = np.nanmedian(fluxes, axis=1)
    delta_mags = -2.5 * np.log10(target_median_fluxes)

    target_median_lcs = np.nanmedian(lcs/obs_transit_models, axis=1)
    
    for i in range(nstars):
        mlc =  np.nanmedian(lcs[i]/obs_transit_models)
        lcs[i] /= mlc
        elcs[i] /= mlc
        
    offset = 0

    lc_mean = np.average(lcs, axis=0, weights=1./(elcs*elcs))
    elc_mean = np.nanstd(lcs-lc_mean, axis=0)
    
    if combine_by_median :
        lc_mean = np.nanmedian(lcs, axis=0)
        elc_mean = np.nanmedian(np.abs(lcs-lc_mean), axis=0) / 0.67449

    #for i in range(len(lc_mean)) :
    #    lc_mean[i], elc_mean[i] = fitlib.odd_ratio_mean(lcs[:,i], elcs[:,i])

    mlc = np.nanmedian(lc_mean/obs_transit_models)
    rms = np.nanmedian(np.abs(lc_mean/obs_transit_models-mlc)) / 0.67449
    
    keep = elc_mean/obs_transit_models < nsig*rms
    keep &= np.abs(1. - lc_mean/obs_transit_models) < nsig*rms

    bin_time, bin_flux, bin_fluxerr = fitlib.bin_data(time[keep], lc_mean[keep], elc_mean[keep], median=False, binsize=binsize)

    if plot :
        bin_transit_models = np.full_like(bin_time, 1.0)
        for j in range(int(posterior["n_planets"])) :
            planet_transit_id = "{0}_{1:03d}".format('transit', j)
            if planet_params[planet_transit_id] :
                bin_transit_models *= exoplanetlib.batman_transit_model(bin_time, planet_params, planet_index=j, instrum_params=instrum_params, instrum_index=instrum_index)

        axs0.errorbar(time[keep], (lc_mean[keep]-1.0)*100, yerr=elc_mean[keep]*100, fmt='o', color="grey", alpha=0.15, label=r"Master: $\sigma$={:.2f}%".format(rms*100),zorder=1)
        
        bin_mlc = np.nanmedian(bin_flux/bin_transit_models)
        binrms = np.nanmedian(np.abs(bin_flux/bin_transit_models-bin_mlc)) / 0.67449
            
        axs0.errorbar(bin_time, (bin_flux-1.0)*100, yerr=bin_fluxerr*100, fmt='o', color="k", alpha=0.8, zorder=2, label=r"Master binned by {:.2f}h: $\sigma$={:.3f}%".format(binsize*24.0,binrms*100))
        
        axs0.set_ylabel(r"relative flux (%)", fontsize=20)
        #axs0.legend(fontsize=16)
        axs0.tick_params(axis='x', labelsize=14)
        axs0.tick_params(axis='y', labelsize=14)
        axs0.minorticks_on()
        axs0.tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
        axs0.tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
    
        if plot_comps :
            offset = np.nanpercentile(lc_mean, 1.0) - offset_nsig*rms

            axs[1].hlines(0, mintime, maxtime, colors='k', linestyle='--', lw=2, zorder=2)
            #plt.hlines(-rms, mintime, maxtime, colors='k', linestyle='--', lw=0.5,zorder=2)
            #plt.hlines(+rms, mintime, maxtime, colors='k', linestyle='--', lw=0.5,zorder=2)

            for i in range(nstars):
                mlc = np.nanmedian(lcs[i]/obs_transit_models)
                rms = np.nanmedian(np.abs(lcs[i]/obs_transit_models-mlc)) / 0.67449

                keep = elcs[i]/obs_transit_models < nsig*rms
                keep &= np.abs(1. - lcs[i]/obs_transit_models) < nsig*rms

                comp_label = "C{:03d}".format(i)
                axs[1].errorbar(time[keep], (lcs[i][keep]/obs_transit_models[keep]-mlc)*100, yerr=(elcs[i][keep]/obs_transit_models[keep])*100, fmt='.', alpha=0.1, label=r"{} $\Delta$mag={:.3f} $\sigma$={:.2f} %".format(comp_label, delta_mags[i], rms*100),zorder=1)

            axs[1].tick_params(axis='x', labelsize=14)
            axs[1].tick_params(axis='y', labelsize=14)
            axs[1].minorticks_on()
            axs[1].tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
            axs[1].tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
            #axs[1].set_ylabel(r"fluxo (%)", fontsize=20)
            axs[1].set_xlabel("time (BJD)", fontsize=20)
        else :
            axs0.set_xlabel("time (BJD)", fontsize=20)
            
        #plt.xlabel(r"time (BJD)", fontsize=16)
        #plt.ylabel(r"$\Delta$mag", fontsize=16)
        #plt.xlabel(r"time (BTJD)", fontsize=16)
        #plt.ylabel(r"Flux ratio", fontsize=16)
        #plt.legend(loc='lower left',fontsize=10)
        plt.show()

    tbl = Table()
    tbl["TIME"] = time
    tbl["TRANSIT_MODEL"] = obs_transit_models
    tbl["FLUX"] = lc_mean
    tbl["FLUXERR"] = elc_mean

    if output != "" :
        header = fits.Header()
        
        for key in planet_params.keys() :
            if not key.endswith("_err") and not key.endswith("_pdf") :
                header.set(key, planet_params[key])

        primary_hdu = fits.PrimaryHDU(header=header)
        hdu_time = fits.ImageHDU(data=time, name="TIME")
        hdu_mag = fits.ImageHDU(data=lc_mean, name="FLUX")
        hdu_magerr = fits.ImageHDU(data=elc_mean, name="FLUXERR")
        hdu_bintime = fits.ImageHDU(data=bin_time, name="BIN_TIME")
        hdu_binmag = fits.ImageHDU(data=bin_flux, name="BIN_FLUX")
        hdu_binmagerr = fits.ImageHDU(data=bin_fluxerr, name="BIN_FLUXERR")

        listofhuds = [primary_hdu, hdu_time, hdu_mag, hdu_magerr, hdu_bintime, hdu_binmag, hdu_binmagerr]
        mef_hdu = fits.HDUList(listofhuds)
        mef_hdu.writeto(options.output_lc, overwrite=True)

    return tbl
        

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
        gls = rvutils.periodogram(time, flux, fluxerr, nyquist_factor=10, probabilities = [0.1,0.0001], plot=plot)
    
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

def errfunc(coeffs, t, flux, fluxerr) :
    """Errfunc.

    Parameters
    ----------
    coeffs
    t
        Array of times.
    flux
        Array of fluxes.
    fluxerr
        Array of flux uncertainties.
    """
    p = np.poly1d(coeffs)
    flux_model = p(t)
    residuals = (flux - flux_model) / fluxerr
    return residuals
    
def remove_systematics(times, fluxes, fluxerrs, posterior,  calib_polyorder=1, binsize=0.005, plot=False, verbose=False) :

    """Remove systematics.

    Parameters
    ----------
    times
        List of time arrays [BJD], one per dataset.
    fluxes
        List of flux arrays, one per dataset.
    fluxerrs
        List of flux uncertainty arrays, one per dataset.
    posterior
        Posterior dictionary as returned by the fitting routines.
    calib_polyorder : int, optional (default: 1)
        Order of the photometric calibration polynomial.
    binsize : float, optional (default: 0.005)
        Bin size in time units [d].
    plot : bool, optional (default: False)
        Show diagnostic plots.
    verbose : bool, optional (default: False)
        Print progress information.
    """
    planet_params = posterior["planet_params"]
    instrum_params = posterior["instrum_params"]
    instrument_indexes = posterior["instrument_indexes"]
    
    out_fluxes, out_fluxerrs = [], []

    for i in range(len(times)) :

        mtime = np.nanmedian(times[i])
        mflux = np.nanmedian(fluxes[i])

        transit_models = np.full_like(times[i], 1.0)
        
        instrum_index = 0
        if instrum_params is not None and instrument_indexes is not None:
            instrum_index = instrument_indexes[i]
        
        for j in range(int(posterior["n_planets"])) :
            planet_transit_id = "{0}_{1:03d}".format('transit', j)
            if planet_params[planet_transit_id] :
                transit_models *= exoplanetlib.batman_transit_model(times[i], planet_params, planet_index=j, instrum_params=instrum_params, instrum_index=instrum_index)
                #plt.plot(times[i], fluxes[i], '.', color="grey",alpha=0.2)
                #plt.plot(times[i], transit_models, 'g-', lw=2)
                #plt.show()

        keep = np.isfinite(fluxes[i]) *  np.isfinite(fluxerrs[i])
        
        flux_without_transit = fluxes[i] / transit_models
        fluxerr_without_transit = fluxerrs[i] / transit_models

        bin_time, bin_flux, bin_fluxerr = fitlib.bin_data(times[i][keep]-mtime, flux_without_transit[keep]-mflux, fluxerr_without_transit[keep], median=False, binsize=binsize)

        init_coeffs = np.array([])
        for j in range(calib_polyorder) :
            init_coeffs = np.append(init_coeffs,-0.1)
        init_coeffs[-1] = np.nanmean(flux_without_transit)

        if len(bin_time) > len(init_coeffs) + 2 :
            coeffs, success = optimize.leastsq(errfunc, init_coeffs, args=(bin_time, bin_flux, bin_fluxerr))
        else :
            coeffs = init_coeffs
            
        if plot :
            inst_label = ""
            if instrument_indexes is not None:
                inst_label = instrument_indexes[i]
            plt.title("Systematics for comparison star C{:03d} Instrum. idx: {}".format(i,inst_label), fontsize=18)
            plt.plot(times[i], flux_without_transit, 'g.',alpha=0.2)
            plt.errorbar(bin_time + mtime, bin_flux+mflux, yerr=bin_fluxerr, fmt='ko')
            plt.plot(times[i], np.poly1d(coeffs)(times[i]-mtime)+mflux, 'b-')
            plt.xlabel(r"time (BTJD)", fontsize=16)
            plt.ylabel(r"Flux ratio / transit model", fontsize=16)
            plt.show()

        out_flux, out_fluxerr = np.full_like(times[i],np.nan),np.full_like(times[i],np.nan)
        out_flux[keep] = (fluxes[i][keep] / (np.poly1d(coeffs)(times[i][keep]-mtime)+mflux)) * mflux
        out_fluxerr[keep] = (fluxerrs[i][keep] / (np.poly1d(coeffs)(times[i][keep]-mtime)+mflux)) * mflux

        out_fluxes.append(out_flux)
        out_fluxerrs.append(out_fluxerr)

    return out_fluxes, out_fluxerrs


def remove_waveplate_modulation_with_gp(times, fluxes, fluxerrs, wppositions, posterior, gp_priors_file, useQPKernel=True, run_mcmc_on_gp=False, combine_by_median=False, plot=True, verbose=False) :

    # get planet parameters from posterior
    """Remove waveplate modulation with gp.

    Parameters
    ----------
    times
        List of time arrays [BJD], one per dataset.
    fluxes
        List of flux arrays, one per dataset.
    fluxerrs
        List of flux uncertainty arrays, one per dataset.
    wppositions
    posterior
        Posterior dictionary as returned by the fitting routines.
    gp_priors_file
    useQPKernel : bool, optional (default: True)
    run_mcmc_on_gp : bool, optional (default: False)
    combine_by_median : bool, optional (default: False)
    plot : bool, optional (default: True)
        Show diagnostic plots.
    verbose : bool, optional (default: False)
        Print progress information.
    """
    planet_params = posterior["planet_params"]
    instrum_params = posterior["instrum_params"]
    instrument_indexes = posterior["instrument_indexes"]
    
    # initialize output data containers
    out_fluxes, out_fluxerrs = [], []

    # initialize temporary data containers
    fluxes_tmp, fluxerrs_tmp = [], []
    mtransits = []
    mfluxes = []
    
    ref_time = times[0]
    median_delta_time = np.nanmedian(np.abs(ref_time[1:] - ref_time[:-1]))
    max_delta_time = median_delta_time * 16
    
    # find out sequences from data for first comparison
    wppos = wppositions[0]
    seq_index = np.full_like(wppos,np.nan)
    nseq = 0
    seq_index[0] = nseq
    for i in range(1,len(ref_time)) :
        delta_wppos = np.abs(wppos[i] - wppos[i-1])
        delta_time = np.abs(ref_time[i] - ref_time[i-1])
        # condition to consider a change in waveplate position
        if delta_wppos != 0 or delta_time > max_delta_time :
            nseq += 1
        seq_index[i] = nseq
    nseq += 1
    
    # loop over each lightcurve
    for i in range(len(times)) :
    
        # calculate transit models
        transit_models = np.full_like(times[i], 1.0)
        
        instrum_index = 0
        if instrum_params is not None and instrument_indexes is not None:
            instrum_index = instrument_indexes[i]
            
        for j in range(int(posterior["n_planets"])) :
            planet_transit_id = "{0}_{1:03d}".format('transit', j)
            if planet_params[planet_transit_id] :
                transit_models *= exoplanetlib.batman_transit_model(times[i], planet_params, planet_index=j, instrum_params=instrum_params, instrum_index=instrum_index)
        # store transit models to return it back to the data later
        mtransits.append(transit_models)
        
        # remove transits
        flux_without_transit = fluxes[i] / transit_models
        fluxerr_without_transit = fluxerrs[i] / transit_models
        
        # calculate mean flux without transit
        mflux = np.nanmedian(flux_without_transit)
        # store mean flux
        mfluxes.append(mflux)
        
        #if plot :
            #plt.plot(times[i], flux_without_transit / mflux, ".", alpha=0.2)
            
        # store temporary fluxes with both transits and mean values removed
        fluxes_tmp.append(flux_without_transit / mflux)
        fluxerrs_tmp.append(fluxerr_without_transit / mflux)
   
    # cast to numpy arrays
    fluxes_tmp = np.array(fluxes_tmp)
    fluxerrs_tmp = np.array(fluxerrs_tmp)

    seq_time, seq_lc, seq_elc = [], [], []
    for k in range(nseq) :
        keep = seq_index == k
        mean_time = np.nanmean(ref_time[keep])
        
        flat_fluxes_tmp = fluxes_tmp[:,keep].flatten()
        flat_fluxerrs_tmp = fluxerrs_tmp[:,keep].flatten()
        
        # calculate mean from all comparison stars
        lc_mean = np.average(flat_fluxes_tmp, weights=1./(flat_fluxerrs_tmp*flat_fluxerrs_tmp))
        elc_mean = np.nanstd(flat_fluxes_tmp-lc_mean)
        if combine_by_median :
            lc_mean = np.nanmedian(flat_fluxes_tmp, axis=0)
            elc_mean = np.nanmedian(np.abs(flat_fluxes_tmp-lc_mean), axis=0) / 0.67449
            
        seq_time.append(mean_time)
        seq_lc.append(lc_mean)
        seq_elc.append(elc_mean)

    seq_time = np.array(seq_time)
    seq_lc = np.array(seq_lc)
    seq_elc = np.array(seq_elc)

    # calculate statistics before gp correction to check efficiency
    sig_before_gp, mad_before_gp = np.nanstd(seq_lc), np.nanmedian(np.abs(seq_lc-np.nanmedian(seq_lc))) / 0.67449
    
    # run GP analysis on mean lightcurve
    gp_phot, gp_feed = run_gp_analysis(seq_time, seq_lc, seq_elc, gp_priors_file=gp_priors_file, walkers=32, nsteps=300, burnin=100, amp=1e-6, run_gls=False, run_mcmc=run_mcmc_on_gp, useQPKernel=useQPKernel, plot_distributions=False, verbose=True, plot=plot)
    gp_phot.compute(gp_feed["t"], gp_feed["yerr"])
    pred_mean_tmp, pred_var_tmp = gp_phot.predict(gp_feed['y'], seq_time, return_var=True)
    pred_mean, pred_var = gp_phot.predict(gp_feed['y'], ref_time, return_var=True)
    pred_std = np.sqrt(pred_var)
       
    # apply correction to mean lightcurve
    corr_lc_mean = seq_lc/pred_mean_tmp
    # measure RMS of corrected lightcurve to check efficiency
    sig_after_gp, mad_after_gp = np.nanstd(corr_lc_mean), np.nanmedian(np.abs(corr_lc_mean-np.nanmedian(corr_lc_mean))) / 0.67449
    
    # print RMS of data  before and after
    print("STATS Before GP: sigma={:.4f}% mad={:.4f}%".format(sig_before_gp*100, mad_before_gp*100))
    print("STATS After GP: sigma={:.4f}% mad={:.4f}%".format(sig_after_gp*100, mad_after_gp*100))

    # loop over each comparison star to apply master GP correction to all lightcurves
    for i in range(len(times)) :
        out_flux, out_fluxerr = np.full_like(times[i],np.nan),np.full_like(times[i],np.nan)
        
        # apply gp correction and recover transit signal and original median flux level
        out_flux = (fluxes_tmp[i] / pred_mean) * mfluxes[i] * mtransits[i]
        out_fluxerr = (fluxerrs_tmp[i] / pred_mean ) * mfluxes[i] * mtransits[i]
        
        #if plot :
        #    plt.plot(times[i], fluxes_tmp[i], ".", alpha=0.2)
            
        # store output fluxes
        out_fluxes.append(out_flux)
        out_fluxerrs.append(out_fluxerr)
    
    return out_fluxes, out_fluxerrs



def remove_waveplate_modulation(times, fluxes, fluxerrs, wppositions, posterior, combine_by_median=False, plot=True, verbose=False) :

    # get planet parameters from posterior
    """Remove waveplate modulation.

    Parameters
    ----------
    times
        List of time arrays [BJD], one per dataset.
    fluxes
        List of flux arrays, one per dataset.
    fluxerrs
        List of flux uncertainty arrays, one per dataset.
    wppositions
    posterior
        Posterior dictionary as returned by the fitting routines.
    combine_by_median : bool, optional (default: False)
    plot : bool, optional (default: True)
        Show diagnostic plots.
    verbose : bool, optional (default: False)
        Print progress information.
    """
    planet_params = posterior["planet_params"]
    instrum_params = posterior["instrum_params"]
    instrument_indexes = posterior["instrument_indexes"]
    
    # initialize output data containers
    out_fluxes, out_fluxerrs = [], []

    # initialize temporary data containers
    fluxes_tmp, fluxerrs_tmp = [], []
    mtransits, mfluxes = [], []
    
    ref_time = times[0]
    median_delta_time = np.nanmedian(np.abs(ref_time[1:] - ref_time[:-1]))
    max_delta_time = median_delta_time * 16
    
    # find out sequences from data for first comparison
    wppos = wppositions[0]
    
    # loop over each lightcurve
    for i in range(len(times)) :
    
        # calculate transit models
        transit_models = np.full_like(times[i], 1.0)
        instrum_index = 0
        if instrum_params is not None and instrument_indexes is not None:
            instrum_index = instrument_indexes[i]
            
        for j in range(int(posterior["n_planets"])) :
            planet_transit_id = "{0}_{1:03d}".format('transit', j)
            if planet_params[planet_transit_id] :
                transit_models *= exoplanetlib.batman_transit_model(times[i], planet_params, planet_index=j, instrum_params=instrum_params, instrum_index=instrum_index)
        # store transit models to return it back to the data later
        mtransits.append(transit_models)
        
        # remove transits
        flux_without_transit = fluxes[i] / transit_models
        fluxerr_without_transit = fluxerrs[i] / transit_models
        
        # calculate mean flux without transit
        mflux = np.nanmedian(flux_without_transit)
        # store mean flux
        mfluxes.append(mflux)
        
        if plot :
            plt.plot(wppositions[i], flux_without_transit / mflux, ".", alpha=0.2)
            
        # store temporary fluxes with both transits and mean values removed
        fluxes_tmp.append(flux_without_transit / mflux)
        fluxerrs_tmp.append(fluxerr_without_transit / mflux)
   
    # cast to numpy arrays
    fluxes_tmp = np.array(fluxes_tmp)
    fluxerrs_tmp = np.array(fluxerrs_tmp)

    wppos_flux, wppos_fluxerr = [], []
    WAVEPLATEPOS = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    
    for k in range(len(WAVEPLATEPOS)) :
        keep = wppos == WAVEPLATEPOS[k]
        
        lc_mean = np.nan
        elc_mean = np.nan
        
        if len(wppos[keep]) :
            flat_fluxes_tmp = fluxes_tmp[:,keep].flatten()
            flat_fluxerrs_tmp = fluxerrs_tmp[:,keep].flatten()
        
            # calculate mean from all comparison stars
            lc_mean = np.average(flat_fluxes_tmp, weights=1./(flat_fluxerrs_tmp*flat_fluxerrs_tmp))
            elc_mean = np.nanstd(flat_fluxes_tmp-lc_mean)
            if combine_by_median :
                lc_mean = np.nanmedian(flat_fluxes_tmp, axis=0)
                elc_mean = np.nanmedian(np.abs(flat_fluxes_tmp-lc_mean), axis=0) / 0.67449
            
        wppos_flux.append(lc_mean)
        wppos_fluxerr.append(elc_mean)
    
    wppos_flux = np.array(wppos_flux)
    wppos_fluxerr = np.array(wppos_fluxerr)

    if plot :
        plt.errorbar(WAVEPLATEPOS, wppos_flux, yerr=wppos_fluxerr, fmt="ko")
        plt.xlabel("waveplate position")
        plt.ylabel("Mean flux")
        plt.show()
    
    ffluxes_tmp = fluxes_tmp.flatten()
    # calculate statistics before gp correction to check efficiency
    sig_before_gp, mad_before_gp = np.nanstd(ffluxes_tmp), np.nanmedian(np.abs(ffluxes_tmp-np.nanmedian(ffluxes_tmp))) / 0.67449
  
    # loop over each comparison star to apply master GP correction to all lightcurves
    for i in range(len(times)) :
        out_flux, out_fluxerr = np.full_like(times[i],np.nan),np.full_like(times[i],np.nan)

        for k in range(len(WAVEPLATEPOS)) :
            keep = wppositions[i] == WAVEPLATEPOS[k]
            if len(wppositions[i][keep]) :
                # apply gp correction and recover transit signal and original median flux level
                fluxes_tmp[i][keep]  /= wppos_flux[k]
                fluxerrs_tmp[i][keep]  /= wppos_flux[k]
                
                out_flux[keep] = fluxes_tmp[i][keep] * mfluxes[i] * mtransits[i][keep]
                out_fluxerr[keep] = fluxerrs_tmp[i][keep] * mfluxes[i] * mtransits[i][keep]
                
        # store output fluxes
        out_fluxes.append(out_flux)
        out_fluxerrs.append(out_fluxerr)
    
    ffluxes_tmp = fluxes_tmp.flatten()
    # measure RMS of corrected lightcurve to check efficiency
    sig_after_gp, mad_after_gp = np.nanstd(ffluxes_tmp), np.nanmedian(np.abs(ffluxes_tmp-np.nanmedian(ffluxes_tmp))) / 0.67449
    
    # print RMS of data  before and after
    print("STATS Before WPPOS detrend: sigma={:.4f}% mad={:.4f}%".format(sig_before_gp*100, mad_before_gp*100))
    print("STATS After WPPOS detrend: sigma={:.4f}% mad={:.4f}%".format(sig_after_gp*100, mad_after_gp*100))

    return out_fluxes, out_fluxerrs


def main() :

    """Main.
    """
    global options

    parser = OptionParser()
    parser.add_option("-0", "--lcdata", dest="lcdata", help='Pattern for input light curve data',type='string',default="")
    parser.add_option("-1", "--input1", dest="input1", help='Pattern for input OPD light curve data ch1',type='string',default="")
    parser.add_option("-2", "--input2", dest="input2", help='Pattern for input OPD light curve data ch2',type='string',default="")
    parser.add_option("-3", "--input3", dest="input3", help='Pattern for input OPD light curve data ch3',type='string',default="")
    parser.add_option("-4", "--input4", dest="input4", help='Pattern for input OPD light curve data ch4',type='string',default="")
    parser.add_option("-t", "--targets", dest="targets", help='Target indexes',type='string',default="0|0|0|0")
    parser.add_option("-f", "--comps", dest="comps", help='Comparison indices ch1',type='string',default="1,2|1,2|1,2|1,2")
    parser.add_option("-q", "--catalog", dest="catalog", help='Catalog extension keyword',type='string',default="CATALOG_PHOT_AP020,CATALOG_PHOT_AP020,CATALOG_PHOT_AP020,CATALOG_PHOT_AP020")
    parser.add_option("-7", "--t0", dest="t0", help='T0: reference time',type='float',default=0.)
    parser.add_option("-x", "--niter", dest="niter", help='Number of iterations for sigma clipping',type='int',default=1)
    parser.add_option("-i", "--instrument_priors", dest="instrument_priors", help='Instrument priors file name',type='string',default="")
    parser.add_option("-r", "--planet_priors", dest="planet_priors", help='Planet priors file name',type='string',default="")
    parser.add_option("-g", "--gp_priors", dest="gp_priors", help='GP priors file name',type='string',default="")
    parser.add_option("-o", "--object", dest="object", help='Object ID',type='string',default="")
    parser.add_option("-c", "--calib_order", dest="calib_order", help='Order of calibration polynomial',type='int',default=1)
    parser.add_option("-n", "--nsteps", dest="nsteps", help="Number of MCMC steps",type='int',default=300)
    parser.add_option("-w", "--walkers", dest="walkers", help="Number of MCMC walkers",type='int',default=32)
    parser.add_option("-u", "--burnin", dest="burnin", help="Number of MCMC burn-in samples",type='int',default=100)
    parser.add_option("-s", "--samples_filename", dest="samples_filename", help='MCMC samples filename',type='string',default="")
    parser.add_option("-z", "--binsize", dest="binsize", help="Light curve binsize [d]",type='float',default=0.1)
    parser.add_option("-k", "--output_lc", dest="output_lc", help='Output reduced lightcurve',type='string',default="")
    parser.add_option("-b", action="store_true", dest="impact_parameter", help="Use impact parameter instead of inclination", default=False)
    parser.add_option("-d", action="store_true", dest="star_density", help="Use star density instead of a/Rs", default=False)
    parser.add_option("-l", action="store_true", dest="ols_fit", help="Perform OLS fit prior to MCMC", default=False)
    parser.add_option("-a", action="store_true", dest="fit_gp_activity", help="Run GP activity analysis",default=False)
    parser.add_option("-e", action="store_true", dest="mode", help="Best fit parameters obtained by the mode instead of median", default=False)
    parser.add_option("-p", action="store_true", dest="plot", help="Display plots",default=False)
    parser.add_option("-m", action="store_true", dest="mcmc_fit", help="Run MCMC",default=False)
    parser.add_option("-j", action="store_true", dest="polarimetry", help="Polarimetry data",default=False)
    #parser.add_option("-y", action="store_true", dest="plot_comps", help="Plot data for comparison stars",default=False)
    parser.add_option("-y", action="store_true", dest="include_tess", help="Include TESS data",default=False)
    parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

    try:
        options,args = parser.parse_args(sys.argv[1:])
    except SystemExit as e :
        # allow clean exits from optparse (e.g. --help)
        if e.code == 0 or e.code is None :
            raise
        print("Error: check usage with fit_sparc4_transits -h"); sys.exit(1);

    if options.verbose:
        print('Pattern for input light curve data for ch1: ', options.input1)
        print('Pattern for input light curve data for ch2: ', options.input2)
        print('Pattern for input light curve data for ch3: ', options.input3)
        print('Pattern for input light curve data for ch4: ', options.input4)
        print('Planet prior parameters file: ', options.planet_priors)
        print('Instrument prior parameters file: ', options.instrument_priors)
        print('Object ID: ', options.object)
        print('Order of calibration polynomial: ', options.calib_order)
        print('Number of MCMC steps: ', options.nsteps)
        print('Number of MCMC walkers: ', options.walkers)
        print('Number of MCMC burn-in samples: ', options.burnin)
        print('MCMC samples filenames: ', options.samples_filename)
        print('Light curve binsize [d]: ', options.binsize)
        print('Output reduced lightcurve: ', options.output_lc)


    product_dir = os.path.dirname(options.planet_priors)

    target_str = options.targets.split("|")
    comps_str = options.comps.split("|")
    catalogs = options.catalog.split(",")

    targets, comps = [], []

    for ch in range(4) :
        comps.append([])
    
    for ch in range(4) :
        targets.append(int(target_str[ch]))
        comps_idx_str = comps_str[ch].split(",")
        for j in comps_idx_str :
            comps[ch].append(int(j))

    print(targets)
    print(comps)

    instrument_indexes, instrument_labels = [], []
    n_instruments = 0

    times, fluxes, fluxerrs = [], [], []

    if options.include_tess :

        planet_index = 0
        min_npoints_per_bin = 30
        min_npoints_within_transit = 500
        transit_window_size = 2.5
        tess_times_in_bjd=True
        timelabel='TBJD'
        if tess_times_in_bjd :
            timelabel = 'BJD'

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
            force_tess_pl_priors = False
            tesslc = tess.load_dvt_files(options.object, priors_dir=priors_dir, save_priors=force_tess_pl_priors, hasrvdata=True, use_star_density=star_density, use_impact_parameter=impact_parameter, convert_times_to_bjd=tess_times_in_bjd, plot=options.plot, verbose=options.verbose)

        #planet = tess.redefine_tessranges(tesslc["PLANETS"][planet_index], tesslc, options.planet_priors, transit_window_size=transit_window_size, verbose=True)
        planet = tesslc["PLANETS"][planet_index]
    
        # select data within certain ranges
        times, fluxes, fluxerrs = planet["times"], planet["fluxes"], planet["fluxerrs"]
    
        instrument_labels.append("TESS")
        # set TESS instrument index to 0
        for i in range(len(times)) :
            instrument_indexes.append(n_instruments)
        n_instruments += 1

    
    ### CHANNEL 1 - Load data ####
    if options.input1 != "" :
        inputdata = sorted(glob.glob(options.input1))
        channel_index = 0
        instrument_labels.append("S4C1")

        for i in range(len(inputdata)) :
            t, f, ef = get_opddata(inputdata[i], target=targets[channel_index], comps=comps[channel_index], extname=catalogs[channel_index], t0=options.t0)
            for j in range(len(t)) :
                times.append(t[j])
                fluxes.append(f[j])
                fluxerrs.append(ef[j])
                instrument_indexes.append(n_instruments)
        n_instruments += 1


    ### CHANNEL 2 - Load data ####
    if options.input2 != "" :
        inputdata = sorted(glob.glob(options.input2))
    
        channel_index = 1
        instrument_labels.append("S4C2")

        for i in range(len(inputdata)) :
            t, f, ef = get_opddata(inputdata[i], target=targets[channel_index], comps=comps[channel_index], extname=catalogs[channel_index], t0=options.t0)
            for j in range(len(t)) :
                times.append(t[j])
                fluxes.append(f[j])
                fluxerrs.append(ef[j])
                instrument_indexes.append(n_instruments)
        n_instruments += 1

    ### CHANNEL 3 - Load data ####
    if options.input3 != "" :
        inputdata = sorted(glob.glob(options.input3))
    
        channel_index = 2
        instrument_labels.append("S4C3")

        for i in range(len(inputdata)) :
            t, f, ef = get_opddata(inputdata[i], target=targets[channel_index], comps=comps[channel_index], extname=catalogs[channel_index], t0=options.t0)
            for j in range(len(t)) :
                times.append(t[j])
                fluxes.append(f[j])
                fluxerrs.append(ef[j])
                instrument_indexes.append(n_instruments)
        n_instruments += 1

            
    ### CHANNEL 4 - Load data ####
    if options.input4 != "" :
        inputdata = sorted(glob.glob(options.input4))
    
        channel_index = 3
        instrument_labels.append("S4C4")

        for i in range(len(inputdata)) :
            t, f, ef = get_opddata(inputdata[i], target=targets[channel_index], comps=comps[channel_index], extname=catalogs[channel_index], t0=options.t0)
            for j in range(len(t)) :
                times.append(t[j])
                fluxes.append(f[j])
                fluxerrs.append(ef[j])
                instrument_indexes.append(n_instruments)
        n_instruments += 1

    # uncomment/comment the two lines below to ignore/consider different instruments for the fit
    #instrument_indexes = None
    #n_instruments = 0

    n_photdatasets = len(times)
    priors = fitlib.read_transit_rv_priors(options.planet_priors, 0, n_photdatasets, planet_index=0, calib_polyorder=options.calib_order, n_instruments=0, instrument_indexes=None, verbose=False)
    posterior = fitlib.guess_calib(priors, times, fluxes, prior_type="Normal", plot=False)

    if options.plot :
        # plot light curves and models in priors
        fitlib.plot_mosaic_of_lightcurves(times, fluxes, fluxerrs, posterior)

    for i in range(options.niter) :
        posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, posterior, calib_post_type="FIXED", verbose=False, plot=False)
        fluxes, fluxerrs = remove_systematics(times, fluxes, fluxerrs, posterior, calib_polyorder=5, binsize=options.binsize, plot=False)
        times, fluxes, fluxerrs = clean_data(times, fluxes, fluxerrs, posterior, n_sigma_clip=4., plot=False, verbose=options.verbose)

    instidx = np.array(instrument_indexes)
    dataset_idx = np.arange(len(instidx))

    ############# FOR POLARIMETRY DATA ##################
    if options.polarimetry :
        input_files = [options.input1,options.input2,options.input3,options.input4]
        for ch in range(1,5) :
            keep = (instidx == ch)
            loctimes, locfluxes, locfluxerrs = [], [], []
            for idx in dataset_idx[keep] :
                loctimes.append(times[idx])
                locfluxes.append(fluxes[idx])
                locfluxerrs.append(fluxerrs[idx])

            wppos = get_wppos(input_files[ch-1], target=targets[ch-1], comps=comps[ch-1], extname=catalogs[ch-1], verbose=True)
            if len(wppos) :
                locfluxes, locfluxerrs = remove_waveplate_modulation(loctimes, locfluxes, locfluxerrs, wppos, posterior, combine_by_median=False, plot=options.plot, verbose=options.verbose)
            
            for i in range(len(dataset_idx[keep])) :
                idx = dataset_idx[keep][i]
                fluxes[idx] = locfluxes[i]
                fluxerrs[idx] = locfluxerrs[i]
            
        for i in range(options.niter) :
            posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, posterior, calib_post_type="FIXED", verbose=False, plot=False)
            fluxes, fluxerrs = remove_systematics(times, fluxes, fluxerrs, posterior, calib_polyorder=5, binsize=options.binsize, plot=False)
            times, fluxes, fluxerrs = clean_data(times, fluxes, fluxerrs, posterior, n_sigma_clip=4., plot=False, verbose=options.verbose)
    #####################################################

    priors = fitlib.read_transit_rv_priors(options.planet_priors, 0, n_photdatasets, planet_index=0, calib_polyorder=1, n_instruments=0, instrument_indexes=None, verbose=False)
    posterior = fitlib.guess_calib(priors, times, fluxes, prior_type="FIXED", plot=False)
    posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, posterior, calib_post_type="FIXED", verbose=False, plot=True)

    if options.mcmc_fit :
        # Final fit with MCMC
        # Make sure the number of walkers is sufficient, and if not assing a new value
        if options.walkers < 2*len(posterior["theta"]):
            print("WARNING: insufficient number of MCMC walkers, resetting nwalkers={}".format(2*len(posterior["theta"])))
            options.walkers = 2*len(posterior["theta"])
        
        posterior = fitlib.fitTransitsWithMCMC(times, fluxes, fluxerrs, posterior, amp=1e-5, nwalkers=options.walkers, niter=options.nsteps, burnin=options.burnin, verbose=True, plot=options.plot, best_fit_from_mode=options.mode, appendsamples=False, plot_individual_transits=False)

        options.planet_priors = posterior["planet_posterior_file"]

    for ch in range(1,5) :

        keep = (instidx == ch)
    
        loctimes, locfluxes, locfluxerrs = [], [], []
        for idx in dataset_idx[keep] :
            loctimes.append(times[idx])
            locfluxes.append(fluxes[idx])
            locfluxerrs.append(fluxerrs[idx])
    
        n_photdatasets = len(loctimes)
        print("n_photdatasets={}".format(n_photdatasets))

        loc_priors = fitlib.read_transit_rv_priors(options.planet_priors, 0, n_photdatasets, planet_index=0, calib_polyorder=1, n_instruments=0, instrument_indexes=None, verbose=False)
        loc_posterior = fitlib.guess_calib(loc_priors, loctimes, locfluxes, prior_type="Normal", plot=False)
        loc_posterior = fitlib.fitTransits_ols(loctimes, locfluxes, locfluxerrs, loc_posterior, calib_post_type="FIXED", verbose=True, plot=False)

        if options.mcmc_fit :
            # Final fit with MCMC
            # Make sure the number of walkers is sufficient, and if not assing a new value
            if options.walkers < 2*len(loc_posterior["theta"]):
                print("WARNING: insufficient number of MCMC walkers, resetting nwalkers={}".format(2*len(loc_posterior["theta"])))
                options.walkers = 2*len(loc_posterior["theta"])

            loc_posterior = fitlib.fitTransitsWithMCMC(loctimes, locfluxes, locfluxerrs, loc_posterior, amp=1e-5, nwalkers=options.walkers, niter=options.nsteps, burnin=options.burnin, verbose=True, plot=options.plot, best_fit_from_mode=options.mode, appendsamples=False, plot_individual_transits=False)

            output_posterior = loc_posterior["planet_posterior_file"].replace(".pars","_ch{}.pars".format(ch))
            os.rename(loc_posterior["planet_posterior_file"], output_posterior)
     
        # figure out output light curve file name
        #output_lc = ""
    
        tbl = diff_light_curve(loctimes, locfluxes, locfluxerrs, loc_posterior, nsig=3, model_time_sampling=0.001, offset_nsig=10, combine_by_median=False, binsize=options.binsize, output=options.output_lc, instrum_index=0, use_calibs=True, plot_comps=True, plot=options.plot, verbose=options.verbose)
    
        tbl.write("{}/{}_ch{}_lc.csv".format(product_dir, options.object.replace(" ",""),ch), overwrite=True)


if __name__ == "__main__" :
    main()
