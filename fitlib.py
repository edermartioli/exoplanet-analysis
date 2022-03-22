# -*- coding: utf-8 -*-
"""
Created on Wed November 4 2020
@author: Eder Martioli
Institut d'Astrophysique de Paris, France.
"""

import priorslib
import modelslib

import numpy as np
import emcee

import astropy.io.fits as fits

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import optimize
from scipy.optimize import minimize
import pylab as pl
import scipy as sp
import scipy.interpolate as sint
import scipy.signal as signal

from copy import deepcopy

#import exoplanet as xo
import corner
#from balrogo import marginals
import matplotlib

import exoplanetlib
from PyAstronomy.pyasl import foldAt

import warnings

import gp_lib

def plot_mosaic_of_lightcurves(times, fluxes, fluxerrs, priors, output='') :
    
    # calcualate flare time tags
    flare_tags = get_flare_tags(priors['flare_params'], times)

    if priors['flare_params'] == {} :
        hasflares = False
        nrows = 3
    else :
        hasflares = True
        nrows = 4

    #font = {'size': 15}
    #matplotlib.rc('font', **font)

    fig, axs = plt.subplots(nrows, len(times), sharex=False, sharey=False, gridspec_kw={'hspace': 0, 'wspace': 0})
    
    for i in range(len(times)):
        calib = modelslib.calib_model(len(times), i, priors['calib_params'], times[i])
        flares = modelslib.flares_model(priors['flare_params'], flare_tags, i, times[i])
        transit_models = np.full_like(times[i], 1.0)
        for j in range(len(priors["planet_params"])) :
            transit_models *= modelslib.batman_transit_model(times[i], priors["planet_params"][j], planet_index=j)

        t0 = priors["planet_params"][0]["tc_000"]
        period = priors["planet_params"][0]["per_000"]
        epoch = np.round((np.mean(times[i]) - t0) / period)
        tc = t0 + period * epoch

        phase = (times[i] - t0 + 0.5 * period) % period - 0.5 * period
        #phase = times[i] - tc

        ####  First plot orignal data and model
        jcol = 0

        axs[jcol, i].set_title(r"Tc={0:.6f}".format(tc))
        axs[jcol, i].errorbar(phase, fluxes[i], yerr=fluxerrs[i], fmt='.', alpha=0.1, zorder=0)
        axs[jcol, i].plot(phase, calib*transit_models + flares, 'k-', lw=2, zorder=1)
        if i==0 :
            axs[jcol, i].set_ylabel(r"Flux")
        
        #axs[jcol, i].set_xlim(-0.5,0.5)
        axs[jcol, i].xaxis.set_ticklabels([])
        axs[jcol, i].yaxis.set_ticklabels([])

        #### Now plot the data and model for the starspot modulation only
        jcol += 1
        #axs[jcol, i].set_title(r"Transit {} - starspot model".format(i))
        axs[jcol, i].errorbar(phase, (fluxes[i]-flares)/transit_models, yerr=fluxerrs[i], fmt='.', alpha=0.1, zorder=0)
        axs[jcol, i].plot(phase, calib, '-', color="brown", lw=2, zorder=1)
        if i==0 :
            axs[jcol, i].set_ylabel(r"Flux")
        #axs[jcol, i].set_xlim(-0.5,0.5)
        axs[jcol, i].xaxis.set_ticklabels([])
        axs[jcol, i].yaxis.set_ticklabels([])

        if hasflares :
            #### Now plot the data and model for the flares only
            jcol += 1
            #axs[2, i].set_title(r"Transit {} - flare model".format(i))
            axs[jcol, i].errorbar(phase, (fluxes[i]/transit_models-calib), yerr=fluxerrs[i], fmt='.', alpha=0.1, zorder=0)
            axs[jcol, i].plot(phase, flares, '-', color="darkblue", lw=2, zorder=1)
            if i==0 :
                axs[jcol, i].set_ylabel(r"Flux [e$^{-}$/s]")

            axs[jcol, i].set_ylim(-500,2000)
            #axs[jcol, i].set_xlim(-0.5,0.5)
            axs[jcol, i].xaxis.set_ticklabels([])
            if i > 0 :
                axs[jcol, i].yaxis.set_ticklabels([])

        #### Now plot the data and model for the transits only
        jcol += 1
        #axs[jcol, i].set_title(r"Transit {} - transit model".format(i))
        axs[jcol, i].errorbar(phase, (fluxes[i]-flares)/calib, yerr=fluxerrs[i]/calib, fmt='.', alpha=0.1, zorder=0)
        axs[jcol, i].plot(phase, transit_models, '-', color="darkgreen", lw=2, zorder=1)
        if i==0 :
            axs[jcol, i].set_ylabel(r"Relative flux")
        else :
            axs[jcol, i].yaxis.set_ticklabels([])
            pass
        #axs[jcol, i].set_xlim(-0.5,0.5)
        #axs[jcol, i].set_ylim(0.995,1.003)
        axs[jcol, i].set_xlabel(r"TBJD - T$_c$")


    if output != '' :
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
    else :
        plt.show()
        plt.clf()
        plt.close()


def fit_continuum(wav, spec, function='polynomial', order=3, nit=5, rej_low=2.0,
    rej_high=2.5, grow=1, med_filt=0, percentile_low=0., percentile_high=100.,
                  min_points=10, xlabel="", ylabel="", return_polycoeffs=False, plot_fit=True, verbose=False):
    
    warnings.simplefilter('ignore', np.RankWarning)
    
    """
    Continuum fitting re-implemented from IRAF's 'continuum' function
    in non-interactive mode only but with additional options.

    :Parameters:
    
    wav: array(float)
        abscissa values (wavelengths, velocities, ...)

    spec: array(float)
        spectrum values

    function: str
        function to fit to the continuum among 'polynomial', 'spline3'

    order: int
        fit function order:
        'polynomial': degree (not number of parameters as in IRAF)
        'spline3': number of knots

    nit: int
        number of iteractions of non-continuum points
        see also 'min_points' parameter

    rej_low: float
        rejection threshold in unit of residul standard deviation for point
        below the continuum

    rej_high: float
        same as rej_low for point above the continuum

    grow: int
        number of neighboring points to reject

    med_filt: int
        median filter the spectrum on 'med_filt' pixels prior to fit
        improvement over IRAF function
        'med_filt' must be an odd integer

    percentile_low: float
        reject point below below 'percentile_low' percentile prior to fit
        improvement over IRAF function
        "percentile_low' must be a float between 0. and 100.

    percentile_high: float
        same as percentile_low but reject points in percentile above
        'percentile_high'
        
    min_points: int
        stop rejection iterations when the number of points to fit is less than
        'min_points'

    plot_fit: bool
        if true display two plots:
            1. spectrum, fit function, rejected points
            2. residual, rejected points

    verbose: bool
        if true fit information is printed on STDOUT:
            * number of fit points
            * RMS residual
    """
    mspec = np.ma.masked_array(spec, mask=np.zeros_like(spec))
    # mask 1st and last point: avoid error when no point is masked
    # [not in IRAF]
    mspec.mask[0] = True
    mspec.mask[-1] = True
    
    mspec = np.ma.masked_where(np.isnan(spec), mspec)
    
    # apply median filtering prior to fit
    # [opt] [not in IRAF]
    if int(med_filt):
        fspec = signal.medfilt(spec, kernel_size=med_filt)
    else:
        fspec = spec
    # consider only a fraction of the points within percentile range
    # [opt] [not in IRAF]
    mspec = np.ma.masked_where(fspec < np.percentile(fspec, percentile_low),
        mspec)
    mspec = np.ma.masked_where(fspec > np.percentile(fspec, percentile_high),
        mspec)
    # perform 1st fit
    if function == 'polynomial':
        coeff = np.polyfit(wav[~mspec.mask], spec[~mspec.mask], order)
        cont = np.poly1d(coeff)(wav)
    elif function == 'spline3':
        knots = wav[0] + np.arange(order+1)[1:]*((wav[-1]-wav[0])/(order+1))
        spl = sint.splrep(wav[~mspec.mask], spec[~mspec.mask], k=3, t=knots)
        cont = sint.splev(wav, spl)
    else:
        raise(AttributeError)
    # iteration loop: reject outliers and fit again
    if nit > 0:
        for it in range(nit):
            res = fspec-cont
            sigm = np.std(res[~mspec.mask])
            # mask outliers
            mspec1 = np.ma.masked_where(res < -rej_low*sigm, mspec)
            mspec1 = np.ma.masked_where(res > rej_high*sigm, mspec1)
            # exlude neighbors cf IRAF's continuum parameter 'grow'
            if grow > 0:
                for sl in np.ma.clump_masked(mspec1):
                    for ii in range(sl.start-grow, sl.start):
                        if ii >= 0:
                            mspec1.mask[ii] = True
                    for ii in range(sl.stop+1, sl.stop+grow+1):
                        if ii < len(mspec1):
                            mspec1.mask[ii] = True
            # stop rejection process when min_points is reached
            # [opt] [not in IRAF]
            if np.ma.count(mspec1) < min_points:
                if verbose:
                    print("  min_points %d reached" % min_points)
                break
            mspec = mspec1
            if function == 'polynomial':
                coeff = np.polyfit(wav[~mspec.mask], spec[~mspec.mask], order)
                cont = np.poly1d(coeff)(wav)
            elif function == 'spline3':
                knots = wav[0] + np.arange(order+1)[1:]*((wav[-1]-wav[0])/(order+1))
                spl = sint.splrep(wav[~mspec.mask], spec[~mspec.mask], k=3, t=knots)
                cont = sint.splev(wav, spl)
            else:
                raise(AttributeError)
    # compute residual and rms
    res = fspec-cont
    sigm = np.std(res[~mspec.mask])
    if verbose:
        print("  nfit=%d/%d" %  (np.ma.count(mspec), len(mspec)))
        print("  fit rms=%.3e" %  sigm)
    # compute residual and rms between original spectrum and model
    # different from above when median filtering is applied
    ores = spec-cont
    osigm = np.std(ores[~mspec.mask])
    if int(med_filt) and verbose:
        print("  unfiltered rms=%.3e" %  osigm)
    # plot fit results
    if plot_fit:
        # overplot spectrum and model + mark rejected points
        fig1 = pl.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.plot(wav[~mspec.mask], spec[~mspec.mask],
            c='tab:blue', lw=1.0)
        # overplot median filtered spectrum
        if int(med_filt):
            ax1.plot(wav[~mspec.mask], fspec[~mspec.mask],
                c='tab:cyan', lw=1.0)
        ax1.scatter(wav[mspec.mask], spec[mspec.mask], s=20., marker='d',
        edgecolors='tab:gray', facecolors='none', lw=0.5)
        ax1.plot(wav, cont, ls='--', c='tab:orange')
        if nit > 0:
            # plot residuals and rejection thresholds
            fig2 = pl.figure(2)
            ax2 = fig2.add_subplot(111)
            ax2.axhline(0., ls='--', c='tab:orange', lw=1.)
            ax2.axhline(-rej_low*sigm, ls=':')
            ax2.axhline(rej_high*sigm, ls=':')
            ax2.scatter(wav[mspec.mask], res[mspec.mask],
                s=20., marker='d', edgecolors='tab:gray', facecolors='none',
                lw=0.5)
            ax2.scatter(wav[~mspec.mask], ores[~mspec.mask],
                marker='o', s=10., edgecolors='tab:blue', facecolors='none',
                lw=.5)
            # overplot median filtered spectrum
            if int(med_filt):
                ax2.scatter(wav[~mspec.mask], res[~mspec.mask],
                    marker='s', s=5., edgecolors='tab:cyan', facecolors='none',
                    lw=.2)
        if xlabel != "" :
            pl.xlabel(xlabel)
        if ylabel != "" :
            pl.ylabel(ylabel)
        pl.show()

    if return_polycoeffs :
        #cont = np.poly1d(coeff)(wav)
        return coeff
    else :
        return cont


def read_priors(planet_priors_files, n_datasets, flare_priorsfile="", calib_priorsfile="", calib_polyorder=1, n_rvdatasets=0, verbose=False) :
    
    loc = {}

    ################################
    ####    PLANET(S) PRIORS    ####
    ################################

    n_planets = len(planet_priors_files)

    planet_priors,  planet_params = [], []
    for i in range(len(planet_priors_files)) :
        if verbose:
            print("Loading exoplanet priors from input file: ",planet_priors_files[i])

        pl_priors = priorslib.read_priors(planet_priors_files[i])
        planet_priors.append(pl_priors)
        
        pl_params = priorslib.read_exoplanet_params(pl_priors, planet_index=i)
        planet_params.append(pl_params)
    
        # print out planet priors
        if verbose:
            print("----------------")
            print("Input parameters for PLANET {}/{}:".format(i,n_planets))
            for key in pl_params.keys() :
                if ("_err" not in key) and ("_pdf" not in key) :
                    pdf_key = "{0}_pdf".format(key)
                    if pl_params[pdf_key] == "FIXED" :
                        print("{0} = {1} ({2})".format(key, pl_params[key], pl_params[pdf_key]))
                    elif pl_params[pdf_key] == "Uniform" or pl_params[pdf_key] == "Jeffreys":
                        error_key = "{0}_err".format(key)
                        min = pl_params[error_key][0]
                        max = pl_params[error_key][1]
                        print("{0} <= {1} <= {2} ({3})".format(min, key, max, pl_params[pdf_key]))
                    elif pl_params[pdf_key] == "Normal" :
                        error_key = "{0}_err".format(key)
                        error = pl_params[error_key][1]
                        print("{0} = {1} +- {2} ({3})".format(key, pl_params[key], error, pl_params[pdf_key]))
            print("----------------")

                
    ################################
    ##    CALIBRATION PRIORS    ####
    ################################

    if calib_priorsfile != "" :
        if verbose:
            print("Loading calibration priors from input file: ",calib_priorsfile)
        # if priors file is provided then load calibration parameters priors
        calib_priors = priorslib.read_priors(calib_priorsfile, calibration=True)
        calib_params = priorslib.read_calib_params(calib_priors)
    else :
        if verbose:
            print("Initializing calibration priors for ndim={0} order={1}: ".format(n_datasets,calib_polyorder))
        #if no priors file is provided then make a guess
        calib_priors = priorslib.init_calib_priors(ndim=n_datasets, order=calib_polyorder)
        calib_params = priorslib.read_calib_params(calib_priors)

        for i in range(n_datasets) :
            coeff_name = 'd{0:02d}c{1}'.format(i, calib_polyorder-1)
            calib_params[coeff_name] = 0.
            calib_priors[coeff_name]['object'].value = 0.
                
    # print out calibration priors
    if verbose:
        print("----------------")
        print("Input CALIBRATION parameters:")
        for key in calib_params.keys() :
            print(key, "=", calib_params[key])
        print("----------------")

    if n_rvdatasets :
        if verbose:
            print("Initializing RV calibration priors for ndim={0}: ".format(n_rvdatasets))
        #if no priors file is provided then make a guess
        rvcalib_priors = priorslib.init_rvcalib_priors(ndim=n_rvdatasets)
        rvcalib_params = priorslib.read_rvcalib_params(rvcalib_priors)

        for i in range(n_rvdatasets) :
            coeff_name = 'rv_d{0:02d}'.format(i)
            rvcalib_params[coeff_name] = 0.
            rvcalib_priors[coeff_name]['object'].value = 0.

        # print out calibration priors
        if verbose:
            print("----------------")
            print("Input RV CALIBRATION parameters:")
            for key in rvcalib_params.keys() :
                print(key, "=", rvcalib_params[key])
            print("----------------")
    else :
        rvcalib_priors = None
    ################################
    ####    FLARES PRIORS    ####
    ################################
    
    # Load flares parameters priors
    if flare_priorsfile != "" :
        if verbose:
            print("Loading flares priors from input file: ",flare_priorsfile)
        flare_priors = priorslib.read_priors(flare_priorsfile, flares=True)
        n_flares = flare_priors["n_flares"]
        flare_params = priorslib.read_flares_params(flare_priors)
        # print out calibration priors
        if verbose:
            print("----------------")
            print("Input FLARE parameters:")
            print("Number of flares:",n_flares)
            for key in flare_params.keys() :
                print(key, "=", flare_params[key])
            print("----------------")
    else :
        flare_priors = {}
        flare_params = {}
        n_flares = 0

    if verbose:
        print("Setting up theta and label variables for MCMC fit ...")
    # Variable "theta" stores only the free parameters, and "labels" stores the corresponding parameter IDs
    theta, labels, theta_priors = priorslib.get_theta_from_priors(planet_priors, calib_priors, flare_priors, rvcalib_priors=rvcalib_priors)

    loc["n_planets"] = n_planets
    loc["planet_priors"] = planet_priors
    loc["planet_params"] = planet_params
    loc["planet_priors_files"] = planet_priors_files

    loc["n_datasets"] = n_datasets
    loc["calib_polyorder"] = calib_polyorder
    loc["calib_priors"] = calib_priors
    loc["calib_params"] = calib_params
    loc["calib_priorsfile"] = calib_priorsfile

    if n_rvdatasets :
        loc["n_rvdatasets"] = n_rvdatasets
        loc["rvcalib_priors"] = rvcalib_priors
        loc["rvcalib_params"] = rvcalib_params

    loc["n_flares"] = n_flares
    loc["flare_priors"] = flare_priors
    loc["flare_params"] = flare_params
    loc["flare_priorsfile"] = flare_priorsfile

    loc["theta"] = theta
    loc["labels"] = labels
    loc["theta_priors"] = theta_priors

    return loc


def guess_calib(priors, times, fluxes, prior_type="FIXED", remove_flares=True, remove_transits=True, plot=False, multiplanetmodel=False, planet_index=0) :
    
    calib_polyorder = priors["calib_polyorder"]
    calib_priors = priors["calib_priors"]
    calib_params = priors["calib_params"]
    
    n_datasets = len(times)
    
    fluxes_copy = deepcopy(fluxes)
    
    if  remove_flares :
        flare_tags = get_flare_tags(priors["flare_params"], times)
    
    for i in range(n_datasets) :
        
        if  remove_flares :
            flares = modelslib.flares_model(priors["flare_params"], flare_tags, i, times[i])
            fluxes_copy[i] -= flares
        
        if remove_transits :
            transit_models = np.full_like(times[i], 1.0)
            
            if multiplanetmodel :
                for j in range(len(priors["planet_params"])) :
                    transit_models *= modelslib.batman_transit_model(times[i], priors["planet_params"][j], planet_index=j)
            else :
                transit_models *= modelslib.batman_transit_model(times[i], priors["planet_params"][planet_index], planet_index=planet_index)

            fluxes_copy[i] /= transit_models

        if plot :
            plt.plot(times[i], fluxes[i], 'r.', alpha=0.1)

        #print("i=",i,np.nanmean(times[i]))

        coeff = fit_continuum(times[i], fluxes_copy[i], function='polynomial', order=calib_polyorder-1, nit=5, rej_low=2.5, rej_high=2.5, grow=1, med_filt=0, percentile_low=0., percentile_high=100.,min_points=10, return_polycoeffs=True, plot_fit=False, verbose=False)

        polymodel = np.poly1d(coeff)(times[i])

        for j in range(calib_polyorder) :
            coeff_id = 'd{0:02d}c{1:1d}'.format(i,j)
            if j < 5 :
                calib_params[coeff_id] = coeff[j]
                calib_priors[coeff_id]['type'] = prior_type
                if prior_type == "Normal" :
                    cen, sig = coeff[j], coeff[j]*0.15
                    calib_priors[coeff_id]['object'] = priorslib.normal_parameter(np.array([cen, sig]))
                elif prior_type == "Uniform" :
                    l_value, u_value = coeff[j] - 0.75*coeff[j], coeff[j] + 0.75*coeff[j]
                    calib_priors[coeff_id]['object'] = priorslib.uniform_parameter(np.array([l_value, u_value]))
                elif prior_type == "FIXED" :
                    calib_priors[coeff_id]['object'] = priorslib.constant_parameter(np.array(coeff[j]))
                else :
                    print("ERROR: prior type {} not recognized, exiting ... ".format(prior_type))
                    exit()
            else :
                calib_params[coeff_id] = 0.
                calib_priors[coeff_id]['object'].value = 0.

        calib = modelslib.calib_model(n_datasets, i, calib_params, times[i])

        if plot :
            plt.plot(times[i], polymodel, 'b--', lw=2, alpha=0.7)
            plt.plot(times[i], calib, 'r-', lw=2, alpha=0.7)

    if plot:
        plt.xlabel(r"Time [BTJD]")
        plt.ylabel(r"Flux [e-/s]")
        plt.show()

    priors["calib_priors"] = calib_priors

    theta, labels, theta_priors = priorslib.get_theta_from_priors(priors["planet_priors"], calib_priors, priors["flare_priors"])

    priors["theta"] = theta
    priors["labels"] = labels
    priors["theta_priors"] = theta_priors

    calib_params = updateParams(calib_params, theta, labels)
    priors["calib_params"] = calib_params

    return priors


def guess_rvcalib(priors, bjds, rvs, prior_type="FIXED", plot=False) :
    
    rvcalib_priors = priors["rvcalib_priors"]
    rvcalib_params = priors["rvcalib_params"]
    planet_params = priors["planet_params"]
    
    n_rvdatasets = len(rvs)
    
    rvs_copy = deepcopy(rvs)
    
    for i in range(n_rvdatasets) :

        if plot :
            plt.plot(bjds[i], rvs[i], 'r.', alpha=0.5)

        prior_rv_model = calculate_rv_model(bjds[i], planet_params, planet_index=0)

        median_rv = np.nanmedian(rvs[i]-prior_rv_model)
        sigma_rv = np.nanstd(rvs[i]-prior_rv_model)
        
        coeff_id = 'rv_d{0:02d}'.format(i)
        if prior_type == "Normal" :
            cen, sig = median_rv, sigma_rv
            rvcalib_priors[coeff_id]['object'] = priorslib.normal_parameter(np.array([cen, sig]))
        elif prior_type == "Uniform" :
            l_value, u_value = median_rv - 10.*sigma_rv, median_rv + 10.*sigma_rv
            rvcalib_priors[coeff_id]['object'] = priorslib.uniform_parameter(np.array([l_value, u_value]))
        elif prior_type == "FIXED" :
            rvcalib_priors[coeff_id]['object'] = priorslib.constant_parameter(median_rv)
        else :
            print("ERROR: prior type {} not recognized, exiting ... ".format(prior_type))
            exit()

        calib = median_rv

        if plot :
            plt.plot(bjds[i], rvs[i]-median_rv, 'b--', lw=2, alpha=0.7)
                
    if plot:
        plt.xlabel(r"BJD")
        plt.ylabel(r"Radial velocity [m/s]")
        plt.show()
    
    priors["rvcalib_priors"] = rvcalib_priors

    theta, labels, theta_priors = priorslib.get_theta_from_priors(priors["planet_priors"], priors["calib_priors"], priors["flare_priors"], rvcalib_priors=rvcalib_priors)

    priors["theta"] = theta
    priors["labels"] = labels
    priors["theta_priors"] = theta_priors

    rvcalib_params = updateParams(rvcalib_params, theta, labels)

    priors["rvcalib_params"] = rvcalib_params

    return priors


def fitTransitsWithMCMC(times, fluxes, fluxerrs, priors, amp=1e-4, nwalkers=32, niter=100, burnin=20, verbose=False, plot=False, plot_prev_model=False, plot_individual_transits=False, samples_filename="default_samples.h5", transitsplot_output="", pairsplot_output="", appendsamples=False) :
    
    warnings.simplefilter('ignore')
    
    posterior = deepcopy(priors)
    
    calib_params = posterior["calib_params"]
    flare_params = posterior["flare_params"]
    planet_params = posterior["planet_params"]
    
    theta, labels, theta_priors = posterior["theta"], posterior["labels"], posterior["theta_priors"]
    
    if verbose :
        print("Free parameters before MCMC fit:")
        for i in range(len(theta)) :
            print(labels[i], "=", theta[i])

    if verbose:
        print("initializing emcee sampler ...")

    #amp, ndim, nwalkers, niter, burnin = 5e-4, len(theta), 50, 2000, 500
    ndim = len(theta)

    # Set up the backend
    backend = emcee.backends.HDFBackend(samples_filename)
    if appendsamples == False :
        backend.reset(nwalkers, ndim)

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = [theta_priors, labels, calib_params, flare_params, planet_params, times, fluxes, fluxerrs], backend=backend)

    pos = [theta + amp * np.random.randn(ndim) for i in range(nwalkers)]
    #--------

    #- run mcmc
    if verbose:
        print("Running MCMC ...")
        print("N_walkers=",nwalkers," ndim=",ndim)

    sampler.run_mcmc(pos, niter, progress=True)
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim)) # burnin : number of first samples to be discard as burn-in
    #--------

    if verbose:
        print("Obtaining best fit calibration parameters from pdfs ...")
    calib_params, calib_theta_fit, calib_theta_labels, calib_theta_err = best_fit_params(posterior["calib_params"], labels, samples)
    if verbose :
        print("CALIBRATION Fit parameters:")
        for i in range(len(calib_theta_fit)) :
            print(calib_theta_labels[i], "=", calib_theta_fit[i], "+", calib_theta_err[i][0], "-", calib_theta_err[i][1])
        print("----------------")

    if posterior["calib_priorsfile"] != "":
        calib_posterior = posterior["calib_priorsfile"].replace(".pars", "_posterior.pars")
    else :
        calib_posterior = "calibration_posterior.pars"
    if verbose:
        print("Output CALIBRATION posterior: ", calib_posterior)
    # save posterior of calibration parameters into file:
    ncoeff=posterior["calib_priors"]['orderOfPolynomial']['object'].value
    priorslib.save_posterior(calib_posterior, calib_params, calib_theta_fit, calib_theta_labels, calib_theta_err, calib=True, ncoeff=ncoeff)

    # update cablib parameters in output posterior
    posterior["calib_params"] = calib_params

    if posterior["n_flares"] :
        if verbose:
            print("Obtaining best fit flare parameters from pdfs ...")
        
        flare_params, flare_theta_fit, flare_theta_labels, flare_theta_err = best_fit_params(posterior["flare_params"], labels, samples)

        if verbose :
            print("FLARE Fit parameters:")
            for i in range(len(flare_theta_fit)) :
                print(flare_theta_labels[i], "=", flare_theta_fit[i], "+", flare_theta_err[i][0], "-", flare_theta_err[i][1])
            print("----------------")

        flare_posterior = posterior["flare_priorsfile"].replace(".pars", "_posterior.pars")
        if verbose:
            print("Output FLARE posterior: ", flare_posterior)
        # save posterior of flare parameters into file:
        priorslib.save_posterior(flare_posterior, flare_params, flare_theta_fit, flare_theta_labels, flare_theta_err)

    # update flare parameters in output posterior
    posterior["flare_params"] = flare_params

    if verbose:
        print("Obtaining best fit planet parameters from pdfs ...")
    planet_params, planet_theta_fit, planet_theta_labels = [], [], []
    posterior["planet_posterior_files"] = []
    for i in range(posterior["n_planets"]) :
        pl_params, pl_theta_fit, pl_theta_labels, pl_theta_err = best_fit_params(posterior["planet_params"][i], labels, samples)
        # update flare parameters in output posterior
        posterior["planet_params"][i] = pl_params

        planet_params.append(pl_params)
        planet_theta_fit = np.concatenate((planet_theta_fit, pl_theta_fit), axis=0)
        planet_theta_labels = np.concatenate((planet_theta_labels, pl_theta_labels), axis=0)
        if verbose:
            # print out best fit parameters and errors
            print("----------------")
            print("PLANET {} Fit parameters:".format(i))
            for j in range(len(planet_theta_fit)) :
                print(pl_theta_labels[j], "=", pl_theta_fit[j], "+", pl_theta_err[j][0], "-", pl_theta_err[j][1])
            print("----------------")

        planet_posterior = posterior["planet_priors_files"][i].replace(".pars", "_posterior.pars")
        posterior["planet_posterior_files"].append(planet_posterior)
        if verbose:
            print("Output PLANET {0} posterior: ".format(i), planet_posterior)
                
        # save posterior of planet parameters into file:
        priorslib.save_posterior(planet_posterior, pl_params, pl_theta_fit, pl_theta_labels, pl_theta_err)

    # Update theta in posterior
    if posterior["n_flares"] :
        theta_tuple  = (calib_theta_fit, flare_theta_fit, planet_theta_fit)
        labels_tuple  = (calib_theta_labels, flare_theta_labels, planet_theta_labels)
    else :
        theta_tuple  = (calib_theta_fit, planet_theta_fit)
        labels_tuple  = (calib_theta_labels, planet_theta_labels)

    posterior['theta'] = np.hstack(theta_tuple)
    posterior['labels'] = np.hstack(labels_tuple)
    posterior["theta_priors"] = posterior['theta']

    if plot :
        if plot_individual_transits :
            plot_posterior_multimodel(times, fluxes, fluxerrs, posterior, plot_prev_model=False)

        plot_all_transits(times, fluxes, fluxerrs, posterior, plot_prev_model=plot_prev_model, bindata=True, output=transitsplot_output)
        #plot model
        #plot_priors_model(times, fluxes, fluxerrs, posterior)
        
        #- make a pairs plot from MCMC output:
        #flat_samples = sampler.get_chain(discard=burnin,flat=True)
        pairs_plot_emcee(samples, labels, calib_params, planet_params[0], output=pairsplot_output, addlabels=True)

    return posterior
    

def pairs_plot_emcee(samples, labels, calib_params, planet_params, output='', addlabels=True, rvcalib_params={}) :
    truths=[]
    font = {'size': 12}
    matplotlib.rc('font', **font)
    
    newlabels = []
    for lab in labels :
        if lab in calib_params.keys():
            truths.append(calib_params[lab])
        
        if lab in rvcalib_params.keys():
            truths.append(rvcalib_params[lab])

        elif lab in planet_params.keys():
            truths.append(planet_params[lab])
        
        
        if lab == 'rp_000':
            newlabels.append(r"R$_{p}$/R$_{\star}$")
        elif lab == 'a_000':
            newlabels.append(r"a/R$_{\star}$")
        elif lab == 'tc_000':
            newlabels.append(r"T$_c$ [BTJD]")
        elif lab == 'per_000':
            newlabels.append(r"P [d]")
        elif lab == 'inc_000':
            newlabels.append(r"$i$ [$^{\circ}$]")
        elif lab == 'u0_000':
            newlabels.append(r"u$_{0}$")
        elif lab == 'u1_000':
            newlabels.append(r"u$_{1}$")
        elif lab == 'ecc_000':
            newlabels.append(r"$e$")
        elif lab == 'w_000':
            newlabels.append(r"$\omega$ [deg]")
        elif lab == 'tp_000':
            newlabels.append(r"T$_p$ [BJD]")
        elif lab == 'k_000':
            newlabels.append(r"K$_p$ [m/s]")
        elif lab == 'rvsys_000':
            newlabels.append(r"$\gamma$ [km/s]")
        elif lab == 'trend_000':
            newlabels.append(r"$\alpha$ [km/s/d]")
        elif lab == 'rv_d00':
            newlabels.append(r"$\gamma$ [km/s]")
        else :
            newlabels.append(lab)

    if addlabels :
        fig = corner.corner(samples, labels = newlabels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], labelpad=1, truths=truths, labelsize=60, label_kwargs={"fontsize": 30}, show_titles=False)
        fig.set_size_inches(40, 45)
        #fig = marginals.corner(samples, labels = newlabels, quantiles=[0.16, 0.5, 0.84], truths=truths)
    else :
        fig = corner.corner(samples, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], labelpad=2.0,truths=truths, labelsize=10, show_titles=False)

    for ax in fig.get_axes():
        plt.setp(ax.get_xticklabels(), ha="left", rotation=60)
        plt.setp(ax.get_yticklabels(), ha="right", rotation=60)
        ax.tick_params(axis='both', labelsize=16)

    if output != '' :
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
    else :
        plt.show()


#- Derive best-fit params and their 1-sigm a error bars
def best_fit_params(params, free_param_labels, samples, use_mean=False, best_fit_from_mode=True, nbins = 30, plot_distributions=False, verbose = False) :

    theta, theta_labels, theta_err = [], [], []
    
    if use_mean :
        npsamples = np.array(samples)
        values = []
        for i in range(len(samples[0])) :
            mean = np.mean(npsamples[:,i])
            err = np.std(npsamples[:,i])
            values.append([mean,err,err])
    else :
        func = lambda v: (v[1], v[2]-v[1], v[1]-v[0])
        percents = np.percentile(samples, [16, 50, 84], axis=0)
        seq = list(zip(*percents))
        values = list(map(func, seq))

        max_values = []
        
        for i in range(len(values)) :
            hist, bin_edges = np.histogram(samples[:,i], bins=nbins, range=(values[i][0]-5*values[i][1],values[i][0]+5*values[i][2]), density=True)
            xcen = (bin_edges[:-1] + bin_edges[1:])/2
            mode = xcen[np.argmax(hist)]
            max_values.append(mode)
                
            if plot_distributions :
                nmax = len(samples[:,i])
                plt.step(xcen, hist, where='mid')
                plt.vlines([values[i][0]], np.min(0), np.max(hist), ls="--", label="median")
                plt.vlines([mode], np.min(0), np.max(hist), ls=":", label="mode")
                plt.ylabel(r"Probability density",fontsize=18)
                plt.xlabel(r"{}".format(free_param_labels[i]),fontsize=18)
                plt.legend()
                plt.show()

                plt.plot(samples[:,i],label="{}".format(labels[i]), alpha=0.5, lw=0.5)
                plt.hlines([], np.min(0), np.max(nmax), ls=":", label="mode",zorder=2)
                plt.hlines([values[i][0]], np.min(0), np.max(nmax), ls="-", label="median",zorder=2)
                plt.ylabel(r"{}".format(free_param_labels[i]),fontsize=18)
                plt.xlabel(r"MCMC iteration",fontsize=18)
                plt.legend(fontsize=18)
                plt.show()
                    
        max_values = np.array(max_values)


    for i in range(len(values)) :
        if free_param_labels[i] in params.keys() :
            if best_fit_from_mode :
                theta.append(max_values[i])
            else :
                theta.append(values[i][0])

            theta_err.append((values[i][1],values[i][2]))
            theta_labels.append(free_param_labels[i])
            
            if verbose :
                print(free_param_labels[i], "=", values[i][0], "+", values[i][1],"-", values[i][2])

    params = updateParams(params, theta, theta_labels)

    return params, theta, theta_labels, theta_err

# prior probability from definitions in priorslib
def lnprior(theta_priors, theta, labels):
    total_prior = 0.0

    for i in range(len(labels)) :
        #theta_priors[labels[i]]['object'].set_value(theta[i])
        
        if theta_priors[labels[i]]['type'] == "Uniform" or theta_priors[labels[i]]['type'] == "Jeffreys" or theta_priors[labels[i]]['type'] == "Normal_positive" :
            
            if not theta_priors[labels[i]]['object'].check_value(theta[i]):
                return -np.inf

        total_prior += theta_priors[labels[i]]['object'].get_ln_prior()

    return total_prior


def updateParams(params, theta, labels) :
    for key in params.keys() :
        for j in range(len(theta)) :
            if key == labels[j]:
                params[key] = theta[j]
                break
    return params


def get_flare_tags(flare_params, times) :
    tags = {}
    n_flares = len(flare_params) / 3

    for i in range(int(n_flares)) :
        tc_id = 'tc{0:04d}'.format(i)
        for j in range(len(times)) :
            if times[j][0] < flare_params[tc_id] < times[j][-1] :
                tags[i] = j
                break
            if j == len(times) - 1 :
                tags[i] = -1
    return tags


#likelihood function
def lnlikelihood(theta, labels, calib_params, flare_params, planet_params, times, fluxes, fluxerrs):
    
    prior_planet_params = deepcopy(planet_params)

    for i in range(len(planet_params)) :
        planet_params[i] = updateParams(planet_params[i], theta, labels)
    
    flare_params = updateParams(flare_params, theta, labels)
    calib_params = updateParams(calib_params, theta, labels)

    flux_models = calculate_models(times, calib_params, flare_params, planet_params)

    sum_of_residuals = 0

    for i in range(len(times)) :
        
        residuals = fluxes[i] - flux_models[i]
        
        for j in range(len(prior_planet_params)) :
            for key in prior_planet_params[j].keys() :
                if ("_err" not in key) and ("_pdf" not in key) :
                    pdf_key = "{0}_pdf".format(key)
                    if prior_planet_params[j][pdf_key] == "Normal" :
                        error_key = "{0}_err".format(key)
                        error = prior_planet_params[j][error_key][1]
                        param_chi2 = ((planet_params[j][key] - prior_planet_params[j][key])/error)**2
                        sum_of_residuals += param_chi2
        
        sum_of_residuals += np.sum((residuals/fluxerrs[i])**2 + np.log(2.0 * np.pi * (fluxerrs[i] * fluxerrs[i])))

    ln_likelihood = -0.5 * (sum_of_residuals)

    return ln_likelihood


#posterior probability
def lnprob(theta, theta_priors, labels, calib_params, flare_params, planet_params, times, fluxes, fluxerrs):
    
    #lp = lnprior(theta)
    lp = lnprior(theta_priors, theta, labels)
    if not np.isfinite(lp):
        return -np.inf
    
    prob = lp + lnlikelihood(theta, labels, calib_params, flare_params, planet_params, times, fluxes, fluxerrs)

    if np.isnan(prob) :
        return -np.inf
    else :
        return prob


#posterior probability
def lnprob_tr_rv(theta, theta_priors, labels, calib_params, rvcalib_params, flare_params, planet_params, tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, rvnull=False):

    #lp = lnprior(theta)
    lp = lnprior(theta_priors, theta, labels)
    if not np.isfinite(lp):
        return -np.inf

    lnlike = lnlikelihood_tr_rv(theta, labels, calib_params, rvcalib_params, flare_params, planet_params, tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, rvnull=rvnull)
    prob = lp + lnlike

    if np.isfinite(prob) :
        return prob
    else :
        return -np.inf


#likelihood function
def lnlikelihood_tr_rv(theta, labels, calib_params, rvcalib_params, flare_params, planet_params, tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, rvnull=False):
    
    prior_planet_params = deepcopy(planet_params)

    for i in range(len(planet_params)) :
        planet_params[i] = updateParams(planet_params[i], theta, labels)
    
    flare_params = updateParams(flare_params, theta, labels)
    calib_params = updateParams(calib_params, theta, labels)

    flux_models = calculate_models(tr_times, calib_params, flare_params, planet_params)

    param_likelihood = 0
    for j in range(len(prior_planet_params)) :
        for key in prior_planet_params[j].keys() :
            if ("_err" not in key) and ("_pdf" not in key) :
                pdf_key = "{0}_pdf".format(key)
                if prior_planet_params[j][pdf_key] == "Normal" or prior_planet_params[j][pdf_key] == "Normal_positive":
                    error_key = "{0}_err".format(key)
                    error = prior_planet_params[j][error_key][1]
                    param_chi2 = ((planet_params[j][key] - prior_planet_params[j][key])/error)**2 + np.log(2.0 * np.pi * error * error)
                    param_likelihood += param_chi2

    tr_likelihood = 0
    for i in range(len(tr_times)) :
        flux_residuals = fluxes[i] - flux_models[i]
        tr_likelihood += np.nansum((flux_residuals/fluxerrs[i])**2 + np.log(2.0 * np.pi * (fluxerrs[i] * fluxerrs[i])))

    rvcalib_params = updateParams(rvcalib_params, theta, labels)
    rv_likelihood = 0
    # Calculate RV model and residuals
    for i in range(len(rvs)) :
        coeff_id = 'rv_d{0:02d}'.format(i)
        rvcalib = rvcalib_params[coeff_id]
        if rvnull :
            rv_model = np.zeros_like(rv_times[i]) + rvcalib
        else :
            rv_model = calculate_rv_model(rv_times[i], planet_params, planet_index=0) + rvcalib
        rv_residuals = rvs[i] - rv_model

        rv_likelihood += np.nansum((rv_residuals/rverrs[i])**2 + np.log(2.0 * np.pi * (rverrs[i] * rverrs[i])))

    #print(-0.5 *param_likelihood, -0.5 *tr_likelihood, -0.5 * rv_likelihood)
    ln_likelihood = -0.5 * (param_likelihood + tr_likelihood + rv_likelihood)

    return ln_likelihood


def calculate_rv_model(time, planets_params, planet_index=0) :
    
    planet_params = planets_params[planet_index]
    
    per = planet_params['per_{0:03d}'.format(planet_index)]
    tt = planet_params['tc_{0:03d}'.format(planet_index)]
    ecc = planet_params['ecc_{0:03d}'.format(planet_index)]
    om = planet_params['w_{0:03d}'.format(planet_index)]
    ks = planet_params['k_{0:03d}'.format(planet_index)]
    rv_sys = planet_params['rvsys_{0:03d}'.format(planet_index)]
    trend = planet_params['trend_{0:03d}'.format(planet_index)]
    
    tp = exoplanetlib.timetrans_to_timeperi(tt, per, ecc, om)
    #tp = planet_params['tp_{0:03d}'.format(planet_index)]

    rv_trend = time * trend + rv_sys
    
    vel = exoplanetlib.rv_model(time, per, tp, ecc, om, ks) + rv_trend
    
    return vel


def calculate_models(times, calib_params, flare_params, planet_params) :
    
    n_datasets = len(times)
    
    flare_tags = get_flare_tags(flare_params, times)
    
    flux_models = []
    
    for i in range(n_datasets) :
        
        transit_models = np.full_like(times[i], 1.0)
        
        for j in range(len(planet_params)) :
            transit_models *= modelslib.batman_transit_model(times[i], planet_params[j], planet_index=j)
    
        calib = modelslib.calib_model(n_datasets, i, calib_params, times[i])
        flares = modelslib.flares_model(flare_params, flare_tags, i, times[i])
        
        flux_model = (calib + flares) * transit_models
        
        flux_models.append(flux_model)

    return flux_models


def plot_priors_model(times, fluxes, fluxerrs, priors) :
    
    n_datasets = priors["n_datasets"]
    n_planets = priors["n_planets"]
    n_flares = priors["n_flares"]

    planet_params = priors["planet_params"]
    calib_params = priors["calib_params"]
    flare_params = priors["flare_params"]
    
    flux_models = calculate_models(times, calib_params, flare_params, planet_params)

    for i in range(len(times)) :
        plt.errorbar(times[i], fluxes[i], yerr=fluxerrs[i], fmt='ko', alpha=0.3)
        plt.plot(times[i], flux_models[i], 'r-', lw=2)

    plt.xlabel(r"Time [BTJD]")
    plt.ylabel(r"Flux [e-/s]")
    plt.show()


#make a pairs plot from MCMC output
def pairs_plot(samples, output='') :

    font = {'size': 15}
    matplotlib.rc('font', **font)

    fig = corner.corner(samples, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])

    if output != '' :
        fig.savefig(output)
        plt.close(fig)
    else :
        plt.show()

def errfunc(theta, labels, calib_params, flare_params, planet_params, times, fluxes) :
    
    for i in range(len(planet_params)) :
        planet_params[i] = updateParams(planet_params[i], theta, labels)

    flare_params = updateParams(flare_params, theta, labels)
    calib_params = updateParams(calib_params, theta, labels)

    flux_models = calculate_models(times, calib_params, flare_params, planet_params)

    residuals = np.array([])
    for i in range(len(times)) :
        residuals = np.append(residuals, fluxes[i] - flux_models[i])

    return residuals


def errfunc_batman(theta, labels, calib_params, flare_params, planet_params, times, fluxes) :
    
    for i in range(len(planet_params)) :
        planet_params[i] = updateParams(planet_params[i], theta, labels)

    flare_params = updateParams(flare_params, theta, labels)
    calib_params = updateParams(calib_params, theta, labels)

    flux_models = calculate_models(times, calib_params, flare_params, planet_params)

    residuals = np.array([])
    for i in range(len(times)) :
        residuals = np.append(residuals, fluxes[i] - flux_models[i])

    return residuals


def update_calib_priors(calib_priors, calib_params, unc=0.05, prior_type="FIXED") :

    calib_polyorder = calib_priors['orderOfPolynomial']['object'].value
    n_datasets = calib_priors['ndatasets']['object'].value
    
    for i in range(n_datasets) :
        for j in range(calib_polyorder) :
            coeff_id = 'd{0:02d}c{1:1d}'.format(i,j)

            calib_priors[coeff_id]['type'] = prior_type
            if prior_type == "Normal" :
                cen, sig = calib_params[coeff_id], calib_params[coeff_id]*unc
                calib_priors[coeff_id]['object'] = priorslib.normal_parameter(np.array([cen, sig]))
            elif prior_type == "Uniform" :
                l_value, u_value = calib_params[coeff_id] - (unc/2)*calib_params[coeff_id], calib_params[coeff_id] + (unc/2)*calib_params[coeff_id]
                calib_priors[coeff_id]['object'] = priorslib.uniform_parameter(np.array([l_value, u_value]))
            elif prior_type == "FIXED" :
                calib_priors[coeff_id]['object'] = priorslib.constant_parameter(float(calib_params[coeff_id]))
            else :
                print("ERROR: prior type {} not recognized, exiting ... ".format(prior_type))
                exit()

    return calib_priors


def update_rvcalib_priors(rvcalib_priors, rvcalib_params, unc=0.05, prior_type="FIXED") :
    
    n_rvdatasets = rvcalib_priors['n_rvdatasets']['object'].value
    
    for i in range(int(n_rvdatasets)) :
        coeff_id = 'rv_d{0:02d}'.format(i)

        rvcalib_priors[coeff_id]['type'] = prior_type
        if prior_type == "Normal" :
            cen, sig = rvcalib_params[coeff_id], rvcalib_params[coeff_id]*unc
            rvcalib_priors[coeff_id]['object'] = priorslib.normal_parameter(np.array([cen, sig]))
        elif prior_type == "Uniform" :
            l_value, u_value = rvcalib_params[coeff_id] - (unc/2)*rvcalib_params[coeff_id], rvcalib_params[coeff_id] + (unc/2)*rvcalib_params[coeff_id]
            rvcalib_priors[coeff_id]['object'] = priorslib.uniform_parameter(np.array([l_value, u_value]))
        elif prior_type == "FIXED" :
            rvcalib_priors[coeff_id]['object'] = priorslib.constant_parameter(float(rvcalib_params[coeff_id]))
        else :
            print("ERROR: prior type {} not recognized, exiting ... ".format(prior_type))
            exit()

    return rvcalib_priors


def update_flare_priors(flare_priors, flare_params, unc=0.05, prior_type="FIXED") :

    n_flares = flare_priors["n_flares"]
    
    for i in range(n_flares) :
        tc_id = 'tc{0:04d}'.format(i)
        fwhm_id = 'fwhm{0:04d}'.format(i)
        amp_id = 'amp{0:04d}'.format(i)
        
        param_ids = [tc_id, fwhm_id, amp_id]
        
        for id in param_ids :
            flare_priors[id]['type'] = prior_type
            if prior_type == "Normal" :
                cen, sig = flare_params[id], flare_params[id]*unc
                flare_priors[id]['object'] = priorslib.normal_parameter(np.array([cen, sig]))
            elif prior_type == "Uniform" :
                l_value, u_value = flare_params[id] - (unc/2)*flare_params[id], flare_params[id] + (unc/2)*flare_params[id]
                flare_priors[id]['object'] = priorslib.uniform_parameter(np.array([l_value, u_value]))
            elif prior_type == "FIXED" :
                flare_priors[id]['object'] = priorslib.constant_parameter(float(flare_params[id]))
            else :
                print("ERROR: prior type {} not recognized, exiting ... ".format(prior_type))
                exit()

    return flare_priors


def fitTransits_ols(times, fluxes, fluxerrs, priors, flare_post_type="FIXED", flares_unc=0.05, calib_post_type="FIXED", calib_unc=0.05, verbose=False, plot=False) :

    posterior = deepcopy(priors)
    
    n_datasets = posterior["n_datasets"]
    n_planets = posterior["n_planets"]
    n_flares = posterior["n_flares"]
    
    calib_params = posterior["calib_params"]
    flare_params = posterior["flare_params"]
    planet_params = posterior["planet_params"]
    
    theta, labels, theta_priors = posterior["theta"], posterior["labels"], posterior["theta_priors"]
    
    if verbose :
        print("Free parameters before OLS fit:")
        for i in range(len(theta)) :
            print(labels[i], "=", theta[i])
    
    theta, success = optimize.leastsq(errfunc_batman, theta, args=(labels, calib_params, flare_params, planet_params, times, fluxes))

    for i in range(len(planet_params)) :
        planet_params[i] = updateParams(planet_params[i], theta, labels)
    flare_params = updateParams(flare_params, theta, labels)
    calib_params = updateParams(calib_params, theta, labels)

    if verbose :
        print("Free parameters after OLS fit:")
        for i in range(len(theta)) :
            print(labels[i], "=", theta[i])

    posterior["calib_params"] = calib_params
    posterior["flare_params"] = flare_params
    posterior["planet_params"] = planet_params

    posterior["theta"] = theta
    posterior["theta_priors"] = deepcopy(theta)

    #posterior["planet_priors"]
    posterior["calib_priors"] = update_calib_priors(posterior["calib_priors"], calib_params, unc=calib_unc, prior_type=calib_post_type)
    if posterior["flare_priors"] != {} :
        posterior["flare_priors"] = update_flare_priors(posterior["flare_priors"], flare_params, unc=flares_unc, prior_type=flare_post_type)

    # generate new theta and label vectors
    theta, labels, theta_priors = priorslib.get_theta_from_priors(posterior["planet_priors"], posterior["calib_priors"], posterior["flare_priors"])

    posterior["theta"] = theta
    posterior["labels"] = labels
    posterior["theta_priors"] = theta_priors

    if plot :
        flux_models = calculate_models(times, calib_params, flare_params, planet_params)
        
        for i in range(len(times)) :
            plt.errorbar(times[i], fluxes[i], yerr=fluxerrs[i], fmt='ko', alpha=0.3)
            plt.plot(times[i], flux_models[i], 'r-', lw=2)
    
        plt.xlabel(r"Time [BTJD]")
        plt.ylabel(r"Flux [e-/s]")
        plt.show()

    if verbose:
        for i in range(len(theta)) :
            print(labels[i],"=",theta[i])
        print(flare_params)
        print(calib_params)
        print(planet_params)

    return posterior


def reduce_lightcurves(times, fluxes, fluxerrs, calib_params, flare_params, remove_calib=True, remove_flares=True) :
    
    n_datasets = len(times)
    
    flare_tags = get_flare_tags(flare_params, times)
    
    outtime, outflux, outfluxerr = np.array([]), np.array([]), np.array([])
    
    for i in range(n_datasets) :

        reduced_flux = fluxes[i]
        reduced_fluxerr = fluxerrs[i]
        
        if remove_flares :
            flares = modelslib.flares_model(flare_params, flare_tags, i, times[i])
            reduced_flux -= flares
        
        if remove_calib :
            calib = modelslib.calib_model(n_datasets, i, calib_params, times[i])
            reduced_flux = reduced_flux / calib - 1.0
            reduced_fluxerr /= calib
        
        outtime = np.append(outtime, times[i])
        outflux = np.append(outflux, reduced_flux)
        outfluxerr = np.append(outfluxerr, reduced_fluxerr)

    return  outtime, outflux, outfluxerr


def plot_posterior_multimodel(times, fluxes, fluxerrs, posterior, output='', plot_prev_model=False) :
    font = {'size': 24}
    matplotlib.rc('font', **font)
    
    # calcualate flare time tags
    flare_tags = get_flare_tags(posterior['flare_params'], times)

    for i in range(len(times)):
        
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=False, gridspec_kw={'hspace': 0})
        
        calib = modelslib.calib_model(len(times), i, posterior['calib_params'], times[i])
        flares = modelslib.flares_model(posterior['flare_params'], flare_tags, i, times[i])
        transit_models = np.full_like(times[i], 1.0)
        for j in range(len(posterior["planet_params"])) :
            transit_models *= modelslib.batman_transit_model(times[i], posterior["planet_params"][j], planet_index=j)

        axs[0].scatter(times[i], fluxes[i], marker='o', s=10., alpha=0.5, edgecolors='tab:blue', facecolors='none', lw=.7, label=r"TESS data")
        if posterior['n_flares'] :
            axs[0].plot(times[i], calib*transit_models+flares, '-', color='orange', lw=2, label=r"Flares model")
        axs[0].plot(times[i], calib*transit_models, '-', color='brown', lw=2, label=r"Transit model")
        axs[0].plot(times[i], calib, '-', color='green', lw=2, label=r"Starspots model")

        epoch = np.round((np.mean(times[i]) - posterior["planet_params"][0]["tc_000"])/posterior["planet_params"][0]["per_000"])
        
        tc = posterior["planet_params"][0]["tc_000"] + epoch * posterior["planet_params"][0]["per_000"]
        
        axs[0].vlines(tc, ymin=np.min(calib), ymax=np.max(calib*transit_models), colors='darkgrey', ls='--', alpha=0.8, label=r"T$_c$={0:.6f}".format(tc))

        axs[0].legend(fontsize=11)
        axs[0].set_ylabel(r"Flux [e-/s]")

        """
            if plot_prev_model :
            # Previous model
            pt0, pper, pb, pr, pu = 1342.22, 30., 0.5, 0.028, [0.21, 0.]
            prev_orbit = xo.orbits.KeplerianOrbit(period=pper, t0=pt0, b=pb, m_star=0.5, r_star=0.75)
            # Compute a limb-darkened light curve using starry
            prev_transit_model = (xo.LimbDarkLightCurve(pu).get_light_curve(orbit=prev_orbit, r=pr, t=times[i]).eval())
            prev_light_curve = np.array(prev_transit_model[:,0], dtype=float) + 1.0
            axs[1].plot(times[i], prev_light_curve, '--', color='orange', lw=2, zorder=0.9, label="Previous model")
            """
        normflux = fluxes[i]/(calib+flares)
        normfluxerr = fluxerrs[i]/(calib+flares)
        axs[1].errorbar(times[i], normflux, yerr=normfluxerr, fmt='.', alpha=0.5, color='tab:blue', lw=.7, zorder=0, label="TESS reduced data")
        axs[1].plot(times[i], transit_models, '-', color='brown', lw=2, zorder=1, label="Transit model")
        resids = normflux-transit_models
        sigma = np.std(resids)
        y  = resids + np.min(transit_models) - 6*sigma
        axs[1].errorbar(times[i], y, yerr=normfluxerr, fmt='.', alpha=0.5, color='olive', lw=.7, zorder=0, label="Residuals")

        print("RMS of residuals: {0:.0f} ppm".format(sigma*1.e6))
        
        axs[1].vlines(tc, ymin=np.min(y), ymax=np.max(normflux)+sigma, colors='darkgrey', ls='--', alpha=0.8, zorder=1.1, label=r"T$_c$={0:.6f}".format(tc))
        axs[1].legend(fontsize=11,loc='upper left')

        axs[1].set_xlabel(r"Time [BTJD]")
        axs[1].set_ylabel(r"Relative flux")
        
        if output != '' :
            fig.savefig(output, bbox_inches='tight')
            plt.close(fig)
        else :
            plt.show()


def plot_all_transits(times, fluxes, fluxerrs, posterior, plot_prev_model=False, bindata=False, binsize=0.01, output='') :
    font = {'size': 24}
    matplotlib.rc('font', **font)
    
    # calcualate flare time tags
    flare_tags = get_flare_tags(posterior['flare_params'], times)

    t0 = posterior["planet_params"][0]["tc_000"]
    period = posterior["planet_params"][0]["per_000"]
    
    time, flux, fluxerr, residuals = [], [], [], []
    
    for i in range(len(times)):
        calib = modelslib.calib_model(len(times), i, posterior['calib_params'], times[i])
        flares = modelslib.flares_model(posterior['flare_params'], flare_tags, i, times[i])
        transit_models = np.full_like(times[i], 1.0)
        for j in range(len(posterior["planet_params"])) :
            transit_models *= modelslib.batman_transit_model(times[i], posterior["planet_params"][j], planet_index=j)
    
        epoch = np.round((np.mean(times[i]) - t0)/period)
        tc = t0 + epoch * period
        
        normflux = fluxes[i]/(calib+flares)
        normfluxerr = fluxerrs[i]/(calib+flares)
        resids = normflux-transit_models
        
        time = np.append(time, times[i]-tc)
        flux = np.append(flux, normflux)
        fluxerr = np.append(fluxerr, normfluxerr)
        residuals = np.append(residuals, resids)

     
    sorted = np.argsort(time)
    
    flux, fluxerr, residuals = flux[sorted], fluxerr[sorted], residuals[sorted]
    time = time[sorted]
    sigma = np.std(residuals)
    print("RMS of residuals: {0:.0f} ppm".format(sigma*1.e6))
    resid_offset = np.min(flux - residuals) - 6*sigma
    
    ti,tf =  time[0]+t0, time[-1]+t0
    highsamptime = np.arange(ti, tf, 0.0005)
    
    highsamp_transit_model = np.full_like(highsamptime, 1.0)
    for j in range(len(posterior["planet_params"])) :
        highsamp_transit_model *= modelslib.batman_transit_model(highsamptime, posterior["planet_params"][j], planet_index=j)

    if bindata :
        bin_time, bin_flux, bin_fluxerr = bin_data(time, flux, fluxerr, median=False, binsize=binsize)
        bin_transit_model = np.full_like(bin_time, 1.0)
        for j in range(len(posterior["planet_params"])) :
            bin_transit_model *= modelslib.batman_transit_model(bin_time+t0, posterior["planet_params"][j], planet_index=j)
    
    if plot_prev_model :
        # Previous model
        diffb, diffc = np.abs(1330.39153 - t0), np.abs(1342.22 - t0)
        if diffb < diffc :
            pt0, pper, pa, pinc, pr, pu = 1330.39153, 8.46321, 19.1, 89.5, 0.0514, [0.21, 0.]
        else :
            pt0, pper, pa, pinc, pr, pu = 1342.22, 30., 40., 89.28, 0.028, [0.21, 0.]

        prev_light_curve = modelslib.batman_model(highsamptime, pper, pt0, pa, pinc, pr, pu[0], u1=pu[1], ecc=0., w=90.)
        
        plt.plot(highsamptime-t0, prev_light_curve, '--', color='darkorange', lw=3, zorder=0.9, label="Previous model")

    plt.errorbar(time, flux, yerr=fluxerr, fmt='.', alpha=0.2, color='tab:blue', lw=.7, zorder=0, label="TESS reduced data")
    #plt.scatter(time, flux, s=10., marker='o', edgecolors='tab:blue', facecolors='none', lw=0.5, label="TESS reduced data")
    plt.errorbar(time, residuals+resid_offset, yerr=fluxerr, fmt='.', alpha=0.2, color='olive', lw=.7, zorder=0, label="Residuals")
    #plt.scatter(time, residuals+resid_offset, s=10., marker='o', edgecolors='olive', facecolors='none', lw=0.5, label="Residuals")

    if bindata :
        plt.errorbar(bin_time, bin_flux, yerr=bin_fluxerr, fmt='o', lw=2, alpha=0.8, color='darkblue', zorder=1.1)
        plt.errorbar(bin_time, (bin_flux-bin_transit_model)+resid_offset, yerr=bin_fluxerr, fmt='o', lw=2, alpha=0.8, color='darkgreen', zorder=1.1)

    plt.plot(highsamptime-t0, highsamp_transit_model, '-', color='darkred', lw=3, zorder=1, label="Transit model")

    plt.legend(fontsize=20)
    plt.xlabel(r"BTJD - T$_c$")
    plt.ylabel(r"Relative flux")
    
    if output != '' :
        plt.savefig(output, bbox_inches='tight')
        plt.clf()
    else :
        plt.show()


def odd_ratio_mean(value, err, odd_ratio = 1e-4, nmax = 10):
    #
    # Provide values and corresponding errors and compute a
    # weighted mean
    #
    #
    # odd_bad -> probability that the point is bad
    #
    # nmax -> number of iterations
    keep = np.isfinite(value)*np.isfinite(err)
    
    if np.sum(keep) == 0:
        return np.nan, np.nan

    value = value[keep]
    err = err[keep]

    guess = np.nanmedian(value)
    guess_err = np.nanmedian(np.abs(value - guess)) / 0.67449

    nite = 0
    while (nite < nmax):
        nsig = (value-guess)/np.sqrt(err**2 + guess_err**2)
        gg = np.exp(-0.5*nsig**2)
        odd_bad = odd_ratio/(gg+odd_ratio)
        odd_good = 1-odd_bad
        
        w = odd_good/(err**2 + guess_err**2)
        
        guess = np.nansum(value*w)/np.nansum(w)
        guess_err = np.sqrt(1/np.nansum(w))
        nite+=1

    return guess, guess_err



def bin_data(x, y, yerr, median=False, binsize = 0.005) :

    xi, xf = x[0], x[-1]
    
    bins = np.arange(xi, xf, binsize)
    digitized = np.digitize(x, bins)
    
    bin_y, bin_yerr = [], []
    bin_x = []
    
    for i in range(0,len(bins)+1):
        if len(x[digitized == i]) :
            if median :
                mean_y = np.nanmedian(y[digitized == i])
                myerr = np.nanmedian(np.abs(y[digitized == i] - mean_y))  / 0.67449
                #weights = 1/(yerr[digitized == i]**2)
                #mean_y = np.average(y[digitized == i], weights=weights)
                #myerr = np.sqrt(np.sum((weights**2)*(yerr[digitized == i]**2)))
            else :
                mean_y, myerr = odd_ratio_mean(y[digitized == i], yerr[digitized == i])
                #mean_y = np.nanmean(y[digitized == i])
                #myerr = np.nanstd(y[digitized == i])
            
            bin_x.append(np.nanmean(x[digitized == i]))
            bin_y.append(mean_y)
            bin_yerr.append(myerr)

    bin_y, bin_yerr = np.array(bin_y), np.array(bin_yerr)
    bin_x = np.array(bin_x)

    return bin_x, bin_y, bin_yerr


def fitTransitsAndRVsWithMCMC(tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, priors, amp=1e-4, nwalkers=32, niter=100, burnin=20, samples_filename="default_samples.h5", appendsamples=False, return_samples=True, rvlabel="",addnparams=0, verbose=False, plot=False, plot_individual_transits=False) :
    
    posterior = deepcopy(priors)
    
    calib_params = posterior["calib_params"]
    rvcalib_params = posterior["rvcalib_params"]
    flare_params = posterior["flare_params"]
    planet_params = posterior["planet_params"]
    
    theta, labels, theta_priors = posterior["theta"], posterior["labels"], posterior["theta_priors"]
    
    if verbose :
        print("Free parameters before MCMC fit:")
        for i in range(len(theta)) :
            print(labels[i], "=", theta[i])

    if verbose:
        print("initializing emcee sampler ...")

    #amp, ndim, nwalkers, niter, burnin = 5e-4, len(theta), 50, 2000, 500
    ndim = len(theta)

    # Make sure the number of walkers is sufficient, and if not passing a new value
    if nwalkers < 2*len(posterior["theta"]):
        nwalkers = 2*len(posterior["theta"])

    # Set up the backend
    backend = emcee.backends.HDFBackend(samples_filename)
    if appendsamples == False :
        backend.reset(nwalkers, ndim)

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_tr_rv, args = [theta_priors, labels, calib_params, rvcalib_params, flare_params, planet_params, tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs], backend=backend)

    pos = [theta + amp * np.random.randn(ndim) for i in range(nwalkers)]
    #--------

    #- run mcmc
    if verbose:
        print("Running MCMC ...")
        print("N_walkers=",nwalkers," ndim=",ndim)

    sampler.run_mcmc(pos, niter, progress=True)
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim)) # burnin : number of first samples to be discard as burn-in
    #--------

    #- append samples to posterior
    if return_samples :
        posterior["samples"] = samples
    #----------

    #- calib fit parameters
    if verbose:
        print("Obtaining best fit calibration parameters from pdfs ...")
    calib_params, calib_theta_fit, calib_theta_labels, calib_theta_err = best_fit_params(posterior["calib_params"], labels, samples)
    if verbose :
        print("CALIBRATION Fit parameters:")
        for i in range(len(calib_theta_fit)) :
            print(calib_theta_labels[i], "=", calib_theta_fit[i], "+", calib_theta_err[i][0], "-", calib_theta_err[i][1])
        print("----------------")

    if posterior["calib_priorsfile"] != "":
        calib_posterior = posterior["calib_priorsfile"].replace(".pars", "_posterior.pars")
    else :
        calib_posterior = "calibration_posterior.pars"
    if verbose:
        print("Output CALIBRATION posterior: ", calib_posterior)
    # save posterior of calibration parameters into file:
    ncoeff=posterior["calib_priors"]['orderOfPolynomial']['object'].value
    priorslib.save_posterior(calib_posterior, calib_params, calib_theta_fit, calib_theta_labels, calib_theta_err, calib=True, ncoeff=ncoeff)

    # update cablib parameters in output posterior
    posterior["calib_params"] = calib_params
    #------------

    #- RV calib fit parameters
    if verbose:
        print("Obtaining best fit RV calibration parameters from pdfs ...")
        rvcalib_params, rvcalib_theta_fit, rvcalib_theta_labels, rvcalib_theta_err = best_fit_params(posterior["rvcalib_params"], labels, samples)
    if verbose :
        print("RV CALIBRATION Fit parameters:")
        for i in range(len(rvcalib_theta_fit)) :
            print(rvcalib_theta_labels[i], "=", rvcalib_theta_fit[i], "+", rvcalib_theta_err[i][0], "-", rvcalib_theta_err[i][1])
        print("----------------")
    rvcalib_posterior = "rv_calibration_posterior.pars"
    if verbose:
        print("Output RV CALIBRATION posterior: ", calib_posterior)
    priorslib.save_posterior(rvcalib_posterior, rvcalib_params, rvcalib_theta_fit, rvcalib_theta_labels, rvcalib_theta_err)
    
    # update cablib parameters in output posterior
    posterior["rvcalib_params"] = rvcalib_params
    #------------

    if posterior["n_flares"] :
        if verbose:
            print("Obtaining best fit flare parameters from pdfs ...")
        
        flare_params, flare_theta_fit, flare_theta_labels, flare_theta_err = best_fit_params(posterior["flare_params"], labels, samples)

        if verbose :
            print("FLARE Fit parameters:")
            for i in range(len(flare_theta_fit)) :
                print(flare_theta_labels[i], "=", flare_theta_fit[i], "+", flare_theta_err[i][0], "-", flare_theta_err[i][1])
            print("----------------")

        flare_posterior = posterior["flare_priorsfile"].replace(".pars", "_posterior.pars")
        if verbose:
            print("Output FLARE posterior: ", flare_posterior)
        # save posterior of flare parameters into file:
        priorslib.save_posterior(flare_posterior, flare_params, flare_theta_fit, flare_theta_labels, flare_theta_err)

    # update flare parameters in output posterior
    posterior["flare_params"] = flare_params

    if verbose:
        print("Obtaining best fit planet parameters from pdfs ...")
    planet_params, planet_theta_fit, planet_theta_labels = [], [], []
    for i in range(posterior["n_planets"]) :
        pl_params, pl_theta_fit, pl_theta_labels, pl_theta_err = best_fit_params(posterior["planet_params"][i], labels, samples)
        # update flare parameters in output posterior
        posterior["planet_params"][i] = pl_params

        planet_params.append(pl_params)
        planet_theta_fit = np.concatenate((planet_theta_fit, pl_theta_fit), axis=0)
        planet_theta_labels = np.concatenate((planet_theta_labels, pl_theta_labels), axis=0)
        if verbose:
            # print out best fit parameters and errors
            print("----------------")
            print("PLANET {} Fit parameters:".format(i))
            for j in range(len(planet_theta_fit)) :
                print(pl_theta_labels[j], "=", pl_theta_fit[j], "+", pl_theta_err[j][0], "-", pl_theta_err[j][1])
                """
                    if "tc" in pl_theta_labels[j] :
                        per = pl_params['per_{0:03d}'.format(i)]
                        tt = pl_params['tc_{0:03d}'.format(i)]
                        ecc = pl_params['ecc_{0:03d}'.format(i)]
                        om = pl_params['w_{0:03d}'.format(i)]
                        tp = exoplanetlib.timetrans_to_timeperi(tt, per, ecc, om)
                        print("Tp = {}".format(tp))
                    """
            print("----------------")

        planet_posterior = posterior["planet_priors_files"][i].replace(".pars", "_posterior.pars")
        if verbose:
            print("Output PLANET {0} posterior: ".format(i), planet_posterior)
        
        # save posterior of planet parameters into file:
        priorslib.save_posterior(planet_posterior, pl_params, pl_theta_fit, pl_theta_labels, pl_theta_err)

    # Update theta in posterior
    if posterior["n_flares"] :
        theta_tuple  = (calib_theta_fit, flare_theta_fit, planet_theta_fit)
        labels_tuple  = (calib_theta_labels, flare_theta_labels, planet_theta_labels)
    else :
        theta_tuple  = (calib_theta_fit, planet_theta_fit)
        labels_tuple  = (calib_theta_labels, planet_theta_labels)

    posterior['theta'] = np.hstack(theta_tuple)
    posterior['labels'] = np.hstack(labels_tuple)
    #posterior["theta_priors"] = ?

    posterior['BIC'] = calculate_bic(posterior, tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, addnparams=addnparams)
    
    posterior['BIC_NOPLANET'] = calculate_bic(posterior, tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, rvnull=True)

    if verbose :
        print("BIC = {}, BIC (no planet)={}".format(posterior['BIC'],posterior['BIC_NOPLANET']))


    if plot :
        if plot_individual_transits :
            plot_posterior_multimodel(tr_times, fluxes, fluxerrs, posterior, plot_prev_model=False)
        plot_all_transits(tr_times, fluxes, fluxerrs, posterior, bindata=True, binsize=0.005)
        
        plot_rv_timeseries(planet_params, rvcalib_params, rv_times, rvs, rverrs, samples=samples, labels=labels, planet_index=0, plot_residuals=True, phasefold=False, rvlabel=rvlabel)
        
        #t0 = posterior["planet_params"][0]['tc_{0:03d}'.format(0)] + 0.5 * posterior["planet_params"][0]['per_{0:03d}'.format(0)]
        t0 = posterior["planet_params"][0]['tc_{0:03d}'.format(0)]
        plot_rv_timeseries(planet_params, rvcalib_params, rv_times, rvs, rverrs, samples=samples, labels=labels, planet_index=0, plot_residuals=True, phasefold=True, plot_bin_data=True, rvlabel=rvlabel, t0=t0)
        #plot model
        #plot_priors_model(tr_times, fluxes, fluxerrs, posterior)
        
        #- make a pairs plot from MCMC output:
        #flat_samples = sampler.get_chain(discard=burnin,flat=True)
        pairs_plot_emcee(samples, labels, calib_params, planet_params[0], output='pairsplot.png', rvcalib_params=rvcalib_params, addlabels=True)

    return posterior


def calculate_bic(posterior, tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, addnparams=0, rvnull=False) :
    
    maximum_likelihood = lnprob_tr_rv(posterior['theta'], posterior['theta_priors'], posterior['labels'], posterior['calib_params'], posterior['rvcalib_params'], posterior['flare_params'], posterior['planet_params'], tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, rvnull=rvnull)
    #maximum_likelihood = lnlikelihood_tr_rv(posterior['theta'], posterior['labels'], posterior['calib_params'], posterior['rvcalib_params'], posterior['flare_params'], posterior['planet_params'], tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs)
    
    number_of_free_params = len(posterior['theta']) + addnparams
    if rvnull :
        number_of_free_params -= 1
    number_of_datapoints = 0
    for i in range(len(fluxes)) :
        number_of_datapoints += len(fluxes[i])
    for i in range(len(rvs)) :
        number_of_datapoints += len(rvs[i])

    print(number_of_free_params,number_of_datapoints,maximum_likelihood)
    
    bic = number_of_free_params * np.log(number_of_datapoints) - 2 * maximum_likelihood

    return bic


def calculate_bic_gp (posterior, tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, gp_params, gp_rv, gp_blong, gp_phot, rv_gp_feed, blong_gp_feed, phot_gp_feed, spec_constrained=False, rvnull=False) :
    
    maximum_likelihood = lnprob_tr_rv_gp(posterior['theta'], posterior['theta_priors'], posterior['labels'], posterior['calib_params'], posterior['rvcalib_params'], posterior['flare_params'], posterior['planet_params'], tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, gp_params, gp_rv, gp_blong, gp_phot, rv_gp_feed, blong_gp_feed, phot_gp_feed, spec_constrained, rvnull=rvnull)
    
    number_of_free_params = len(posterior['theta'])
    if rvnull :
        number_of_free_params -= 1

    number_of_datapoints = 0
    for i in range(len(fluxes)) :
        number_of_datapoints += len(fluxes[i])
    for i in range(len(rvs)) :
        number_of_datapoints += len(rvs[i])

    print(number_of_free_params,number_of_datapoints,maximum_likelihood)
    
    bic = number_of_free_params * np.log(number_of_datapoints) - 2 * maximum_likelihood

    return bic


def errfunc_transit_rv(theta, labels, calib_params, rvcalib_params, flare_params, planet_params, tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs) :
    
    for i in range(len(planet_params)) :
        planet_params[i] = updateParams(planet_params[i], theta, labels)

    flare_params = updateParams(flare_params, theta, labels)
    calib_params = updateParams(calib_params, theta, labels)
    rvcalib_params = updateParams(rvcalib_params, theta, labels)

    flux_models = calculate_models(tr_times, calib_params, flare_params, planet_params)

    residuals = np.array([])
    for i in range(len(tr_times)) :
        residuals = np.append(residuals, (fluxes[i] - flux_models[i])/fluxerrs[i])

    for i in range(len(rvs)) :
        coeff_id = 'rv_d{0:02d}'.format(i)
        rvcalib = rvcalib_params[coeff_id]
        rvmodel = calculate_rv_model(rv_times[i], planet_params, planet_index=0) + rvcalib
        rv_residuals = (rvs[i] - rvmodel) / rverrs[i]
        residuals = np.append(residuals, rv_residuals)

    return residuals



def fitTransits_and_RVs_ols(tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, priors, fix_eccentricity=False, flare_post_type="FIXED", flares_unc=0.05, calib_post_type="FIXED", rvcalib_post_type="FIXED", calib_unc=0.05, verbose=False, plot=False, rvlabel="") :

    posterior = deepcopy(priors)
    
    n_datasets = posterior["n_datasets"]
    n_rvdatasets = posterior["n_rvdatasets"]
    n_planets = posterior["n_planets"]
    n_flares = posterior["n_flares"]
    
    calib_params = posterior["calib_params"]
    rvcalib_params = posterior["rvcalib_params"]
    flare_params = posterior["flare_params"]
    planet_params = posterior["planet_params"]
    
    theta, labels, theta_priors = posterior["theta"], posterior["labels"], posterior["theta_priors"]
    
    if fix_eccentricity :
        theta_copy, labels_copy = np.array([]), []
        for i in range(len(labels)) :
            #if ("ecc_" not in labels[i]) and ("w_" not in labels[i]) :
            if ("ecc_" not in labels[i]) and ("w_" not in labels[i]) and ("u0_" not in labels[i]) and ("u1_" not in labels[i])  :
                theta_copy = np.append(theta_copy,theta[i])
                labels_copy.append(labels[i])
        print("Free parameters before OLS fit:")
        for i in range(len(theta_copy)) :
            print(labels_copy[i], "=", theta_copy[i])
        theta_copy, success = optimize.leastsq(errfunc_transit_rv, theta_copy, args=(labels_copy, calib_params, rvcalib_params, flare_params, planet_params,tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs))

        for i in range(len(labels)) :
            for j in range(len(labels_copy)) :
                if labels_copy[j] == labels[i] :
                    theta[i] = theta_copy[j]
                    break
    else :
        print("Free parameters before OLS fit:")
        for i in range(len(theta)) :
            print(labels[i], "=", theta[i])
        theta, success = optimize.leastsq(errfunc_transit_rv, theta, args=(labels, calib_params, rvcalib_params, flare_params, planet_params, tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs))


    for i in range(len(planet_params)) :
        planet_params[i] = updateParams(planet_params[i], theta, labels)
    flare_params = updateParams(flare_params, theta, labels)
    calib_params = updateParams(calib_params, theta, labels)
    rvcalib_params = updateParams(rvcalib_params, theta, labels)

    print("Free parameters after OLS fit:")
    for i in range(len(theta)) :
        print(labels[i], "=", theta[i])

    posterior["calib_params"] = calib_params
    posterior["rvcalib_params"] = rvcalib_params
    posterior["flare_params"] = flare_params
    posterior["planet_params"] = planet_params

    #posterior["planet_priors"]
    posterior["calib_priors"] = update_calib_priors(posterior["calib_priors"], calib_params, unc=calib_unc, prior_type=calib_post_type)
    posterior["rvcalib_priors"] = update_rvcalib_priors(posterior["rvcalib_priors"], rvcalib_params, unc=calib_unc, prior_type=rvcalib_post_type)
    if posterior["flare_priors"] != {} :
        posterior["flare_priors"] = update_flare_priors(posterior["flare_priors"], flare_params, unc=flares_unc, prior_type=flare_post_type)

    # generate new theta and label vectors
    theta, labels, theta_priors = priorslib.get_theta_from_priors(posterior["planet_priors"], posterior["calib_priors"], posterior["flare_priors"], rvcalib_priors=posterior["rvcalib_priors"])

    # update initial guess in the priors of free parameters with the OLS fit values
    for key in theta_priors.keys() :
        for i in range(len(planet_params)) :
            if key in planet_params[i].keys() :
                #print(key,theta_priors[key]['type'], theta_priors[key]['object'].value, theta_priors[key]['object'].get_ln_prior())
                theta_priors[key]['object'].value = planet_params[i][key]

    """
    for i in range(len(labels)) :
        for j in range(len(planet_params)) :
            if labels[i] in planet_params[j].keys() :
                theta[i] = planet_params[j][labels[i]]
    """
    posterior["theta"] = theta
    posterior["labels"] = labels
    posterior["theta_priors"] = theta_priors

    if plot :
        flux_models = calculate_models(tr_times, calib_params, flare_params, planet_params)
        for i in range(len(tr_times)) :
            plt.errorbar(tr_times[i], fluxes[i], yerr=fluxerrs[i], fmt='ko', alpha=0.3)
            plt.plot(tr_times[i], flux_models[i], 'r-', lw=2)
    
        plt.xlabel(r"Time [BTJD]")
        plt.ylabel(r"Flux [e-/s]")
        plt.show()

        plot_rv_timeseries(planet_params, rvcalib_params, rv_times, rvs, rverrs, planet_index=0, plot_residuals=True,  phasefold=False, rvlabel=rvlabel)
        
        #t0 = posterior["planet_params"][0]['tc_{0:03d}'.format(0)] + 0.5 * posterior["planet_params"][0]['per_{0:03d}'.format(0)]
        t0 = posterior["planet_params"][0]['tc_{0:03d}'.format(0)]
        plot_rv_timeseries(planet_params, rvcalib_params, rv_times, rvs, rverrs, planet_index=0, plot_residuals=True,  phasefold=True, plot_bin_data=True, rvlabel=rvlabel,t0=t0)

    if verbose:
        for i in range(len(theta)) :
            print(labels[i],"=",theta[i])
        print(flare_params)
        print(calib_params)
        print(rvcalib_params)
        print(planet_params)

    return posterior


def get_rv_model(planets_params, T0, ti, tf, timesampling=0.001, planet_index=0, phasefold=False, plot=False):

    #print("get_rv_model() -> T0={0:.6f} ti={1:.6f} tf={2:.6f}".format(T0, ti, tf))
    
    planet_params = planets_params[planet_index]
    
    per = planet_params['per_{0:03d}'.format(planet_index)]
    tt = planet_params['tc_{0:03d}'.format(planet_index)]
    ecc = planet_params['ecc_{0:03d}'.format(planet_index)]
    om = planet_params['w_{0:03d}'.format(planet_index)]
    #tp = planet_params['tp_{0:03d}'.format(planet_index)]
    tp = exoplanetlib.timetrans_to_timeperi(tt, per, ecc, om)
    inc = planet_params['inc_{0:03d}'.format(planet_index)]
    k = planet_params['k_{0:03d}'.format(planet_index)]
    mstar = planet_params['ms_{0:03d}'.format(planet_index)]
    rstar = planet_params['rs_{0:03d}'.format(planet_index)]
    rp = planet_params['rp_{0:03d}'.format(planet_index)]
    rv_sys = planet_params['rvsys_{0:03d}'.format(planet_index)]
    trend = planet_params['trend_{0:03d}'.format(planet_index)]
    
    #print(T0, per, tp, ecc, om, k)
    
    time = np.arange(ti, tf, timesampling)
    rvmodel = exoplanetlib.rv_model(time, per, tp, ecc, om, k)

    if phasefold :
        phases = foldAt(time, per, T0=T0)
        obs_sort = np.argsort(phases)
        if plot :
            plt.plot(phases[obs_sort], rvmodel[obs_sort], '-', color='red', lw=0.2, alpha=0.2)
        return phases[obs_sort], rvmodel[obs_sort]
    else :
        model_rvsys = rv_sys + time*trend
        rvmodel += model_rvsys
        if plot :
            plt.plot(time, rvmodel, '-', color='red', lw=0.4, alpha=0.4)
        return time, rvmodel


def plot_rv_timeseries(planets_params, rvcalib_params, bjds, rvs, rverrs, samples=None, labels=None, nsamples=100, planet_index=0, plot_residuals=False, plot_bin_data=True, phasefold=False, t0=0, rvlabel='') :

    legends, fmts, colors = [],[],[]
    if len(rvs) == 1 :
        legends=[rvlabel]
        fmts = ["o"]
        colors = ["k"]
    elif len(rvs) == 2 :
        legends=["SPIRou CCF RV", "SPIRou LBL RV"]
        fmts = ["o","o"]
        colors = ["k",'#1f77b4']
    else :
        for i in range(len(rvs)) :
            legends.append("RV")
            fmts.append("o")
            colors.append("k")

    min_bjd, max_bjd = 1e20, -1e20
    for i in range(len(bjds)) :
        if np.nanmin(bjds[i]) < min_bjd :
            min_bjd = np.nanmin(bjds[i])
        if np.nanmax(bjds[i]) > max_bjd :
            max_bjd = np.nanmax(bjds[i])
    
    if t0 == 0:
        t0 = min_bjd

    rv_sys = planets_params[planet_index]['rvsys_{0:03d}'.format(planet_index)]
    trend = planets_params[planet_index]['trend_{0:03d}'.format(planet_index)]

    time, meanrvmodel = get_rv_model(planets_params, t0, min_bjd, max_bjd, timesampling=0.001, phasefold=phasefold)

    if samples is not None and labels is not None:
        rvsamplemodels = []
        copy_planet_params = deepcopy(planets_params[planet_index])
        for theta in samples[np.random.randint(len(samples), size=nsamples)]:
            copy_planet_params = updateParams(copy_planet_params, theta, labels)
            _, rvm = get_rv_model([copy_planet_params], t0, min_bjd, max_bjd, timesampling=0.001,phasefold=phasefold)
            rvsamplemodels.append(rvm)
        rvsamplemodels = np.array(rvsamplemodels, dtype=float)
        rms = np.nanstd(rvsamplemodels,axis=0)
    else :
        rms = np.zeros_like(meanrvmodel)

    # set up fig
    if plot_residuals :
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=False, gridspec_kw={'hspace': 0, 'height_ratios': [2, 1]})
        axs0 = axs[0]
    else :
        fig, axs0 = plt.subplots(1, 1, figsize=(12, 6), sharex=True, sharey=False, gridspec_kw={'hspace': 0})

    color = 'green'
    #color = "#ff7f0e"
    axs0.plot(time, meanrvmodel, color=color, label='fit model')
    axs0.fill_between(time, meanrvmodel+rms, meanrvmodel-rms, color=color, alpha=0.3,edgecolor="none")

    rv_res = np.array([])

    for i in range(len(rvs)) :
        
        bjd, rv, rverr = bjds[i], rvs[i], rverrs[i]

        coeff_id = 'rv_d{0:02d}'.format(i)
        rvcalib = rvcalib_params[coeff_id]
        
        obs_rvsys = rv_sys + bjd * trend
        
        rvmodel = calculate_rv_model(bjd, planets_params, planet_index=0)
        obs_model = rvmodel + rvcalib
        reduced_rv = rv - obs_rvsys - rvcalib
        residuals =  reduced_rv - rvmodel

        sig_res = np.nanstd(residuals)
        rv_res = np.append(rv_res, residuals)
        
        if phasefold :
            per = planets_params[planet_index]['per_{0:03d}'.format(planet_index)]
            obs_phases = foldAt(bjd, per, T0=t0)
            obs_sort = np.argsort(obs_phases)
            
            if plot_bin_data :
                alpha, lw = 0.1, 0.5
                bin_ph, bin_rv, bin_rverr = bin_data(obs_phases[obs_sort], reduced_rv[obs_sort], rverr[obs_sort], median=False, binsize = 0.05)
                resbin_ph, bin_residuals, resbin_rverr = bin_data(obs_phases[obs_sort], residuals[obs_sort], rverr[obs_sort], median=False, binsize = 0.05)
            else :
                alpha, lw = 1, 1
            
            if i == len(rvs) - 1 :
                if plot_bin_data :
                    axs0.errorbar(bin_ph, bin_rv, yerr=bin_rverr, fmt=fmts[i], color=colors[i], lw=3, label=legends[i])
                    axs0.errorbar(obs_phases[obs_sort], reduced_rv[obs_sort], yerr=rverr[obs_sort], fmt=fmts[i], color=colors[i], alpha=alpha, lw=lw)
                else :
                    axs0.errorbar(obs_phases[obs_sort], reduced_rv[obs_sort], yerr=rverr[obs_sort], fmt=fmts[i], color=colors[i], label=legends[i], alpha=alpha, lw=lw)
                
                if plot_residuals :
                    if plot_bin_data :
                        axs[1].errorbar(resbin_ph, bin_residuals, yerr=resbin_rverr, fmt=fmts[i], color=colors[i], lw=3)
                    axs[1].errorbar(obs_phases[obs_sort], residuals[obs_sort], yerr=rverr[obs_sort], fmt=fmts[i], color=colors[i], alpha=alpha, lw=lw)
                    axs[1].set_xlabel("phase (P={0:.3f} d)".format(per), fontsize=16)
                else :
                    axs0.set_xlabel("phase (P={0:.3f} d)".format(per), fontsize=16)
            else :
                if legends[i] == 'RV' :
                    axs0.errorbar(obs_phases[obs_sort], reduced_rv[obs_sort], yerr=rverr[obs_sort], fmt=fmts[i], color=colors[i], alpha=alpha, lw=lw)
                else :
                    axs0.errorbar(obs_phases[obs_sort], reduced_rv[obs_sort], yerr=rverr[obs_sort], fmt=fmts[i], color=colors[i], label=legends[i], alpha=alpha, lw=lw)

                if plot_residuals :
                    if plot_bin_data :
                        axs[1].errorbar(resbin_ph, bin_residuals, yerr=resbin_rverr, fmt=fmts[i], color=colors[i], lw=3)
                    axs[1].errorbar(obs_phases[obs_sort], residuals[obs_sort], yerr=rverr[obs_sort], fmt=fmts[i], color=colors[i], alpha=alpha, lw=lw)

        else :
            if i == len(rvs) - 1 :
                axs0.errorbar(bjd, rv-rvcalib, yerr=rverr, fmt=fmts[i], color=colors[i], label=legends[i])
                if plot_residuals :
                    axs[1].errorbar(bjd, residuals, yerr=rverr, fmt=fmts[i], color=colors[i])
                    axs[1].set_xlabel("Time [BJD]", fontsize=16)
                else :
                    axs0.set_xlabel("Time [BJD]", fontsize=16)
            else :
                if legends[i] == 'RV' :
                    axs0.errorbar(bjd, rv-rvcalib, yerr=rverr, fmt=fmts[i], color=colors[i])
                else :
                    axs0.errorbar(bjd, rv-rvcalib, yerr=rverr, fmt=fmts[i], color=colors[i], label=legends[i])

                if plot_residuals :
                    axs[1].errorbar(bjd, residuals, yerr=rverr, fmt=fmts[i], color=colors[i])

                    
    axs0.set_ylabel(r"RV [m/s]", fontsize=16)
    axs0.legend(fontsize=16)
    axs0.tick_params(axis='x', labelsize=14)
    axs0.tick_params(axis='y', labelsize=14)
    
    if plot_residuals :
        if phasefold :
            axs[1].hlines(0., 0.0, 1.0, color="k", linestyles=":", lw=0.6)
        else :
            axs[1].hlines(0., min_bjd, max_bjd, color="k", linestyles=":", lw=0.6)

        axs[1].set_ylim(-5*sig_res,+5*sig_res)
        axs[1].set_ylabel(r"Residuals [m/s]", fontsize=16)
        axs[1].tick_params(axis='x', labelsize=14)
        axs[1].tick_params(axis='y', labelsize=14)

    print("RMS of RV residuals: {:.2f} m/s".format(np.nanstd(rv_res)))
    n = len(residuals)
    m = 0 # the correct would be to pull the number of free parameters in the fit
    chi2 = np.sum((residuals/rverr)**2) / (n - m)
    print("Reduced chi-square (n={}, DOF={}): {:.2f}".format(n,n-m,chi2))

    plt.show()


def plot_gp_photometry(lc, posterior, gp, y, binsize, phaseplot=False) :

    x = np.linspace(lc['time'][0], lc['time'][-1], 10000)

    bin_tbjd, bin_flux, bin_fluxerr = bin_data(lc['time'], lc['nflux'], lc['nfluxerr'], median=False, binsize=binsize)

    transit_model_obs = np.full_like(lc['time'], 1.0)
    transit_model_binobs = np.full_like(bin_tbjd, 1.0)
    transit_model = np.full_like(x, 1.0)
    for j in range(len(posterior["planet_params"])) :
        transit_model *= modelslib.batman_transit_model(x, posterior["planet_params"][j], planet_index=j)
        transit_model_obs *= modelslib.batman_transit_model(lc['time'], posterior["planet_params"][j], planet_index=j)
        transit_model_binobs *= modelslib.batman_transit_model(bin_tbjd, posterior["planet_params"][j], planet_index=j)
    
    pred_mean, pred_var = gp.predict(y, x, return_var=True)
    pred_std = np.sqrt(pred_var)
    
    pred_mean_obs, pred_var_obs = gp.predict(y, lc['time'], return_var=True)
    pred_std_obs = np.sqrt(pred_var_obs)

    pred_mean_binobs, pred_var_binobs = gp.predict(y, bin_tbjd, return_var=True)
    pred_std_binobs = np.sqrt(pred_var_binobs)

    residuals = lc['nflux'] - pred_mean_obs * transit_model_obs
    binresiduals = bin_flux - pred_mean_binobs * transit_model_binobs

    print("RMS of Flux residuals: {:.5f} ppm".format(np.nanstd(residuals)*1e6))
    print("RMS of binned Flux residuals: {:.5f} ppm".format(np.nanstd(binresiduals)*1e6))

    plt.plot(lc['time'], lc['nflux'], ".", color="darkgrey", alpha=0.1, label="TESS data", zorder=0.5)
    plt.errorbar(bin_tbjd, bin_flux, yerr=bin_fluxerr, fmt=".k", alpha=0.5, label="TESS data binned by {0:.2f} d".format(binsize), zorder=1)

    color = "#ff7f0e"
    plt.plot(x, pred_mean * transit_model, color=color, label="Transit + GP model")
    plt.fill_between(x, (pred_mean+pred_std) * transit_model, (pred_mean-pred_std) * transit_model, color=color, alpha=0.3, edgecolor="none", zorder=2)

    plt.legend(fontsize=20)
    plt.xlabel(r"Time [BTJD]", fontsize=26)
    plt.ylabel(r"Relative flux", fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=20)
    plt.show()

    if phaseplot :
        # PHASE PLOT
        t0 = posterior["planet_params"][0]["tc_000"]
        period = posterior["planet_params"][0]["per_000"]

        phases = foldAt(lc['time'], period, T0=t0+0.5*period)
        sortIndi = np.argsort(phases)

        red_phases = phases[sortIndi] - 0.5
        red_phased_flux = (lc['nflux'] / pred_mean_obs)[sortIndi]
        red_phased_fluxerr = (lc['nfluxerr'] / pred_mean_obs)[sortIndi]

        plt.plot(red_phases, red_phased_flux, ".", color="darkgrey", alpha=0.1, label=r"TESS data", zorder=0.5)

        rbin_phases, rbin_flux, rbin_fluxerr = bin_data(red_phases, red_phased_flux, red_phased_fluxerr, median=False, binsize=binsize/period)

        plt.errorbar(rbin_phases, rbin_flux, yerr=rbin_fluxerr, fmt=".k", alpha=0.5, label="TESS data binned by {0:.2f} d".format(binsize), zorder=1)

        plt.plot(red_phases, transit_model_obs[sortIndi], '-', color=color, lw=2, label=r"Transit model")
    
        plt.ylabel(r"Relative flux", fontsize=26)
        plt.xlabel("phase (P={0:.3f} d)".format(period), fontsize=26)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=20)
        plt.show()


def plot_gp_rv(posterior, gp_rv, gp_feed, bjds, rvs, rverrs, samples=None, labels=None, nsamples=100, planet_index=0, default_legend="SPIRou RV", t0=0) :

    legends, fmts, colors = [],[],[]
    if len(rvs) == 1 :
        legends=[default_legend]
        fmts = ["o"]
        colors = ["k"]
    elif len(rvs) == 2 :
        legends=["CCF RV","LBL RV"]
        fmts = ["o","o"]
        colors = ["k",'#1f77b4']
    else :
        for i in range(len(rvs)) :
            if i == 0:
                legends.append("RV")
            else :
                legends.append(None)
            fmts.append("o")
            colors.append("k")

    rvcalib_params = posterior["rvcalib_params"]
    planets_params = posterior["planet_params"]

    bjd_all, rv_all, rverr_all = np.array([]), np.array([]), np.array([])
    
    rv_sys = planets_params[planet_index]['rvsys_{0:03d}'.format(planet_index)]
    trend = planets_params[planet_index]['trend_{0:03d}'.format(planet_index)]

    fig, axes = plt.subplots(4, 1, figsize=(16, 8), sharex=True, sharey=False, gridspec_kw={'hspace': 0, 'height_ratios': [2, 2, 2, 1.5]})

    min_bjd, max_bjd = 1e20, -1e20
    for i in range(len(bjds)) :
        if np.nanmin(bjds[i]) < min_bjd :
            min_bjd = np.nanmin(bjds[i])
        if np.nanmax(bjds[i]) > max_bjd :
            max_bjd = np.nanmax(bjds[i])
            
    if t0 == 0:
        t0 = min_bjd
        
    time, meanrvmodel = get_rv_model(planets_params, t0, min_bjd, max_bjd, timesampling=0.001, phasefold=False)

    if samples is not None and labels is not None:
        rvsamplemodels = []
        copy_planet_params = deepcopy(planets_params[planet_index])
        for theta in samples[np.random.randint(len(samples), size=nsamples)]:
            copy_planet_params = updateParams(copy_planet_params, theta, labels)
            _, rvm = get_rv_model([copy_planet_params], t0, min_bjd, max_bjd, timesampling=0.001,phasefold=False)
            rvsamplemodels.append(rvm)
        rvsamplemodels = np.array(rvsamplemodels, dtype=float)
        rms = np.nanstd(rvsamplemodels,axis=0)
    else :
        rms = np.zeros_like(meanrvmodel)

    gp_rv.compute(gp_feed["t"], gp_feed["yerr"])

    for i in range(len(rvs)) :
        
        bjd, rv, rverr = bjds[i], rvs[i], rverrs[i]
        
        coeff_id = 'rv_d{0:02d}'.format(i)
        rvcalib = rvcalib_params[coeff_id]
        
        rv_model = calculate_rv_model(bjd, planets_params, planet_index=planet_index)

        pred_mean_obs = gp_rv.predict(gp_feed["y"], bjd, return_cov=False)
            
        residuals =  rv - rvcalib - rv_model - pred_mean_obs

        bjd_all = np.append(bjd_all, bjd)
        rv_all = np.append(rv_all, residuals)
        rverr_all = np.append(rverr_all, rverr)

        axes[0].errorbar(bjd, rv-rvcalib, yerr=rverr, alpha=0.6, fmt=fmts[i], color=colors[i], label='{}'.format(legends[i]))
        axes[1].errorbar(bjd, rv-rvcalib - rv_model, yerr=rverr, alpha=0.6, fmt=fmts[i], color=colors[i], label='{} - orbit'.format(legends[i]))
        axes[2].errorbar(bjd, rv-rvcalib - pred_mean_obs, yerr=rverr, alpha=0.6, fmt=fmts[i], color=colors[i], label='{} - GP model'.format(legends[i]))
        axes[3].errorbar(bjd, residuals, yerr=rverr, alpha=0.6, fmt=fmts[i], color=colors[i], label='{} residuals'.format(legends[i]))

    rvsorted = np.argsort(bjd_all)
    rv_t, rv_res, rv_err = bjd_all[rvsorted], rv_all[rvsorted], rverr_all[rvsorted]

    pred_mean, pred_var = gp_rv.predict(gp_feed["y"], time, return_var=True)
    pred_std = np.sqrt(pred_var)

    # Plot the models
    color ='#1f77b4'
    axes[0].plot(time, meanrvmodel+pred_mean, "-", color=color, lw=2, label="GP + orbit model")
    axes[0].fill_between(time,meanrvmodel+pred_mean+pred_std, meanrvmodel+pred_mean-pred_std, color=color, alpha=0.3, edgecolor="none")
    axes[0].set_ylabel(r"RV [m/s]", fontsize=13)
    axes[0].legend(fontsize=12, loc='upper left')
    axes[0].tick_params(axis='x', labelsize=14)
    axes[0].tick_params(axis='y', labelsize=14)

    color = "#ff7f0e"
    axes[1].plot(time, pred_mean, "-", color=color, lw=2, label="GP model")
    axes[1].fill_between(time,pred_mean+pred_std, pred_mean-pred_std, color=color, alpha=0.3, edgecolor="none")
    axes[1].set_ylabel(r"RV [m/s]", fontsize=13)

    axes[1].legend(fontsize=12, loc='upper left')
    axes[1].tick_params(axis='x', labelsize=13)
    axes[1].tick_params(axis='y', labelsize=13)

    color = 'green'
    axes[2].plot(time, meanrvmodel, "-", color=color, lw=2, label="Orbit model")
    axes[2].fill_between(time,meanrvmodel+pred_std, meanrvmodel-pred_std, color=color, alpha=0.3, edgecolor="none")
    axes[2].set_ylabel(r"RV [m/s]", fontsize=13)
    axes[2].legend(fontsize=12, loc='upper left')
    axes[2].tick_params(axis='x', labelsize=13)
    axes[2].tick_params(axis='y', labelsize=13)

    sig_res = np.nanstd(rv_res)
    axes[3].hlines(0., min_bjd, max_bjd, color="k", linestyles=":", lw=0.6)
    axes[3].set_ylim(-5*sig_res,+5*sig_res)
    axes[3].set_ylabel(r"RV [m/s]", fontsize=13)
    axes[3].tick_params(axis='x', labelsize=13)
    axes[3].tick_params(axis='y', labelsize=13)
    axes[3].legend(fontsize=12, loc='upper left')
    axes[3].set_xlabel("Time [BJD]", fontsize=15)
    
    print("RMS of RV residuals: {:.2f} m/s".format(sig_res))

    plt.show()


def get_rv_residuals(posterior, bjds, rvs, rverrs, planet_index=0) :
    
    rvcalib_params = posterior["rvcalib_params"]
    planets_params = posterior["planet_params"]

    bjd_all, rv_all, rverr_all = np.array([]), np.array([]), np.array([])
    
    rv_sys = planets_params[planet_index]['rvsys_{0:03d}'.format(planet_index)]
    trend = planets_params[planet_index]['trend_{0:03d}'.format(planet_index)]

    for i in range(len(rvs)) :
        
        bjd, rv, rverr = bjds[i], rvs[i], rverrs[i]
        
        coeff_id = 'rv_d{0:02d}'.format(i)
        rvcalib = rvcalib_params[coeff_id]
        
        rv_model = calculate_rv_model(bjd, planets_params, planet_index=planet_index)
        residuals =  rv - rvcalib - rv_model

        bjd_all = np.append(bjd_all, bjd)
        rv_all = np.append(rv_all, residuals)
        rverr_all = np.append(rverr_all, rverr)

    rvsorted = np.argsort(bjd_all)
    
    rv_t, rv_res, rv_err = bjd_all[rvsorted], rv_all[rvsorted], rverr_all[rvsorted]

    return rv_t, rv_res, rv_err


def set_star_rotation_priors(gp_phot, gp_blong, gp_rv, fix_phot_params=False, fix_blong_params=False, fix_rv_params=False, shared_Prot=False, ini_prot=0., period_lim=(4,100), decaytime_lim=(50,1000), fix_phot_decaytime=False, fix_spec_decaytime=False, smoothfactor_lim=(0.25,1.25), fix_phot_smoothfactor=False, fix_spec_smoothfactor=False) :
    
    phot_params = gp_lib.get_star_rotation_gp_params(gp_phot)
    blong_params = gp_lib.get_star_rotation_gp_params(gp_blong)
    rv_params = gp_lib.get_star_rotation_gp_params(gp_rv)
    
    params = {}
    priortypes = {}
    values = {}
    
    if fix_phot_params :
        priortypes["phot_mean"] = 'FIXED'
        priortypes["phot_white_noise"] = 'FIXED'
        priortypes["phot_amplitude"] = 'FIXED'
    else :
        priortypes["phot_mean"] = 'Normal'
        priortypes["phot_white_noise"] = 'Uniform'
        priortypes["phot_amplitude"] = 'Uniform'

    values["phot_mean"] = (phot_params["mean"], np.abs(phot_params["mean"])*0.5)
    values["phot_white_noise"] = (0, np.abs(phot_params["white_noise"])*10)
    values["phot_amplitude"] = (0, np.abs(phot_params["amplitude"])*10)

    # Set initial phot parameters
    params["phot_mean"] = phot_params["mean"]
    params["phot_white_noise"] = phot_params["white_noise"]
    params["phot_amplitude"] = phot_params["amplitude"]

    if fix_blong_params :
        priortypes["blong_mean"] = 'FIXED'
        priortypes["blong_white_noise"] = 'FIXED'
        priortypes["blong_amplitude"] = 'FIXED'
    else :
        priortypes["blong_mean"] = 'Normal'
        priortypes["blong_white_noise"] = 'Uniform'
        priortypes["blong_amplitude"] = 'Uniform'

    values["blong_mean"] = (blong_params["mean"], np.abs(blong_params["mean"])*0.5)
    values["blong_white_noise"] = (0, np.abs(blong_params["white_noise"])*10)
    values["blong_amplitude"] = (0, np.abs(blong_params["amplitude"])*10)

    params["blong_mean"] = blong_params["mean"]
    params["blong_white_noise"] = blong_params["white_noise"]
    params["blong_amplitude"] = blong_params["amplitude"]

    if fix_rv_params :
        priortypes["rv_mean"] = 'FIXED'
        priortypes["rv_white_noise"] = 'FIXED'
        priortypes["rv_amplitude"] = 'FIXED'
    else :
        priortypes["rv_mean"] = 'Normal'
        priortypes["rv_white_noise"] = 'Uniform'
        priortypes["rv_amplitude"] = 'Uniform'

    values["rv_mean"] = (rv_params["mean"], np.abs(rv_params["mean"])*0.5)
    values["rv_white_noise"] = (0, np.abs(rv_params["white_noise"])*10)
    values["rv_amplitude"] = (0, np.abs(rv_params["amplitude"])*10)

    params["rv_mean"] = rv_params["mean"]
    params["rv_white_noise"] = rv_params["white_noise"]
    params["rv_amplitude"] = rv_params["amplitude"]

    # Set photometric decay time parameter
    params["phot_decaytime"] = phot_params["decaytime"]
    if fix_phot_decaytime or fix_phot_params :
        priortypes["phot_decaytime"] = 'FIXED'
    else :
        priortypes["phot_decaytime"] = 'Uniform'
    values["phot_decaytime"] = decaytime_lim
    #values["phot_decaytime"] = (params["phot_decaytime"], params["phot_decaytime"] )

    # Set spectroscopic decay time parameter
    params["spec_decaytime"] = blong_params["decaytime"]
    if fix_spec_decaytime :
        priortypes["spec_decaytime"] = 'FIXED'
    else :
        priortypes["spec_decaytime"] = 'Uniform'
    values["spec_decaytime"] = decaytime_lim
    #values["spec_decaytime"] = (params["spec_decaytime"], params["spec_decaytime"] )

    # Set photometric gamma -> smooth factor parameter
    params["phot_smoothfactor"] = phot_params["smoothfactor"]
    if fix_phot_smoothfactor or fix_phot_params :
        priortypes["phot_smoothfactor"] = 'FIXED'
    else :
        priortypes["phot_smoothfactor"] = 'Uniform'
    values["phot_smoothfactor"] = smoothfactor_lim

    # Set spectroscopic gamma -> smooth factor parameter
    params["spec_smoothfactor"] = blong_params["smoothfactor"]
    if fix_spec_smoothfactor :
        priortypes["spec_smoothfactor"] = 'FIXED'
    else :
        priortypes["spec_smoothfactor"] = 'Uniform'
    values["spec_smoothfactor"] = smoothfactor_lim

    # Set rotation  period
    if shared_Prot :
        if fix_phot_params and fix_blong_params and fix_rv_params :
            priortypes["prot"] = 'FIXED'
        else :
            priortypes["prot"] = 'Uniform'

        if ini_prot == 0 :
            ini_prot = (phot_params["period"] + blong_params["period"] + rv_params["period"]) / 3
        params["prot"] = ini_prot
        
        if (params["prot"] < period_lim[0]) or (params["prot"] > period_lim[1]) :
            params["prot"] = (period_lim[0] + period_lim[1]) / 2

        values["prot"] = period_lim
    else :
        if fix_phot_params :
            priortypes["phot_prot"] = 'FIXED'
        else :
            priortypes["phot_prot"] = 'Uniform'

        params["phot_prot"] = phot_params["period"]
        values["phot_prot"] = period_lim

        if fix_blong_params :
            priortypes["blong_prot"] = 'FIXED'
        else :
            priortypes["blong_prot"] = 'Uniform'

        params["blong_prot"] = blong_params["period"]
        values["blong_prot"] = period_lim

        if fix_rv_params :
            priortypes["rv_prot"] = 'FIXED'
        else :
            priortypes["rv_prot"] = 'Uniform'

        params["rv_prot"] = rv_params["period"]
        values["rv_prot"] = period_lim

    labels, theta, theta_priors = gp_lib.set_theta_priors(params, priortypes, values)

    return params, labels, theta, theta_priors


def set_star_rotation_gp(posterior, gp_priors, lc, bjds, rv, rverr, bbjds, blong, blongerr, binsize = 0.1, gp_ls_fit=True, run_gp_mcmc=False, amp=1e-4, nwalkers=32, niter=300, burnin=50, remove_transits=False, plot=False, verbose=False, spec_constrained=False) :

    gp_params = priorslib.read_starrot_gp_params(gp_priors, spec_constrained=spec_constrained)

    param_lim, param_fixed = {}, {}
    # print out gp priors
    print("----------------")
    print("Input GP parameters:")
    for key in gp_params.keys() :
        if ("_err" not in key) and ("_pdf" not in key) and key != "param_ids":
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

    rot_period, period_lim, fix_period = gp_params["prot"], param_lim["prot"], param_fixed["prot"]

    if spec_constrained :
        decaytime, decaytime_lim, fix_decaytime = gp_params["spec_decaytime"], param_lim["spec_decaytime"], param_fixed["spec_decaytime"]
        smoothfactor, smoothfactor_lim, fix_smoothfactor = gp_params["spec_smoothfactor"], param_lim["spec_smoothfactor"], param_fixed["spec_smoothfactor"]
    else :
        decaytime, decaytime_lim, fix_decaytime = gp_params["blong_decaytime"], param_lim["blong_decaytime"], param_fixed["blong_decaytime"]
        smoothfactor, smoothfactor_lim, fix_smoothfactor = gp_params["blong_smoothfactor"], param_lim["blong_smoothfactor"], param_fixed["blong_smoothfactor"]

    amplitude = gp_params["blong_amplitude"]
    fit_mean = True
    if param_fixed["blong_mean"] : fit_mean = False
    fit_white_noise = True
    if param_fixed["blong_white_noise"] : fit_white_noise = False

    # Run GP on B-long data with a QP kernel
    if len(blong) :
        #output_pairsplot="gpblong_activity_pairsplot.png"
        output_pairsplot=""
        gp_blong = gp_lib.star_rotation_gp(bbjds, blong, blongerr, run_optimization=gp_ls_fit, period=rot_period, period_lim=period_lim, fix_period=fix_period, amplitude=amplitude, decaytime=decaytime, decaytime_lim=decaytime_lim, fix_decaytime=fix_decaytime, smoothfactor=smoothfactor, smoothfactor_lim=smoothfactor_lim, fix_smoothfactor=fix_smoothfactor, fixpars_before_fit=True, fit_mean=fit_mean, fit_white_noise=fit_white_noise, period_label=r"Prot$_B$ [d]", amplitude_label=r"$\alpha_B$", decaytime_label=r"$l_B$", smoothfactor_label=r"$\beta_B$", mean_label=r"$\mu_b$", white_noise_label=r"$\sigma_B$", output_pairsplot=output_pairsplot, run_mcmc=run_gp_mcmc, amp=amp, nwalkers=nwalkers, niter=niter, burnin=burnin, x_label="BJD", y_label="B$_l$ [G]", plot=plot, verbose=verbose)

        gp_blong_params = gp_lib.get_star_rotation_gp_params(gp_blong)

        gp_params["blong_mean"] = gp_blong_params["mean"]
        gp_params["blong_white_noise"] = gp_blong_params["white_noise"]
        gp_params["blong_amplitude"] = gp_blong_params["amplitude"]
        gp_params["prot"] = gp_blong_params["period"]
        
        if spec_constrained :
            gp_params["spec_decaytime"] = gp_blong_params["decaytime"]
            gp_params["spec_smoothfactor"] = gp_blong_params["smoothfactor"]
        else :
            gp_params["blong_decaytime"] = gp_blong_params["decaytime"]
            gp_params["blong_smoothfactor"] = gp_blong_params["smoothfactor"]
    else :
        gp_blong = None
        
    decaytime, decaytime_lim, fix_decaytime = gp_params["rv_decaytime"], param_lim["rv_decaytime"], param_fixed["rv_decaytime"]
    smoothfactor, smoothfactor_lim, fix_smoothfactor = gp_params["rv_smoothfactor"], param_lim["rv_smoothfactor"], param_fixed["rv_smoothfactor"]

    amplitude = gp_params["rv_amplitude"]
    fit_mean = True
    if param_fixed["rv_mean"] : fit_mean = False
    fit_white_noise = True
    if param_fixed["rv_white_noise"] : fit_white_noise = False

    # Run GP on RV residuals with a QP kernel
    #run_gp_rv_optimization = gp_ls_fit
    run_gp_rv_optimization = False
    #output_pairsplot="gprv_activity_pairsplot.png"
    output_pairsplot=""
    gp_rv = gp_lib.star_rotation_gp(bjds, rv, rverr, period=rot_period, run_optimization=run_gp_rv_optimization, period_lim=period_lim, fix_period=fix_period, amplitude=amplitude, decaytime=decaytime, decaytime_lim=decaytime_lim, fix_decaytime=fix_decaytime, smoothfactor=smoothfactor, smoothfactor_lim=smoothfactor_lim, fix_smoothfactor=fix_smoothfactor, fixpars_before_fit=True, fit_mean=fit_mean, fit_white_noise=fit_white_noise, period_label=r"Prot$_v$ [d]", amplitude_label=r"$\alpha_v$", decaytime_label=r"$l_v$", smoothfactor_label=r"$\beta_v$", mean_label=r"$\mu_v$", white_noise_label=r"$\sigma_v$", output_pairsplot=output_pairsplot, run_mcmc=run_gp_mcmc, amp=amp, nwalkers=nwalkers, niter=niter, burnin=burnin, x_label="BJD", y_label="RV [m/s]", plot=plot, verbose=verbose)

    gp_rv_params = gp_lib.get_star_rotation_gp_params(gp_rv)

    gp_params["rv_mean"] = gp_rv_params["mean"]
    gp_params["rv_white_noise"] = gp_rv_params["white_noise"]
    gp_params["rv_amplitude"] = gp_rv_params["amplitude"]
    if spec_constrained :
        #gp_params["spec_decaytime"] = gp_rv_params["decaytime"]
        #gp_params["spec_smoothfactor"] = gp_rv_params["smoothfactor"]
        #gp_params["prot"] = gp_rv_params["period"]
        pass
    else :
        gp_params["rv_decaytime"] = gp_rv_params["decaytime"]
        gp_params["rv_smoothfactor"] = gp_rv_params["smoothfactor"]

    amplitude = gp_params["phot_amplitude"]
    fit_mean = True
    if param_fixed["phot_mean"] : fit_mean = False
    fit_white_noise = True
    if param_fixed["phot_white_noise"] : fit_white_noise = False

    decaytime, decaytime_lim, fix_decaytime = gp_params["phot_decaytime"], param_lim["phot_decaytime"], param_fixed["phot_decaytime"]
    smoothfactor, smoothfactor_lim, fix_smoothfactor = gp_params["phot_smoothfactor"], param_lim["phot_smoothfactor"], param_fixed["phot_smoothfactor"]

    if remove_transits :
        transit_models = np.full_like(lc['time'], 1.0)
        for j in range(len(posterior["planet_params"])) :
            transit_models *= modelslib.batman_transit_model(lc['time'], posterior["planet_params"][j], planet_index=j)
        # Remove transits before running gp
        nflux = lc['nflux'] / transit_models
        nfluxerr = lc['nfluxerr'] / transit_models
    else :
        nflux = lc['nflux']
        nfluxerr = lc['nfluxerr']

    # Bin data to make GP more efficient
    bin_tbjd, bin_flux, bin_fluxerr = bin_data(lc['time'], nflux, nfluxerr, median=False, binsize=binsize)

    #output_pairsplot="gpphot_activity_pairsplot.png"
    output_pairsplot=""
    # Run GP on photometry data with a SHO kernel
    gp_phot = gp_lib.star_rotation_gp(bin_tbjd, bin_flux, bin_fluxerr, run_optimization=gp_ls_fit, period=rot_period, period_lim=period_lim, fix_period=fix_period, amplitude=amplitude, decaytime=decaytime, decaytime_lim=decaytime_lim, fix_decaytime=fix_decaytime, smoothfactor=smoothfactor, smoothfactor_lim=smoothfactor_lim, fix_smoothfactor=fix_smoothfactor, fixpars_before_fit=True, fit_mean=fit_mean, fit_white_noise=fit_white_noise, period_label=r"Prot$_p$ [d]", amplitude_label=r"$\alpha_p$", decaytime_label=r"$l_p$", smoothfactor_label=r"$\beta_p$", mean_label=r"$\mu_p$", white_noise_label=r"$\sigma_p$", output_pairsplot=output_pairsplot, run_mcmc=run_gp_mcmc, amp=amp, nwalkers=nwalkers, niter=niter, burnin=burnin, x_label="BTJD", y_label="Flux", plot=plot, verbose=verbose)

    gp_phot_params = gp_lib.get_star_rotation_gp_params(gp_phot)

    gp_params["phot_mean"] = gp_phot_params["mean"]
    gp_params["phot_white_noise"] = gp_phot_params["white_noise"]
    gp_params["phot_amplitude"] = gp_phot_params["amplitude"]
    gp_params["phot_decaytime"] = gp_phot_params["decaytime"]
    gp_params["phot_smoothfactor"] = gp_phot_params["smoothfactor"]
    #gp_params["prot"] = gp_phot_params["period"]

    rv_gp_feed, blong_gp_feed, phot_gp_feed = {}, {}, {}
    rv_gp_feed["t"], rv_gp_feed["y"], rv_gp_feed["yerr"] = bjds, rv, rverr
    blong_gp_feed["t"], blong_gp_feed["y"], blong_gp_feed["yerr"] = bbjds, blong, blongerr
    phot_gp_feed["t"], phot_gp_feed["y"], phot_gp_feed["yerr"] = bin_tbjd, bin_flux, bin_fluxerr

    for key in gp_params['param_ids'] :
        gp_priors[key]['object'].value = gp_params[key]

    return gp_rv, gp_blong, gp_phot, rv_gp_feed, blong_gp_feed, phot_gp_feed, gp_priors


def plot_gp_blong(bjds, blong, blongerr, posterior, gp_blong, fold_period, phase_plot=True, ylabel=r"B$_l$ [G]", timesampling=0.001, number_of_free_params=0) :

    #gp_blong.compute(gp_feed["t"], gp_feed["yerr"])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=False, gridspec_kw={'hspace': 0, 'height_ratios': [2, 1]})

    ti, tf = np.min(bjds), np.max(bjds)
    time = np.arange(ti, tf, timesampling)
    
    pred_mean, pred_var = gp_blong.predict(blong, time, return_var=True)
    pred_std = np.sqrt(pred_var)
    
    pred_mean_obs, _ = gp_blong.predict(blong, bjds, return_var=True)

    residuals = blong - pred_mean_obs
    
    # Plot the data
    color = "#ff7f0e"
    axes[0].plot(time, pred_mean, "-", color=color, lw=2, label="GP model")
    axes[0].fill_between(time, pred_mean+pred_std, pred_mean-pred_std, color=color, alpha=0.3, edgecolor="none")
    axes[0].errorbar(bjds, blong, yerr=blongerr, fmt='o', color='k', label='SPIRou data')
    axes[1].errorbar(bjds, residuals, yerr=blongerr, fmt='o', color='k')
    axes[1].set_xlabel("BJD", fontsize=16)

    axes[0].set_ylabel(ylabel, fontsize=16)
    axes[0].legend(fontsize=16)
    axes[0].tick_params(axis='x', labelsize=14)
    axes[0].tick_params(axis='y', labelsize=14)

    sig_res = np.nanstd(residuals)
    axes[1].hlines(0., ti, tf, color="k", linestyles=":", lw=0.6)
    axes[1].set_ylim(-5*sig_res,+5*sig_res)
    axes[1].set_ylabel(r"Residuals [G]", fontsize=16)
    axes[1].tick_params(axis='x', labelsize=14)
    axes[1].tick_params(axis='y', labelsize=14)
        
    print("RMS of {} residuals: {:.2f}".format(ylabel, sig_res))
    n = len(residuals)
    m = number_of_free_params
    chi2 = np.sum((residuals/blongerr)**2) / (n - m)
    print("Reduced chi-square (n={}, DOF={}): {:.2f}".format(n,n-m,chi2))

    plt.show()

    if phase_plot :
        plt.clf()
        
        phases = foldAt(bjds, fold_period, T0=ti)
        sortIndi = np.argsort(phases)
        plt.errorbar(phases[sortIndi], blong[sortIndi], yerr=blongerr[sortIndi], fmt='o', color='k', label='SPIRou data')
        
        mphases = foldAt(time, fold_period, T0=ti)
        msortIndi = np.argsort(mphases)
        plt.plot(mphases[msortIndi], pred_mean[msortIndi], "-", color=color, lw=2, alpha=0.5, label="GP model")
        #plt.fill_between(mphases[msortIndi], pred_mean[msortIndi]+pred_std[msortIndi], pred_mean[msortIndi]-pred_std[msortIndi], color=color, alpha=0.3, edgecolor="none")
        plt.ylabel(ylabel, fontsize=16)
        plt.xlabel("phase (P={0:.2f} d)".format(fold_period), fontsize=16)
        plt.legend(fontsize=16)
        plt.show()


#likelihood function
def lnlikelihood_tr_rv_gp(theta, labels, calib_params, rvcalib_params, flare_params, planet_params, tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, gp_rv, gp_phot, rv_gp_feed, phot_gp_feed, rvnull=False):

    prior_planet_params = deepcopy(planet_params)

    for i in range(len(planet_params)) :
        planet_params[i] = updateParams(planet_params[i], theta, labels)
    
    flare_params = updateParams(flare_params, theta, labels)
    calib_params = updateParams(calib_params, theta, labels)

    flux_models = calculate_models(tr_times, calib_params, flare_params, planet_params)

    sum_of_residuals = 0

    for j in range(len(prior_planet_params)) :
        for key in prior_planet_params[j].keys() :
            if ("_err" not in key) and ("_pdf" not in key) :
                pdf_key = "{0}_pdf".format(key)
                if prior_planet_params[j][pdf_key] == "Normal" :
                    error_key = "{0}_err".format(key)
                    error = prior_planet_params[j][error_key][1]
                    param_chi2 = ((planet_params[j][key] - prior_planet_params[j][key])/error)**2
                    sum_of_residuals += param_chi2


    for i in range(len(tr_times)) :
        pred_mean_phot = gp_phot.predict(phot_gp_feed["y"], tr_times[i], return_cov=False, return_var=False)
        flux_residuals = (fluxes[i] - flux_models[i]) + 1.0 - pred_mean_phot
        sum_of_residuals += np.nansum((flux_residuals/fluxerrs[i])**2 + np.log(2.0 * np.pi * (fluxerrs[i] * fluxerrs[i])))

    rvcalib_params = updateParams(rvcalib_params, theta, labels)

    # Calculate RV model and residuals
    for i in range(len(rvs)) :
        pred_mean_rv = gp_rv.predict(rv_gp_feed["y"], rv_times[i], return_cov=False, return_var=False)
        coeff_id = 'rv_d{0:02d}'.format(i)
        rvcalib = rvcalib_params[coeff_id]
        if rvnull :
            rv_model = np.zeros_like(rv_times[i]) + rvcalib
        else :
            rv_model = calculate_rv_model(rv_times[i], planet_params, planet_index=0) + rvcalib
        rv_residuals = rvs[i] - pred_mean_rv - rv_model

        sum_of_residuals += np.nansum((rv_residuals/rverrs[i])**2 + np.log(2.0 * np.pi * (rverrs[i] * rverrs[i])))

    ln_likelihood = -0.5 * (sum_of_residuals)

    return ln_likelihood


#posterior probability
def lnprob_tr_rv_gp(theta, theta_priors, labels, calib_params, rvcalib_params, flare_params, planet_params, tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, gp_params, gp_rv, gp_blong, gp_phot, rv_gp_feed, blong_gp_feed, phot_gp_feed, spec_constrained, rvnull=False):
        
    lp = lnprior(theta_priors, theta, labels)
    if not np.isfinite(lp):
        return -np.inf

    gp_phot, gp_blong, gp_rv = update_star_rotation_gp_params(gp_params, labels, theta, gp_phot, gp_blong, gp_rv, spec_constrained=spec_constrained)

    gp_phot.compute(phot_gp_feed["t"], yerr=phot_gp_feed["yerr"])
    gp_blong.compute(blong_gp_feed["t"], yerr=blong_gp_feed["yerr"])
    gp_rv.compute(rv_gp_feed["t"], yerr=rv_gp_feed["yerr"])

    prob = lp + gp_blong.log_likelihood(blong_gp_feed["y"]) + gp_phot.log_likelihood(phot_gp_feed["y"]) + lnlikelihood_tr_rv_gp(theta, labels, calib_params, rvcalib_params, flare_params, planet_params, tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, gp_rv, gp_phot, rv_gp_feed, phot_gp_feed, rvnull=rvnull)
    
    if np.isfinite(prob) :
        return prob
    else :
        return -np.inf


def update_star_rotation_gp_params(params, labels, theta, gp_phot, gp_blong, gp_rv, spec_constrained=False) :

    phot_params = gp_lib.get_star_rotation_gp_params(gp_phot)
    blong_params = gp_lib.get_star_rotation_gp_params(gp_blong)
    rv_params = gp_lib.get_star_rotation_gp_params(gp_rv)

    for i in range(len(labels)) :
        # we don't want to create new keys in params when labels has more than the gp parameters in it
        if labels[i] in params.keys() :
            params[labels[i]] = theta[i]

    # Set mean parameter
    phot_params["mean"] = params["phot_mean"]
    blong_params["mean"] = params["blong_mean"]
    rv_params["mean"] = params["rv_mean"]
    
    # Set white noise parameter
    phot_params["white_noise"] = params["phot_white_noise"]
    blong_params["white_noise"] = params["blong_white_noise"]
    rv_params["white_noise"] = params["rv_white_noise"]

    # Set amplitude parameter
    phot_params["amplitude"] = params["phot_amplitude"]
    blong_params["amplitude"] = params["blong_amplitude"]
    rv_params["amplitude"] = params["rv_amplitude"]
    
    # Set decay time parameter
    phot_params["decaytime"] = params["phot_decaytime"]
    if spec_constrained :
        blong_params["decaytime"] = params["spec_decaytime"]
        rv_params["decaytime"] = params["spec_decaytime"]
    else :
        blong_params["decaytime"] = params["blong_decaytime"]
        rv_params["decaytime"] = params["rv_decaytime"]

    # Set decay smooth factor
    phot_params["smoothfactor"]  = params["phot_smoothfactor"]
    if spec_constrained :
        blong_params["smoothfactor"]  = params["spec_smoothfactor"]
        rv_params["smoothfactor"]  = params["spec_smoothfactor"]
    else :
        blong_params["smoothfactor"]  = params["blong_smoothfactor"]
        rv_params["smoothfactor"]  = params["rv_smoothfactor"]

    phot_params["period"]  = params["prot"]
    blong_params["period"]  = params["prot"]
    rv_params["period"]  = params["prot"]
        
    gp_rv = gp_lib.set_star_rotation_gp_params(gp_rv, rv_params)
    gp_phot = gp_lib.set_star_rotation_gp_params(gp_phot, phot_params)
    gp_blong = gp_lib.set_star_rotation_gp_params(gp_blong, blong_params)

    return gp_phot, gp_blong, gp_rv



def fitTransitsAndRVsWithMCMCAndGP(tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, b_times, blong, blongerr, priors, gp_priors, lc, phot_binsize=0.1, amp=1e-4, nwalkers=32, niter=100, burnin=20, samples_filename="default_samples.h5", appendsamples=False, return_samples=False, verbose=False, plot=False, plot_individual_transits=False, rvlabel="", gp_spec_constrained=False) :
    
    posterior = deepcopy(priors)
    
    calib_params = posterior["calib_params"]
    rvcalib_params = posterior["rvcalib_params"]
    flare_params = posterior["flare_params"]
    planet_params = posterior["planet_params"]
    
    # Make RV calib equals zero since GP already account for this in its mean parameter
    for i in range(len(rvs)) :
        coeff_id = 'rv_d{0:02d}'.format(i)
        rvcalib_params[coeff_id] = 0.

    theta, labels, theta_priors = posterior["theta"], posterior["labels"], posterior["theta_priors"]

    gp_rv, gp_blong, gp_phot, rv_gp_feed, blong_gp_feed, phot_gp_feed, gp_priors = set_star_rotation_gp (posterior, gp_priors, lc, rv_times[0], rvs[0], rverrs[0], b_times, blong, blongerr, binsize=phot_binsize, gp_ls_fit=True, run_gp_mcmc=False, amp=1e-4, nwalkers=32, niter=500, burnin=100, remove_transits=True, plot=True, verbose=verbose, spec_constrained=gp_spec_constrained)

    gp_params = priorslib.read_starrot_gp_params(gp_priors, spec_constrained=gp_spec_constrained)
    
    gp_theta, gp_labels, gp_theta_priors = priorslib.get_gp_theta_from_priors(gp_priors)
    
    posterior["gp_params"] = gp_params
    
    labels = labels + gp_labels
    #print("labels=",labels)
    theta = np.hstack([theta,gp_theta])
    #print("theta=",theta)
    for key in gp_theta_priors :
        #print("Adding key {} from gp to theta priors".format(key))
        theta_priors[key] = gp_theta_priors[key]

    # Make sure the number of walkers is sufficient, and if not assing a new value
    if nwalkers < 2*len(theta):
        nwalkers = 2*len(theta)
    
    if verbose :
        print("Free parameters before MCMC fit:")
        for i in range(len(theta)) :
            print(labels[i], "=", theta[i])

    if verbose:
        print("initializing emcee sampler ...")

    #amp, ndim, nwalkers, niter, burnin = 5e-4, len(theta), 50, 2000, 500
    ndim = len(theta)

    # Set up the backend
    backend = emcee.backends.HDFBackend(samples_filename)
    if appendsamples == False :
        backend.reset(nwalkers, ndim)
    

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_tr_rv_gp, args = [theta_priors, labels, calib_params, rvcalib_params, flare_params, planet_params, tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, gp_params, deepcopy(gp_rv), deepcopy(gp_blong), deepcopy(gp_phot), rv_gp_feed, blong_gp_feed, phot_gp_feed, gp_spec_constrained], backend=backend)

    pos = [theta + amp * np.random.randn(ndim) for i in range(nwalkers)]
    #--------

    #- run mcmc
    if verbose:
        print("Running MCMC ...")
        print("N_walkers=",nwalkers," ndim=",ndim)

    sampler.run_mcmc(pos, niter, progress=True)
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim)) # burnin : number of first samples to be discard as burn-in
    #--------

    #- calib fit parameters
    if verbose:
        print("Obtaining best fit calibration parameters from pdfs ...")
    calib_params, calib_theta_fit, calib_theta_labels, calib_theta_err = best_fit_params(posterior["calib_params"], labels, samples)
    if verbose :
        print("CALIBRATION Fit parameters:")
        for i in range(len(calib_theta_fit)) :
            print(calib_theta_labels[i], "=", calib_theta_fit[i], "+", calib_theta_err[i][0], "-", calib_theta_err[i][1])
        print("----------------")

    if posterior["calib_priorsfile"] != "":
        calib_posterior = posterior["calib_priorsfile"].replace(".pars", "_posterior.pars")
    else :
        calib_posterior = "calibration_posterior.pars"
    if verbose:
        print("Output CALIBRATION posterior: ", calib_posterior)
    # save posterior of calibration parameters into file:
    ncoeff=posterior["calib_priors"]['orderOfPolynomial']['object'].value
    priorslib.save_posterior(calib_posterior, calib_params, calib_theta_fit, calib_theta_labels, calib_theta_err, calib=True, ncoeff=ncoeff)

    # update cablib parameters in output posterior
    posterior["calib_params"] = calib_params
    #------------

    #- RV calib fit parameters
    if verbose:
        print("Obtaining best fit RV calibration parameters from pdfs ...")
    rvcalib_params, rvcalib_theta_fit, rvcalib_theta_labels, rvcalib_theta_err = best_fit_params(posterior["rvcalib_params"], labels, samples)
    if verbose :
        print("RV CALIBRATION Fit parameters:")
        for i in range(len(rvcalib_theta_fit)) :
            print(rvcalib_theta_labels[i], "=", rvcalib_theta_fit[i], "+", rvcalib_theta_err[i][0], "-", rvcalib_theta_err[i][1])
        print("----------------")
    rvcalib_posterior = "rv_calibration_posterior.pars"
    if verbose:
        print("Output RV CALIBRATION posterior: ", calib_posterior)
    priorslib.save_posterior(rvcalib_posterior, rvcalib_params, rvcalib_theta_fit, rvcalib_theta_labels, rvcalib_theta_err)
    
    # update cablib parameters in output posterior
    posterior["rvcalib_params"] = rvcalib_params
    #------------

    if posterior["n_flares"] :
        if verbose:
            print("Obtaining best fit flare parameters from pdfs ...")
        
        flare_params, flare_theta_fit, flare_theta_labels, flare_theta_err = best_fit_params(posterior["flare_params"], labels, samples)

        if verbose :
            print("FLARE Fit parameters:")
            for i in range(len(flare_theta_fit)) :
                print(flare_theta_labels[i], "=", flare_theta_fit[i], "+", flare_theta_err[i][0], "-", flare_theta_err[i][1])
            print("----------------")

        flare_posterior = posterior["flare_priorsfile"].replace(".pars", "_posterior.pars")
        if verbose:
            print("Output FLARE posterior: ", flare_posterior)
        # save posterior of flare parameters into file:
        priorslib.save_posterior(flare_posterior, flare_params, flare_theta_fit, flare_theta_labels, flare_theta_err)

    # update flare parameters in output posterior
    posterior["flare_params"] = flare_params

    if verbose:
        print("Obtaining best fit planet parameters from pdfs ...")
    planet_params, planet_theta_fit, planet_theta_labels = [], [], []
    for i in range(posterior["n_planets"]) :
        pl_params, pl_theta_fit, pl_theta_labels, pl_theta_err = best_fit_params(posterior["planet_params"][i], labels, samples)
        # update flare parameters in output posterior
        posterior["planet_params"][i] = pl_params

        planet_params.append(pl_params)
        planet_theta_fit = np.concatenate((planet_theta_fit, pl_theta_fit), axis=0)
        planet_theta_labels = np.concatenate((planet_theta_labels, pl_theta_labels), axis=0)
        if verbose:
            # print out best fit parameters and errors
            print("----------------")
            print("PLANET {} Fit parameters:".format(i))
            for j in range(len(planet_theta_fit)) :
                print(pl_theta_labels[j], "=", pl_theta_fit[j], "+", pl_theta_err[j][0], "-", pl_theta_err[j][1])
            print("----------------")

        planet_posterior = posterior["planet_priors_files"][i].replace(".pars", "_posterior.pars")
        if verbose:
            print("Output PLANET {0} posterior: ".format(i), planet_posterior)
        
        # save posterior of planet parameters into file:
        priorslib.save_posterior(planet_posterior, pl_params, pl_theta_fit, pl_theta_labels, pl_theta_err)

    #- GP fit parameters
    if verbose:
        print("Obtaining best fit GP parameters from pdfs ...")
    gp_params, gp_theta_fit, gp_theta_labels, gp_theta_err = best_fit_params(gp_params, labels, samples)
    if verbose :
        print("GP Fit parameters:")
        for i in range(len(gp_theta_fit)) :
            print(gp_theta_labels[i], "=", gp_theta_fit[i], "+", gp_theta_err[i][0], "-", gp_theta_err[i][1])
        print("----------------")
    gp_posterior = gp_priors["gp_priorsfile"].replace(".pars", "_posterior.pars")
    if verbose:
        print("Output GP  posterior: ", gp_posterior)
    priorslib.save_posterior(gp_posterior, gp_params, gp_theta_fit, gp_theta_labels, gp_theta_err)

    # update gp parameters in output posterior
    posterior["gp_params"] = gp_params

    gp_phot, gp_blong, gp_rv = update_star_rotation_gp_params(gp_params, gp_theta_labels, gp_theta_fit, gp_phot, gp_blong, gp_rv, spec_constrained=gp_spec_constrained)

    gp_phot.compute(phot_gp_feed['t'], phot_gp_feed['yerr'])
    gp_blong.compute(blong_gp_feed['t'], blong_gp_feed['yerr'])
    gp_rv.compute(rv_gp_feed['t'], rv_gp_feed['yerr'])
    #------------

    # Update theta in posterior
    if posterior["n_flares"] :
        theta_tuple  = (calib_theta_fit, flare_theta_fit, planet_theta_fit, gp_theta_fit)
        labels_tuple  = (calib_theta_labels, flare_theta_labels, planet_theta_labels, gp_theta_labels)
    else :
        theta_tuple  = (calib_theta_fit, planet_theta_fit, gp_theta_fit)
        labels_tuple  = (calib_theta_labels, planet_theta_labels, gp_theta_labels)

    posterior['theta'] = np.hstack(theta_tuple)
    posterior['labels'] = np.hstack(labels_tuple)

    posterior['BIC'] = calculate_bic_gp (posterior, tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, gp_params, gp_rv, gp_blong, gp_phot, rv_gp_feed, blong_gp_feed, phot_gp_feed, spec_constrained=gp_spec_constrained)

    posterior['BIC_NOPLANET'] = calculate_bic_gp (posterior, tr_times, fluxes, fluxerrs, rv_times, rvs, rverrs, gp_params, gp_rv, gp_blong, gp_phot, rv_gp_feed, blong_gp_feed, phot_gp_feed, spec_constrained=gp_spec_constrained, rvnull=True)
    
    if verbose :
        print("BIC = {}, BIC (no planet)={}".format(posterior['BIC'],posterior['BIC_NOPLANET']))

    if return_samples :
        posterior["samples"] = samples

    if plot :
        prot_key_blong = "prot"

        # plot Blong gp posterior
        plot_gp_blong(b_times, blong, blongerr, posterior, gp_blong, gp_params[prot_key_blong], phase_plot=True)

        # plot photometry gp posterior
        plot_gp_photometry(lc, posterior, gp_phot, phot_gp_feed["y"], phot_binsize)

        # plot RV gp posterior
        plot_gp_rv(posterior, gp_rv, rv_gp_feed, rv_times, rvs, rverrs, samples=samples, labels=labels)
        
        # reduce gp activity model in phot time series
        red_fluxes, red_fluxerrs = reduce_gp(gp_phot, tr_times, fluxes, fluxerrs, phot_gp_feed)

        # reduce gp activity model in RV time series
        red_rvs, red_rverrs = reduce_gp(gp_rv, rv_times, rvs, rverrs, rv_gp_feed)

        if plot_individual_transits :
            plot_posterior_multimodel(tr_times, red_fluxes, red_fluxerrs, posterior, plot_prev_model=False)
        plot_all_transits(tr_times, red_fluxes, red_fluxerrs, posterior, bindata=True)
        
        plot_rv_timeseries(planet_params, rvcalib_params, rv_times, red_rvs, red_rverrs, samples=samples, labels=labels, planet_index=0, plot_residuals=True, phasefold=False, rvlabel=rvlabel)
        
        #t0 = posterior["planet_params"][0]['tc_{0:03d}'.format(0)] + 0.5 * posterior["planet_params"][0]['per_{0:03d}'.format(0)]
        t0 = posterior["planet_params"][0]['tc_{0:03d}'.format(0)]
        plot_rv_timeseries(planet_params, rvcalib_params, rv_times, red_rvs, red_rverrs, samples=samples, labels=labels, planet_index=0, plot_residuals=True, phasefold=True, plot_bin_data=True, rvlabel=rvlabel,t0=t0)
        #plot model
        #plot_priors_model(tr_times, fluxes, fluxerrs, posterior)
        
        #- make a pairs plot from MCMC output:
        #flat_samples = sampler.get_chain(discard=burnin,flat=True)
        pairs_plot_gp_emcee(samples, labels, calib_params, planet_params[0], gp_params, output='pairsplot.png', addlabels=True)

    return posterior


def reduce_gp(gp, x, y, yerr, gp_feed) :

    gp.compute(gp_feed["t"], gp_feed["yerr"])
    yout, yerrout = deepcopy(y), deepcopy(yerr)
    for i in range(len(x)) :
        pred_mean, pred_var = gp.predict(gp_feed['y'], x[i], return_var=True)
        pred_std = np.sqrt(pred_var)
        yout[i] -= pred_mean
        #yerrout[i] = np.sqrt(yerrout[i]*yerrout[i] + pred_std*pred_std)

    return yout, yerrout


def pairs_plot_gp_emcee(samples, labels, calib_params, planet_params, gp_params, output='', addlabels=True) :
    truths=[]
    font = {'size': 12}
    matplotlib.rc('font', **font)
    
    newlabels = []
    for lab in labels :
        if lab in calib_params.keys():
            truths.append(calib_params[lab])
        elif lab in planet_params.keys():
            truths.append(planet_params[lab])
        elif lab in gp_params.keys():
            truths.append(gp_params[lab])

        if lab == 'rp_000':
            newlabels.append(r"R$_{p}$/R$_{\star}$")
        elif lab == 'a_000':
            newlabels.append(r"a/R$_{\star}$")
        elif lab == 'tc_000':
            newlabels.append(r"T$_c$ [BTJD]")
        elif lab == 'per_000':
            newlabels.append(r"P [d]")
        elif lab == 'inc_000':
            newlabels.append(r"$i$ [$^{\circ}$]")
        elif lab == 'u0_000':
            newlabels.append(r"u$_{0}$")
        elif lab == 'u1_000':
            newlabels.append(r"u$_{1}$")
        elif lab == 'ecc_000':
            newlabels.append(r"$e$")
        elif lab == 'w_000':
            newlabels.append(r"$\omega$ [deg]")
        elif lab == 'tp_000':
            newlabels.append(r"T$_p$ [BJD]")
        elif lab == 'k_000':
            newlabels.append(r"K$_p$ [m/s]")
        elif lab == 'rvsys_000':
            newlabels.append(r"$\gamma$ [km/s]")
        elif lab == 'trend_000':
            newlabels.append(r"$\alpha$ [km/s/d]")
    
        # Below are SH GP related variables:
        elif lab == 'log_S0_RV':
            newlabels.append(r"$\log(S_v)$")
        elif lab == 'log_S0_Blong':
            newlabels.append(r"$\log(S_B)$")
        elif lab == 'log_S0_Phot':
            newlabels.append(r"$\log(S_p)$")

        # Below are all star rot GP related variables:
        elif lab == 'phot_mean':
            newlabels.append(r"$\mu_p$")
        elif lab == 'blong_mean':
            newlabels.append(r"$\mu_B$ [G]")
        elif lab == 'rv_mean':
            newlabels.append(r"$\mu_v$  [m/s]")
        elif lab == 'phot_white_noise':
            newlabels.append(r"$\sigma_p$")
        elif lab == 'blong_white_noise':
            newlabels.append(r"$\sigma_B$ [G]")
        elif lab == 'rv_white_noise':
            newlabels.append(r"$\sigma_v$  [m/s]")
        elif lab == 'phot_amplitude':
            newlabels.append(r"$\alpha_p$")
        elif lab == 'blong_amplitude':
            newlabels.append(r"$\alpha_B$ [G]")
        elif lab == 'rv_amplitude':
            newlabels.append(r"$\alpha_v$  [m/s]")
        elif lab == 'phot_decaytime':
            newlabels.append(r"$l_p$ [d]")
        elif lab == 'spec_decaytime':
            newlabels.append(r"$l_s$ [d]")
        elif lab == 'blong_decaytime':
            newlabels.append(r"$l_B$ [d]")
        elif lab == 'rv_decaytime':
            newlabels.append(r"$l_v$ [d]")
        elif lab == 'phot_smoothfactor':
            newlabels.append(r"$\beta_p$ [d]")
        elif lab == 'spec_smoothfactor':
            newlabels.append(r"$\beta_s$ [d]")
        elif lab == 'blong_smoothfactor':
            newlabels.append(r"$\beta_B$ [d]")
        elif lab == 'rv_smoothfactor':
            newlabels.append(r"$\beta_v$ [d]")
        elif lab == 'phot_prot':
            newlabels.append(r"Prot$_p$ [d]")
        elif lab == 'spec_prot':
            newlabels.append(r"Prot$_s$ [d]")
        elif lab == 'blong_prot':
            newlabels.append(r"Prot$_B$ [d]")
        elif lab == 'rv_prot':
            newlabels.append(r"Prot$_v$ [d]")
        elif lab == 'prot':
            newlabels.append(r"Prot [d]")
        else :
            newlabels.append(lab)
    
    if addlabels :
        fig = corner.corner(samples, show_titles=True, labels = newlabels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], truths=truths, labelsize=12, labelpad=2.0)

        #fig = corner.corner(samples, labels = newlabels, quantiles=[0.16, 0.5, 0.84], truths=truths, label_size=10, max_n_ticks=4, tick_size=10, colormain='tab:blue', truth_color=(1,102/255,102/255),colorhist='tab:blue', colorbackgd=(240/255,240/255,240/255),tick_rotate=30)
    else :
        fig = corner.corner(samples, show_titles=True, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], labelpad=2.0, truths=truths, labelsize=12)
        #fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labelpad=1.0,truths=truths, label_size=10, max_n_ticks=4, tick_size=10,colormain='tab:blue', truth_color=(1,102/255,102/255),colorhist='tab:blue', colorbackgd=(240/255,240/255,240/255),tick_rotate=30)

    for ax in fig.get_axes():
        plt.setp(ax.get_xticklabels(), ha="left", rotation=45)
        plt.setp(ax.get_yticklabels(), ha="right", rotation=45)
        ax.tick_params(axis='both', labelsize=8)

    plt.savefig("pairsplot_allparams.png",bbox_inches='tight')

    plt.show()

    if output != '' :
        fig.savefig(output)
        plt.close(fig)
