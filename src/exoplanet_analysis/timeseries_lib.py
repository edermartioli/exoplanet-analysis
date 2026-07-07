# -*- coding: utf-8 -*-
"""
    Created on Oct 18 2022
    
    Description: library with utilities for the analysis of time series
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy.timeseries import LombScargle
from PyAstronomy.pyasl import foldAt

import mpmath  # https://code.google.com/p/mpmath/
    
from exoplanet_analysis import rvutils, gp_lib, priorslib
    
def plot_bgls(periods, power, period=0, npeaks=1, y_label='y', phaseplot=True) :

    """Plot bgls.

    Parameters
    ----------
    periods
    power
    period : int, optional (default: 0)
        Period [d].
    npeaks : int, optional (default: 1)
    y_label : str, optional (default: 'y')
    phaseplot : bool, optional (default: True)
    """
    if period == 0 :
        sorted = np.argsort(power)
        best_periods = periods[sorted][-npeaks:]
        best_powers = power[sorted][-npeaks:]
    else :
        best_periods = [period]
        best_powers = [np.nanmax(power)]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i in range(len(best_periods)) :
        best_period, best_power = best_periods[i], best_powers[i]
        plt.vlines(best_period, np.min(power), best_power, ls="--", color=colors[i], label="Max power at P={0:.1f} d".format(best_period), zorder=2)
    
    plt.plot(periods, power, color="k", zorder=1)
    
    plt.xlabel("Period [d]", fontsize=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("Power", fontsize=15)
    plt.legend(fontsize=16)
    plt.show()

    
def sbglsp(x, y, yerr, niter=1, min_nobs=10, period=0, min_period=1.0, max_period=1.0, period_sampling=0.1, ofac=100, plot_snr=True) :
    #stacked Bayesian general Lomb-Scargle periodogram (SBGLSP) analysis

    """Sbglsp.

    Parameters
    ----------
    x
        Array of x values.
    y
        Array of y values.
    yerr
        Array of y uncertainties.
    niter : int, optional (default: 1)
        Number of MCMC steps.
    min_nobs : int, optional (default: 10)
    period : int, optional (default: 0)
        Period [d].
    min_period : float, optional (default: 1.0)
    max_period : float, optional (default: 1.0)
    period_sampling : float, optional (default: 0.1)
    ofac : int, optional (default: 100)
    plot_snr : bool, optional (default: True)
    """
    my, myerr, totalnobs = np.median(y), np.median(yerr), len(x)

    plow=min_period
    phigh=max_period
    
    if period < min_period or period > max_period:
        print("ERROR: expected period must be between min and max period, exiting ...")
        exit()

    per_arr = np.arange(plow,phigh,period_sampling)
    
    kperiod = np.abs(per_arr - period).argmin()
    
    nobs = np.arange(min_nobs,len(x))

    power, noise = [], []
    
    # set array of indexes
    idx = np.arange(len(x))
    
    for i in range(len(nobs)) :
        print("Calculating SBGLSP for {} of {} observations".format(nobs[i] + 1, nobs[-1] + 1))

        # set mask to select a number of observations
        mask = idx < nobs[i]
    
        pow_arr = np.zeros_like(per_arr)
        noisepow_arr = np.zeros_like(per_arr)

        for j in range(niter) :
            # shuffle mask array to get independent points in each iteration
            np.random.shuffle(mask)
            
            # generate simulated noise array
            noise_arr = np.random.normal(my, myerr, totalnobs)

            # select a number of observations
            short_x, short_y, short_yerr = x[mask], y[mask], yerr[mask]
            short_noise = noise_arr[mask]
            
            # calculate periodogram
            per, powr = bgls(short_x, short_y, short_yerr, plow=plow, phigh=phigh, ofac=ofac)
            _, noise_powr = bgls(short_x, short_noise, short_yerr, plow=plow, phigh=phigh, ofac=ofac)

            for k in range(len(per_arr)) :
                keep = (per > per_arr[k]-period_sampling/2) & (per <= per_arr[k]+period_sampling/2)
                pow_arr[k] += np.mean(powr[keep])
                noisepow_arr[k] += np.mean(noise_powr[keep])
          
        power.append(pow_arr)
        noise.append(noisepow_arr)
    
    power = np.array(power, dtype=float)
    noise = np.array(noise, dtype=float)
    
    plt.title("Significance of detection at P={:.3f} d".format(per_arr[kperiod]), fontsize=18)
    plt.plot(nobs, power[:,kperiod]/noise[:,kperiod], color='k')
    plt.hlines(1.0, np.min(nobs), np.max(nobs), ls="--", color='red', zorder=2, label="SNR=1")
    plt.ylabel("SNR", fontsize=18)
    plt.xlabel("Number of observations", fontsize=18)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()
    
    if plot_snr :
        snr_arr = power[-1]/noise[-1]
        plt.plot(per_arr,snr_arr, color='k')
        plt.hlines(1.0, np.min(per_arr), np.max(per_arr), ls="--", color='red', zorder=2, label="SNR=1")
        plt.vlines(period, 0, 1.2*np.max(snr_arr), ls=":", color='darkblue', zorder=2, label="P = {:.3f} d".format(period))
        plt.ylabel("SNR", fontsize=18)
    else :
        plt.plot(per_arr,power[-1], color='k', label='Data')
        plt.plot(per_arr,noise[-1],'--', color='k', label='Simulated noise')
        plt.vlines(period, 0, 1.2*np.max(power[-1]), ls=":", color='darkblue', zorder=2, label="P = {:.3f} d".format(period))
        plt.ylabel("Power", fontsize=18)
        
    plt.legend(fontsize=16)
    plt.yscale('log')
    plt.xlabel("Period [d]", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()
    
    x_lab = r"Period [d]"
    y_lab = r"Number of observations"
    coolwarm_color_map = plt.cm.get_cmap('coolwarm')
    
    if plot_snr :
        snr = np.log(power/noise)

        z_lab = r"$\log{SNR}$"
        LAB = [x_lab,y_lab,z_lab]
        plot_2d(per_arr, nobs, snr, test_period=period, LAB=LAB, use_index_in_y=False, cmap=coolwarm_color_map)

    else :
        z_lab = r"$\log{P}$"
        LAB = [x_lab,y_lab,z_lab]
        plot_2d(per_arr, nobs, np.log(power), test_period=period, LAB=LAB, use_index_in_y=False, cmap=coolwarm_color_map)
   
        z_lab = r"$\log{\sigma}$"
        LAB = [x_lab,y_lab,z_lab]
        plot_2d(per_arr, nobs, np.log(noise), test_period=period, LAB=LAB, use_index_in_y=False, cmap=coolwarm_color_map)


def periodogram(x, y, yerr, period=0., nyquist_factor=20, probabilities = [0.01, 0.001], y_label="y", check_period=0, npeaks=1, phaseplot=False, plot=False, plot_frequencies=False) :
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
        plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=18)    # fontsize of the tick labels

        if plot_frequencies :
            plt.plot(frequency, power, color="k", zorder=1)
        else :
            plt.plot(periods, power, color="k", zorder=1)
            
        for i in range(len(best_frequencies)) :
            best_frequency, best_power = best_frequencies[i], best_powers[i]
            if plot_frequencies :
                plt.vlines(best_frequency, np.min(power), best_power, ls="--", color=colors[i], label="F={0:.4f} 1/d".format(best_frequency), zorder=2)
            else :
                plt.vlines(1/best_frequency, np.min(power), best_power, ls="--", color=colors[i], label="P={0:.4f} d".format(1/best_frequency), zorder=2)

            #plt.hlines(best_power, np.min(periods), 1/best_frequency, ls="--", color=colors[i], zorder=2)
 
        if plot_frequencies :
            plt.xlabel("Frequency [1/d]", fontsize=18)
        else :
            if check_period :
                plt.vlines(check_period, np.min(power), np.max(power), color="red",ls="--", label="P={0:.4f} d".format(check_period))

            for i in range(len(fap)) :
                plt.text(np.min(periods),fap[i]+0.01,r"FAP={0:.3f}%".format(100*probabilities[i]),horizontalalignment='left', fontsize=15)
                plt.hlines([fap[i]], np.min(periods), np.max(periods),ls=":", lw=0.5)
                plt.xlabel("Period [d]", fontsize=18)
            
        #plt.yscale('log')
        plt.xscale('log')
        plt.ylabel("Power", fontsize=18)
        plt.legend(fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()
    else :
        for i in range(len(best_frequencies)) :
            best_frequency, best_power = best_frequencies[i], best_powers[i]

    phases = foldAt(x, 1/best_frequency, T0=x[0])
    sortIndi = np.argsort(phases)

    if plot and phaseplot:
        plt.errorbar(phases[sortIndi],y[sortIndi],yerr=yerr[sortIndi],fmt='o', color="k")
        plt.ylabel(r"{}".format(y_label), fontsize=16)
        plt.xlabel("phase (P={0:.3f} d)".format(1/best_frequency), fontsize=16)
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
    
    """Phase plot.

    Parameters
    ----------
    x
        Array of x values.
    y
        Array of y values.
    yerr
        Array of y uncertainties.
    gp
    fold_period
        Period used to phase-fold the data [d].
    ylabel : str, optional (default: "y")
    t0 : int, optional (default: 0)
    alpha : float, optional (default: 0.7)
    timesampling : float, optional (default: 0.001)
    """
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




def bgls(t, y, err, plow=0.5, phigh=100, ofac=1):
    # -*- coding: utf-8 -*-
    #================================================================================
    # Copyright (c) 2014-2015 Annelies Mortier, João Faria
    # Distributed under the MIT License.
    # (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
    #================================================================================

    """Compute the Bayesian Generalized Lomb-Scargle periodogram.

    Parameters
    ----------
    t
        Array of times.
    y
        Array of y values.
    err
    plow : float, optional (default: 0.5)
    phigh : int, optional (default: 100)
    ofac : int, optional (default: 1)
    """
    n_steps = int(ofac*len(t)*(1./plow - 1./phigh))
    f = np.linspace(1./phigh, 1./plow, n_steps)
    
    omegas = 2. * np.pi * f

    err2 = err * err
    w = 1./err2
    W = sum(w)

    bigY = sum(w*y)  # Eq. (10)

    p = []
    constants = []
    exponents = []

    for i, omega in enumerate(omegas):
        theta = 0.5 * np.arctan2(sum(w*np.sin(2.*omega*t)), sum(w*np.cos(2.*omega*t)))
        x = omega*t - theta
        cosx = np.cos(x)
        sinx = np.sin(x)
        wcosx = w*cosx
        wsinx = w*sinx

        C = sum(wcosx)
        S = sum(wsinx)

        YCh = sum(y*wcosx)
        YSh = sum(y*wsinx)
        CCh = sum(wcosx*cosx)
        SSh = sum(wsinx*sinx)

        if (CCh != 0 and SSh != 0):
            K = (C*C*SSh + S*S*CCh - W*CCh*SSh)/(2.*CCh*SSh)

            L = (bigY*CCh*SSh - C*YCh*SSh - S*YSh*CCh)/(CCh*SSh)

            M = (YCh*YCh*SSh + YSh*YSh*CCh)/(2.*CCh*SSh)

            constants.append(1./np.sqrt(CCh*SSh*abs(K)))

        elif (CCh == 0):
            K = (S*S - W*SSh)/(2.*SSh)

            L = (bigY*SSh - S*YSh)/(SSh)

            M = (YSh*YSh)/(2.*SSh)

            constants.append(1./np.sqrt(SSh*abs(K)))

        elif (SSh == 0):
            K = (C*C - W*CCh)/(2.*CCh)

            L = (bigY*CCh - C*YCh)/(CCh)

            M = (YCh*YCh)/(2.*CCh)

            constants.append(1./np.sqrt(CCh*abs(K)))

        if K > 0:
            raise RuntimeError('K is positive. This should not happen.')

        exponents.append(M - L*L/(4.*K))

    constants = np.array(constants)
    exponents = np.array(exponents)

    logp = np.log10(constants) + (exponents * np.log10(np.exp(1.)))

    p = [10**mpmath.mpf(x) for x in logp]

    p = np.array(p) / max(p)  # normalize

    p[p < (sys.float_info.min * 10)] = 0
    p = np.array([float(pp) for pp in p])

    return 1./f, p



def plot_2d(x, y, z, test_period=0, LIM=None, LAB=None, z_lim=None, use_index_in_y=False, title="", pfilename="", cmap="gist_heat"):
    """
    Use pcolor to display sequence of spectra
    
    Inputs:
    - x:        x array of the 2D map (if x is 1D vector, then meshgrid; else: creation of Y)
    - y:        y 1D vector of the map
    - z:        2D array (sequence of spectra; shape: (len(x),len(y)))
    - LIM:      list containing: [[lim_inf(x),lim_sup(x)],[lim_inf(y),lim_sup(y)],[lim_inf(z),lim_sup(z)]]
    - LAB:      list containing: [label(x),label(y),label(z)] - label(z) -> colorbar
    - title:    title of the map
    - **kwargs: **kwargs of the matplolib function pcolor
    
    Outputs:
    - Display 2D map of the sequence of spectra z
    
    """
    
    if use_index_in_y :
        y = np.arange(len(y))
    
    if len(np.shape(x))==1:
        X,Y  = np.meshgrid(x,y)
    else:
        X = x
        Y = []
        for n in range(len(x)):
            Y.append(y[n] * np.ones(len(x[n])))
        Y = np.array(Y,dtype=float)
    Z = z

    if LIM == None :
        x_lim = [np.min(X),np.max(X)] #Limits of x axis
        y_lim = [np.min(Y),np.max(Y)] #Limits of y axis
        if z_lim == None :
            z_lim = [np.min(Z),np.max(Z)]
        LIM   = [x_lim,y_lim,z_lim]

    if LAB == None :
        ### Labels of the map
        x_lab = r"$Velocity$ [km/s]"     #Wavelength axis
        y_lab = r"Time [BJD]"         #Time axis
        z_lab = r"CCF"     #Intensity (exposures)
        LAB   = [x_lab,y_lab,z_lab]

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (10,7)
    ax = plt.subplot(111)

    if test_period != 0 :
        P = np.full_like(X,test_period)
        ax.plot(P, Y, ls='--',color='k', lw=2)

    cc = ax.pcolor(X, Y, Z, vmin=LIM[2][0], vmax=LIM[2][1], cmap=cmap)
    cb = plt.colorbar(cc,ax=ax)
    
    ax.set_xlim(LIM[0][0],LIM[0][1])
    ax.set_ylim(LIM[1][0],LIM[1][1])
    
    ax.set_xlabel(LAB[0], fontsize=20)
    ax.set_ylabel(LAB[1],labelpad=15, fontsize=20)
    cb.set_label(LAB[2],rotation=270,labelpad=30, fontsize=20)

    ax.set_title(title,pad=15, fontsize=20)

    if pfilename=="" :
        plt.show()
    else :
        plt.savefig(pfilename, format='png')
    plt.clf()
    plt.close()



def run_gp_analysis(time, flux, fluxerr, gp_priors_file="", run_gls=False,  run_mcmc=True, walkers=32, nsteps=1000, burnin=300, amp=1e-4, best_fit_from_mode=True, plot_distributions=False, verbose=False, plot=False) :

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
        gls = rvutils.periodogram(time, flux, fluxerr, nyquist_factor=0.02, probabilities = [0.00001], plot=plot)
    
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

    #if plot :
    #    phase_plot(time, flux, fluxerr, gp, best_period, ylabel=ylabel, t0=t0, alpha=1)

    gp_feed = {}
    gp_feed["t"], gp_feed["y"], gp_feed["yerr"] = time, flux, fluxerr

    return gp, gp_feed
