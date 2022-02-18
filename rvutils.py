# -*- coding: utf-8 -*-
"""
Created on Wed May 19 2021
@author: Eder Martioli
Institut d'Astrophysique de Paris, France.
"""
import os,sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.timeseries import LombScargle
from PyAstronomy.pyasl import foldAt
from astropy.io import ascii
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy import ndimage

import exoplanetlib

from celerite.modeling import Model
from scipy.optimize import minimize
import celerite
from celerite import terms
import emcee
import corner


def read_rv_time_series(filename) :
    """
        Description: function to read RV data from *.rdb file
        """
    try :
        if ("lbl_" in filename) or ("lbl2_" in filename) :
            try :
                tbl = Table.read(filename, format="rdb", data_start=2)
                bjd, rv, erv = tbl["BJD"].data, tbl["vrad"].data, tbl["svrad"].data
            except :
                rvdata = ascii.read(filename, data_start=2)
                bjd = np.array(rvdata['rjd']) + 2400000.
                rv, erv = 1000. * np.array(rvdata["vrad"]), 1000. * np.array(rvdata["svrad"])
            if "XavierRVcorr" in filename :
                rv, erv = rv/1000, erv/1000
            
        else :
            rvdata = ascii.read(filename, data_start=2)
            bjd = np.array(rvdata['rjd']) + 2400000.
            rv, erv = 1000. * np.array(rvdata["vrad"]), 1000. * np.array(rvdata["svrad"])
    except :
        tbl = Table.read(filename, format="rdb", data_start=2)
        bjd, rv, erv = tbl["BJD"].data, tbl["vrad"].data, tbl["svrad"].data

    finite = np.isfinite(bjd) * np.isfinite(rv) * np.isfinite(erv)

    return np.array(bjd[finite], dtype='float'), np.array(rv[finite], dtype='float'), np.array(erv[finite], dtype='float')


def calculate_lbl_drifts(filename, bjd, plot=True) :

    drifts = np.zeros_like(bjd)
    
    rvdata = ascii.read(filename, data_start=2)

    bjd_data = np.array(rvdata['rjd']) + 2400000.
    vrad = np.array(rvdata['vrad'])
    densfilt = np.array(rvdata['SBCDEN_P'])
    mjdtime = np.array(rvdata['MJDMID'])
    wavetime = np.array(rvdata['WAVETIME'])
    
    df_values = []
    for i in range(len(densfilt)) :
        if densfilt[i] not in df_values :
            df_values.append(densfilt[i])
    df_values.sort()
    #df_values = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]
                
    for i in range(len(df_values)) :
        df = df_values[i]
        #color = [i/len(df_values),1-i/len(df_values),1-i/len(df_values)]
        
        dfmask = densfilt == df
        #plt.plot(bjd_data[dfmask],densfilt[dfmask],'.')
        plt.plot(bjd_data[dfmask],vrad[dfmask],'o',alpha=0.35,label="D={0:.2f}".format(df))

        #dfmask = densfilt > df
        #dfmask &= densfilt <= df + 0.5
        #plt.plot(time_from_wave[dfmask],vrad[dfmask],'.',label="{0:.1f}<df<{1:.1f}".format(df,df+0.5))

    plt.ylabel("LBL drift velocity of FP [m/s]")
    plt.xlabel("BJD")
    plt.legend(fontsize=5, ncol=4)
    plt.show()

    return drifts


def periodogram(bjd, rv, rverr, period=0., nyquist_factor=20, probabilities = [0.01, 0.001], plot=False) :
    """
        Description: calculate GLS periodogram
        """
            
    ls = LombScargle(bjd, rv, rverr)

    frequency, power = ls.autopower(nyquist_factor=nyquist_factor)

    fap = ls.false_alarm_level(probabilities)
    
    if period == 0 :
        best_frequency = frequency[np.argmax(power)]
        period = 1./best_frequency
    else :
        best_frequency = 1./period

    if plot :
        periods = 1/frequency
        plt.plot(periods, power)
        plt.vlines(1/best_frequency, np.min(power), np.max(power),ls="--", label="Max power at P={0:.4f} d".format(1/best_frequency))
        
        for i in range(len(fap)) :
            plt.text(np.max(periods),fap[i]+0.01,r"FAP={0:.1f}%".format(100*probabilities[i]),horizontalalignment='right')
            plt.hlines([fap[i]], np.min(periods), np.max(periods),ls=":", lw=0.5)
        plt.xscale('log')
        plt.xlabel("Period [d]")
        plt.ylabel("Power")
        plt.legend()
        plt.show()

    phases = foldAt(bjd, 1/best_frequency, T0=bjd[0])
    sortIndi = np.argsort(phases)

    if plot :
        plt.errorbar(phases[sortIndi],rv[sortIndi],yerr=rverr[sortIndi],fmt='o')
        plt.ylabel(r"RV [km/s]")
        plt.xlabel("phase (P={0:.3f} d)".format(1/best_frequency))
        plt.show()

    loc = {}
    loc['best_frequency'] = best_frequency
    loc['period'] = period
    loc['power'] = power
    loc['frequency'] = frequency
    loc['phases'] = phases
    loc['fap'] = fap

    return loc

def calculate_rvmodel(params, bjd) :
    
    per = params['per']
    tp = params['tp']
    tt = params['tt']
    ecc = params['ecc']
    om = params['om']
    inc = params['inc']
    ks = params['ks']
    mstar = params['mstar']
    rstar = params['rstar']
    rp = params['rp']
    
    vel = exoplanetlib.rv_model(bjd, per, tp, ecc, om, ks) + params['rv_sys']

    return vel


def fit_rv_timeseries_with_trend(params, bjd, rv, rverr) :
    
    per = params['per']
    tp = params['tp']
    ecc = params['ecc']
    om = params['om']
    inc = params['inc']
    k = params['ks']
    mstar = params['mstar']
    rstar = params['rstar']
    rp = params['rp']
    rv_sys = params['rv_sys']
    trend = params['trend']
    
    def rv_orbit(t, shift, a, tpp, kk) :
        vel = exoplanetlib.rv_model(t, per, tpp, ecc, om, kk)
        return vel + a*t + shift
    guess = [rv_sys, trend, tp, k]
    
    pfit, pcov = curve_fit(rv_orbit, bjd, rv, p0=guess)

    rv_fit_model = rv_orbit(bjd, *pfit)

    params['rv_sys'] = pfit[0]
    params['trend'] = pfit[1]
    params['tp'], params['ks'] = pfit[2], pfit[3]

    params['tt'] = exoplanetlib.timeperi_to_timetrans(params['tp'], params['per'], params['ecc'], params['om'])

    residuals = rv - rv_fit_model
    
    params["chi2"] = np.nansum((residuals**2)/(rverr*rverr))
    params["rms"] = np.nanstd(residuals)
    params["mad"] = np.nanmedian(np.abs(residuals)) / 0.67449

    return params


def fit_rv_timeseries(params, bjd, rv, rverr) :
    
    per = params['per']
    tp = params['tp']
    ecc = params['ecc']
    om = params['om']
    inc = params['inc']
    k = params['ks']
    mstar = params['mstar']
    rstar = params['rstar']
    rp = params['rp']
    rv_sys = params['rv_sys']
    
    if ecc == 0. :
        def rv_orbit(t, shift, tpp, kk) :
            vel = exoplanetlib.rv_model(t, per, tpp, ecc, om, kk)
            return vel + shift
        guess = [rv_sys, tp, k]
        #bounds = ([0,tp-per/2,0],[10000,tp+per/2,1000])
        pfit, pcov = curve_fit(rv_orbit, bjd, rv, p0=guess)
    else :
        def rv_orbit(t, shift, tpp, kk, omm) :
            vel = exoplanetlib.rv_model(t, per, tpp, ecc, omm, kk)
            return vel + shift
        guess = [rv_sys, tp, k, om]
        bounds = ([0,tp-per/2,0,0],[10000,tp+per/2,1000,360])
        
        pfit, pcov = curve_fit(rv_orbit, bjd, rv, p0=guess,bounds=bounds)
        
        params['om']= pfit[3]

    rv_fit_model = rv_orbit(bjd, *pfit)

    params['rv_sys'] = pfit[0]
    params['tp'], params['ks'] = pfit[1], pfit[2]

    params['tt'] = exoplanetlib.timeperi_to_timetrans(params['tp'], params['per'], params['ecc'], params['om'])

    residuals = rv - rv_fit_model
    
    params["chi2"] = np.nansum((residuals**2)/(rverr*rverr))
    params["rms"] = np.nanstd(residuals)
    params["mad"] = np.nanmedian(np.abs(residuals)) / 0.67449

    return params


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


def plot_rv_timeseries_modelset(params, bjd, rv, rverr, ks_set=[1.,5.,20.], datalabel=None, xlabel="BJD", ylabel=r"RV [km/s]", phasefold=False) :

    per = params['per']
    tp = params['tp']
    tt = params['tt']
    ecc = params['ecc']
    om = params['om']
    inc = params['inc']
    k = params['ks']
    mstar = params['mstar']
    rstar = params['rstar']
    rp = params['rp']
    rv_sys = params['rv_sys']
    trend = params['trend']
    
    phases_tt = foldAt(bjd, per, T0=tt)

    obs_rvsys = rv_sys + bjd*trend
    
    ti,tf = bjd[0],bjd[-1]
    time = np.arange(ti, tf, 0.001)
    model_rvsys = rv_sys + time*trend

    if phasefold :
        obs_phases = foldAt(bjd, per, T0=bjd[0])
        obs_sort = np.argsort(obs_phases)
        plt.errorbar(obs_phases[obs_sort], (rv-obs_rvsys)[obs_sort], yerr=rverr[obs_sort], fmt='o', alpha=0.7, label=datalabel)
        plt.xlabel("phase (P={0:.3f} d)".format(per),fontsize=18)
    else :
        plt.errorbar(bjd, rv, yerr=rverr, fmt='o', alpha=0.7, label=datalabel)
        plt.xlabel(xlabel,fontsize=18)


    for k in ks_set :
        obs_rvmodel = exoplanetlib.rv_model(bjd, per, tp, ecc, om, k) + obs_rvsys
        rvmodel = exoplanetlib.rv_model(time, per, tp, ecc, om, k) + model_rvsys
        mp_earth = exoplanetlib.planet_mass(per, inc, ecc, mstar, k, units='mearth')
        if phasefold :
            plt.plot(obs_phases[obs_sort], (obs_rvmodel-obs_rvsys)[obs_sort], '-', label=r"model M$_p$={:.0f} M$_E$".format(mp_earth))
        else :
            plt.plot(time, rvmodel, '-', label=r"model M$_p$={:.0f} M$_E$".format(mp_earth))

    plt.legend(fontsize=18)
    plt.ylabel(ylabel,fontsize=18)
    plt.show()



def plot_rv_timeseries(params, bjd, rv, rverr, bindata=True, datalabel=None, xlabel="BJD", ylabel=r"RV [km/s]", phasefold=False) :

    per = params['per']
    tp = params['tp']
    tt = params['tt']
    ecc = params['ecc']
    om = params['om']
    inc = params['inc']
    k = params['ks']
    mstar = params['mstar']
    rstar = params['rstar']
    rp = params['rp']
    rv_sys = params['rv_sys']
    trend = params['trend']
    
    phases_tt = foldAt(bjd, per, T0=tt)

    obs_rvsys = rv_sys + bjd*trend
    
    obs_rvmodel = exoplanetlib.rv_model(bjd, per, tp, ecc, om, k) + obs_rvsys

    ti,tf = bjd[0],bjd[-1]
    time = np.arange(ti, tf, 0.001)
    model_rvsys = rv_sys + time*trend
    rvmodel = exoplanetlib.rv_model(time, per, tp, ecc, om, k) + model_rvsys

    if phasefold :
        obs_phases = foldAt(bjd, per, T0=bjd[0])
        obs_sort = np.argsort(obs_phases)

        plt.errorbar(obs_phases[obs_sort], (rv-obs_rvsys)[obs_sort], yerr=rverr[obs_sort], fmt='o', alpha=0.7, label=datalabel)

        if bindata :
            bin_ph, bin_rv, bin_rverr = bin_data(obs_phases[obs_sort], (rv-obs_rvsys)[obs_sort], rverr[obs_sort], median=False, binsize = 0.02)
        
            plt.errorbar(bin_ph, bin_rv, yerr=bin_rverr, fmt='ko', lw=3)
        
        plt.plot(obs_phases[obs_sort], (obs_rvmodel-obs_rvsys)[obs_sort], '-', color="darkgreen", label='model')
        plt.xlabel("phase (P={0:.3f} d)".format(per))

    else :
        
        plt.errorbar(bjd, rv, yerr=rverr, fmt='o', alpha=0.7, label=datalabel)

        if bindata :
            bin_bjd, bin_rv, bin_rverr = bin_data(bjd, rv, rverr, median=False, binsize = 0.5)
            plt.errorbar(bin_bjd, bin_rv, yerr=bin_rverr, fmt='ko', lw=3, label="Binned data")

        plt.plot(time, rvmodel, '-', color="darkgreen", label='model')
        plt.xlabel(xlabel)

        #plt.xlim(2459450,2459550)
    plt.legend()
    plt.ylabel(ylabel)
    plt.show()


def plot_calibrv_timeseries(params, bjd, rv, rverr, calibrv, calibrverr, phasefold=False) :

    per, tp, ecc, om, k = params['per'], params['tp'], params['ecc'], params['om'], params['ks']
    rv_sys = params['rv_sys']
    trend = params['trend']

    obs_rvmodel = exoplanetlib.rv_model(bjd, per, tp, ecc, om, k)

    if phasefold :
        obs_phases = foldAt(bjd, per, T0=bjd[0])
        obs_sort = np.argsort(obs_phases)
        plt.errorbar(obs_phases[obs_sort], rv[obs_sort]-(rv_sys+bjd*trend), yerr=rverr[obs_sort], fmt='ro', alpha=0.3, label="uncalibrated")
        plt.errorbar(obs_phases[obs_sort], calibrv[obs_sort]-(rv_sys+bjd*trend), yerr=calibrverr[obs_sort], fmt='ko', label="calibrated")
        plt.plot(obs_phases[obs_sort], obs_rvmodel[obs_sort], '-', lw=2, label='model')
        plt.xlabel("phase (P={0:.3f} d)".format(per))

    else :
        ti,tf = bjd[0],bjd[-1]
        time = np.arange(ti, tf, 0.001)
        rvmodel = exoplanetlib.rv_model(time, per, tp, ecc, om, k) + (rv_sys + time*trend)

        plt.errorbar(bjd, rv, yerr=rverr, fmt='ro', alpha=0.3, label="uncalibrated")
        plt.errorbar(obs_phases, calibrv, yerr=calibrverr, fmt='ko', label="calibrated")
        plt.plot(time, rvmodel, '-', lw=2, label='model')
        plt.xlabel("BJD")

    plt.legend()
    plt.ylabel(r"RV [km/s]")
    plt.show()


def gp_kernel(y) :
    # A non-periodic component
    Q = 1.0 / np.sqrt(2.0)
    w0 = 3.0
    S0 = np.var(y) / (w0 * Q)
    bounds = dict(log_S0=(-15, 15), log_Q=(-15, 15), log_omega0=(-15, 15))
    kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                       bounds=bounds)
    kernel.freeze_parameter("log_Q")  # We don't want to fit for "Q" in this term

    # A periodic component
    Q = 1.0
    w0 = 3.0
    S0 = np.var(y) / (w0 * Q)
    kernel += terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                        bounds=bounds)
    return kernel


# Define the model
class MeanModel(Model):
    parameter_names = ("alpha", "ell", "log_sigma2")
    
    def get_value(self, t):
        return self.alpha * np.exp(-0.5*(t-self.ell)**2 * np.exp(-self.log_sigma2))
    
    # This method is optional but it can be used to compute the gradient of the
    # cost function below.
    def compute_gradient(self, t):
        e = 0.5*(t-self.ell)**2 * np.exp(-self.log_sigma2)
        dalpha = np.exp(-e)
        dell = self.alpha * dalpha * (t-self.ell) * np.exp(-self.log_sigma2)
        dlog_s2 = self.alpha * dalpha * e
        return np.array([dalpha, dell, dlog_s2])

# Define a cost function
def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

def grad_neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.grad_log_likelihood(y)[1]

def gp_calib_fit_model(x, y, dy, fit_mean=True, jitter=True):
    """
    Return a model using an (approximate) Matern 3/2 kernel.
    - Input -
    x:          Sampling locations.
    y:          Observations.
    dy:         Uncertainties.
    fit_mean:   Fit the mean of the observations.
    jitter:     Include a Jitter term in the GP model.
    return_var: Return the standard deviation of the fit.
    oversample: If true, sample the GP at a higher resolution. If true will use
                a default of 5, otherwise a value can be specified. Good for
                making plots look nice.
    - Returns -
    xx:         Sampling points of the GP
    yy:         Samples of the GP.
    zz:         Standard deviation of the GP fit if return_var is True.
    """

    # Make sure arrays are ordered for celerite.
    idx = np.argsort(x)
    x, y, dy = x[idx], y[idx], dy[idx]

    # Define the Matern-3/2 Kernel.
    lnsigma, lnrho = np.log(np.nanstd(y)), np.log(np.nanmean(abs(np.diff(x))))
    kernel = celerite.terms.Matern32Term(log_sigma=lnsigma, log_rho=lnrho)
    
    # Include a noise kernel if appropriate.
    if jitter:
        if np.nanmean(dy) == 0.0:
            noise = celerite.terms.JitterTerm(log_sigma=np.log(np.nanmean(dy)))
        else:
            noise = celerite.terms.JitterTerm(log_sigma=np.log(np.nanstd(y)))
        kernel = kernel + noise
    
    # Compute the GP.
    gp = celerite.GP(kernel, mean=np.nanmean(y), fit_mean=fit_mean)
    gp.compute(x, dy)

    # Minimize the results.
    params = gp.get_parameter_vector()
    params += 1e-2 * np.random.randn(len(params))
    soln = minimize(neg_log_like, params, jac=grad_neg_log_like,
                    args=(y, gp), method='L-BFGS-B')
    gp.set_parameter_vector(soln.x)

    # If the fit was successful, return the fit, otherwise just return
    # the provided values.
    if soln.success:
        print('Solution Found =)')
    else:
        print('No Solution Found =(')

    print("Final log-likelihood: {0}".format(-soln.fun))

    return gp

"""
def gp_calib_fit_model(cbjds, crvs, crverrs) :
  
    #mean_model = MeanModel(alpha=-1.0, ell=0.1, log_sigma2=np.log(0.4))
    #gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
    gp = celerite.GP(kernel, mean=np.mean(crvs), fit_mean=True)
    gp.compute(cbjds, crverrs)
    #print("Initial log-likelihood: {0}".format(gp.log_likelihood(crvs)))
    
    # Fit for the maximum likelihood parameters
    initial_params = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                    method="L-BFGS-B", bounds=bounds, args=(crvs, gp))
    gp.set_parameter_vector(soln.x)
    #print("Final log-likelihood: {0}".format(-soln.fun))
    return gp
"""

def calibrate_rvs(calibfiles, bjd, rv, rverr, sig_clip=0.,median_filter=False, med_filter_size=30, fiber='AB', suffix='', plot=False, verbose=False) :
    
    bjds, rvs, rverrs, medrvs = [], [], [], []
    
    c1bjds, c1rvs, c1rverrs = np.array([]), np.array([]), np.array([])
    
    for i in range(len(calibfiles)) :
        
        lbjds, lrvs, lrverrs = read_rv_time_series(calibfiles[i])
    
        mask = np.isfinite(lrvs) * np.isfinite(lrverrs)
            
        medianlrv = np.nanmedian(lrvs[mask])
        
        if sig_clip > 0:
            sigma = np.nanmedian(np.abs(lrvs[mask]-medianlrv)) / 0.67449
            mask &= lrvs > medianlrv - sig_clip * sigma
            mask &= lrvs < medianlrv + sig_clip * sigma

        bjds.append(lbjds[mask])
        rvs.append(lrvs[mask])
        rverrs.append(lrverrs[mask])
        medrvs.append(medianlrv)
        
        c1bjds = np.append(c1bjds, lbjds[mask])
        c1rvs = np.append(c1rvs, lrvs[mask] - medianlrv)
        c1rverrs = np.append(c1rverrs, lrverrs[mask])

    sort1Ind = np.argsort(c1bjds)
    c1bjds, c1rvs, c1rverrs = c1bjds[sort1Ind], c1rvs[sort1Ind], c1rverrs[sort1Ind]

    # generate a first calibration model using GP
    gp1 = gp_calib_fit_model(c1bjds, c1rvs, c1rverrs)
    calib1_model, calib1_var = gp1.predict(c1rvs, c1bjds, return_var=True)
    calib1_std = np.sqrt(calib1_var)

    if verbose:
        print("RMS of 1st calibration residuals: {0:.1f} m/s".format(1000.*np.nanstd(c1rvs - calib1_model)))

    # Start over calibration now using prior GP model
    
    cbjds, crvs, crverrs = np.array([]), np.array([]), np.array([])

    for i in range(len(calibfiles)) :
        objname = os.path.basename(calibfiles[i]).replace(".rdb","")
        if '_ccf' in objname :
            objname = objname.replace("_spirou_ccf","")
        elif 'lbl_' in objname :
            objname = objname.replace("lbl_","")
            objname = objname.replace("_drift","")

        if verbose:
            print(i, objname, medrvs[i])
    
        mu, var = gp1.predict(c1rvs, bjds[i], return_var=True)
        
        def rv_func(t, shift) :
            return mu + shift
    
        # Fit all data to match first GP model
        guess = [medrvs[i]]
        pfit, pcov = curve_fit(rv_func, bjds[i], rvs[i], p0=guess)

        medrvs[i] = pfit[0]
        rv_calib = rvs[i] - medrvs[i]
        
        cbjds = np.append(cbjds, bjds[i])
        crvs = np.append(crvs, rv_calib)
        crverrs = np.append(crverrs, rverrs[i])
        
        if plot :
            plt.errorbar(bjds[i], rv_calib, yerr=rverrs[i], fmt='o', alpha=0.3, label="{0} RV={1:.3f} km/s".format(objname, medrvs[i]))

    sortInd = np.argsort(cbjds)
    cbjds, crvs, crverrs = cbjds[sortInd], crvs[sortInd], crverrs[sortInd]
    
    if median_filter :
        #plt.errorbar(cbjds, crvs, yerr=crverrs, fmt='o', alpha=0.3)
        crvs = ndimage.median_filter(crvs, size=med_filter_size)
        crverrs /= np.sqrt(med_filter_size)
        if plot :
            plt.plot(cbjds, crvs, 'ko', label="median filtered")

    # generate final calibration model using GP
    gp = gp_calib_fit_model(cbjds, crvs, crverrs)

    calib_model, calib_var = gp.predict(crvs, cbjds, return_var=True)
    calib_std = np.sqrt(calib_var)
    if verbose :
        print("RMS of calibration residuals: {0:.1f} m/s".format(1000.*np.nanstd(crvs - calib_model)))

    if plot :
        # Make the maximum likelihood prediction
        t = np.linspace(cbjds[0], cbjds[-1], 10000)
        mu, var = gp.predict(crvs, t, return_var=True)
        std = np.sqrt(var)

        # Plot the data
        color = "#ff7f0e"
        #plt.errorbar(cbjds, crvs, yerr=crverrs, fmt=".k", capsize=0)
        plt.plot(t, mu, color=color, label="GP model")
        plt.fill_between(t, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
        plt.ylabel(r"Radial velocity [km/s]")
        plt.xlabel(r"BJD")
        plt.xlim(cbjds[0], cbjds[-1])
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
        plt.title("maximum likelihood prediction")
        plt.legend()
        plt.show()

    mu, var = gp.predict(crvs, bjd, return_var=True)
    calib_rv  = rv - mu
    calib_rverr = np.sqrt(var + rverr*rverr)
    
    if plot :
        rv_sys = np.nanmedian(rv)
        plt.plot(bjd, np.full_like(bjd,rv_sys), '--', color='darkgreen', label="Systemic RV = {0:.3f} km/s".format(rv_sys))
        # Plot the data
        plt.errorbar(bjd, rv, yerr=rverr, fmt=".r", capsize=0, label='uncalibrated', alpha=0.3)
        plt.errorbar(bjd, calib_rv, yerr=calib_rverr, fmt=".k", capsize=0, label='calibrated')
        
        # Make the maximum likelihood prediction
        t = np.linspace(bjd[0], bjd[-1], 10000)
        mu, var = gp.predict(crvs, t, return_var=True)
        std = np.sqrt(var)

        plt.plot(t, mu+rv_sys, color=color, label="Instrumental calibration from GP")
        plt.fill_between(t, mu+std+rv_sys, mu-std+rv_sys, color=color, alpha=0.3, edgecolor="none")
        plt.ylabel(r"Radial velocity [km/s]")
        plt.xlabel(r"BJD")
        plt.xlim(bjd[0], bjd[-1])
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
        plt.title("Gaussian process calibration")
        plt.legend()
        plt.show()

    return bjd, calib_rv, calib_rverr


def save_rv_time_series(output, bjd, rv, rverr, time_in_rjd=True, rv_in_mps=False) :
    
    outfile = open(output,"w+")
    outfile.write("rjd\tvrad\tsvrad\n")
    outfile.write("---\t----\t-----\n")
    
    for i in range(len(bjd)) :
        if time_in_rjd :
            rjd = bjd[i] - 2400000.
        else :
            rjd = bjd[i]
        
        if rv_in_mps :
            outfile.write("{0:.10f}\t{1:.2f}\t{2:.2f}\n".format(rjd, 1000. * rv[i], 1000. * rverr[i]))
        else :
            outfile.write("{0:.10f}\t{1:.5f}\t{2:.5f}\n".format(rjd, rv[i], rverr[i]))

    outfile.close()


def periodgram_rv_check_signal(bjd, rv, rverr, period=[], period_labels=[], period_range=[5,45], nbins=3, phase_plot=False) :

    binsize = int(np.floor(len(bjd) / nbins))

    maxpower = 0

    for i in range(nbins) :

        last_point = (i+1)*binsize
        if last_point > len(bjd) :
            last_point = len(bjd)
        print("Subset {}/{}: considering points from 0:{}".format(i+1,nbins,last_point))
        
        frequency, power = LombScargle(bjd[:last_point], rv[:last_point]).autopower()
    
        periods = 1/frequency

        if np.nanmax(power) > maxpower :
            maxpower = np.nanmax(power)

        color = [i/nbins,1-i/nbins,1-i/nbins]

        plt.plot(periods, power, color=color, label="N points: {}".format(last_point))

    if period == [] :
        best_period = [1./frequency[np.argmax(power)]]
        plt.vlines(best_period[0], 0, maxpower,ls="--", label="Max power at P={0:.4f} d".format(best_period))
    else :
        colors = ["darkgreen","steelblue","coral"]
    
        best_period = period
        if len(period_labels) != len(period) :
            period_labels = []
            for i in range(len(period)) :
                period_labels.append("")
        for i in range(len(period)) :
            plt.vlines(period[i], 0, maxpower, ls="--", color=[colors[i]], label="{0} P={1:.2f} d".format(period_labels[i],period[i]))

    min_per, max_per = period_range[0],period_range[1]

    #plt.xscale('log')
    plt.xlabel("Period [d]")
    plt.ylabel("Power")
    plt.xlim(min_per, max_per)
    plt.legend()
    plt.show()

    if phase_plot :
        for per in best_period :
            phases = foldAt(bjd, per, T0=bjd[0])
            sortIndi = np.argsort(phases)
            plt.errorbar(phases[sortIndi],rv[sortIndi],yerr=rverr[sortIndi],fmt='o')
            plt.ylabel(r"RV [km/s]")
            plt.xlabel("phase (P={0:.3f} d)".format(per))
            plt.show()

