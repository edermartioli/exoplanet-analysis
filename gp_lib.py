# -*- coding: utf-8 -*-


# -*- coding: iso-8859-1 -*-
"""
    Created on Jun 21 2021
    
    Description: library with several utilities using Gaussian Process
    
    @authors:  Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import emcee
import modelslib
from PyAstronomy.pyasl import foldAt
import george
from george import kernels
import corner
#from balrogo import marginals
import priorslib
from copy import deepcopy

#########################
###### START GP QP Kernel
#########################

def gnll(p, y, gp):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def ggrad_nll(p, y, gp):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)


def set_theta_priors(params, priortypes, values) :

    theta, labels, theta_priors  = np.array([]), [], {}

    for key in params.keys() :
        
        if priortypes[key] == 'Normal' or priortypes[key] == 'Normal_positive' or priortypes[key] == 'Uniform' or priortypes[key] == 'Jeffreys':
            
            theta = np.append(theta,params[key])
            labels.append(key)

            theta_priors[key] = {}
            theta_priors[key]['type'] = priortypes[key]

            if priortypes[key] == 'Normal':
                theta_priors[key]['object'] = priorslib.normal_parameter(np.array(values[key]).astype('float64'))
            elif priortypes[key] == 'Normal_positive':
                theta_priors[key]['object'] = priorslib.normal_parameter(np.array(values[key]).astype('float64'), True)
            elif priortypes[key] == 'Uniform':
                theta_priors[key]['object'] = priorslib.uniform_parameter(np.array(values[key]).astype('float64'))
                print(theta_priors[key]['object'])
            elif priortypes[key] == 'Jeffreys':
                theta_priors[key]['object'] = priorslib.jeffreys_parameter(np.array(values[key]).astype('float64'))

        elif priortypes[key] == 'FIXED' :
            theta_priors[key] = {}
            theta_priors[key]['type'] = priortypes[key]
            theta_priors[key]['object'] = priorslib.constant_parameter(params[key])
        
        else :
            print("WARNING: couldn't recognize prior type", priortypes[key], "for variable", key)

    return  labels, theta, theta_priors


def set_star_rotation_priors(gp, period_lim=(1,500), aplitude_lim=(0,1e10), decaytime_lim=(10,300), smoothfactor_lim=(0.1,1.0)) :

    params = get_star_rotation_gp_params(gp)
    
    priortypes = {}
    values = {}

    priortypes["mean"] = 'Normal'
    values["mean"] = (params["mean"],np.abs(params["mean"])*0.5)

    priortypes["white_noise"] = 'Normal_positive'
    values["white_noise"] = (params["white_noise"],np.abs(params["white_noise"])*0.5)

    priortypes["amplitude"] = 'Uniform'
    values["amplitude"] = aplitude_lim
    
    if "kernel:k1:k2:metric:log_M_0_0" in gp.get_parameter_names() :
        priortypes["decaytime"] = 'Uniform'
        values["decaytime"] = decaytime_lim
    else :
        priortypes["decaytime"] = 'FIXED'
        values["decaytime"] = (params["decaytime"],params["decaytime"])

    if "kernel:k2:gamma" in gp.get_parameter_names() :
        priortypes["smoothfactor"] = 'Uniform'
        values["smoothfactor"] = smoothfactor_lim
    else:
        priortypes["smoothfactor"] = 'FIXED'
        values["smoothfactor"] = (params["smoothfactor"],params["smoothfactor"])

    if "kernel:k2:log_period" in gp.get_parameter_names() :
        priortypes["period"] = 'Uniform'
        values["period"] = period_lim
    else :
        priortypes["period"] = 'FIXED'
        values["period"] = (params["period"],params["period"])

    return params, priortypes, values


def update_params(params, labels, theta) :
    for i in range(len(labels)) :
        params[labels[i]] = theta[i]
    return params

# prior probability from definitions in priorslib
def lnprior(theta_priors, theta, labels):
    total_prior = 0.0
    for i in range(len(theta)) :
        #theta_priors[labels[i]]['object'].set_value(theta[i])
        if theta_priors[labels[i]]['type'] == "Uniform" or theta_priors[labels[i]]['type'] == "Jeffreys" or theta_priors[labels[i]]['type'] == "Normal_positive" :

            if not theta_priors[labels[i]]['object'].check_value(theta[i]):
                return -np.inf
            
            if labels[i] == 'white_noise' and theta[i] < 0:
                print("Why am I here?", theta[i], theta_priors[labels[i]]['object'].check_value(theta[i]))
        total_prior += theta_priors[labels[i]]['object'].get_ln_prior()

    return total_prior


def gprotlnprob(theta, gp, x, y, yerr, params, labels, theta_priors) :
            
    #lp = lnprior(theta)
    #lp = lnprior(theta_priors, theta, labels) + gp.log_prior()
    lp = lnprior(theta_priors, theta, labels)
    if not np.isfinite(lp):
        return -np.inf
    # Update the kernel gp parameters
    params = update_params(params, labels, theta)

    gp = set_star_rotation_gp_params(gp, params)

    #gp.compute(x, yerr)

    return gp.lnlikelihood(y, quiet=True) + lp


def star_rotation_gp(t, y, yerr,
                     run_optimization=True,
                     period=10., period_lim=(1,500), fix_period=False,
                     amplitude=1.0, aplitude_lim=(0,1e10), fix_amplitude=False,
                     decaytime=70, decaytime_lim=(10,300), fix_decaytime=False,
                     smoothfactor=0.5, smoothfactor_lim=(0.1,1.0), fix_smoothfactor=False,
                     fit_mean=False, fit_white_noise=False,
                     fix_mean=False, fix_white_noise=False,
                     period_label ="Prot [d]",
                     amplitude_label =r"$\alpha$",
                     decaytime_label =r"$l$",
                     smoothfactor_label =r"$\beta$",
                     mean_label =r"$\mu$",
                     white_noise_label =r"$\sigma$",
                     fixpars_before_fit=True,
                     x_label="time", y_label="y", output_pairsplot="",
                     run_mcmc=False, amp=1e-4, nwalkers=32, niter=500, burnin=100,
                     best_fit_from_mode = True, plot_distributions=False,
                     plot=False, verbose=False) :

    # define star rotation kernel:
    # Eq. 2 of Angus et al. 2017
    #k1 = kernels.ConstantKernel(log_constant=np.log(amplitude**2), bounds=dict(log_constant=(np.log(aplitude_lim[0]**2),np.log(aplitude_lim[1]**2))))
    k1 = kernels.ConstantKernel(log_constant=np.log(amplitude**2))
    
    #k2 = kernels.ExpSquaredKernel(metric=np.log(decaytime**2), metric_bounds=[(np.log(decaytime_lim[0]**2),np.log(decaytime_lim[1]**2))])
    k2 = kernels.ExpSquaredKernel(metric=decaytime**2, metric_bounds=[(np.log(decaytime_lim[0]**2),np.log(decaytime_lim[1]**2))])

    k3 = kernels.ExpSine2Kernel(gamma=(1/smoothfactor)**2, log_period=np.log(period), bounds=dict(gamma=((1/smoothfactor_lim[1])**2, (1/smoothfactor_lim[0])**2),log_period=(np.log(period_lim[0]), np.log(period_lim[1]))))
    #k3 = kernels.ExpSine2Kernel(gamma=(1/smoothfactor)**2, log_period=np.log(period))
    
    kernel = k1 * k2 * k3

    # Below are the labels for the hyperparameters in the gp
    """
    'mean:value'
    'white_noise:value'
    'kernel:k1:k1:log_constant'
    'kernel:k1:k2:metric:log_M_0_0'
    'kernel:k2:gamma'
    'kernel:k2:log_period'
    """
    
    if fixpars_before_fit :
        if fix_amplitude :
            kernel.freeze_parameter("k1:k1:log_constant")
        if fix_period :
            kernel.freeze_parameter("k2:log_period")
        if fix_decaytime :
            kernel.freeze_parameter("k1:k2:metric:log_M_0_0")
        if fix_smoothfactor :
            kernel.freeze_parameter("k2:gamma")

    initial_sigma = np.nanmedian(yerr)

    gp = george.GP(kernel, mean=np.mean(y), fit_mean=fit_mean, white_noise=np.log(initial_sigma**2), fit_white_noise=fit_white_noise)

    gp.compute(t, yerr)  # You always need to call compute once.
    if verbose :
        print("Initial log likelihood: {0}".format(gp.log_likelihood(y)))
        print("parameter_dict:\n{0}\n".format(gp.get_parameter_dict()))
    
    initial_params = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()

    if run_optimization :
        # run optimization:
        soln = minimize(gnll, initial_params, method="L-BFGS-B", jac=ggrad_nll, bounds=bounds, args=(y, gp))

        # print the value of the optimised parameters:
        if verbose :
            print("Fit parameters: {0}".format(soln.x))
            print("Final log likelihood: {0}".format(gp.log_likelihood(y)))
    
        # pass the parameters to the kernel:
        gp.set_parameter_vector(soln.x)

        params = get_star_rotation_gp_params(gp)

        if verbose :
            print("Fit parameters:")
            for key in params :
                print(key,"=",params[key])

    if plot :
        x = np.linspace(t[0], t[-1], 10000)
        pred_mean, pred_var = gp.predict(y, x, return_var=True)
        pred_std = np.sqrt(pred_var)

        color = "#ff7f0e"
        plt.errorbar(t, y, yerr=yerr, fmt=".k", alpha=0.5, capsize=0)
        plt.plot(x, pred_mean, color=color)
        plt.fill_between(x, pred_mean+pred_std, pred_mean-pred_std, color=color, alpha=0.3,
                 edgecolor="none")
        plt.xlabel(x_label, fontsize=15)
        plt.ylabel(y_label, fontsize=15)
        plt.show()

    if fix_mean :
        gp.freeze_parameter("mean:value")
    if fix_white_noise :
        gp.freeze_parameter("white_noise:value")
    if fix_amplitude :
        gp.freeze_parameter("kernel:k1:k1:log_constant")
    if fix_period :
        gp.freeze_parameter("kernel:k2:log_period")
    if fix_decaytime :
        gp.freeze_parameter("kernel:k1:k2:metric:log_M_0_0")
    if fix_smoothfactor :
        gp.freeze_parameter("kernel:k2:gamma")

    if run_mcmc :

        params, priortypes, priorvalues = set_star_rotation_priors(gp, period_lim=period_lim, aplitude_lim=aplitude_lim, decaytime_lim=decaytime_lim, smoothfactor_lim=smoothfactor_lim)

        labels, theta, theta_priors = set_theta_priors(params, priortypes, priorvalues)

        #gp = star_rotation_gp_freeze_all_params(gp)
        
        # Set up the sampler.
        ndim = len(theta)
        # Make sure the number of walkers is sufficient, and if not passing a new value
        if nwalkers < 2*ndim:
            nwalkers = 2*ndim
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, gprotlnprob, args = [gp, t, y, yerr, params, labels, theta_priors])

        # Initialize the walkers.
        pos = [theta + amp * np.random.randn(ndim) for i in range(nwalkers)]

        if verbose :
            print("Running MCMC ... ")
        sampler.run_mcmc(pos, niter, progress=True)
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim)) # burnin : number of first samples to be discard as burn-in
        
        func = lambda v: (v[1], v[2]-v[1], v[1]-v[0])
        percents = np.percentile(samples, [16, 50, 84], axis=0)
        seq = list(zip(*percents))
        values = list(map(func, seq))

        if best_fit_from_mode :
            max_values = []
            nbins = 30
            
            for i in range(len(labels)) :
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
                    plt.xlabel(r"{}".format(labels[i]),fontsize=18)
                    plt.legend()
                    plt.show()

                    plt.plot(samples[:,i],label="{}".format(labels[i]), alpha=0.5, lw=0.5)
                    plt.hlines([], np.min(0), np.max(nmax), ls=":", label="mode",zorder=2)
                    plt.hlines([values[i][0]], np.min(0), np.max(nmax), ls="-", label="median",zorder=2)
                    plt.ylabel(r"{}".format(labels[i]),fontsize=18)
                    plt.xlabel(r"MCMC iteration",fontsize=18)
                    plt.legend(fontsize=18)
                    plt.show()
                    
            max_values = np.array(max_values)
            
        mean_params, max_params, min_params = [], [], []
        for i in range(len(labels)) :
            if best_fit_from_mode :
                mean_params.append(max_values[i])
            else :
                mean_params.append(values[i][0])
            max_params.append(values[i][1])
            min_params.append(values[i][2])
        min_params, max_params = np.array(min_params), np.array(max_params)
        mean_params = np.array(mean_params)
        err_params = (min_params + max_params)/2

        # Update the kernel gp parameters
        params = update_params(params, labels, mean_params)
        gp = set_star_rotation_gp_params(gp, params)
        gp.compute(t, yerr)

        if verbose :
            for i in range(len(labels)) :
                mean = mean_params[i]
                min = min_params[i]
                max = max_params[i]
                print("{} = {:.8f} + {:.8f} - {:.8f}".format(labels[i], mean, max, min))

        if plot :
            cornerlabels, corner_ranges = [], []
            for i in range(len(labels)) :
                if labels[i] == 'period' :
                    cornerlabels.append(period_label)
                    corner_ranges.append((5,15))
                if labels[i] == 'amplitude' :
                    cornerlabels.append(amplitude_label)
                    corner_ranges.append((mean_params[i]-3.5*err_params[i],mean_params[i]+3.5*err_params[i]))
                if labels[i] == 'decaytime' :
                    cornerlabels.append(decaytime_label)
                    corner_ranges.append((mean_params[i]-3.5*err_params[i],mean_params[i]+3.5*err_params[i]))
                if labels[i] == 'smoothfactor' :
                    cornerlabels.append(smoothfactor_label)
                    corner_ranges.append((mean_params[i]-3.5*err_params[i],mean_params[i]+3.5*err_params[i]))
                if labels[i] == 'mean' :
                    cornerlabels.append(mean_label)
                    corner_ranges.append((mean_params[i]-3.5*err_params[i],mean_params[i]+3.5*err_params[i]))
                if labels[i] == 'white_noise' :
                    cornerlabels.append(white_noise_label)
                    corner_ranges.append((mean_params[i]-3.5*err_params[i],mean_params[i]+3.5*err_params[i]))

            fig = corner.corner(sampler.flatchain, truths=mean_params, labels=cornerlabels, quantiles=[0.16, 0.5, 0.84],tick_size=14,labelsize=30,tick_rotate=30,label_kwargs=dict(fontsize=18))
            #marginals.corner(sampler.flatchain, truths=mean_params, labels=cornerlabels, quantiles=[0.16, 0.5, 0.84], tick_size=14, label_size=18, max_n_ticks=4, colormain='tab:blue', truth_color=(1,102/255,102/255),colorhist='tab:blue', colorbackgd=(240/255,240/255,240/255),tick_rotate=30)


            for ax in fig.get_axes():
                #ax.tick_params(axis='both', which='major', labelsize=14)
                #ax.tick_params(axis='both', which='minor', labelsize=12)
                ax.tick_params(axis='both', labelsize=14)

            if output_pairsplot != "":
                plt.savefig(output_pairsplot,bbox_inches='tight')
            plt.show()
            
            x = np.linspace(t[0], t[-1], 5000)

            plt.errorbar(t, y, yerr=yerr, fmt=".k", alpha=0.5, capsize=0)
            pred_mean, pred_var = gp.predict(y, x, return_var=True)
            pred_std = np.sqrt(pred_var)
            color = "#ff7f0e"
            plt.plot(x, pred_mean, color=color, lw=2)
            plt.fill_between(x, pred_mean+pred_std, pred_mean-pred_std, color=color, alpha=0.3,edgecolor="none")

            """
                gp_copy = deepcopy(gp)
            for i in range(50):
                # Choose a random walker and step.
                w = np.random.randint(sampler.chain.shape[0])
                n = np.random.randint(sampler.chain.shape[1])
                
                # Update the kernel gp parameters
                params = update_params(params, labels, sampler.chain[w, n])
                gp_copy = set_star_rotation_gp_params(gp_copy, params)
                gp_copy.compute(t, yerr)
                pred_sample_mean = gp_copy.predict(y, x, return_cov=False, return_var=False)
                plt.plot(x, pred_sample_mean, "g", alpha=0.1)
            """
            plt.xlabel(x_label, fontsize=15)
            plt.ylabel(y_label, fontsize=15)
            plt.show()

    return gp


def star_rotation_gp_freeze_all_params(gp) :
    gp.freeze_parameter('mean:value')
    gp.freeze_parameter('white_noise:value')
    gp.freeze_parameter('kernel:k1:k1:log_constant')
    gp.freeze_parameter('kernel:k1:k2:metric:log_M_0_0')
    gp.freeze_parameter('kernel:k2:gamma')
    gp.freeze_parameter('kernel:k2:log_period')
    return gp

def star_rotation_gp_thaw_all_params(gp) :
    gp.thaw_parameter('mean:value')
    gp.thaw_parameter('white_noise:value')
    gp.thaw_parameter('kernel:k1:k1:log_constant')
    gp.thaw_parameter('kernel:k1:k2:metric:log_M_0_0')
    gp.thaw_parameter('kernel:k2:gamma')
    gp.thaw_parameter('kernel:k2:log_period')
    return gp

def get_star_rotation_gp_params(gp) :

    mean = gp.get_parameter('mean:value')
    white_noise = np.sqrt(np.exp(gp.get_parameter('white_noise:value')))
    amplitude = np.sqrt(np.exp(gp.get_parameter('kernel:k1:k1:log_constant')))
    decaytime = np.sqrt(np.exp(gp.get_parameter('kernel:k1:k2:metric:log_M_0_0')))
    smoothfactor = np.sqrt(1/gp.get_parameter('kernel:k2:gamma'))
    period = np.exp(gp.get_parameter('kernel:k2:log_period'))

    params = {}

    params["mean"] = mean
    params["white_noise"] = white_noise
    params["amplitude"] = amplitude
    params["decaytime"] = decaytime
    params["smoothfactor"] = smoothfactor
    params["period"] = period

    return params


def set_star_rotation_gp_params(gp, params) :
    
    gp.set_parameter('mean:value',params['mean'])
    gp.set_parameter('white_noise:value', np.log(params['white_noise']**2))
    gp.set_parameter('kernel:k1:k1:log_constant',np.log(params['amplitude']**2))
    gp.set_parameter('kernel:k1:k2:metric:log_M_0_0',np.log(params['decaytime']**2))
    gp.set_parameter('kernel:k2:gamma',1/(params['smoothfactor']**2))
    gp.set_parameter('kernel:k2:log_period',np.log(params['period']))
    
    return gp

