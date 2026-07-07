"""
    Created on Nov 4 2022
    
    Description: This routine performs a Gaussian Process analysis of the data, remove the GP component and save residuals
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    gp_analysis --input=/Volumes/Samsung_T5/Science/TOI-1736/TOI-1736_sophie_000.rdb --gp_priors=/Volumes/Samsung_T5/Science/TOI-1736/GPAnalysis/gp_rv_priors.pars --nsteps=10000 --burnin=3000 -pvm --output=/Volumes/Samsung_T5/Science/TOI-1736/GPAnalysis/TOI-1736_sophie_gpremov.rdb
    
    gp_analysis --input=/Volumes/Samsung_T5/Science/TOI-2141/TOI-2141_sophie_ccf.rdb --gp_priors=/Volumes/Samsung_T5/Science/TOI-2141/GPAnalysis/gp_rv_priors.pars --nsteps=10000 --burnin=3000 -pvm --output=/Volumes/Samsung_T5/Science/TOI-2141/GPAnalysis/TOI-2141_sophie_gpremov.rdb
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os, sys

from optparse import OptionParser

import numpy as np
import scipy
import matplotlib.pyplot as plt

from exoplanet_analysis import timeseries_lib as tslib
from exoplanet_analysis import rvutils, priorslib, gp_lib, fitlib

from astropy.io import ascii
from PyAstronomy.pyasl import foldAt


def run_gp_analysis(time, y, yerr, gp_priors_file="", run_gls=False,  run_mcmc=False, walkers=32, nsteps=500, burnin=100, amp=1e-4, ylabel=r"y", best_fit_from_mode=True, plot_distributions=False, verbose=False, plot=False) :

    """Run gp analysis.

    Parameters
    ----------
    time
        Array of times [BJD].
    y
        Array of y values.
    yerr
        Array of y uncertainties.
    gp_priors_file : str, optional (default: "")
    run_gls : bool, optional (default: False)
    run_mcmc : bool, optional (default: False)
    walkers : int, optional (default: 32)
    nsteps : int, optional (default: 500)
    burnin : int, optional (default: 100)
        Number of MCMC burn-in steps discarded from the chain.
    amp : float, optional (default: 1e-4)
        Amplitude of the initial ball of walkers around the starting point.
    ylabel : str, optional (default: r"y")
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
    if run_gls :
        gls = tslib.periodogram(time, y, yerr, nyquist_factor=0.02, probabilities = [0.00001], npeaks=1, y_label=ylabel, plot=plot, phaseplot=plot)
    
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
    gp = gp_lib.star_rotation_gp(time, y, yerr, period=best_period, period_lim=period_lim, fix_period=fix_period, amplitude=amplitude, amplitude_lim=amplitude_lim, fix_amplitude=fix_amplitude, decaytime=decaytime, decaytime_lim=decaytime_lim, fix_decaytime=fix_decaytime, smoothfactor=smoothfactor, smoothfactor_lim=smoothfactor_lim, fix_smoothfactor=fix_smoothfactor, fixpars_before_fit=True, fit_mean=fit_mean, fit_white_noise=fit_white_noise, period_label=r"Prot [d]", amplitude_label=r"$\alpha$", decaytime_label=r"$l$ [d]", smoothfactor_label=r"$\beta$", mean_label=r"$\mu$", white_noise_label=r"$\sigma$", output_pairsplot=output_pairsplot, run_mcmc=run_mcmc, amp=amp, nwalkers=walkers, niter=nsteps, burnin=burnin, x_label="BJD", y_label=ylabel, output=gp_posterior, best_fit_from_mode = best_fit_from_mode, plot_distributions = plot_distributions, plot=plot, verbose=verbose)

    gp_params = gp_lib.get_star_rotation_gp_params(gp)
    best_period = gp_params["period"]

    gp_feed = {}
    gp_feed["t"], gp_feed["y"], gp_feed["yerr"] = time, y, yerr

    return gp, gp_feed


def save_time_series(output, time, y, yerr) :
    
    """Save time series.

    Parameters
    ----------
    output
        Output file path.
    time
        Array of times [BJD].
    y
        Array of y values.
    yerr
        Array of y uncertainties.
    """
    outfile = open(output,"w+")
    #outfile.write("btjd\ty\tyerr\n")
    #outfile.write("---\t----\t-----\n")
    
    for i in range(len(time)) :
        outfile.write("{0:.10f}\t{1:.5f}\t{2:.5f}\n".format(time[i], y[i], yerr[i]))

    outfile.close()


def main() :

    """Main.
    """
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", help='Input data file',type='string',default="")
    parser.add_option("-g", "--gp_priors", dest="gp_priors", help='GP priors file',type='string',default="")
    parser.add_option("-y", "--ylabel", dest="ylabel", help='Y variable label',type='string',default="RV [m/s]")
    parser.add_option("-o", "--output", dest="output", help='Output reduced data',type='string',default="")
    parser.add_option("-n", "--nsteps", dest="nsteps", help="Number of MCMC steps",type='int',default=500)
    parser.add_option("-w", "--walkers", dest="walkers", help="Number of MCMC walkers",type='int',default=50)
    parser.add_option("-b", "--burnin", dest="burnin", help="Number of MCMC burn-in samples",type='int',default=100)
    parser.add_option("-r", action="store_true", dest="periodogram", help="Run GLS periodogram",default=False)
    parser.add_option("-m", action="store_true", dest="run_mcmc", help="Run MCMC",default=False)
    parser.add_option("-p", action="store_true", dest="plot", help="plot",default=False)
    parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

    try:
        options,args = parser.parse_args(sys.argv[1:])
    except SystemExit as e :
        # allow clean exits from optparse (e.g. --help)
        if e.code == 0 or e.code is None :
            raise
        print("Error: check usage with gp_analysis -h "); sys.exit(1);

    if options.verbose:
        print('Input data file: ', options.input)
        print('GP priors file: ', options.gp_priors)
        print('Y variable label: ', options.ylabel)
        print('Output reduced data: ', options.output)
        print('Number of MCMC steps: ', options.nsteps)
        print('Number of MCMC walkers: ', options.walkers)
        print('Number of MCMC burn-in samples: ', options.burnin)

    ##########################################
    ### LOAD input data
    ##########################################
    if options.verbose:
        print("Loading time series data ...")
        
    if options.input.endswith(".rdb") :
        try :
            time, y, yerr = rvutils.read_rv_time_series(options.input)
        except :
            timeseriesdata = ascii.read(options.input)
            time, y, yerr = timeseriesdata["col1"], timeseriesdata["col2"], timeseriesdata["col3"]
    else :
        timeseriesdata = ascii.read(options.input)
        time, y, yerr = timeseriesdata["col1"], timeseriesdata["col2"], timeseriesdata["col3"]

    my, medy, myerr = np.mean(y), np.median(y), np.median(yerr)
    sig, mad = np.std(y), scipy.stats.median_abs_deviation(y, scale=0.67449)

    if options.verbose:
        print("Statistics for the input data set:")
        print("mean = {:.3f}  median = {:.3f}  median_err = {:.3f} stddev = {:.3f}  mad = {:.3f}".format(my, medy, myerr, sig, mad))

    #if options.periodogram :
    #    glsperiodogram = tslib.periodogram(time, y, yerr, nyquist_factor=20, probabilities = [0.01, 0.001], y_label=options.ylabel, npeaks=1, phaseplot=options.plot, plot=options.plot, plot_frequencies=False)

    if options.verbose:
        print("Calculating GP...")

    options.ylabel = r"B$_\ell [G]$"

    gp, gp_feed = run_gp_analysis(time, y, yerr, gp_priors_file=options.gp_priors, run_gls=options.periodogram,  run_mcmc=options.run_mcmc, walkers=options.walkers, nsteps=options.nsteps, burnin=options.burnin, amp=1e-4, ylabel=options.ylabel, best_fit_from_mode=True, plot_distributions=False, verbose=options.verbose, plot=options.plot)

    if options.plot:
        gp_params = gp_lib.get_star_rotation_gp_params(gp)
        best_period = gp_params["period"]
        phases = foldAt(time, best_period, T0=time[0])
        sortIndi = np.argsort(phases)
        plt.errorbar(phases[sortIndi],y[sortIndi],yerr=yerr[sortIndi],fmt='o', color="k")
        plt.ylabel(r"{}".format(options.ylabel), fontsize=16)
        plt.xlabel("phase (P={0:.3f} d)".format(best_period), fontsize=16)
        plt.show()
  
    reduced_y, reduced_yerr = fitlib.reduce_gp(gp, time, y, yerr, gp_feed, subtract=True)

    #if options.periodogram :
    #    glsperiodogram = tslib.periodogram(time, reduced_y, reduced_yerr, nyquist_factor=20, probabilities = [0.01, 0.001], y_label=options.ylabel, npeaks=1, phaseplot=options.plot, plot=options.plot, plot_frequencies=False)

    if options.output != "" :
        save_time_series(options.output, time-2400000., reduced_y/1000, reduced_yerr/1000)


if __name__ == "__main__" :
    main()
