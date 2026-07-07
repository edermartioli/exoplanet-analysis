"""
    Created on Jun 7 2023
    
    Description: This routine performs an analysis of the time series of RVs and activity indices
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    python time_series_analysis.py --input=/Volumes/Samsung_T5/Data/SOPHIE/TOI-1736/analysis/e2ds/TOI-1736_sophie_results.txt --planet_posterior=/Volumes/Samsung_T5/Science/TOI-1736/RV+TRANSITS_ANALYSIS/TOI-1736_posterior.pars -pv
    
    python time_series_analysis.py --input=/Volumes/Samsung_T5/Data/SOPHIE/TOI-2141/analysis/e2ds/TOI-2141_sophie_results.txt --planet_posterior=/Volumes/Samsung_T5/Science/TOI-2141/RV+TRANSITS_ANALYSIS/TOI-2141_posterior.pars
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os, sys

from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt

from exoplanet_analysis import timeseries_lib as tslib
from exoplanet_analysis import rvutils, fitlib

from astropy.io import ascii


def main() :

    """Main.
    """
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", help='Input data file',type='string',default="")
    parser.add_option("-l", "--planet_posterior", dest="planet_posterior", help='Planet posterior parameters file',type='string',default="")
    parser.add_option("-p", action="store_true", dest="plot", help="plot",default=False)
    parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

    try:
        options,args = parser.parse_args(sys.argv[1:])
    except SystemExit as e :
        # allow clean exits from optparse (e.g. --help)
        if e.code == 0 or e.code is None :
            raise
        print("Error: check usage with time_series_analysis.py -h "); sys.exit(1);

    if options.verbose:
        print('Input data file: ', options.input)
        print('Planet posterior parameters file: ', options.planet_posterior)

    ##########################################
    ### LOAD input data
    ##########################################
    if options.verbose:
        print("Loading time series data ...")

    tbl = ascii.read(options.input)

    bjd = np.array(tbl['rjd']) + 2400000
    vrad, svrad = np.array(tbl['vrad'])*1000, np.array(tbl['svrad'])*1000
    biss = np.array(tbl['bis_span'])*1000
    #fwhm = np.array(tbl['fwhm'])
    sindex = np.array(tbl['rhk'])
    halpha = np.array(tbl['ha'])

    ns = len(bjd)
    n_rvdatasets = 1
    dataset_index = 0
    rvdatalabels = ["SOPHIE data"]

    # read priors from input file
    priors = fitlib.read_rv_priors(options.planet_posterior, n_rvdatasets, verbose=options.verbose)

    # Fit RV calibration parameters for initial guess
    posterior = fitlib.guess_rvcalib(priors, [bjd], [vrad], prior_type="FIXED", plot=False)
    rvcalib_params = posterior["rvcalib_params"]

    # Fit jitter
    rverrs, jitter, jitter_err = rvutils.fit_RV_jitter(posterior, [bjd], [vrad], [svrad], rvdatalabels=rvdatalabels)
    # update rv errors with fitted jitter
    svrad = rverrs[0]

    # OLS fit involving all priors
    posterior = fitlib.fit_RVs_ols([bjd], [vrad], [svrad], posterior, fix_eccentricity=False, rvcalib_post_type="Normal", calib_unc=0.01, verbose=False)
    if options.plot :
        fitlib.plot_rv_global_timeseries(posterior["planet_params"], posterior["rvcalib_params"], [bjd], [vrad], [svrad], samples=None, labels=None, nsamples=100, plot_residuals=True, rvdatalabels=rvdatalabels)
        n_planets = int(posterior["planet_params"]["n_planets"])
        for planet_index in range(n_planets) :
            fitlib.plot_rv_perplanet_timeseries(posterior["planet_params"], posterior["rvcalib_params"], [bjd], [vrad], [svrad], planet_index=planet_index, samples=None, labels=None, nsamples=100, plot_residuals=True, rvdatalabels=rvdatalabels, phase_plot=True)

    # calculate rv residuals
    rvcalib = rvcalib_params['rv_d{0:02d}'.format(dataset_index)]
    rvmodel, model_rvsys, otherrvms = fitlib.calculate_rv_model_per_planet(bjd, posterior["planet_params"], planet_index=0)
    obs_model = rvmodel + model_rvsys + rvcalib + otherrvms
    vrad_res =  vrad - obs_model

    ## START PLOTTING TIME SERIES
    fig, axs = plt.subplots(5, 1, figsize=(12, 6), sharex=True, sharey=False, gridspec_kw={'hspace': 0})

    nsig = 4
    ###########
    ### RVs ###
    ###########
    axs[0].set_ylabel(r"$\Delta$RV [m/s]", fontsize=16)
    axs[0].errorbar(bjd, vrad_res, yerr=svrad, fmt='o', ecolor='black', color='k', capthick=2, elinewidth=2, mec='k')
    m, sig = np.nanmedian(vrad_res), np.nanstd(vrad_res)
    axs[0].set_ylim(m-nsig*sig,m+nsig*sig)
    ###########
    ### FWHM ###
    ###########
    axs[1].set_ylabel(r"FWHM [km/s]", fontsize=16)
    axs[1].errorbar(bjd, fwhm, yerr=fwhmerr, fmt='o', ecolor='darkblue', color='darkblue', capthick=2, elinewidth=2, mec='k')
    m, sig = np.nanmedian(fwhm), np.nanstd(fwhm)
    axs[1].set_ylim(m-nsig*sig,m+nsig*sig)
    ###########
    ### BIS ###
    ###########
    axs[2].set_ylabel(r"BIS [m/s]", fontsize=16)
    axs[2].errorbar(bjd, biss, yerr=bisserr, fmt='o', ecolor='darkgreen', color='darkgreen', capthick=2, elinewidth=2, mec='k')
    m, sig = np.nanmedian(biss), np.nanstd(biss)
    axs[2].set_ylim(m-nsig*sig,m+nsig*sig)
    ###########
    ### s-index ###
    ###########
    axs[3].set_ylabel(r"S$_{\rm MW}$", fontsize=16)
    axs[3].errorbar(bjd, sindex, yerr=sindexerr, fmt='o', ecolor='darkred', color='darkred', capthick=2, elinewidth=2, mec='k')
    m, sig = np.nanmedian(sindex), np.nanstd(sindex)
    axs[3].set_ylim(m-nsig*sig,m+nsig*sig)
    ###########
    ### h-alpha ###
    ###########
    axs[4].set_ylabel(r"H$_{\alpha}$", fontsize=16)
    axs[4].errorbar(bjd, halpha, yerr=halphaerr, fmt='o', ecolor='orange', color='orange', capthick=2, elinewidth=2, mec='k')
    m, sig = np.nanmedian(halpha), np.nanstd(halpha)
    axs[4].set_ylim(m-nsig*sig,m+nsig*sig)

    axs[4].set_xlabel("Time [BJD]", fontsize=16)

    for i in range(5) :
        axs[i].tick_params(axis='x', labelsize=14)
        axs[i].tick_params(axis='y', labelsize=14)
        axs[i].minorticks_on()
        axs[i].tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
        axs[i].tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)

    plt.show()

    plow, phigh, ofac = 0.5, 700, 1
    probability = 0.001

    ## START PLOTTING PERIODOGRAMS
    #per, powr = tslib.bgls(bjd, vrad_res, svrad, plow=plow, phigh=phigh, ofac=ofac)

    fig, axs = plt.subplots(5, 1, figsize=(12, 6), sharex=True, sharey=False, gridspec_kw={'hspace': 0})

    ###########
    ### RVs ###
    ###########
    gls = tslib.periodogram(bjd, vrad_res, svrad, nyquist_factor=20, probabilities = [probability], y_label="y", check_period=0, npeaks=1, plot=False)
    per, powr, fap, fappwrs = 1/gls['frequency'], gls['power'], gls['probabilities'], gls['fap']

    axs[0].plot(per, powr, '-', color='k', zorder=1, label=r"$\Delta$RV")
    #axs[0].text(np.min(per),fappwrs[0]+0.005,r"FAP={0:.3f}%".format(100*fap[0]),horizontalalignment='left', fontsize=12)
    axs[0].hlines([fappwrs[0]], np.min(per), np.max(per),ls=":", lw=0.5)

    ###########
    ### FWHM ###
    ###########
    gls = tslib.periodogram(bjd, fwhm, fwhmerr, nyquist_factor=20, probabilities = [probability], y_label="y", check_period=0, npeaks=1, plot=False)
    per, powr, fap, fappwrs = 1/gls['frequency'], gls['power'], gls['probabilities'], gls['fap']

    axs[1].plot(per, powr, '-', color='darkblue', zorder=1, label=r"FWHM")
    #axs[1].text(np.min(per),fappwrs[0]+0.005,r"FAP={0:.3f}%".format(100*fap[0]),horizontalalignment='left', fontsize=12)
    axs[1].hlines([fappwrs[0]], np.min(per), np.max(per),ls=":", lw=0.5)

    ###########
    ### BIS ###
    ###########
    gls = tslib.periodogram(bjd, biss, bisserr, nyquist_factor=20, probabilities = [probability], y_label="y", check_period=0, npeaks=1, plot=False)
    per, powr, fap, fappwrs = 1/gls['frequency'], gls['power'], gls['probabilities'], gls['fap']

    axs[2].plot(per, powr, '-', color='darkgreen', zorder=1, label=r"BIS")
    #axs[2].text(np.min(per),fappwrs[0]+0.005,r"FAP={0:.3f}%".format(100*fap[0]),horizontalalignment='left', fontsize=12)
    axs[2].hlines([fappwrs[0]], np.min(per), np.max(per),ls=":", lw=0.5)

    ###########
    ### S-index ###
    ###########
    gls = tslib.periodogram(bjd, sindex, sindexerr, nyquist_factor=20, probabilities = [probability], y_label="y", check_period=0, npeaks=1, plot=False)
    per, powr, fap, fappwrs = 1/gls['frequency'], gls['power'], gls['probabilities'], gls['fap']

    axs[3].plot(per, powr, '-', color='darkred', zorder=1, label=r"S$_{\rm MW}$")
    #axs[3].text(np.min(per),fappwrs[0]+0.005,r"FAP={0:.3f}%".format(100*fap[0]),horizontalalignment='left', fontsize=12)
    axs[3].hlines([fappwrs[0]], np.min(per), np.max(per),ls=":", lw=0.5)

    ###########
    ### H-alpha ###
    ###########
    gls = tslib.periodogram(bjd, halpha, halphaerr, nyquist_factor=20, probabilities = [probability], y_label="y", check_period=0, npeaks=1, plot=False)
    per, powr, fap, fappwrs = 1/gls['frequency'], gls['power'], gls['probabilities'], gls['fap']

    axs[4].plot(per, powr, '-', color='orange', zorder=1, label=r"H$_{\alpha}$")
    #axs[4].text(np.min(per),fappwrs[0]+0.005,r"FAP={0:.3f}%".format(100*fap[0]),horizontalalignment='left', fontsize=12)
    axs[4].hlines([fappwrs[0]], np.min(per), np.max(per),ls=":", lw=0.5)

    axs[4].set_xlabel("Period [d]", fontsize=18)

    for i in range(5) :
        axs[i].set_ylabel("Power", fontsize=14)
        axs[i].set_xscale('log')
        axs[i].set_xlim(plow, phigh)
        axs[i].tick_params(axis='x', labelsize=14)
        axs[i].tick_params(axis='y', labelsize=14)
        axs[i].minorticks_on()
        axs[i].tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
        axs[i].tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
        axs[i].legend(fontsize=16, loc='upper right')
    
    plt.show()


if __name__ == "__main__" :
    main()
