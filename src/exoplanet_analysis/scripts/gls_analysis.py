"""
    Created on Oct 13 2022
    
    Description: This routine performs a Generalized Lomb-Scargle periodogram analysis of the data
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    gls_analysis --input=/Volumes/Samsung_T5/Science/TOI-1736/TOI-1736_sophie_rv_minus_planetc.rdb --period=7.0731267 --min_period=3. --max_period=12. --niter=100 --min_nobs=30 --ofac=1000 -pv
        
    gls_analysis --input=/Volumes/Samsung_T5/Science/TOI-2141/TOI-2141_sophie_clean_rv.rdb --period=18.258946986954 --min_period=5. --max_period=50. --niter=20 --min_nobs=50 --ofac=1000 -pv
    

    gls_analysis --input=/Volumes/Samsung_T5/Science/TOI-2141/TOI-2141_sophie_clean.rdb --period=18.258946986954 --min_period=5. --max_period=50. --niter=20 --min_nobs=50 --ofac=1000 -pv


    gls_analysis --input=/Volumes/Samsung_T5/Data/SOPHIE/TOI-1736/analysis/e2ds/TOI-1736_sophie_results.rdb --period=31 --min_period=16 --max_period=46 --niter=10 --min_nobs=50 --ofac=500 --ylabel="BIS [m/s]" --timelabel="rjd" --varlabel="bis_span" --varerrlabel="sig_bis_span" -pv
    gls_analysis --input=/Volumes/Samsung_T5/Data/SOPHIE/TOI-1736/analysis/e2ds/TOI-1736_sophie_results.rdb --period=31 --min_period=16 --max_period=46 --niter=10 --min_nobs=50 --ofac=500 --ylabel="H-alpha" --timelabel="rjd" --varlabel="s_mw" --varerrlabel="sig_s" -pv

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
from exoplanet_analysis import rvutils

from astropy.io import ascii


def main() :

    """Main.
    """
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", help='Input data file',type='string',default="")
    parser.add_option("-y", "--ylabel", dest="ylabel", help='Y variable label',type='string',default="RV [m/s]")
    parser.add_option("-t", "--timelabel", dest="timelabel", help='Table time label',type='string',default="bjds")
    parser.add_option("-l", "--varlabel", dest="varlabel", help='Table variable label',type='string',default="rvs")
    parser.add_option("-e", "--varerrlabel", dest="varerrlabel", help='Table error variable label',type='string',default="rverrs")
    parser.add_option("-o", "--period", dest="period", help='Period of expected signal',type='float',default=0.)
    parser.add_option("-1", "--min_period", dest="min_period", help='Minimum period',type='float',default=0)
    parser.add_option("-2", "--max_period", dest="max_period", help='Maximum period',type='float',default=0)
    parser.add_option("-n", "--niter", dest="niter", help='Number of iterations',type='int',default=1)
    parser.add_option("-m", "--min_nobs", dest="min_nobs", help='Minimum number of observations to calculate SBGLSP',type='int',default=10)
    parser.add_option("-f", "--ofac", dest="ofac", help='OFAC',type='int',default=10)
    parser.add_option("-p", action="store_true", dest="plot", help="plot",default=False)
    parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

    try:
        options,args = parser.parse_args(sys.argv[1:])
    except SystemExit as e :
        # allow clean exits from optparse (e.g. --help)
        if e.code == 0 or e.code is None :
            raise
        print("Error: check usage with gls_analysis -h "); sys.exit(1);

    if options.verbose:
        print('Input data file: ', options.input)
        print('Y variable label: ', options.ylabel)
        print('Period of expected signal: ', options.period)
        print('Minimum period: ', options.min_period)
        print('Maximum period: ', options.max_period)
        print('Number of iterations: ', options.niter)
        print('Minimum number of observations: ', options.min_nobs)
        print('OFAC: ', options.ofac)

    ##########################################
    ### LOAD input data
    ##########################################
    if options.verbose:
        print("Loading time series data ...")
    
    # load rdb data file
    data = ascii.read(options.input)
    time, y, yerr = data[options.timelabel], data[options.varlabel], data[options.varerrlabel]

    #glsperiodogram = tslib.periodogram(time, y, yerr, nyquist_factor=20, probabilities = [0.01, 0.001], y_label=options.ylabel, check_period=options.period, npeaks=1, phaseplot=options.plot, plot=options.plot, plot_frequencies=False)

    glsperiodogram = tslib.periodogram(time, y, yerr, nyquist_factor=20, probabilities = [0.01, 0.001], y_label=options.ylabel, check_period=0, npeaks=1, phaseplot=options.plot, plot=options.plot, plot_frequencies=False)

    if options.verbose:
        print("Calculating SBGLSP...")

    tslib.sbglsp(time, y, yerr, niter=options.niter, min_nobs=options.min_nobs, period=options.period, min_period=options.min_period, max_period=options.max_period, ofac=options.ofac)


if __name__ == "__main__" :
    main()
