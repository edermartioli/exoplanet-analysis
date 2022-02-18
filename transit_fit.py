"""
    Created on Feb 14 2022
    
    Description: This routine fits planetary transits data using MCMC
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    Paper: python transit_fit.py --object="HATS-24" -vp
    
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

ExoplanetAnalysis_dir = os.path.dirname(__file__)
priors_dir = os.path.join(ExoplanetAnalysis_dir, 'priors/')


parser = OptionParser()
parser.add_option("-o", "--object", dest="object", help='Object ID',type='string',default="")
parser.add_option("-t", "--sector", dest="sector", help='Select TESS sector',type='int',default=0)
parser.add_option("-c", "--calib_order", dest="calib_order", help='Order of calibration polynomial',type='string',default="1")
parser.add_option("-n", "--nsteps", dest="nsteps", help="Number of MCMC steps",type='int',default=1000)
parser.add_option("-w", "--walkers", dest="walkers", help="Number of MCMC walkers",type='int',default=32)
parser.add_option("-b", "--burnin", dest="burnin", help="Number of MCMC burn-in samples",type='int',default=300)
parser.add_option("-s", "--samples_filename", dest="samples_filename", help='MCMC samples filename',type='string',default="")
parser.add_option("-e", action="store_true", dest="mode", help="Best fit parameters obtained by the mode instead of median", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="verbose",default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with transit_fit.py -h "); sys.exit(1);

if options.verbose:
    print('Object ID: ', options.object)
    if options.sector :
        print('TESS sector selected: ', options.sector)
    print('Order of calibration polynomial: ', options.calib_order)
    print('Number of MCMC steps: ', options.nsteps)
    print('Number of MCMC walkers: ', options.walkers)
    print('Number of MCMC burn-in samples: ', options.burnin)
    print('MCMC samples filenames: ', options.samples_filename)

# Download TESS DVT products and return a list of input data files
dvt_filenames = tess.retrieve_tess_data_files(options.object, sector=options.sector, products_wanted_keys = ["DVT"], verbose=options.verbose)

if options.verbose:
    print("Loading TESS lightcurves ...")
# Load TESS data
tesslc = tess.load_dvt_files(options.object, priors_dir=priors_dir, save_priors=True, plot=options.plot, verbose=options.verbose)


for planet in tesslc["PLANETS"] :

    # select data within certain ranges
    times, fluxes, fluxerrs = planet["times"], planet["fluxes"], planet["fluxerrs"]

    #Load priors information:
    planet_priors_files = sorted(glob.glob(planet["prior_file"]))

    calib_polyorder = int(options.calib_order)

    # read priors from input files
    priors = fitlib.read_priors(planet_priors_files, len(times), calib_polyorder=calib_polyorder, verbose=False)

    # Fit calibration parameters for initial guess
    priors = fitlib.guess_calib(priors, times, fluxes, prior_type="Normal", batman=True)

    if options.plot :
        # plot light curves and models in priors
        fitlib.plot_mosaic_of_lightcurves(times, fluxes, fluxerrs, priors)

    posterior = priors

    # OLS fit involving all priors
    posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, posterior, calib_post_type="Normal", calib_unc=0.01, batman=True, verbose=False, plot=False)
    # OLS fit involving all priors
    posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, posterior, calib_post_type="FIXED", batman=True, verbose=False, plot=False)

    if options.plot :
        fitlib.plot_mosaic_of_lightcurves(times, fluxes, fluxerrs, posterior)

    # Make sure the number of walkers is sufficient, and if not assing a new value
    if options.walkers < 2*len(posterior["theta"]):
        print("WARNING: insufficient number of MCMC walkers, resetting nwalkers={}".format(2*len(posterior["theta"])))
        options.walkers = 2*len(posterior["theta"])

    if options.samples_filename == "" :
        options.samples_filename = planet["prior_file"].replace(".pars","_mcmc_samples.h5")

    posterior = fitlib.fitTransitsWithMCMC(times, fluxes, fluxerrs, posterior, amp=1e-5, nwalkers=options.walkers, niter=options.nsteps, burnin=options.burnin, verbose=True, plot=True, samples_filename=options.samples_filename, appendsamples=False, plot_individual_transits=False)
