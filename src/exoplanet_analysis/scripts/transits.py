"""
    Created on Feb 14 2022
    
    Description: This routine fits planetary transits data using MCMC
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    transits_analysis --object="WASP-108" --tessdata=/Volumes/Samsung_T5/Science/WASP-108/TESS/*lc.fits --opddata=/Volumes/Samsung_T5/Science/WASP-108/IAG_photometry/WASP-108_reduced_lc.fits --planet_priors=/Volumes/Samsung_T5/Science/WASP-108/WASP-108.pars --nsteps=100 --walkers=32 --burnin=30  -vp

    transits_analysis --object="WASP-108" --tessdata=/Users/eder/Science/WASP-108/TESS/*lc.fits --opddata=/Users/eder/Science/WASP-108/IAG_photometry/WASP-108_reduced_lc.fits --planet_priors=/Users/eder/Science/WASP-108/WASP-108.pars --nsteps=1000 --walkers=32 --burnin=300  -vp
    
    
    transits_analysis --object="TOI-6235" --tessffidata=/Users/eder/Observations/Gemini2024B/TOI-6235_TESS_S57.txt --planet_priors=/Users/eder/Observations/Gemini2024B/TOI-6235.pars --nsteps=10000 --walkers=52 --burnin=3000 -vp

    transits_analysis --object="TOI-6342" --tessffidata=/Users/eder/Observations/Gemini2024B/TOI-6342_S58.txt --planet_priors=/Users/eder/Observations/Gemini2024B/TOI-6342.pars --nsteps=10000 --walkers=52 --burnin=3000 --calib_order=3 -vp

    transits_analysis --object="K2-65" --tessffidata=/Users/eder/Observations/Gemini2024B/K2-65_kepler.txt --planet_priors=/Users/eder/Observations/Gemini2024B/K2-65.pars --nsteps=1000 --walkers=52 --burnin=300 --calib_order=3 -vp
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
from exoplanet_analysis import fitlib, priorslib
import glob

from copy import deepcopy
from exoplanet_analysis import exoplanetlib, gp_lib
from exoplanet_analysis import timeseries_lib as ts
from exoplanet_analysis import tess
from exoplanet_analysis import rvutils

from exoplanet_analysis.config import priors_dir

from astropy.io import fits, ascii
from uncertainties import ufloat,umath

def get_opddata(tsfile, target=0, comps=[], extname="BEST_APERTURES") :

    """Get opddata.

    Parameters
    ----------
    tsfile
    target : int, optional (default: 0)
    comps : list, optional (default: [])
    extname : str, optional (default: "BEST_APERTURES")
    """
    times, fluxes, fluxerrs = [], [], []
    
    hdul = fits.open(tsfile)

    tbl = hdul[extname].data
    
    targettbl = tbl[tbl['SRCINDEX']==target]
    bjd = targettbl['TIME']
    
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
        
        for i in range(len(comptbl['MAG'])) :
            umag = ufloat(comptbl['MAG'][i],comptbl['EMAG'][i])
            uflux = 10 ** (- 0.4 * umag)
            fr = target_uflux[i] / uflux
            
            fratio = np.append(fratio,fr.nominal_value)
            fratioerr = np.append(fratioerr,fr.std_dev)
            
        mfratio = np.nanmedian(fratio)
        fluxes.append(fratio/mfratio)
        fluxerrs.append(fratioerr/mfratio)
        times.append(bjd)
        
    return times, fluxes, fluxerrs


def main() :

    """Main.
    """
    parser = OptionParser()
    parser.add_option("-i", "--tessdata", dest="tessdata", help='Pattern for input TESS light curve data',type='string',default="")
    parser.add_option("-f", "--tessffidata", dest="tessffidata", help='Pattern for input TESS light curve data',type='string',default="")
    parser.add_option("-m", "--opddata", dest="opddata", help='Pattern for input OPD light curve data',type='string',default="")
    parser.add_option("-o", "--object", dest="object", help='Object ID',type='string',default="")
    parser.add_option("-t", "--sector", dest="sector", help='Select TESS sector',type='int',default=0)
    parser.add_option("-c", "--calib_order", dest="calib_order", help='Order of calibration polynomial',type='string',default="1")
    parser.add_option("-r", "--planet_priors", dest="planet_priors", help='Planet priors file name',type='string',default="")
    parser.add_option("-n", "--nsteps", dest="nsteps", help="Number of MCMC steps",type='int',default=1000)
    parser.add_option("-w", "--walkers", dest="walkers", help="Number of MCMC walkers",type='int',default=32)
    parser.add_option("-u", "--burnin", dest="burnin", help="Number of MCMC burn-in samples",type='int',default=300)
    parser.add_option("-s", "--samples_filename", dest="samples_filename", help='MCMC samples filename',type='string',default="")
    parser.add_option("-z", "--binsize", dest="binsize", help="Light curve binsize [d]",type='float',default=0.1)
    parser.add_option("-1", "--output_binned_lc", dest="output_binned_lc", help='Output binned reduced lightcurve',type='string',default="")
    parser.add_option("-b", action="store_true", dest="impact_parameter", help="Use impact parameter instead of inclination", default=False)
    parser.add_option("-d", action="store_true", dest="star_density", help="Use star density instead of a/Rs", default=False)
    parser.add_option("-l", action="store_true", dest="ols_fit", help="Perform OLS fit prior to MCMC", default=False)
    parser.add_option("-a", action="store_true", dest="fit_gp_activity", help="Run GP activity analysis",default=False)
    parser.add_option("-e", action="store_true", dest="mode", help="Best fit parameters obtained by the mode instead of median", default=False)
    parser.add_option("-p", action="store_true", dest="plot", help="verbose",default=False)
    parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

    try:
        options,args = parser.parse_args(sys.argv[1:])
    except SystemExit as e :
        # allow clean exits from optparse (e.g. --help)
        if e.code == 0 or e.code is None :
            raise
        print("Error: check usage with transits_analysis -h "); sys.exit(1);

    if options.verbose:
        if options.tessdata != '' :
            print('Pattern for input light curve data: ', options.tessdata)
        print('Object ID: ', options.object)
        if options.sector :
            print('TESS sector selected: ', options.sector)
        print('Planet prior parameters file: ', options.planet_priors)
        print('Order of calibration polynomial: ', options.calib_order)
        print('Number of MCMC steps: ', options.nsteps)
        print('Number of MCMC walkers: ', options.walkers)
        print('Number of MCMC burn-in samples: ', options.burnin)
        print('MCMC samples filenames: ', options.samples_filename)
        print('Light curve binsize [d]: ', options.binsize)
        print('Output binned reduced lightcurve: ', options.output_binned_lc)

    min_npoints_per_bin = 30
    min_npoints_within_transit = 300
    transit_window_size = 2

    #min_npoints_per_bin = 3
    #min_npoints_within_transit = 10
    #transit_window_size = 7

    if options.verbose:
        print("Loading TESS lightcurves ...")
    
    tesslc = {}
    tesslc["time"] = np.array([])
    tesslc["flux"] = np.array([])
    tesslc["fluxerr"] = np.array([])
    
    if options.tessdata != "" :
        # Load TESS data
        input_tessdata = sorted(glob.glob(options.tessdata))
        # load TESS data from input lc files
        tesslc = tess.load_lc(input_tessdata, object_name=options.object, transit_window_size=transit_window_size, min_npoints_within_transit=min_npoints_within_transit, binbymedian=False, binsize=options.binsize, min_npoints_per_bin=min_npoints_per_bin, convert_times_to_bjd=True, plot=options.plot, verbose=options.verbose)
    else :
        tesslc["PLANETS"] = []
        loc = {}
        tesslc["PLANETS"].append(loc)
    
    if options.tessffidata != "" :

        planet = tesslc["PLANETS"][0]
        planet["times"] = []
        planet["fluxes"] = []
        planet["fluxerrs"] = []

        tbl = ascii.read(options.tessffidata)
        time = np.array(tbl["col1"])
        flux = np.array(tbl["col2"])
        #fluxerr = np.full_like(flux,0.002)
        fluxerr = np.full_like(flux,0.00005)

        # TOI-6235
        #tcs = tess.calculate_tcs_within_range(time[0], time[-1], 2856.72839, 7.8209807)
    
        # TOI-6342
        #tcs = tess.calculate_tcs_within_range(time[0], time[-1], 2895.572362, 9.3568116)
    
        # K2-65
        tcs = tess.calculate_tcs_within_range(time[0], time[-1], 2456986.3277, 12.647815)
    
        tdur = 0.05
    
        for i in range(len(tcs)) :
            keep = (time > tcs[i] - tdur * 5) & (time < tcs[i] + tdur * 5)
            if len(time[keep]) > 10 :
                planet["times"].append(time[keep])
                planet["fluxes"].append(flux[keep])
                planet["fluxerrs"].append(fluxerr[keep])

    # Load OPD data
    if options.opddata != "" :
        #opdtimes, opdfluxes, opdfluxerrs = get_opddata(options.opddata, target=1, comps=[0,2,3,4,5,6], extname="CATALOG_PHOT_AP010")
        #tesslc["time"] = np.append(tesslc["time"],opdtimes[1])
        #tesslc["flux"] = np.append(tesslc["flux"],opdfluxes[1])
        #tesslc["fluxerr"] = np.append(tesslc["fluxerr"],opdfluxerrs[1])
    
        hdul = fits.open(options.opddata)

        #tesslc["time"] = np.append(tesslc["time"],hdul["TIME"].data)
        #tesslc["flux"] = np.append(tesslc["flux"],hdul["FLUX"].data)
        #tesslc["fluxerr"] = np.append(tesslc["fluxerr"],hdul["FLUXERR"].data)
    
        tesslc["time"] = np.append(tesslc["time"],hdul["BIN_TIME"].data)
        tesslc["flux"] = np.append(tesslc["flux"],hdul["BIN_FLUX"].data)
        tesslc["fluxerr"] = np.append(tesslc["fluxerr"],hdul["BIN_FLUXERR"].data)

    posterior = None
    planet_index=0
        
    for planet in tesslc["PLANETS"] :
        #Load priors information:
    
        if options.planet_priors != "" :
            tesslc["PRIOR_FILE"] = options.planet_priors
            #planet = tess.redefine_tessranges(planet, tesslc, tesslc["PRIOR_FILE"], planet_index=planet_index, transit_window_size=transit_window_size, verbose=True)
        
        planet_priors_file = tesslc["PRIOR_FILE"]

        # select data within certain ranges
        times, fluxes, fluxerrs = planet["times"], planet["fluxes"], planet["fluxerrs"]
    
        """
        for i in range(len(opdtimes)) :
            times.append(opdtimes[i])
            fluxes.append(opdfluxes[i])
            fluxerrs.append(opdfluxerrs[i])
        """
        calib_polyorder = int(options.calib_order)

        # read priors from input files
        #priors = fitlib.read_priors(planet_priors_file, len(times), calib_polyorder=calib_polyorder, verbose=False)
        priors = fitlib.read_transit_rv_priors(planet_priors_file, 0, len(times), planet_index=planet_index, calib_polyorder=calib_polyorder,verbose=False)
    
        # Fit calibration parameters for initial guess
        if options.ols_fit :
            posterior = fitlib.guess_calib(priors, times, fluxes, prior_type="Normal")
        else :
            posterior = fitlib.guess_calib(priors, times, fluxes, prior_type="FIXED")

        if options.plot :
            # plot light curves and models in priors
            fitlib.plot_mosaic_of_lightcurves(times, fluxes, fluxerrs, posterior)

        if options.ols_fit :
            # OLS fit involving all priors
            posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, posterior, calib_post_type="Normal", calib_unc=0.01, verbose=False, plot=False)
            # OLS fit involving all priors
            posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, posterior, calib_post_type="FIXED", verbose=False, plot=False)

            if options.plot :
                fitlib.plot_mosaic_of_lightcurves(times, fluxes, fluxerrs, posterior)

        # Make sure the number of walkers is sufficient, and if not assing a new value
        if options.walkers < 2*len(posterior["theta"]):
            print("WARNING: insufficient number of MCMC walkers, resetting nwalkers={}".format(2*len(posterior["theta"])))
            options.walkers = 2*len(posterior["theta"])

        if options.samples_filename == "" :
            options.samples_filename = priorslib.derive_filename(planet_priors_file, "_mcmc_samples.h5")

        amp=1e-6
    
        posterior = fitlib.fitTransitsWithMCMC(times, fluxes, fluxerrs, posterior, amp=amp, nwalkers=options.walkers, niter=options.nsteps, burnin=options.burnin, verbose=True, plot=True, samples_filename=options.samples_filename, best_fit_from_mode=options.mode, appendsamples=False, plot_individual_transits=False)
    
        planet_index += 1


if __name__ == "__main__" :
    main()
