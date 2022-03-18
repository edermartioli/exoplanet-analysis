"""
    Created on Feb 14 2022
    
    Description: This routine fits planetary transits data using MCMC
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    python tess_ttv.py --object="HATS-24" -vp
    
    python tess_ttv.py --object="HATS-24" --burnin=500 --nsteps=3000 -vpo
    
    python tess_ttv.py --object="TOI-1201" --calib_order=3 --burnin=200 --nsteps=1000 -vpo
    
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

from uncertainties import ufloat
import tess

ExoplanetAnalysis_dir = os.path.dirname(__file__)
priors_dir = os.path.join(ExoplanetAnalysis_dir, 'priors/')
ttvs_dir = os.path.join(ExoplanetAnalysis_dir, 'ttvs/')

def global_transit_fit (tesslc, calib_polyorder=1, walkers=32, niter=500, burnin=100, plot=False, save_plots=False) :

    planets_posteriors = []

    for planet_index in range(len(tesslc["PLANETS"])) :

        planet = tesslc["PLANETS"][planet_index]
        
        # select data within certain ranges
        times, fluxes, fluxerrs = planet["times"], planet["fluxes"], planet["fluxerrs"]

        #Load priors information:
        planet_priors_files = sorted(glob.glob(planet["prior_file"]))

        # read priors from input files
        priors = fitlib.read_priors(planet_priors_files, len(times), calib_polyorder=calib_polyorder, verbose=False)

        # Fit calibration parameters for initial guess
        priors = fitlib.guess_calib(priors, times, fluxes, prior_type="Normal", planet_index=planet_index)

        #if plot :
        # plot light curves and models in priors
        #    fitlib.plot_mosaic_of_lightcurves(times, fluxes, fluxerrs, priors)

        # OLS fit involving all priors
        posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, priors, calib_post_type="Normal", calib_unc=0.01, verbose=False, plot=False)
        # OLS fit involving all priors
        posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, posterior, calib_post_type="FIXED", verbose=False, plot=False)
        
        if plot :
            plot_mosaic_filename = ""
            if save_plots :
                plot_mosaic_filename = planet["prior_file"].replace(".pars","mosaic_lighcurves.png")

            fitlib.plot_mosaic_of_lightcurves(times, fluxes, fluxerrs, posterior, output=plot_mosaic_filename)

        # Make sure the number of walkers is sufficient, and if not assing a new value
        if walkers < 2*len(posterior["theta"]):
            print("WARNING: insufficient number of MCMC walkers, resetting nwalkers={}".format(2*len(posterior["theta"])))
            walkers = 2*len(posterior["theta"])

        samples_filename = planet["prior_file"].replace(".pars","_mcmc_samples.h5")
        
        pairsplot_filename, transitsplot_filename = "", ""
        if save_plots :
            pairsplot_filename = planet["prior_file"].replace(".pars","_pairsplot.png")
            transitsplot_filename = planet["prior_file"].replace(".pars","_transitsplot.png")

        posterior = fitlib.fitTransitsWithMCMC(times, fluxes, fluxerrs, posterior, amp=1e-5, nwalkers=walkers, niter=niter, burnin=burnin, verbose=True, plot=True, samples_filename=samples_filename, appendsamples=False, plot_individual_transits=False, transitsplot_output=transitsplot_filename, pairsplot_output=pairsplot_filename)

        planets_posteriors.append(posterior["planet_posterior_files"][planet_index])

    return planets_posteriors
 
 
def set_priors_from_global_fit(input, output, new_tc=0., planet_index=0, nsig_around_tc=10) :

    priors = priorslib.read_priors(input)
    pl_params = priorslib.read_exoplanet_params(priors, planet_index=planet_index)
    
    outfile = open(output,"w+")
    outfile.write("# Parameter_ID\tPrior_Type\tValues\n")
    
    tc_key = "{0}_{1:03d}".format('tc', planet_index)
    
    for key in pl_params.keys() :
        if ("_err" not in key) and ("_pdf" not in key) :
            pdf_key = "{0}_pdf".format(key)
            if key == tc_key :
                tc_error_key = "{0}_err".format(tc_key)
                if new_tc == 0:
                    # E. Martioli 18/03/2022 -> would be nice to calculate tc from ephemeris tc = t0 + Ep * per
                    new_tc = pl_params[tc_key]
                    
                min = new_tc - nsig_around_tc * pl_params[tc_error_key][1]
                max = new_tc + nsig_around_tc * pl_params[tc_error_key][1]
                    
                outfile.write("{}\tUniform\t{:.10f},{:.10f},{:.10f}\n".format(tc_key,pl_params[tc_key],min,max))

            else :
                outfile.write("{}\tFIXED\t{:.10f}\n".format(key,pl_params[key]))
    outfile.close()


def caculate_tcs(times, fluxes, per, per_err, t0, t0_err) :

    tcs, tcerrs, epochs = [], [], []
    
    uper = ufloat(per, per_err)
    ut0 = ufloat(t0, t0_err)
    
    for i in range(len(times)) :
        
        min_time = times[i][np.nanargmin(fluxes[i])]

        epoch = np.round((min_time - t0) / per)
        
        utc = ut0 + epoch * uper
        
        tcs.append(utc.nominal_value)
        tcerrs.append(utc.std_dev)
        epochs.append(epoch)
        
    tcs = np.array(tcs)
    tcerrs = np.array(tcerrs)
    epochs = np.array(epochs)

    return tcs, tcerrs, epochs


def ttv_pipeline(object, calib_order=1, walkers=32, burnin=30, nsteps=100, save_output=False, verbose=False, plot=False) :

    # Download TESS DVT products and return a list of input data files
    dvt_filenames = tess.retrieve_tess_data_files(object, products_wanted_keys = ["DVT"], verbose=verbose)

    if verbose:
        print("Loading TESS lightcurves ...")
    
    # Load TESS data
    tesslc = tess.load_dvt_files(object, priors_dir=priors_dir, save_priors=True, plot=plot, verbose=verbose)

    if verbose:
        print("Performing global fit to all transits observed by TESS ...")
        
    planets_posteriors = global_transit_fit(tesslc, calib_polyorder=calib_order, walkers=walkers, niter=nsteps, burnin=burnin, plot=plot, save_plots=True)

    for planet_index in range(len(planets_posteriors)) :

        output_str = ''
    
        planet = tesslc["PLANETS"][planet_index]
        # select data within certain ranges
        times, fluxes, fluxerrs = planet["times"], planet["fluxes"], planet["fluxerrs"]

        tc_key = "{0}_{1:03d}".format('tc', planet_index)
        tc_error_key = "{0}_err".format(tc_key)
        per_key = "{0}_{1:03d}".format('per', planet_index)
        per_error_key = "{0}_err".format(per_key)
    
        global_posterior = priorslib.read_priors(planets_posteriors[planet_index])
        planet_params = priorslib.read_exoplanet_params(global_posterior, planet_index=planet_index)
    
        tcs, tcerrs, epochs = caculate_tcs(times, fluxes, planet_params[per_key], planet_params[per_error_key][1], planet_params[tc_key], planet_params[tc_error_key][1])

        obs_tc, obs_tcerr = [], []
        omc, omcerr = [], []
    
        # loop over each individual transit observed by TESS
        for i in range(len(times)) :
            if verbose:
                print("Performing individual transit fit to planet {}/{} transit {}/{} (tc={:.8f} BTJD)".format(planet_index+1, len(planets_posteriors), i+1, len(times), tcs[i]))

            #transitpriorfile = planet["prior_file"].replace(".pars","_{0:05d}_ttvtmp.pars".format(i))
            tmppriorfile = planet["prior_file"].replace(".pars","_ttvtmp.pars")
                
            set_priors_from_global_fit(planets_posteriors[planet_index],tmppriorfile, new_tc=tcs[i],planet_index=planet_index, nsig_around_tc=10)
        
            #Load priors information:
            planet_priors_files = sorted(glob.glob(tmppriorfile))

            # read priors from input files
            priors = fitlib.read_priors(planet_priors_files, 1, calib_polyorder=calib_order, verbose=False)

            # Fit calibration parameters for initial guess
            priors = fitlib.guess_calib(priors, [times[i]], [fluxes[i]], prior_type="Normal", planet_index=planet_index)
        
            # OLS fit involving all priors
            posterior = fitlib.fitTransits_ols([times[i]], [fluxes[i]], [fluxerrs[i]], priors, calib_post_type="FIXED", calib_unc=0.01, verbose=False, plot=False)
            # OLS fit involving all priors
            #posterior = fitlib.fitTransits_ols([times[i]], [fluxes[i]], [fluxerrs[i]], posterior, calib_post_type="FIXED", verbose=False, plot=False)
        
            samples_filename = planet["prior_file"].replace(".pars","_ttvtmp_mcmc_samples.h5")
            pairsplot_filename = planet["prior_file"].replace(".pars","_ttvtmp_pairsplot.png")
            transitsplot_filename = planet["prior_file"].replace(".pars","_ttvtmp_transitsplot.png")

            posterior = fitlib.fitTransitsWithMCMC([times[i]], [fluxes[i]], [fluxerrs[i]], posterior, amp=1e-5, nwalkers=walkers, niter=nsteps, burnin=burnin, verbose=False, plot=False, samples_filename=samples_filename, appendsamples=False, plot_individual_transits=False, transitsplot_output=transitsplot_filename, pairsplot_output=pairsplot_filename)

            results = priorslib.read_priors(posterior["planet_posterior_files"][planet_index])
            planet_params = priorslib.read_exoplanet_params(results, planet_index=planet_index)
        
            fit_tc = planet_params[tc_key]
            fit_tcerr = planet_params[tc_error_key][1]
      
            ufit_tc = ufloat(fit_tc,fit_tcerr)
            u_tc = ufloat(tcs[i], tcerrs[i])
            uomc = ufit_tc-u_tc
        
            omc.append(uomc.nominal_value)
            omcerr.append(uomc.std_dev)
          
            obs_tc.append(fit_tc)
            obs_tcerr.append(fit_tcerr)
        
            output_str += "{}\t{}\t{:.0f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\n".format(planet_index, i, epochs[i], tcs[i], tcerrs[i], fit_tc, fit_tcerr, uomc.nominal_value, uomc.std_dev)
        
        obs_tc = np.array(obs_tc)
        obs_tcerr = np.array(obs_tcerr)
        omc = np.array(omc)*24*60*60
        omcerr = np.array(omcerr)*24*60*60

        if plot :
            plt.errorbar(obs_tc, omc, yerr=obs_tcerr*24*60*60, fmt='o')
            plt.xlabel("time [BTJD]")
            plt.ylabel("O-C [s]")
            plt.show()

        if save_output :
            output_ttv_filename = os.path.join(ttvs_dir, '{}_{:03d}.ttv'.format(object.replace(" ",""),planet_index))
            outttvfile = open(output_ttv_filename,"w+")
            outttvfile.write("#PLANET_INDEX\tTRANSIT_INDEX\tEPOCH\tTC\tTCERRR\tOBS_TC\tOBSTCERR\t(O-C)\t(O-C)ERR\n")
            outttvfile.write(output_str)
            outttvfile.close()

        if verbose :
            print("#PLANET_INDEX\tTRANSIT_INDEX\tEPOCH\tTC\tTCERRR\tOBS_TC\tOBSTCERR\t(O-C)\t(O-C)ERR")
            print(output_str)



parser = OptionParser()
parser.add_option("-j", "--object", dest="object", help='Object ID',type='string',default="")
parser.add_option("-c", "--calib_order", dest="calib_order", help='Order of calibration polynomial',type='int',default=1)
parser.add_option("-w", "--walkers", dest="walkers", help="Number of MCMC walkers",type='int',default=32)
parser.add_option("-b", "--burnin", dest="burnin", help="Number of MCMC burn-in samples",type='int',default=30)
parser.add_option("-n", "--nsteps", dest="nsteps", help="Number of MCMC steps",type='int',default=100)
parser.add_option("-o", action="store_true", dest="save_output", help="Save output TTV", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="verbose",default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with tess_ttv.py -h "); sys.exit(1);

if options.verbose:
    print('Object ID: ', options.object)
    print('Order of calibration polynomial: ', options.calib_order)
    print('Number of MCMC walkers: ', options.walkers)
    print('Number of MCMC burn-in samples: ', options.burnin)
    print('Number of MCMC steps: ', options.nsteps)
    
ttv_pipeline(options.object, calib_order=options.calib_order, walkers=options.walkers, burnin=options.burnin, nsteps=options.nsteps, save_output=options.save_output, verbose=options.verbose, plot=options.plot)

