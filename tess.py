# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 2022
@author: Eder Martioli
Laboratório Nacional de Astrofísica, Brazil.
Institut d'Astrophysique de Paris, France.
"""

import sys, os

from astroquery.mast import Observations
from astroquery.mast import Catalogs
from astropy.io import fits
from astropy import table
from astropy.io import ascii

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from scipy import constants

import json

tess_lib_dir = os.path.dirname(__file__)
claret_coeff_path = os.path.join(tess_lib_dir, 'tess_claret_ldc/table25.dat')
objects_db = os.path.join(tess_lib_dir,'tess_objects.json')

def parse_manifest(manifest):
    """
    Parse manifest and add back columns that are useful for TESS DV exploration.
    """
    results = deepcopy(manifest)
    filenames = []
    sector_range = []
    exts = []
    for i,f in enumerate(manifest['Local Path']):
        file_parts = np.array(np.unique(f.split(sep = '-')))
        sectors = list( map ( lambda x: x[0:2] == 's0', file_parts))
        s1 = file_parts[sectors][0]
        try:
            s2 = file_parts[sectors][1]
        except:
            s2 = s1
        sector_range.append("%s-%s" % (s1,s2))
        path_parts = np.array(f.split(sep = '/'))
        filenames.append(path_parts[-1])
        exts.append(path_parts[-1][-8:])

    results.add_column(table.Column(name = "filename", data = filenames))
    results.add_column(table.Column(name = "sectors", data = sector_range))
    results.add_column(table.Column(name = "fileType", data = exts))
    results.add_column(table.Column(name = "index", data = np.arange(0,len(manifest))))
    
    return results

def plot_detrended_timeseries(time, relflux, fluxerr, model, star_name="UNKNOWN") :
    # Plot the detrended photometric time series in the first binary table.
    plt.figure(figsize = (18,6))
    plt.title('Data Validation Detrended Light Curve for %s' % (star_name))
    
    #plt.errorbar (time, relflux, yerr=fluxerr, fmt='.', color="darkblue", alpha=0.3)
    plt.plot(time, relflux, '.', color="darkblue", alpha=0.7,zorder=1)
    plt.plot(time, model,'r-', lw=0.7, alpha=0.5, zorder=2)
    plt.ylim(1.5* np.nanpercentile(relflux, .5) , 1.5 * np.nanpercentile(relflux, 99.5))
    plt.xlabel(r"Time [BTJD]", fontsize=20)
    plt.ylabel(r"Rel. flux", fontsize=20)
    plt.show()


def plot_folded(phase, flux, model, ext, period, sector) :
    isort = phase.argsort()
    
    plt.plot(phase[isort], flux[isort],'.', ms = .5, alpha=0.5, zorder=1)
    plt.plot(phase[isort], model[isort], '-', lw = 2, label = "TCE {} Sector {}".format(ext,sector), zorder=2)
    plt.xlabel('Phase (Period = %5.2f days)' % period, fontsize=18)
    plt.ylim(1.5 * np.nanpercentile(flux, .5) , 1.4 * np.nanpercentile(flux,99.5))
    plt.legend(loc = "lower right")
    plt.ylabel("rel. flux", fontsize=18)

    
def plot_folded_light_curve (dvt_filenames) :

    plt.figure(figsize = (18,6))

    for dvt_filename in dvt_filenames :
    
        phdr = fits.getheader(dvt_filename, 0)
        try :
            sector = phdr['SECTOR']
        except :
            sector = "ALL"
        
        nTCEs = fits.getheader(dvt_filename)['NEXTEND'] - 2
        
        for ext in range(1, nTCEs + 1):
            data = fits.getdata(dvt_filename, ext)
            
            head = fits.getheader(dvt_filename, ext)
            period = head['TPERIOD']
            phase = data['PHASE'] / (np.max(data['PHASE']) - np.min(data['PHASE']))
            #flux = data['LC_INIT']
            flux = data['LC_DETREND']
            fluxerr = data['LC_INIT_ERR']
            model = data['MODEL_INIT']
            plt.subplot(nTCEs, 1, ext)
            plot_folded(phase, flux, model, ext, period, sector)

    #plt.legend(fontsize=18)
    plt.show()


def get_sector_labels(results, sector=0, get_individual_sectors=False, verbose=False) :

    sectors_set = []
    for i in range(len(results['sectors'])) :
        if results['sectors'][i] not in sectors_set :
            sectors_set.append(results['sectors'][i])

    labels = []
    individual_sector_labels = []
    sec1min, sec2max = 1e10,0
    selected_sector_found = False
    for lab in sectors_set :
        if verbose :
            print("Data set label:",lab)
        sec1,sec2 = lab.split("-")
        sec1int,sec2int = int(sec1[1:]),int(sec2[1:])
        #print("sec1=",sec1,"sec2=",sec2)
        
        # the condition below is for a single selected sector
        if sec1int == sector and sec2int == sector :
            labels.append(lab)
            selected_sector_found = True
            break

        if sec1 == sec2 :
            individual_sector_labels.append(lab)

        # the condition below is for the widest range of sectors
        if sec1int <= sec1min and sec2int >= sec2max :
            sec1min = sec1int
            sec2max = sec2int
            labels.append(lab)

    if not selected_sector_found and sector != 0:
        print("WARNING: selected TESS sector = {} not found. Selecting all sectors.".format(sector))

    if get_individual_sectors :
        labels = individual_sector_labels
       
    if verbose:
        print("TESS Sectors labels:",labels)
    return labels


def retrieve_tess_data_files(object, sector=0, products_wanted_keys = ["DVT","DVM","DVS","DVR"], individual_sectors=False, get_all_dvt_files=True, verbose=False) :

    # first check if database exists
    if os.path.exists(objects_db) :
        with open(objects_db, 'r') as json_file:
            object_data = json.load(json_file)

        # check if object exists in db keys
        if object in object_data.keys() :
            if "dvt_filenames" in object_data[object].keys() :
                dvt_filenames = object_data[object]["dvt_filenames"]
                return dvt_filenames
        else :
            object_data[object] = {}
    else :
        print("Could not find data base, a new one will be created ... ")
        object_data = {}
        object_data[object] = {}

    if verbose:
        print('Object ID: ', object)
        if sector :
            print('TESS sector selected: ', sector)

    star_name = object

    observations = Observations.query_object(star_name,radius = "0 deg")
    obs_wanted = (observations['dataproduct_type'] == 'timeseries') & (observations['obs_collection'] == 'TESS')
    
    if verbose :
        print(observations[obs_wanted]['obs_collection', 'project', 'obs_id'] )

    data_products = Observations.get_product_list(observations[obs_wanted])
    products_wanted = Observations.filter_products(data_products,productSubGroupDescription=products_wanted_keys)
    
    if verbose:
        print("Downloading TESS products ... ")
        print(products_wanted["productFilename"])
        
    manifest = Observations.download_products(products_wanted)
    
    if verbose:
        print("*****************")
        print("Printing MANIFEST")
        print("-----------------")
        print(manifest['Local Path'])
    
    # run parser
    results = parse_manifest(manifest)

    if verbose :
        print("*****************")
        print("Printing RESULTS")
        print("-----------------")
        print(results['index','sectors','fileType'])

    if get_all_dvt_files :
    
        want = results['fileType'] == "dvt.fits"
        dvt_filenames = results['Local Path'][want]
        
    else :
        dvt_filenames = []

        # Get label corresponding to all sectors that observed the target
        sectors_labels = get_sector_labels(results, sector=sector, get_individual_sectors=individual_sectors, verbose=verbose)

        if verbose :
            #Print the DVT File
            print("*****************")
            print("Printing SECTORS")
            print("-----------------")
        for sectors_label in sectors_labels :
            print(results['index', 'sectors', 'fileType'][results['sectors'] == sectors_label])
            
        for sectors_lab in sectors_labels :
            want = (results['sectors'] == sectors_lab) & (results['fileType'] == "dvt.fits")
            dvt_filename = manifest[want]['Local Path'][0]
            if dvt_filename not in dvt_filenames :
                dvt_filenames.append(dvt_filename)

    object_data[object]["dvt_filenames"] = list(dvt_filenames)
    
    if verbose :
        print("Dumping object data into the database -> ", objects_db)
    with open(objects_db, 'w') as json_file :
        json.dump(object_data, json_file, sort_keys=True, indent=4)

    return dvt_filenames


def get_claret_ld_coeffs(teff, logg=5.0, zsun=0, method="LSM") :
    """
        Description: Function to retrieve quadratic limb darkening
        coefficients from Claret (2018)
        
        https://ui.adsabs.harvard.edu/abs/2017A%26A...600A..30C/abstract
        
        teff: (float), star effective temperature in K
        logg: (float), logarithm of star surface gravity
        zsun: (float), star metallicity
        
        return (float,float) tuple with linear and quadratic coefficients
    """


    lddata = ascii.read(claret_coeff_path)
    
    rounded_logg = round(logg * 2) / 2
    
    keep = (lddata['col1'] == rounded_logg) & (lddata['col3'] == zsun)
    
    dtmin = 1e20
    a, b = np.nan, np.nan
    
    if len(lddata['col1'][keep]) == 0 :
        print("ERROR: select 0.0 < logg < 5.0 and -5 < zsun < 1")
        return a, b
    
    for i in range(len(lddata['col2'][keep])) :
        dt = np.abs(lddata['col2'][keep][i]-teff)
        if dt < dtmin :
            dtmin = dt
            if method == "LSM" :
                a = lddata['col5'][keep][i]
                b = lddata['col6'][keep][i]
            elif method == "PCM" :
                a = lddata['col7'][keep][i]
                b = lddata['col8'][keep][i]
            else :
                print("ERROR: invalid limb-darkening method. Select LSM or PCM, exiting...")
                exit()
    return a,b


def save_planet_prior(output, teff, ms, rs, tc, per, a, rp, inc, u0, u1, ecc=0, w=90, k=0., hasrvdata=False, circular_orbit=True, planet_index=0, append=True, all_parameters_fixed=False) :
        
    if append :
        outfile = open(output,"a+")
    else :
        outfile = open(output,"w+")
        
    if planet_index==0 or not append :
        outfile.write("# Parameter_ID\tPrior_Type\tValues\n")
        outfile.write("teff_{:03d}\tFIXED\t{:.0f}\n".format(planet_index,teff))
        outfile.write("ms_{:03d}\tFIXED\t{:.3f}\n".format(planet_index,ms))
        outfile.write("rs_{:03d}\tFIXED\t{:.3f}\n".format(planet_index,rs))
        
    if hasrvdata and not all_parameters_fixed :
        outfile.write("k_{:03d}\tUniform\t-100,100.,{:.3f}\n".format(planet_index,k))
    else :
        outfile.write("k_{:03d}\tFIXED\t{:.3f}\n".format(planet_index,k))

    outfile.write("rvsys_{:03d}\tFIXED\t0.\n".format(planet_index))
    outfile.write("trend_{:03d}\tFIXED\t0.\n".format(planet_index))
    
    if all_parameters_fixed :
        outfile.write("tc_{:03d}\tFIXED\t{:.8f}\n".format(planet_index,tc))
        outfile.write("per_{:03d}\tFIXED\t{:.8f}\n".format(planet_index,per))
        outfile.write("a_{:03d}\tFIXED\t{:.2f}\n".format(planet_index,a))
        outfile.write("rp_{:03d}\tFIXED\t{:.5f}\n".format(planet_index,rp))
        outfile.write("inc_{:03d}\tFIXED\t{:.2f}\n".format(planet_index,inc))
        outfile.write("u0_{:03d}\tFIXED\t{:.4f}\n".format(planet_index,u0))
        outfile.write("u1_{:03d}\tFIXED\t{:.4f}\n".format(planet_index,u1))
    else :
        outfile.write("tc_{:03d}\tUniform\t{:.8f},{:.8f},{:.8f}\n".format(planet_index,tc-0.1*per,tc+0.1*per,tc))
        outfile.write("per_{:03d}\tUniform\t{:.8f},{:.8f},{:.8f}\n".format(planet_index,per*0.8,per*1.2,per))
        outfile.write("a_{:03d}\tUniform\t{:.2f},{:.2f},{:.2f}\n".format(planet_index,a*0.8,a*1.2,a))
        outfile.write("rp_{:03d}\tUniform\t{:.5f},{:.5f},{:.5f}\n".format(planet_index,rp*0.8,rp*1.2,rp))
        outfile.write("inc_{:03d}\tUniform\t{:.2f},90,{:.2f}\n".format(planet_index,inc*0.8,inc))
    
        outfile.write("u0_{:03d}\tUniform\t0.,3.,{:.4f}\n".format(planet_index,u0))
        outfile.write("u1_{:03d}\tUniform\t0.,3.,{:.4f}\n".format(planet_index,u1))
    
    if circular_orbit or all_parameters_fixed :
        outfile.write("ecc_{:03d}\tFIXED\t0.\n".format(planet_index))
        outfile.write("w_{:03d}\tFIXED\t90.\n".format(planet_index))
    else :
        outfile.write("ecc_{:03d}\tUniform\t0,1,{:.5f}\n".format(planet_index,ecc))
        outfile.write("w_{:03d}\tUniform\t0.,360.,{:.3f}\n".format(planet_index,w))

    outfile.close()


def calculate_tcs_within_range(tmin, tmax, tepoch, tperiod) :
    t0 = tepoch
    if tepoch > tmin :
        while t0 > tmin :
            t0 -= tperiod
    tcs = []
    while t0 < tmax :
        t0 += tperiod
        tcs.append(t0)
    tcs = np.array(tcs)
    return tcs
    
    
def select_transit_windows(selected_dvt_files, tcs, twindow, tdur_days, tess_cadence=1.388888888e-3, flux_base_line=1.0, recenter_tcs=True, plot=False) :

    t = f = ef = np.array([])
    model = np.array([])
    for dvt_filename in selected_dvt_files :
        data = fits.getdata(dvt_filename, 1)
        t = np.append(t,data['TIME'])
        f = np.append(f,data['LC_DETREND'] + flux_base_line)
        ef = np.append(ef,data['LC_INIT_ERR'])
        model = np.append(model,data['MODEL_INIT'] + flux_base_line)

    sorted = np.argsort(t)
    t = t[sorted]
    f, ef = f[sorted], ef[sorted]
    model = model[sorted]

    times, fluxes, fluxerrs = [], [], []

    if plot :
        plt.plot(t, f, 'r.', alpha=0.1, label="TESS data")
    
    if recenter_tcs :
        for i in range(len(tcs)) :
            t1 = tcs[i] - twindow/2
            t2 = tcs[i] + twindow/2
        
            window = (t > t1) & (t < t2)
            tcs[i] = t[window][np.argmin(model[window])]
    
    valid_tcs = []
    
    for i in range(len(tcs)) :
        t1 = tcs[i] - twindow/2
        t2 = tcs[i] + twindow/2
        
        window = (t > t1) & (t < t2) & (np.isfinite(f)) & (np.isfinite(ef))
        
        # first test if there is any valid data within the time window
        if len(t[window]) :
        
            window_size = t[window][-1] - t[window][0]
        
            np_expected_within_window = tdur_days*2 / tess_cadence
                
            if window_size > tdur_days*2 and len(t[window]) > np_expected_within_window :
        
                times.append(t[window])
                fluxes.append(f[window])
                fluxerrs.append(ef[window])
            
                valid_tcs.append(tcs[i])
            
                if plot :
                    plt.errorbar(t[window], f[window], yerr=ef[window], fmt='o')
    
    valid_tcs = np.array(valid_tcs)
    
    if plot:
        plt.xlabel(r"Time [BTJD]")
        plt.ylabel(r"Flux [e-/s]")
        plt.show()

    return valid_tcs, times, fluxes, fluxerrs
    
    
def load_dvt_files(object, priors_dir=".", transit_window_size = 5, save_priors=True, planet_to_analyze=-1, hasrvdata=False, force_circular_orbit=True, plot=False, verbose=False) :
    
    # first check if database exists
    if os.path.exists(objects_db) :
        with open(objects_db, 'r') as json_file:
                object_data = json.load(json_file)

        # check if object exists in db keys
        if object not in object_data.keys() :
            print("ERROR: object {} not found in database, exiting ... ".format(object))
            exit()
    else :
        print("ERROR: could not find objects database file: {}, exiting ... ".format(objects_db))
        exit()

    dvt_filenames = object_data[object]["dvt_filenames"]

    loc = {}

    # FALTA :
    # - SELEÇÃO AUTOMÁTICA DOS INTERVALOS AO REDOR DOS TRÂNSITOS BASEADO NOS PARÂMETROS DE AJUSTE DO TRÂNSITO

    if not os.path.exists(priors_dir):
        os.makedirs(priors_dir)

    tspan_max = 0
    widestdvt = dvt_filenames[0]
    # loop over each dvt file in the list
    for dvt_filename in dvt_filenames :
        hdr = fits.getheader(dvt_filename, 0)
        tstart = hdr['TSTART']  # observation start time in BTJD
        tstop = hdr['TSTOP']  # observation stop time in BTJD
        tspan = tstop - tstart
        if tspan > tspan_max :
            tspan_max = tspan
            widestdvt = dvt_filename
            tmin,tmax = tstart,tstop

    selected_dvt_files = [widestdvt]
    for dvt_filename in dvt_filenames :
        hdr = fits.getheader(dvt_filename, 0)
        tstart = hdr['TSTART']  # observation start time in BTJD
        tstop = hdr['TSTOP']  # observation stop time in BTJD
        if tstop < tmin :
            selected_dvt_files.append(dvt_filename)
            tmin = tstart
        elif tstart > tmax :
            selected_dvt_files.append(dvt_filename)
            tmax = tstop

    loc["selected_dvt_files"] = selected_dvt_files

    # save list of selected dvt files into db
    object_data[object]["selected_dvt_files"] = selected_dvt_files

    if verbose :
        print("Selected DVT files:",selected_dvt_files)
    
    loc["base_dvt_file"] = selected_dvt_files[0]
    # get basic information from the base file:
    bhdr = fits.getheader(selected_dvt_files[0], 0)
    #object_name = bhdr['OBJECT']  #target id
    object_name = object.replace(" ","").upper()
    
    tessmag = bhdr['TESSMAG']  #TESS magnitude
    teff = bhdr['TEFF']  #[K] Effective temperature
    logg = bhdr['LOGG']  #[cm/s2] log10 surface gravity
    rstar = bhdr['RADIUS']  #[solar radii] stellar radius
    G_cgs = constants.G * 1e3
    radius_cm = rstar * 6.957e10
    msun_g = 1.989e+33
    #logg = log(GM/r^2) -> M = 10^logg * r^2 / G
    mstar = (10**(logg) * radius_cm * radius_cm / G_cgs) / msun_g #[solar mass] stellar mass
    if verbose :
        print("Derived stellar mass: {:.3f} Msun".format(mstar))

    u0, u1 = get_claret_ld_coeffs(teff, logg=logg)
    if verbose :
        print("Quadratic limb-darkening coefficients (Claret, 2018): {:.3f} {:.3f}".format(u0, u1))

    loc["TESSMAG"] = tessmag
    loc["TEFF"] = teff
    loc["LOGG"] = logg
    loc["RADIUS"] = rstar
    loc["MASS"] = mstar
    loc["u0"] = u0
    loc["u1"] = u1

    # save object info into db
    object_data[object]["TESSMAG"] = tessmag
    object_data[object]["TEFF"] = teff
    object_data[object]["LOGG"] = logg
    object_data[object]["RADIUS"] = rstar
    object_data[object]["MASS"] = mstar
    object_data[object]["u0"] = u0
    object_data[object]["u1"] = u1

    # get number of planet extensions
    nTCEs = bhdr['NEXTEND'] - 2
        
    loc["NPLANETS"] = nTCEs
    
    # save number of transiting objects into db
    object_data[object]["NPLANETS"] = nTCEs
        
    if verbose :
        print("Number of transit objects: ", nTCEs)
    
    loc["PLANETS"] = []
    object_data[object]["PLANETS"] = []

    output_prior_file = priors_dir + "/{}.pars".format(object_name)
    loc["PRIOR_FILE"] = output_prior_file
                
    # loop over each planet extension to create the priors files and
    # to select the transit ranges based on the planet parameters in the header
    for ext in range(1, nTCEs + 1) :
        pl_loc = {}
        pl_name = "{}_{:03}".format(object_name,ext-1)
        pl_loc["PLANET_NAME"] = pl_name
        
        object_data[object]["PLANETS"].append(pl_name)
        object_data[object][pl_name] = {}
            
        exthdr = fits.getheader(selected_dvt_files[0], ext)
        tperiod = object_data[object][pl_name]["per"] = pl_loc["TPERIOD"] = exthdr['TPERIOD']  # transit period [days]
        tepoch = object_data[object][pl_name]["tc"] = pl_loc["TEPOCH"] = exthdr['TEPOCH']  # transit epoch in BTJD
        tdepth = object_data[object][pl_name]["tdepth"] = pl_loc["TDEPTH"] = exthdr['TDEPTH'] # fitted transit depth [ppm]
        tsnr = object_data[object][pl_name]["tsnr"] = pl_loc["TSNR"] = exthdr['TSNR']  # transit signal-to-noise ratio
        tdur = object_data[object][pl_name]["tdur"] = pl_loc["TDUR"] = exthdr['TDUR']  # transit duration [hr]
        idur = object_data[object][pl_name]["idur"] = pl_loc["INDUR"] = exthdr['INDUR']  # ingress duration [hr]
        impact = object_data[object][pl_name]["impact"] = pl_loc["IMPACT"] = exthdr['IMPACT']  # impact parameter
        inclin = object_data[object][pl_name]["inc"] = pl_loc["INCLIN"] = exthdr['INCLIN']  # inclination [deg]
        drratio = object_data[object][pl_name]["a"] = pl_loc["DRRATIO"] = exthdr['DRRATIO']  # ratio of planet distance to star radius
        radratio = object_data[object][pl_name]["rp"] = pl_loc["RADRATIO"] = exthdr['RADRATIO']  # ratio of planet radius to star radius
        pradius = object_data[object][pl_name]["rp_earth"] = pl_loc["PRADIUS"] = exthdr['PRADIUS']  # planet radius in earth radii
        maxmes = pl_loc["MAXMES"] = exthdr['MAXMES']  # maximum multi-event statistic
        maxses = pl_loc["MAXSES"] = exthdr['MAXSES']  # maximum single-event statistic
        ecc = object_data[object][pl_name]["ecc"] = pl_loc["ECC"] = 0. # orbital eccentricity
        omega = object_data[object][pl_name]["omega"] = pl_loc["OMEGA"] = 90. # longitude of periastron in degrees

        # check variable periods:
        recenter_tcs = False
        periods = np.array([])
        for dvt_filename in selected_dvt_files :
            loc_exthdr = fits.getheader(dvt_filename, ext)
            periods = np.append(periods,loc_exthdr['TPERIOD'])
            if verbose :
                print(dvt_filename,"-> Period=", loc_exthdr['TPERIOD'], "d")
        tperiod = np.nanmean(periods)
        if np.any(periods != tperiod) :
            pl_loc["TPERIOD"] = tperiod
            recenter_tcs = True
        
        # create priors files:
        pl_output = priors_dir + "/{}_{:03}.pars".format(object_name,ext-1)
        pl_loc["prior_file"] = pl_output

        if save_priors :
            if verbose :
                print("Saving priors file:",pl_output)
                
            all_parameters_fixed = True
            if ext-1 == planet_to_analyze or planet_to_analyze == -1 :
                all_parameters_fixed = False
                
            append = False
            if ext > 1 :
                append = True
    
            # Save priors for each individual planet
            save_planet_prior(pl_output, teff, mstar, rstar, tepoch, tperiod, drratio, radratio, inclin, u0, u1, ecc=ecc, w=omega, hasrvdata=hasrvdata, circular_orbit=force_circular_orbit, planet_index=0, append=False, all_parameters_fixed=False)
            # save priors for all planets into one file
            save_planet_prior(output_prior_file, teff, mstar, rstar, tepoch, tperiod, drratio, radratio, inclin, u0, u1, ecc=ecc, w=omega, hasrvdata=hasrvdata, circular_orbit=force_circular_orbit, planet_index=ext-1, append=append, all_parameters_fixed=all_parameters_fixed)
            
        object_data[object][pl_name]["planet_priors"] = pl_output
        object_data[object][pl_name]["hasrvdata"] = hasrvdata
        object_data[object][pl_name]["force_circular_orbit"] = force_circular_orbit

        tcs = calculate_tcs_within_range(tmin, tmax, tepoch, tperiod)
        if verbose :
            print("tmin={} tmax={} tepoch={} tperiod={} d".format(tmin, tmax, tepoch, tperiod))
            print("N_transits={} 1st_transit={} last_transit={} BTJD".format(len(tcs),tcs[0],tcs[-1]))

        # initialize mask with all False
        observed = tcs == 0
        
        for dvt_filename in selected_dvt_files :
            loc_hdr = fits.getheader(dvt_filename, 0)
            tstart = loc_hdr['TSTART']  # observation start time in BTJD
            tstop = loc_hdr['TSTOP']  # observation stop time in BTJD
            observed ^= (tcs > tstart) & (tcs < tstop)
        
        if verbose :
            print("N_transits_observed={} 1st_transit_obs={} last_transit_obs={} BTJD".format(len(tcs[observed]),tcs[observed][0],tcs[observed][-1]))
        
        object_data[object][pl_name]["n_transits_observed_by_TESS"] = len(tcs[observed])
        
        tdur_days = tdur/24.
        # select data within certain ranges
        tcs, times, fluxes, fluxerrs = select_transit_windows(selected_dvt_files, tcs[observed], transit_window_size*tdur_days, tdur_days, recenter_tcs=recenter_tcs, plot=False)
        if verbose :
            print("Number of transit windows selected:",len(times))
        
        object_data[object][pl_name]["n_transits_windows_selected"] = len(times)

        pl_loc["tcs"] = tcs
        pl_loc["times"] = times
        pl_loc["fluxes"] = fluxes
        pl_loc["fluxerrs"] = fluxerrs

        loc["PLANETS"].append(pl_loc)


    time = np.array([])
    relflux, fluxerr = np.array([]), np.array([])
    phase, model = np.array([]), np.array([])
    #[('TIME', '>f8'), ('TIMECORR', '>f4'), ('CADENCENO', '>i4'), ('PHASE', '>f4'), ('LC_INIT', '>f4'), ('LC_INIT_ERR', '>f4'), ('LC_WHITE', '>f4'), ('LC_DETREND', '>f4'), ('MODEL_INIT', '>f4'), ('MODEL_WHITE', '>f4')]))

    for dvt_filename in selected_dvt_files :
        data = fits.getdata(dvt_filename, 1)
        keep = np.isfinite(data['LC_DETREND']) *  np.isfinite(data['LC_INIT_ERR'])
        time = np.append(time,data['TIME'][keep])
        relflux = np.append(relflux,data['LC_DETREND'][keep])
        fluxerr = np.append(fluxerr,data['LC_INIT_ERR'][keep])
        model = np.append(model,data['MODEL_INIT'][keep])
        phase = np.append(phase,data['PHASE'][keep])

    if plot :
        # Plot the detrended photometric time series
        plot_detrended_timeseries(time, relflux, fluxerr, model, star_name=object)
        plot_folded_light_curve (selected_dvt_files)

    sorted = np.argsort(time)
    loc["object_name"] = object_name
    loc["time"] = time[sorted]
    loc["flux"] = relflux[sorted]
    loc["fluxerr"] = fluxerr[sorted]
    loc["model"] = model[sorted]
    loc["phase"] = phase[sorted]
    loc['nflux'] = loc["flux"]
    loc['nfluxerr'] = loc["fluxerr"]

    if verbose :
        print("Dumping object data into the database -> ", objects_db)
    with open(objects_db, 'w') as json_file :
        json.dump(object_data, json_file, sort_keys=True, indent=4)

    return loc
