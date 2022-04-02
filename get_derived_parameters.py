"""
    Created on Feb 18 2022
    
    Description:  This routine calculate the derived parameters
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    python get_derived_parameters.py --input=priors
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import sys

from optparse import OptionParser
import priorslib
import numpy as np

from uncertainties import ufloat
from uncertainties import umath
from scipy import constants

from astropy import units as u
from astropy.modeling.models import BlackBody

import matplotlib.pyplot as plt

def read_planet_prior(planet_priors_file, verbose=False) :

    if verbose:
        print("Loading exoplanet priors from input file: ",planet_priors_file)
    
    planet_priors = priorslib.read_priors(planet_priors_file)
        
    planet_params = priorslib.read_exoplanet_params(planet_priors)

    loc = {}
    
    # print out planet priors
    if verbose:
        print("----------------")
        print("Input parameters:")
    for key in planet_params.keys() :
        if ("_err" not in key) and ("_pdf" not in key) :
            pdf_key = "{0}_pdf".format(key)
            error_key = "{0}_err".format(key)

            if planet_params[pdf_key] == "FIXED" :
                cen = planet_params[key]
                error = 0.
                if verbose :
                    print("{0} = {1} ({2})".format(key, cen, planet_params[pdf_key]))
            elif planet_params[pdf_key] == "Uniform" or planet_params[pdf_key] == "Jeffreys":
                min = planet_params[error_key][0]
                max = planet_params[error_key][1]
                if verbose :
                    print("{0} <= {1} <= {2} ({3})".format(min, key, max, planet_params[pdf_key]))
                error = np.abs(max - min)/2
                cen = (max + min)/2
            elif planet_params[pdf_key] == "Normal" :
                error = planet_params[error_key][1]
                cen = planet_params[key]
                if verbose :
                    print("{0} = {1} +- {2} ({3})".format(key, cen, error, planet_params[pdf_key]))

            loc[key.replace("_000","")] = ufloat(cen,error)
    if verbose :
        print("----------------")

    return loc

def semi_major_axis(params, fitvalue=False, over_rs=True) :
    
    period = params["per"]
    mstar = params["ms"]
    fit_a_over_r = params["a"]
    
    #G = 6.67408e-11
    G = constants.G
    msun = 1.989e+30
    ms = mstar * msun
    d2s = 24.*60.*60.
    
    a = (((d2s*period * d2s*period * G * ms)/(4. * np.pi * np.pi))**(1/3))
    
    if over_rs :
        if fitvalue :
            return fit_a_over_r
        else :
            rstar = params["rs"]
            rsun = 696.34e6
            rs = rstar * rsun
            return a / rs
    else :
        if fitvalue :
            rstar = params["rs"]
            rsun = 696.34e6
            rs = rstar * rsun
            au_m = 1.496e+11
            return fit_a_over_r * rs / au_m
        else:
            au_m = 1.496e+11
            return a / au_m


def impact_parameter(params, fitvalue=False) :
    
    if fitvalue :
        a = params["a"]
    else :
        a  = semi_major_axis(params, over_rs=True)
    
    inc = params["inc"]
    inc_rad = inc * np.pi / 180
    
    b = umath.cos(inc_rad) * a
    return b


def planet_radius(params, units='rjup') :
    
    rp_over_rs = params["rp"]
    
    rstar = params["rs"]
    rsun = 696.34e6
    rs = rstar * rsun
    rp = rp_over_rs * rs
    
    if units == 'rjup' :
        rjup = 69911000.
        return rp/rjup
    if units == 'rnep' :
        rnep = 24622000.
        return rp/rnep
    elif units == 'rearth' :
        rearth = 6371000.
        return rp/rearth


def planet_density(params, k) :

    rjup = 69911000.
    mjup = 1.898e27
    rp = planet_radius(params, units='rjup') * rjup
    mp = planet_mass(params, k, units='mjup') * mjup

    volume = (4/3) * np.pi * rp**3
    density = (mp*1000.) / (volume*(100*100*100))

    return density


def transit_duration(params, fitvalue=False):
    
    rp_over_rs = params["rp"]
    period = params["per"]
    if fitvalue :
        sma_over_rs = params["a"]
    else :
        sma_over_rs = semi_major_axis(params, over_rs=True)
    inclination = params["inc"]
    eccentricity = params["ecc"]
    periastron = params["w"]
    
    ww = periastron * np.pi / 180
    ii = inclination * np.pi / 180
    ee = eccentricity
    aa = sma_over_rs
    ro_pt = (1 - ee ** 2) / (1 + ee * umath.sin(ww))
    b_pt = aa * ro_pt * umath.cos(ii)
    if b_pt.nominal_value > 1:
        b_pt = 0.5
    s_ps = 1.0 + rp_over_rs
    df = umath.asin(umath.sqrt((s_ps ** 2 - b_pt ** 2) / ((aa ** 2) * (ro_pt ** 2) - b_pt ** 2)))
    abs_value = (period * (ro_pt ** 2)) / (np.pi * umath.sqrt(1 - ee ** 2)) * df
    
    return abs_value

def teq(params, geom_albedo = 0.1, f = 0.5) :
    
    #f = 1/2 uniform heat redistribution
    #f = 2/3 # no heat redistribution

    teff = params["teff"]
    
    Teq = teff * umath.sqrt( f / semi_major_axis(params, over_rs=True, fitvalue=False) ) * (1.0 - geom_albedo)**(0.25)

    return Teq


def rv_semi_amplitude(params, mp) :

    # per in days
    # mpsini and mp in jupiter mass
    # mstar in solar mass

    mstar=params["ms"]
    inc=params["inc"]
    ecc=params["ecc"]
    
    G = constants.G # constant of gravitation in m^3 kg^-1 s^-2
    
    per_s = params["per"] * 24. * 60. * 60. # in s
    
    mjup = 1.898e27 # mass of Jupiter in Kg
    msun = 1.989e30 # mass of the Sun in Kg
    
    mstar_kg = mstar*msun
    mp_kg = mp*mjup
    
    inc_rad = inc * np.pi/180. # inclination in radians
    
    p1 = (2. * np.pi * G / per_s)**(1/3)
    p2 = umath.sin(inc_rad) / (mstar_kg + mp_kg)**(2/3)
    p3 = 1./umath.sqrt(1 - ecc*ecc)
    
    ks = mp_kg * p1*p2*p3
    kp = mstar_kg * p1*p2*p3
    
    # return semi-amplitude in km/s
    return ks, kp

def planet_mass(params, k, units='mjup') :
    
    # per in days
    # mpsini and mp in jupiter mass
    # mstar in solar mass
    
    mstar=params["ms"]
    inc=params["inc"]
    ecc=params["ecc"]
    
    G = constants.G # constant of gravitation in m^3 kg^-1 s^-2
    
    per_s = params["per"] * 24. * 60. * 60. # in s
    
    msun = 1.989e30 # mass of the Sun in Kg
    
    mstar_kg = mstar*msun
    inc_rad = inc * np.pi/180. # inclination in radians
    p1 = (2. * np.pi * G / per_s)**(1/3)
    p3 = 1./umath.sqrt(1 - ecc*ecc)

    mp_kg = 0.
    for i in range(10) :
        p2 = umath.sin(inc_rad) / (mstar_kg + mp_kg)**(2/3)
        mp_kg = 1000. * k / (p1*p2*p3)
    
    if units == 'mjup':
        mjup = 1.898e27 # mass of Jupiter in Kg
        return mp_kg/mjup
    elif units == 'mnep':
        mnep = 1.024e26
        return mp_kg/mnep
    elif units == 'mearth' :
        mearth = 5.972e24
        return mp_kg/mearth
    elif units == 'msun' :
        return mp_kg/msun
    else :
        return mp_kg


def effective_radius(params, update_rp=True, spot_filling_factor=0.2, spot_relteff=0.86, wl0=600, wlf=1000, modulation_noise=0.05) :

    Teff = params["teff"].nominal_value
    
    bb_phot = BlackBody(temperature=Teff * u.K)
    bb_spot = BlackBody(temperature=Teff * spot_relteff * u.K)
    
    wav = np.arange(wl0*10, wlf*10) * u.AA
    
    flux_phot = bb_phot(wav)
    flux_spot = bb_spot(wav)

    int_flux_phot = np.trapz(flux_phot)
    int_flux_spot = np.trapz(flux_spot)

    spot_phot_flux_fraction = int_flux_phot / int_flux_spot
    
    rp_eff = params['rp'] * np.sqrt(1.0 - spot_filling_factor * (1. - 0.46))

    if update_rp :
        
        params["rp"] = rp_eff * umath.sqrt(ufloat(1.0,modulation_noise/2))
        return params
    else :
        return rp_eff


parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help='Input planet parameters',type='string',default="")
parser.add_option("-p", action="store_true", dest="plot", help="verbose",default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with get_derived_parameters.py -h "); sys.exit(1);

if options.verbose:
    print('Input planet parameters: ', options.input)

params = read_planet_prior(options.input)

params['k'] /= 1000.

params = effective_radius(params, update_rp=True)

for key in params.keys() :
    print(key,"=",params[key])
print("R [Rjup] = ", planet_radius(params, units='rjup'))
print("R [Rnep] = ", planet_radius(params, units='rnep'))
print("R [Rearth] = ", planet_radius(params, units='rearth'))
print("a/Rs = ",semi_major_axis(params, over_rs=True))
print("a [AU] = ",semi_major_axis(params, over_rs=False))
print("Transit duration [h]: ",transit_duration(params)*24)
print("Fit a [AU] = ",semi_major_axis(params, fitvalue=True, over_rs=False))
print("Fit transit duration [h]: ",transit_duration(params, fitvalue=True)*24)
print("Teq [K] = ",teq(params))
print("Mp [mjup] = ",planet_mass(params, params['k'], units='mjup'))
print("Mp [mnep] = ",planet_mass(params, params['k'], units='mnep'))
print("Mp [mearth] = ",planet_mass(params, params['k'], units='mearth'))
print("Density [g/cm^3] = ",planet_density(params, params['k']))
print("Impact parameter [Rs] = ",impact_parameter(params))
print("Fit impact parameter [Rs] = ",impact_parameter(params,fitvalue=True))

#masses = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
#masses = [0.04616-0.0409, 7.0*5.972e24/1.898e27,  0.04616, 0.04616+0.0409]
#masses = [0.0428-0.036, 7.0*5.972e24/1.898e27, 0.0428, 0.0428+0.036]


masses = np.array([0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03])
for mp in masses :
    k = rv_semi_amplitude(params, mp)[0]
    rho = planet_density(params, k)
    mpnep = mp * 1.898e27 / 1.024e26
    mpear = mp * 1.898e27 / 5.972e24
    print("Mp={0:.3f}Mjup, {1:.2f}Mnep, {2:.1f}Mearth -> K = {3} m/s  density = {4} g/cm^3".format(mp,mpnep,mpear,k,rho))
