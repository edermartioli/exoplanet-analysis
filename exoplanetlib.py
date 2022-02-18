# -*- coding: utf-8 -*-
"""
Created on Wed May 19 2021
@author: Eder Martioli
Institut d'Astrophysique de Paris, France.
"""

import numpy as np
from scipy import constants
import yaml


def load_params_from_yaml(filename) :
    yaml_file = open(filename)
    loc = yaml.load(yaml_file, Loader=yaml.FullLoader)
    
    if ('tt' in loc.keys()) and ('tp' not in loc.keys()) :
        loc['tp'] = timetrans_to_timeperi(loc['tt'], loc['per'], loc['ecc'], loc['om'])
    
    if ('tp' in loc.keys()) and ('tt' not in loc.keys()) :
        loc['tt'] = timeperi_to_timetrans(loc['tp'], loc['per'], loc['ecc'], loc['om'])

    return loc


def write_params_to_yaml(params, outfilename) :
    with open(outfilename, 'w') as outfile:
        yaml.dump(params, outfile, default_flow_style=False)


def rv_model(t, per, tp, ecc, om, k):
    """RV Drive
    Args:
        t (array of floats): times of observations (JD)
        per (float): orbital period (days)
        tp (float): time of periastron (JD)
        ecc (float): eccentricity
        om (float): argument of periatron (degree)s
        k (float): radial velocity semi-amplitude (m/s)
        
    Returns:
        rv: (array of floats): radial velocity model
    """

    omega = np.pi * om / 180.
    # Performance boost for circular orbits
    if ecc == 0.0:
        m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
        return k * np.cos(m + omega)

    if per < 0:
        per = 1e-4
    if ecc < 0:
        ecc = 0
    if ecc > 0.99:
        ecc = 0.99

    # Calculate the approximate eccentric anomaly, E1, via the mean anomaly  M.
    nu = true_anomaly(t, tp, per, ecc)
    rv = k * (np.cos(nu + omega) + ecc * np.cos(omega))

    return rv


def kepler(Marr, eccarr):
    """Solve Kepler's Equation
    Args:
        Marr (array): input Mean anomaly
        eccarr (array): eccentricity
    Returns:
        array: eccentric anomaly
    """

    conv = 1.0e-12  # convergence criterion
    k = 0.85

    Earr = Marr + np.sign(np.sin(Marr)) * k * eccarr  # first guess at E
    # fiarr should go to zero when converges
    fiarr = ( Earr - eccarr * np.sin(Earr) - Marr)
    convd = np.where(np.abs(fiarr) > conv)[0]  # which indices have not converged
    nd = len(convd)  # number of unconverged elements
    count = 0

    while nd > 0:  # while unconverged elements exist
        count += 1

        M = Marr[convd]  # just the unconverged elements ...
        ecc = eccarr[convd]
        E = Earr[convd]

        fi = fiarr[convd]  # fi = E - e*np.sin(E)-M    ; should go to 0
        fip = 1 - ecc * np.cos(E)  # d/dE(fi) ;i.e.,  fi^(prime)
        fipp = ecc * np.sin(E)  # d/dE(d/dE(fi)) ;i.e.,  fi^(\prime\prime)
        fippp = 1 - fip  # d/dE(d/dE(d/dE(fi))) ;i.e.,  fi^(\prime\prime\prime)

        # first, second, and third order corrections to E
        d1 = -fi / fip
        d2 = -fi / (fip + d1 * fipp / 2.0)
        d3 = -fi / (fip + d2 * fipp / 2.0 + d2 * d2 * fippp / 6.0)
        E = E + d3
        Earr[convd] = E
        fiarr = ( Earr - eccarr * np.sin( Earr ) - Marr) # how well did we do?
        convd = np.abs(fiarr) > conv  # test for convergence
        nd = np.sum(convd is True)

    if Earr.size > 1:
        return Earr
    else:
        return Earr[0]


def timetrans_to_timeperi(tc, per, ecc, om):
    """
    Convert Time of Transit to Time of Periastron Passage
    Args:
        tc (float): time of transit
        per (float): period [days]
        ecc (float): eccentricity
        omega (float): longitude of periastron (degree)
    Returns:
        float: time of periastron passage
    """
    try:
        if ecc >= 1:
            return tc
    except ValueError:
        pass
    omega = om * np.pi / 180.
    f = np.pi/2 - omega
    ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))  # eccentric anomaly
    tp = tc - per/(2*np.pi) * (ee - ecc*np.sin(ee))      # time of periastron

    return tp


def timeperi_to_timetrans(tp, per, ecc, om, secondary=False) :
    """
    Convert Time of Periastron to Time of Transit
    Args:
        tp (float): time of periastron
        per (float): period [days]
        ecc (float): eccentricity
        omega (float): argument of peri (radians)
        secondary (bool): calculate time of secondary eclipse instead
    Returns:
        float: time of inferior conjunction (time of transit if system is transiting)
    """
    try:
        if ecc >= 1:
            return tp
    except ValueError:
        pass

    omega = om * np.pi / 180.
    
    if secondary:
        f = 3*np.pi/2 - omega                                       # true anomaly during secondary eclipse
        ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))  # eccentric anomaly

        # ensure that ee is between 0 and 2*pi (always the eclipse AFTER tp)
        if isinstance(ee, np.float64):
            ee = ee + 2 * np.pi
        else:
            ee[ee < 0.0] = ee + 2 * np.pi
    else:
        f = np.pi/2 - omega                                         # true anomaly during transit
        ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))  # eccentric anomaly

    tc = tp + per/(2*np.pi) * (ee - ecc*np.sin(ee))         # time of conjunction

    return tc


def true_anomaly(t, tp, per, ecc):
    """
    Calculate the true anomaly for a given time, period, eccentricity.
    Args:
        t (array): array of times in JD
        tp (float): time of periastron, same units as t
        per (float): orbital period in days
        ecc (float): eccentricity
    Returns:
        array: true anomoly at each time
    """

    # f in Murray and Dermott p. 27
    m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
    eccarr = np.zeros(t.size) + ecc
    e1 = kepler(m, eccarr)
    n1 = 1.0 + ecc
    n2 = 1.0 - ecc
    nu = 2.0 * np.arctan((n1 / n2)**0.5 * np.tan(e1 / 2.0))

    return nu


def semi_major_axis(per, mstar, rstar, over_rs=False) :
    """
        Calculate the semi-major axis
        Args:
        per (float): orbital period in days
        mstar (float): stellar mass in Msun
        rstar (float): stellar radius in Rsun
        over_rs (bool): switch to output units in relative units to star radius
        Returns:
        a (float): semi-major axis in units of au
        """
        
    G = constants.G
    msun = 1.989e+30
    rsun = 696.34e6
    ms = mstar * msun
    rs = rstar * rsun
    d2s = 24.*60.*60.
    
    a = (((d2s*per * d2s*per * G * ms)/(4. * np.pi * np.pi))**(1/3))
    
    if over_rs :
        return a / rs
    else :
        au_m = 1.496e+11
        return a / au_m


def rv_semi_amplitude(per, inc, ecc, mstar, mp) :
    """
        Calculate the radial velocity semi-amplitude
        Args:
        per (float): orbital period (day)
        inc (float): orbital inclination (degree)
        ecc (float): eccentricity
        mstar (float): stellar mass (Msun)
        mp (float): planet mass (Mjup)
        Returns:
        ks,kp ([float,float]): ks : star radial velocity semi-amplitude
                               kp : planet radial velocity semi-amplitude
        """

    # mpsini and mp in jupiter mass
    # mstar in solar mass
    
    G = constants.G # constant of gravitation in m^3 kg^-1 s^-2
    
    per_s = per * 24. * 60. * 60. # in s
    
    mjup = 1.898e27 # mass of Jupiter in Kg
    msun = 1.989e30 # mass of the Sun in Kg
    
    mstar_kg = mstar*msun
    mp_kg = mp * mjup
    
    inc_rad = inc * np.pi/180. # inclination in radians
    
    p1 = (2. * np.pi * G / per_s)**(1/3)
    p2 = np.sin(inc_rad) / (mstar_kg + mp_kg)**(2/3)
    p3 = 1./np.sqrt(1 - ecc*ecc)
    
    ks = mp_kg * p1*p2*p3
    kp = mstar_kg * p1*p2*p3
    
    # return semi-amplitudes in km/s
    return ks, kp


def planet_mass(per, inc, ecc, mstar, k, units='') :
    """
        Calculate the planet mass
        Args:
        per (float): orbital period (day)
        inc (float): orbital inclination (degree)
        ecc (float): eccentricity
        mstar (float): stellar mass (Msun)
        k (float): radial velocity semi-amplitude (km/s)
        units (string) : define the output units for the planet mass
        supported units: 'mjup', 'mnep', 'mearth', 'msun', or in kg by default
        Returns:
        mp (float): planet mass
        """

    G = constants.G # constant of gravitation in m^3 kg^-1 s^-2
    
    per_s = per * 24. * 60. * 60. # in s

    msun = 1.989e30 # mass of the Sun in Kg
    
    mstar_kg = mstar*msun
    inc_rad = inc * np.pi/180. # inclination in radians
    p1 = (2. * np.pi * G / per_s)**(1/3)
    p3 = 1./np.sqrt(1 - ecc*ecc)

    mp_kg = 0.
    for i in range(10) :
        p2 = np.sin(inc_rad) / (mstar_kg + mp_kg)**(2/3)
        mp_kg = k / (p1*p2*p3)
    
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


def transit_duration(per, inc, ecc, om, rp, mstar, rstar):
    """
        Calculate the transit duration
        Args:
        per (float): orbital period (day)
        inc (float): orbital inclination (degree)
        ecc (float): eccentricity
        om (float): argument of periastron (degree)
        rp (float): planet radius (Rjup)
        mstar (float): stellar mass (Msun)
        rstar (float): stellar radius (Rsun)
        Returns:
        tdur (float): transit duration (days)
        """
    rsun = 696.34e6
    rs = rstar * rsun
    rjup = 69.911e6
    rp_over_rs = rp * rjup / rs
    
    sma_over_rs = semi_major_axis(per, mstar, rstar, over_rs=True)

    ww = om * np.pi / 180
    ii = inc * np.pi / 180
    ee = ecc
    aa = sma_over_rs
    ro_pt = (1 - ee ** 2) / (1 + ee * np.sin(ww))
    b_pt = aa * ro_pt * np.cos(ii)
    if b_pt > 1:
        b_pt = 0.5
    s_ps = 1.0 + rp_over_rs
    df = np.arcsin(np.sqrt((s_ps ** 2 - b_pt ** 2) / ((aa ** 2) * (ro_pt ** 2) - b_pt ** 2)))
    tdur = (per * (ro_pt ** 2)) / (np.pi * np.sqrt(1 - ee ** 2)) * df
    
    return tdur

