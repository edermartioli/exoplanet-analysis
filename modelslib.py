# -*- coding: utf-8 -*-
"""
Created on Wed November 4 2020
@author: Eder Martioli
Institut d'Astrophysique de Paris, France.
"""

import exoplanet as xo
import numpy as np
import batman

def aflare(t, p):
    """
    This is the Analytic Flare Model from the flare-morphology paper.
    Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    Note: this model assumes the flux before the flare is zero centered
    Note: many sub-flares can be modeled by this method by changing the
    number of parameters in "p". As a result, this routine may not work
    for fitting with methods like scipy.optimize.curve_fit, which require
    a fixed number of free parameters. Instead, for fitting a single peak
    use the aflare1 method.
    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    p : 1-d array
        p == [tpeak, fwhm (units of time), amplitude (units of flux)] x N
    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
    """
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    Nflare = int( np.floor( (len(p)/3.0) ) )
    #print(Nflare, p)
    flare = np.zeros_like(t)
    # compute the flare model for each flare
    for i in range(Nflare):
        outm = np.piecewise(t, [(t<= p[0+i*3]) * (t-p[0+i*3])/p[1+i*3] > -1.,
                                (t > p[0+i*3])],
                            [lambda x: (_fr[0]+                             # 0th order
                                        _fr[1]*((x-p[0+i*3])/p[1+i*3])+     # 1st order
                                        _fr[2]*((x-p[0+i*3])/p[1+i*3])**2.+  # 2nd order
                                        _fr[3]*((x-p[0+i*3])/p[1+i*3])**3.+  # 3rd order
                                        _fr[4]*((x-p[0+i*3])/p[1+i*3])**4. ),# 4th order
                             lambda x: (_fd[0]*np.exp( ((x-p[0+i*3])/p[1+i*3])*_fd[1] ) +
                                        _fd[2]*np.exp( ((x-p[0+i*3])/p[1+i*3])*_fd[3] ))]
                            ) * p[2+i*3] # amplitude
        flare = flare + outm

    return flare


def calib_model(n, i, params, time) :
    #polynomial model
    ncoefs = int(len(params) / n)
    coefs = []
    for c in range(int(ncoefs)):
        coeff_id = 'd{0:02d}c{1:1d}'.format(i,c)
        coefs.append(params[coeff_id])
    #p = np.poly1d(np.flip(coefs))
    p = np.poly1d(coefs)
    out_model = p(time)
    return out_model


def flares_model(flare_params, flare_tags, index, time) :
    n_flares = int(len(flare_params) / 3)
    pflares = []
    for i in range(n_flares) :
        if flare_tags[i] == index :
            tc_id = 'tc{0:04d}'.format(i)
            fwhm_id = 'fwhm{0:04d}'.format(i)
            amp_id = 'amp{0:04d}'.format(i)
            pflares.append(flare_params[tc_id])
            pflares.append(flare_params[fwhm_id])
            pflares.append(flare_params[amp_id])

    flare_model = aflare(time, pflares)
    return flare_model


def transit_model(time, planet_params, planet_index=0) :

    m_star=planet_params['ms_{0:03d}'.format(planet_index)]
    r_star=planet_params['rs_{0:03d}'.format(planet_index)]
    period = planet_params['per_{0:03d}'.format(planet_index)]
    tc = planet_params['tc_{0:03d}'.format(planet_index)]
    b = planet_params['b_{0:03d}'.format(planet_index)]
    rp = planet_params['rp_{0:03d}'.format(planet_index)]
    u0 = planet_params['u0_{0:03d}'.format(planet_index)]
    u1 = planet_params['u1_{0:03d}'.format(planet_index)]
    
    u = [u0,u1]
    
    # The light curve calculation requires an orbit
    orbit = xo.orbits.KeplerianOrbit(period=period, t0=tc, b=b, m_star=m_star, r_star=r_star)
        
    #if len(time[orbit.in_transit(time, r=rp)]) :
    
    # Compute a limb-darkened light curve using starry
    light_curve = (
                   xo.LimbDarkLightCurve(u)
                   .get_light_curve(orbit=orbit, r=rp, t=time)
                   .eval()
                   )
    out_light_curve = np.array(light_curve[:,0], dtype=float) + 1.
    
    return out_light_curve


def batman_transit_model(time, planet_params, planet_index=0) :

    """
        Function for computing transit models for the set of 8 free paramters
        x - time array
        """
    params = batman.TransitParams()
    
    params.per = planet_params['per_{0:03d}'.format(planet_index)]
    params.t0 = planet_params['tc_{0:03d}'.format(planet_index)]
    params.inc = planet_params['inc_{0:03d}'.format(planet_index)]
    
    params.a = planet_params['a_{0:03d}'.format(planet_index)]

    params.ecc = planet_params['ecc_{0:03d}'.format(planet_index)]
    params.w = planet_params['w_{0:03d}'.format(planet_index)]
    params.rp = planet_params['rp_{0:03d}'.format(planet_index)]
    u0 = planet_params['u0_{0:03d}'.format(planet_index)]
    u1 = planet_params['u1_{0:03d}'.format(planet_index)]
    params.u = [u0,u1]
    params.limb_dark = "quadratic"       #limb darkening model

    m = batman.TransitModel(params, time)    #initializes model
        
    flux_m = m.light_curve(params)          #calculates light curve

    return np.array(flux_m)


def batman_model(time, per, t0, a, inc, rp, u0, u1=0., ecc=0., w=90.) :
    
    """
        Function for computing transit models for the set of 8 free paramters
        x - time array
        """
    params = batman.TransitParams()
    
    params.per = per
    params.t0 = t0
    params.inc = inc
    params.a = a
    params.ecc = ecc
    params.w = w
    params.rp = rp
    params.u = [u0,u1]
    params.limb_dark = "quadratic"       #limb darkening model
    
    m = batman.TransitModel(params, time)    #initializes model
    
    flux_m = m.light_curve(params)          #calculates light curve
    
    return np.array(flux_m)


def semi_major_axis(period, mstar, rstar) :
    G = 6.67408e-11
    rsun = 696.34e6
    msun = 1.989e+30
    ms = mstar * msun
    rs = rstar * rsun
    d2s = 24.*60.*60.
    a = (((d2s*period * d2s*period * G * ms)/(4. * np.pi * np.pi))**(1/3)) / rs
    
    return a
