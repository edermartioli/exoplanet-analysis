# -*- coding: utf-8 -*-
"""
    Description: Rossiter-McLaughlin (RM) library.

    This module implements the classical (analytical) Rossiter-McLaughlin
    radial-velocity anomaly of Ohta, Taruya & Suto (2005), adapted to the
    parameter conventions of the exoplanet_analysis package (per-planet
    parameters carry a 3-digit index suffix, e.g. ``per_000``). It reuses the
    Keplerian machinery and eccentricity helpers of ``exoplanetlib`` and is
    designed to plug directly into the MCMC fitting infrastructure of
    ``fitlib``.

    The RM anomaly for planet ``planet_index`` is controlled by the following
    per-planet parameters, in addition to the standard orbital ones
    (``per``, ``tc``, ``k``, ``a``, ``inc``, ``rp``, ``esinw``, ``ecosw``):

    - ``lambda_{iii}``   : sky-projected spin-orbit obliquity [deg]
    - ``vsini_{iii}``    : projected stellar rotation velocity [same velocity
                           units as the input RVs, e.g. km/s or m/s]
    - ``omega_rm_{iii}`` : argument of periastron used in the RM geometry [deg]
                           (usually identical to the orbital omega)
    - ``ldc_{iii}``      : OPTIONAL dedicated linear limb-darkening coefficient
                           for the RM model. If ABSENT, the RM model shares the
                           transit limb darkening (the quadratic ``u0_{iii}`` /
                           ``u1_{iii}`` pair, converted to an effective linear
                           coefficient), meaning the RM data and the transit are
                           treated as the SAME bandpass. If PRESENT, the RM data
                           is treated as a DIFFERENT bandpass with its own
                           coefficient. See ``resolve_rm_ldc`` and
                           ``rm_ldc_report``. Different photometric bandpasses
                           are handled with per-instrument ``u0_inst``/``u1_inst``
                           coefficients, which the RM model can tie to via an
                           instrument (bandpass) index.

    @author: Eder Martioli, Shweta Dalal, Alexandre Teissier

    Original pyRM authors: E. Martioli, S. Dalal, A. Teissier
    (Institut d'Astrophysique de Paris).
"""

import numpy as np

from exoplanet_analysis import exoplanetlib


# RM-specific per-planet parameters (those not already present in the standard
# transit/RV parameter set). Used by priorslib to extend the planet parameter
# list and by the fit routines to recognize an RM model.
RM_PLANET_PARAM_IDS = ['lambda', 'vsini', 'omega_rm', 'ldc']


def quadratic_to_linear_ld(u0, u1):
    """Convert a quadratic limb-darkening pair to an effective linear coefficient.

    The transit model (batman) uses the quadratic law
    ``I(mu)/I(1) = 1 - u0 (1-mu) - u1 (1-mu)^2``, while the analytical RM model
    of Ohta et al. (2005) uses the linear law ``I(mu)/I(1) = 1 - eps (1-mu)``.
    When both describe the *same* stellar surface in the *same* bandpass, the
    single RM coefficient is taken as the flux-weighted equivalent linear
    coefficient of the quadratic law:

        eps = u0 + 2/3 u1

    This matches the disk-integrated limb-darkening of the quadratic law with a
    linear one and is the standard choice for feeding quadratic coefficients
    into the Ohta RM formula.

    Parameters
    ----------
    u0 : float
        Linear term of the quadratic limb-darkening law.
    u1 : float
        Quadratic term of the quadratic limb-darkening law.

    Returns
    -------
    float
        The equivalent linear (RM) limb-darkening coefficient.
    """
    return u0 + (2.0 / 3.0) * u1


def resolve_rm_ldc(planet_params, planet_index=0, instrum_params=None, instrum_index=None):
    """Resolve the RM linear limb-darkening coefficient ``eps``.

    The RM effect and the transit share the same stellar limb darkening. This
    function returns the linear coefficient the RM model should use, following a
    clear precedence that lets the user decide whether the RM and transit LD are
    tied together (same bandpass) or independent (different bandpass):

    1. If the planet has a dedicated ``ldc_{iii}`` parameter, use it directly
       (the RM data is treated as an independent bandpass with its own
       coefficient). This is the case, e.g., when the spectroscopic RM bandpass
       differs from the photometric one and the user wants a separate value.
    2. Otherwise, tie the RM coefficient to the transit limb darkening,
       converting the quadratic pair to an effective linear coefficient with
       ``quadratic_to_linear_ld``. The coefficients are taken from the matching
       bandpass: a per-instrument pair ``u0_inst/u1_inst`` when an instrument
       (bandpass) index is supplied, otherwise the planet-level ``u0/u1``.
    3. If no limb-darkening information is available at all, return 0.

    In other words: **omit** ``ldc_{iii}`` from the priors to make the RM model
    use the same limb darkening as the transit (same bandpass); **include**
    ``ldc_{iii}`` to give the RM data its own coefficient (different bandpass).

    Parameters
    ----------
    planet_params
        Dictionary of planet parameters (internal ids, e.g. per_000).
    planet_index : int, optional (default: 0)
        Index of the planet (0-based).
    instrum_params : dict, optional (default: None)
        Per-instrument (per-bandpass) parameters, containing ``u0_inst_{iiiii}``
        and ``u1_inst_{iiiii}``. When given together with ``instrum_index``, the
        transit LD of that bandpass is used to tie the RM coefficient.
    instrum_index : int, optional (default: None)
        Index of the instrument (bandpass) whose transit limb darkening should
        be tied to the RM model.

    Returns
    -------
    eps : float
        The linear limb-darkening coefficient for the RM model.
    tied : bool
        True if the coefficient was tied to the transit limb darkening, False
        if a dedicated ``ldc`` was used (or no LD was available).
    """
    ldc_key = 'ldc_{0:03d}'.format(planet_index)
    if ldc_key in planet_params:
        return planet_params[ldc_key], False

    # tie to the transit limb darkening of the matching bandpass
    if instrum_params is not None and instrum_index is not None:
        u0_key = 'u0_inst_{0:05d}'.format(instrum_index)
        u1_key = 'u1_inst_{0:05d}'.format(instrum_index)
        if u0_key in instrum_params and u1_key in instrum_params:
            return quadratic_to_linear_ld(instrum_params[u0_key], instrum_params[u1_key]), True

    u0_key = 'u0_{0:03d}'.format(planet_index)
    u1_key = 'u1_{0:03d}'.format(planet_index)
    if u0_key in planet_params:
        u0 = planet_params[u0_key]
        u1 = planet_params.get(u1_key, 0.0)
        return quadratic_to_linear_ld(u0, u1), True

    return 0.0, False


def has_rm_parameters(planet_params, planet_index=0):
    """Return True if the planet parameters contain RM parameters.

    Parameters
    ----------
    planet_params
        Dictionary of planet parameters (internal ids, e.g. per_000).
    planet_index : int, optional (default: 0)
        Index of the planet (0-based).

    Returns
    -------
    bool
        True if both ``lambda_{iii}`` and ``vsini_{iii}`` are present.
    """
    lam = 'lambda_{0:03d}'.format(planet_index)
    vsini = 'vsini_{0:03d}'.format(planet_index)
    return (lam in planet_params) and (vsini in planet_params)


def _gfunction(x, etap, gamma):
    """Auxiliary geometric function of the Ohta et al. (2005) RM model.

    Parameters
    ----------
    x : float or array
        Integration variable.
    etap : float or array
        Normalized planet-star separation term.
    gamma : float
        Planet-to-star radius ratio.

    Returns
    -------
    float or array
        Value of the auxiliary function.
    """
    return ((1. - x ** 2) * np.arcsin(np.sqrt((gamma ** 2 - (x - 1. - etap) ** 2) / (1. - x ** 2)))
            + np.sqrt((gamma ** 2 - (x - 1. - etap) ** 2) * (1. - x ** 2 - gamma ** 2 + (x - 1. - etap) ** 2)))


def rm_anomaly(dvrad, lbda, vsini, a_R, inc, r_R, omega, eps, ecc, anovraie, per, tau, t, phi0):
    """Compute the Rossiter-McLaughlin RV anomaly (Ohta et al. 2005).

    Parameters
    ----------
    dvrad : array
        Time derivative (sign) of the Keplerian RV, used to select the
        transit (approaching) branch.
    lbda : float
        Sky-projected spin-orbit obliquity [rad].
    vsini : float
        Projected stellar rotation velocity [velocity units of the RVs].
    a_R : float
        Scaled semi-major axis a/Rstar.
    inc : float
        Orbital inclination [rad].
    r_R : float
        Planet-to-star radius ratio Rp/Rstar.
    omega : float
        Argument of periastron for the RM geometry [rad].
    eps : float
        Linear limb-darkening coefficient.
    ecc : float
        Orbital eccentricity.
    anovraie : array
        True anomaly at each time.
    per : float
        Orbital period [d].
    tau : float
        Time of transit center [BJD].
    t : array
        Times of observation [BJD].
    phi0 : float
        Reference time (time of inferior conjunction/ephemeris zero) [BJD].

    Returns
    -------
    v : array
        RM radial-velocity anomaly at each time.

    Notes
    -----
    This is a direct port of the analytical model in the pyRM package. The
    argument of periastron is offset internally by +pi/2 to follow the Ohta
    convention.
    """
    # offsetting omega because of the Ohta specific convention
    omega = omega + np.pi / 2

    # computing the anomaly at the reference transit time
    phase0 = ((tau - phi0) % per) / per
    phase = np.where(np.less(phase0, 0), phase0 + 1, phase0)

    anomoy = 2. * np.pi * phase
    anoexc = anomoy * 1.

    anoexc1 = anoexc + (anomoy + ecc * np.sin(anoexc) - anoexc) / (1. - ecc * np.cos(anoexc))
    while np.max(np.abs(anoexc1 - anoexc)) > 1.e-8:
        anoexc = anoexc1 * 1.
        anoexc1 = anoexc + (anomoy + ecc * np.sin(anoexc) - anoexc) / (1. - ecc * np.cos(anoexc))

    anovraiep = 2. * np.arctan(np.sqrt((1. + ecc) / (1. - ecc)) * np.tan(anoexc / 2.))

    # anomaly offset required to properly center the transit
    offset = anovraiep - anovraie[np.argmin(np.abs(anovraie + omega - np.pi))]
    anovraie = anovraie - offset

    # planet position relative to the true anomaly
    rp = (a_R) * (1. - ecc ** 2) / (1. + ecc * np.cos(anovraie))
    xp = rp * (-np.cos(lbda) * np.sin(anovraie + omega) - np.sin(lbda) * np.cos(inc) * np.cos(anovraie + omega))
    zp = rp * (np.sin(lbda) * np.sin(anovraie + omega) - np.cos(lbda) * np.cos(inc) * np.cos(anovraie + omega))
    R = np.sqrt(xp ** 2 + zp ** 2)

    # RV anomaly computation
    ind1 = 1. - r_R
    ind2 = 1. + r_R
    g = r_R
    g2 = (r_R) ** 2
    v = np.zeros(len(R), 'd')

    for j in range(len(R)):
        # Ingress / egress phase
        if (R[j] > ind1) and (R[j] < ind2) and (dvrad[j] < 0.):
            n_p = R[j] - 1.
            x0 = 1. - (g2 - n_p ** 2) / (2. * (1. + n_p))
            z0 = np.sqrt(1. - x0 ** 2)
            zeta = 1. + n_p - x0
            xc = x0 + (zeta - g) / 2.
            w2 = np.sqrt(1. - (1. - g) ** 2)
            w3 = 0.
            w4 = (np.pi / 2.) * g * (g - zeta) * xc * w2 * _gfunction(xc, n_p, g) / _gfunction(1 - g, -g, g)
            v[j] = -1. * (vsini) * xp[j] * ((1. - eps) * (-1. * z0 * zeta + g2 * np.arccos(zeta / g)) + (eps * w4 / (1. + n_p))) / (np.pi * (1. - (1. / 3.) * eps) - (1. - eps) * (np.arcsin(z0) - (1. + n_p) * z0 + g2 * np.arccos(zeta / g)) - eps * w3)

        # Complete transit phase
        if (R[j] < ind1) and dvrad[j] < 0.:
            n_p = R[j] - 1.
            rho = n_p + 1.
            w2 = np.sqrt(1. - rho ** 2)
            w1 = 0.
            v[j] = -1 * (vsini) * xp[j] * g2 * (1. - eps * (1. - w2)) / (1. - g2 - eps * (1. / 3. - g2 * (1. - w1)))

        # Outside transit
        if R[j] > ind2:
            v[j] = 0.

    return v


def rm_rv_anomaly(time, planet_params, planet_index=0, instrum_params=None, instrum_index=None):
    """Compute the RM radial-velocity anomaly for one planet.

    This function reads the per-planet parameters from a planet parameters
    dictionary (using the standard ``_{iii}`` index convention), computes the
    Keplerian true anomaly with ``exoplanetlib``, and evaluates the RM
    anomaly. It does NOT include the Keplerian orbit itself (use
    ``exoplanetlib.rv_model`` or ``fitlib.calculate_rv_model_new`` for that).

    The limb-darkening coefficient of the RM model is resolved with
    ``resolve_rm_ldc``: it is tied to the transit limb darkening (converting the
    quadratic ``u0/u1`` pair of the matching bandpass to an effective linear
    coefficient) unless the planet has its own ``ldc_{iii}`` parameter, in which
    case the RM data is treated as an independent bandpass. See
    ``resolve_rm_ldc`` for details.

    Parameters
    ----------
    time
        Array of times [BJD].
    planet_params
        Dictionary of planet parameters (internal ids, e.g. per_000).
    planet_index : int, optional (default: 0)
        Index of the planet (0-based).
    instrum_params : dict, optional (default: None)
        Per-instrument (per-bandpass) parameters, used to tie the RM limb
        darkening to the transit limb darkening of a specific bandpass.
    instrum_index : int, optional (default: None)
        Index of the bandpass whose transit limb darkening is tied to the RM
        model.

    Returns
    -------
    array
        The RM radial-velocity anomaly at each input time (same velocity
        units as ``vsini``).
    """
    per = planet_params['per_{0:03d}'.format(planet_index)]
    tc = planet_params['tc_{0:03d}'.format(planet_index)]
    a_R = planet_params['a_{0:03d}'.format(planet_index)]
    inc = planet_params['inc_{0:03d}'.format(planet_index)] * np.pi / 180.
    r_R = planet_params['rp_{0:03d}'.format(planet_index)]

    esinw, ecosw = exoplanetlib.get_esinw_ecosw(planet_params, planet_index)
    ecc, om_deg = exoplanetlib.get_ecc_omg(esinw, ecosw)

    lbda = planet_params['lambda_{0:03d}'.format(planet_index)] * np.pi / 180.
    vsini = planet_params['vsini_{0:03d}'.format(planet_index)]

    # RM argument of periastron (defaults to the orbital omega if absent)
    omrm_key = 'omega_rm_{0:03d}'.format(planet_index)
    if omrm_key in planet_params:
        omega_rm = planet_params[omrm_key] * np.pi / 180.
    else:
        omega_rm = om_deg * np.pi / 180.

    # limb-darkening coefficient for the RM model, tied to the transit LD unless
    # a dedicated ldc is present (see resolve_rm_ldc)
    eps, _tied = resolve_rm_ldc(planet_params, planet_index=planet_index,
                                instrum_params=instrum_params, instrum_index=instrum_index)

    # time of periastron and the reference ephemeris time
    tp = exoplanetlib.timetrans_to_timeperi(tc, per, ecc, om_deg)
    phi0 = tc

    # Keplerian velocity (used only to get the true anomaly and its derivative
    # sign, which selects the transit branch of the RM effect)
    ksemi = planet_params['k_{0:03d}'.format(planet_index)]
    nu = exoplanetlib.true_anomaly(time, tp, per, ecc)
    vrad = ksemi * (np.cos(nu + om_deg * np.pi / 180.) + ecc * np.cos(om_deg * np.pi / 180.))
    dvrad = np.concatenate((np.array([vrad[1] - vrad[0]]), vrad[1:] - vrad[:-1]))

    v = rm_anomaly(dvrad, lbda, vsini, a_R, inc, r_R, omega_rm, eps, ecc, nu,
                   per, tc, time, phi0)
    return v


def rm_model(time, planet_params, planet_index=0, include_orbit=True, include_trend=False, instrum_params=None, instrum_index=None):
    """Compute the full RM radial-velocity model (Keplerian orbit + RM anomaly).

    Parameters
    ----------
    time
        Array of times [BJD].
    planet_params
        Dictionary of planet parameters (internal ids, e.g. per_000).
    planet_index : int, optional (default: 0)
        Index of the planet (0-based).
    include_orbit : bool, optional (default: True)
        Add the Keplerian orbital RV to the RM anomaly.
    include_trend : bool, optional (default: False)
        Add the systemic velocity and linear/quadratic RV trends.
    instrum_params : dict, optional (default: None)
        Per-instrument (per-bandpass) parameters, used to tie the RM limb
        darkening to the transit limb darkening of a specific bandpass.
    instrum_index : int, optional (default: None)
        Index of the bandpass whose transit limb darkening is tied to the RM
        model.

    Returns
    -------
    array
        The total RV model at each input time.
    """
    vel = np.zeros_like(time, dtype='float64')

    if include_orbit:
        per = planet_params['per_{0:03d}'.format(planet_index)]
        tc = planet_params['tc_{0:03d}'.format(planet_index)]
        esinw, ecosw = exoplanetlib.get_esinw_ecosw(planet_params, planet_index)
        ecc, om = exoplanetlib.get_ecc_omg(esinw, ecosw)
        ks = planet_params['k_{0:03d}'.format(planet_index)]
        tp = exoplanetlib.timetrans_to_timeperi(tc, per, ecc, om)
        vel = vel + exoplanetlib.rv_model(time, per, tp, ecc, om, ks)

        if include_trend:
            rv_sys = planet_params.get('rvsys_{0:03d}'.format(planet_index), 0.)
            trend = planet_params.get('trend_{0:03d}'.format(planet_index), 0.)
            quadtrend = planet_params.get('quadtrend_{0:03d}'.format(planet_index), 0.) / (1000 * (365.25 ** 2))
            vel = vel + rv_sys + time * trend + time * time * quadtrend

    vel = vel + rm_rv_anomaly(time, planet_params, planet_index=planet_index,
                              instrum_params=instrum_params, instrum_index=instrum_index)

    return vel


def rm_ldc_report(planet_params, planet_index=0, instrum_params=None, instrum_index=None):
    """Return a human-readable description of how the RM limb darkening is set.

    Use this to make explicit, for a given set of parameters, whether the RM
    model shares its limb darkening with the transit model (same bandpass) or
    uses an independent coefficient (different bandpass).

    Parameters
    ----------
    planet_params
        Dictionary of planet parameters (internal ids, e.g. per_000).
    planet_index : int, optional (default: 0)
        Index of the planet (0-based).
    instrum_params : dict, optional (default: None)
        Per-instrument (per-bandpass) transit LD parameters.
    instrum_index : int, optional (default: None)
        Bandpass index to tie the RM limb darkening to.

    Returns
    -------
    str
        A one-line description of the RM limb-darkening configuration.
    """
    eps, tied = resolve_rm_ldc(planet_params, planet_index=planet_index,
                               instrum_params=instrum_params, instrum_index=instrum_index)
    if not tied:
        if 'ldc_{0:03d}'.format(planet_index) in planet_params:
            return ("RM limb darkening: INDEPENDENT dedicated coefficient "
                    "ldc_{0:03d} = {1:.4f} (RM bandpass treated as different "
                    "from the transit bandpass).".format(planet_index, eps))
        return ("RM limb darkening: none available (eps = 0).")
    if instrum_params is not None and instrum_index is not None:
        return ("RM limb darkening: TIED to the transit limb darkening of "
                "bandpass/instrument {0} (u0_inst/u1_inst -> linear eps = "
                "{1:.4f}); same bandpass as that photometry.".format(instrum_index, eps))
    return ("RM limb darkening: TIED to the transit limb darkening "
            "(u0_{0:03d}/u1_{0:03d} -> linear eps = {1:.4f}); RM and transit "
            "share the same bandpass. Add an ldc_{0:03d} prior to use an "
            "independent coefficient instead.".format(planet_index, eps))


def rm_transit_duration(planet_params, planet_index=0):
    """Estimate the transit duration [days] from the planet parameters.

    Parameters
    ----------
    planet_params
        Dictionary of planet parameters (internal ids, e.g. per_000).
    planet_index : int, optional (default: 0)
        Index of the planet (0-based).

    Returns
    -------
    float
        Approximate total transit duration in days.
    """
    per = planet_params['per_{0:03d}'.format(planet_index)]
    a_R = planet_params['a_{0:03d}'.format(planet_index)]
    inc = planet_params['inc_{0:03d}'.format(planet_index)] * np.pi / 180.
    r_R = planet_params['rp_{0:03d}'.format(planet_index)]
    b = a_R * np.cos(inc)
    arg = ((1. + r_R) ** 2 - b ** 2)
    if arg <= 0:
        return 0.
    return (per / np.pi) * np.arcsin((1. / a_R) * np.sqrt(arg) / np.sin(inc))
