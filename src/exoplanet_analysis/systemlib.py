# -*- coding: utf-8 -*-
"""
    Description: Library to handle planetary system parameters in JSON format.

    This module defines a JSON representation of a full planetary system
    (star + planets) that serves both as *input* (priors) and *output*
    (posteriors) for all analysis modules in the package.

    Value formats accepted for each parameter:
      - [value, error]                : error null or 0 -> FIXED parameter;
                                        error > 0 -> Normal prior (for fittable
                                        parameters) or uncertainty carried
                                        through to derived quantities (for
                                        non-fittable parameters).
      - [value, err_minus, err_plus]  : posterior style, asymmetric errors.
      - {"value": v, "error": e,
         "prior": "Uniform"|"Normal"|"Normal_positive"|"Jeffreys"|"FIXED",
         "min": a, "max": b}          : full control over the prior.
      - null or [null, null]          : parameter unknown; a sensible FIXED
                                        default is used where one is required
                                        by the models.

    Units: times in BJD, periods in days, angles in degrees, RV quantities in
    km/s, stellar mass/radius in solar units, planet radius ratio Rp/Rstar,
    semi-major axis in units of Rstar (orbital_sma_rstar) or au
    (orbital_sma_au), stellar density in g/cm^3.

    @author: Eder Martioli
    """

import os
import json
import warnings
from copy import deepcopy

import numpy as np

PLANET_LETTERS = "bcdefghijklmnopqrstuvwxyz"

FORMAT_ID = "exoplanet_analysis system parameters"
SCHEMA_VERSION = "1.1"

# ----------------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------------

STAR_TEMPLATE = {
    "name": None,
    "object_type": "star",

    "_comment_stellar": "Fundamental stellar parameters: teff [K], mass [Msun], radius [Rsun], density [g/cm^3]",
    "teff": [None, None],
    "logg": [None, None],
    "mass": [None, None],
    "radius": [None, None],
    "density": [None, None],
    "metallicity": [None, None],
    "luminosity": [None, None],

    "_comment_rotation": "Rotation and spin geometry: vsini [km/s], angles [deg], period [d]",
    "vsini": [None, None],
    "stellar_inclination": [None, None],
    "differential_rotation": [None, None],
    "rotation_period_days": [None, None],

    "_comment_obliquity": "Spin-orbit alignment (projected, sky-plane) [deg]",
    "spinorbit_obliquity": [None, None],

    "_comment_ld": "Limb-darkening (quadratic law): [[u0, err], [u1, err]]",
    "limb_darkening_law": "quadratic",
    "limb_darkening_coeffs": [[None, None], [None, None]],

    "_comment_atm": "Atmospheric / spectroscopic [km/s]",
    "microturbulence": [None, None],
    "macroturbulence": [None, None],

    "_comment_activity": "Stellar activity",
    "log_rhk": [None, None],
    "prot_gyro_days": [None, None],
}

PLANET_TEMPLATE = {
    "name": None,
    "object_type": "planet",
    "transit": True,

    "_comment_size": "Planet dimensions: radius_rstar = Rp/Rstar; others derived",
    "radius_rstar": [None, None],
    "radius_jupiter": [None, None],
    "radius_earth": [None, None],
    "mass_jupiter": [None, None],
    "mass_earth": [None, None],
    "density_cgs": [None, None],
    "surface_gravity": [None, None],

    "_comment_orbit": "Keplerian orbital elements: sma in Rstar or au, period [d], angles [deg]",
    "orbital_sma_rstar": [None, None],
    "orbital_sma_au": [None, None],
    "orbital_period_days": [None, None],
    "orbital_ecc": [None, None],
    "orbital_omega": [None, None],
    "orbital_Omega": [None, None],
    "orbital_inc": [None, None],
    "impact_parameter": [None, None],
    "esinw": None,
    "ecosw": None,

    "_comment_obliquity": "Spin-orbit obliquity [deg] and Rossiter-McLaughlin controls",
    "spinorbit_obliquity": [None, None],
    "rm_omega": None,
    "rm_limb_darkening": None,

    "_comment_rv": "Radial-velocity observables [km/s]; trends in km/s/d and km/s/d^2",
    "rv_semi_amplitude": [None, None],
    "systemic_velocity": [None, None],
    "rv_linear_trend": [None, None],
    "rv_quadratic_trend": [None, None],

    "_comment_timing": "Transit and periastron timing (BJD_TDB)",
    "transit_time_bjd": [None, None],
    "time_of_periastron_bjd": [None, None],

    "_comment_transit_geometry": "Transit observables (derived)",
    "transit_depth": [None, None],
    "transit_duration_hours": [None, None],

    "_comment_ld": "Per-planet limb-darkening override: [[u0, err], [u1, err]] (default: star block)",
    "limb_darkening_coeffs": None,

    "_comment_atmosphere": "Atmospheric",
    "equilibrium_temperature": [None, None],
    "insolation_earth": [None, None],
    "transmission_spectrum_ref": None,
}

# Defaults used when a required internal parameter is not set in the JSON
PLANET_DEFAULTS = {
    "k": 0.0,
    "rvsys": 0.0,
    "trend": 0.0,
    "quadtrend": 0.0,
    "u0": 0.3,
    "u1": 0.2,
    "esinw": 0.0,
    "ecosw": 0.0,
    "rp": 0.01,
    "a": 10.0,
    "inc": 90.0,
    "tc": 0.0,
    "per": 1.0,
}

# JSON key -> internal id for star-level parameters. "fittable" means the
# parameter may become a free parameter of the fit (Normal prior) when an
# uncertainty is provided; non-fittable parameters are always written FIXED.
STAR_PARAM_MAP = {
    "teff": ("teff", False),
    "mass": ("ms", False),
    "radius": ("rs", False),
    "density": ("rhos", True),
}

# JSON key -> internal per-planet id (suffix _{index:03d} added). Special
# cases handled separately: eccentricity/omega -> esinw/ecosw, limb darkening,
# and the transit flag.
PLANET_PARAM_MAP = {
    "orbital_period_days": ("per", True),
    "transit_time_bjd": ("tc", True),
    "orbital_sma_rstar": ("a", True),
    "radius_rstar": ("rp", True),
    "orbital_inc": ("inc", True),
    "impact_parameter": ("b", True),
    "rv_semi_amplitude": ("k", True),
    "systemic_velocity": ("rvsys", True),
    "rv_linear_trend": ("trend", True),
    "rv_quadratic_trend": ("quadtrend", True),
    # Rossiter-McLaughlin: sky-projected spin-orbit obliquity (lambda)
    "spinorbit_obliquity": ("lambda", True),
}

# Inverse map: internal id -> JSON key (for posterior -> JSON updates)
INTERNAL_TO_STAR_KEY = {v[0]: k for k, v in STAR_PARAM_MAP.items()}
INTERNAL_TO_PLANET_KEY = {v[0]: k for k, v in PLANET_PARAM_MAP.items()}


# ----------------------------------------------------------------------------
# Template creation, loading and saving
# ----------------------------------------------------------------------------

def create_template(n_planets=1, system_name="MY-SYSTEM"):
    """Create a full system parameters template with all supported fields.

    Parameters
    ----------
    n_planets : int
        Number of planets in the system.
    system_name : str
        Name of the system (used to name the star and planets).

    Returns
    -------
    dict
        System parameters dictionary with all values set to null.
    """
    system = {
        "_format": FORMAT_ID,
        "_schema_version": SCHEMA_VERSION,
        "_comment": "Planetary system parameters. Format: [value, 1-sigma uncertainty] "
                    "or [value, err_minus, err_plus] (posteriors), or a dict "
                    "{'value','error','prior','min','max'} for full prior control. "
                    "null = unknown/not applicable.",
        "_reference": None,
        "system_name": system_name,
        "components": [],
    }
    star = deepcopy(STAR_TEMPLATE)
    star["name"] = system_name
    system["components"].append("star A")
    system["star A"] = star
    for i in range(n_planets):
        letter = PLANET_LETTERS[i % len(PLANET_LETTERS)]
        key = "planet {}".format(letter)
        planet = deepcopy(PLANET_TEMPLATE)
        planet["name"] = "{} {}".format(system_name, letter)
        system["components"].append(key)
        system[key] = planet
    return system


def load_system(filename):
    """Load a system parameters JSON file."""
    with open(filename, "r") as f:
        system = json.load(f)
    return system


def save_system(system, filename):
    """Save a system parameters dictionary to a JSON file."""
    with open(filename, "w") as f:
        json.dump(system, f, indent=4)
    return filename


def get_star_key(system):
    """Return the key of the (first) star component."""
    for key in system.get("components", []):
        block = system.get(key, {})
        if isinstance(block, dict) and block.get("object_type") == "star":
            return key
    # fallback: any block declaring itself a star
    for key, block in system.items():
        if isinstance(block, dict) and block.get("object_type") == "star":
            return key
    return None


def get_planet_keys(system):
    """Return the ordered list of planet component keys."""
    keys = []
    for key in system.get("components", []):
        block = system.get(key, {})
        if isinstance(block, dict) and block.get("object_type") == "planet":
            keys.append(key)
    if not keys:
        for key, block in system.items():
            if isinstance(block, dict) and block.get("object_type") == "planet":
                keys.append(key)
    return keys


# ----------------------------------------------------------------------------
# Parameter value parsing
# ----------------------------------------------------------------------------

def parse_value(entry):
    """Parse a JSON parameter entry into (value, error, prior_type, vmin, vmax).

    prior_type is None when not explicitly specified (i.e. the default
    FIXED/Normal rule applies).
    """
    if entry is None:
        return None, None, None, None, None
    if isinstance(entry, dict):
        value = entry.get("value", None)
        error = entry.get("error", None)
        prior = entry.get("prior", None)
        vmin = entry.get("min", None)
        vmax = entry.get("max", None)
        return value, error, prior, vmin, vmax
    if isinstance(entry, (list, tuple)):
        if len(entry) == 0 or entry[0] is None:
            return None, None, None, None, None
        value = float(entry[0])
        if len(entry) == 1 or entry[1] is None:
            return value, None, None, None, None
        if len(entry) >= 3 and entry[2] is not None:
            # posterior style: [value, err_minus, err_plus]
            error = (abs(float(entry[1])) + abs(float(entry[2]))) / 2.0
        else:
            error = abs(float(entry[1]))
        return value, error, None, None, None
    # scalar
    return float(entry), None, None, None, None


def _fmt(x):
    """Fmt.

    Parameters
    ----------
    x
        Array of x values.
    """
    return "{:.15g}".format(float(x))


def _pars_line(param_id, value, error, prior, vmin, vmax, fittable):
    """Build one .pars-format line for a parameter."""
    if prior is not None:
        ptype = str(prior)
        if ptype.upper() == "FIXED":
            return "{}\tFIXED\t{}\n".format(param_id, _fmt(value))
        if ptype in ("Uniform", "Jeffreys"):
            if vmin is None or vmax is None:
                raise ValueError("Parameter '{}': prior '{}' requires 'min' and 'max'".format(param_id, ptype))
            start = value if value is not None else 0.5 * (vmin + vmax)
            return "{}\t{}\t{},{},{}\n".format(param_id, ptype, _fmt(vmin), _fmt(vmax), _fmt(start))
        if ptype in ("Normal", "Normal_positive"):
            if error is None or error == 0:
                raise ValueError("Parameter '{}': prior '{}' requires a non-zero 'error'".format(param_id, ptype))
            return "{}\t{}\t{},{}\n".format(param_id, ptype, _fmt(value), _fmt(error))
        raise ValueError("Parameter '{}': unknown prior type '{}'".format(param_id, ptype))
    # default rule
    if fittable and error is not None and error > 0:
        return "{}\tNormal\t{},{}\n".format(param_id, _fmt(value), _fmt(error))
    return "{}\tFIXED\t{}\n".format(param_id, _fmt(value))


def _get_ld_coeffs(system, planet_block):
    """Return ((u0,e0,prior...), (u1,e1,prior...)) from planet or star block."""
    ld = planet_block.get("limb_darkening_coeffs", None)
    if ld is None:
        star_key = get_star_key(system)
        if star_key is not None:
            ld = system[star_key].get("limb_darkening_coeffs", None)
    if ld is None:
        return None, None
    u0 = parse_value(ld[0]) if len(ld) > 0 else (None,) * 5
    u1 = parse_value(ld[1]) if len(ld) > 1 else (None,) * 5
    return u0, u1


def _ecc_omega_to_esinw_ecosw(planet_block):
    """Convert (ecc, omega) [deg] with uncertainties into esinw/ecosw entries.

    Explicit 'esinw'/'ecosw' entries in the planet block take precedence.
    Returns two tuples (value, error, prior, vmin, vmax).
    """
    es_entry = planet_block.get("esinw", None)
    ec_entry = planet_block.get("ecosw", None)
    if es_entry is not None or ec_entry is not None:
        es = parse_value(es_entry) if es_entry is not None else (0.0, None, None, None, None)
        ec = parse_value(ec_entry) if ec_entry is not None else (0.0, None, None, None, None)
        return es, ec

    ecc, decc, eprior, emin, emax = parse_value(planet_block.get("orbital_ecc", None))
    w, dw, wprior, wmin, wmax = parse_value(planet_block.get("orbital_omega", None))
    if ecc is None:
        return (0.0, None, None, None, None), (0.0, None, None, None, None)
    if w is None:
        w, dw = 90.0, None
    wrad = np.deg2rad(w)
    es = ecc * np.sin(wrad)
    ec = ecc * np.cos(wrad)
    des = dec = None
    if (decc is not None and decc > 0) or (dw is not None and dw > 0):
        decc_ = decc if decc is not None else 0.0
        dw_ = np.deg2rad(dw) if dw is not None else 0.0
        des = np.sqrt((np.sin(wrad) * decc_) ** 2 + (ecc * np.cos(wrad) * dw_) ** 2)
        dec = np.sqrt((np.cos(wrad) * decc_) ** 2 + (ecc * np.sin(wrad) * dw_) ** 2)
        if des == 0:
            des = None
        if dec == 0:
            dec = None
    return (es, des, None, None, None), (ec, dec, None, None, None)


# ----------------------------------------------------------------------------
# JSON system -> .pars priors
# ----------------------------------------------------------------------------

def system_to_pars_lines(system, planet_to_analyze=-1):
    """Convert a system parameters dict into .pars-format prior lines.

    The output is fully compatible with priorslib.read_priors and all the
    fitting routines. Parameters with no value receive documented FIXED
    defaults; a warning is issued for the ones that affect the models.

    Parameters
    ----------
    system : dict
        System parameters (as returned by load_system or create_template).
    planet_to_analyze : int
        If >= 0, only this planet index is included.

    Returns
    -------
    list of str
        Lines in .pars format (including header).
    """
    lines = ["# Parameter_ID\tPrior_Type\tValues\n"]

    star_key = get_star_key(system)
    star = system.get(star_key, {}) if star_key else {}
    planet_keys = get_planet_keys(system)
    if planet_to_analyze >= 0:
        if planet_to_analyze >= len(planet_keys):
            raise ValueError("planet_to_analyze={} but system has {} planets".format(planet_to_analyze, len(planet_keys)))
        planet_keys = [planet_keys[planet_to_analyze]]
    n_planets = len(planet_keys)
    if n_planets == 0:
        raise ValueError("No planet components found in system parameters")

    # --- star-level parameters ---
    for json_key, (pid, fittable) in STAR_PARAM_MAP.items():
        value, error, prior, vmin, vmax = parse_value(star.get(json_key, None))
        if value is None:
            continue
        if pid == "rhos":
            # only use stellar density when some planet lacks orbital_sma_rstar
            planets_lack_a = any(
                parse_value(system[pk].get("orbital_sma_rstar", None))[0] is None
                for pk in planet_keys
            )
            if not planets_lack_a:
                continue
        lines.append(_pars_line(pid, value, error, prior, vmin, vmax, fittable))
    lines.append("n_planets\tFIXED\t{}\n".format(n_planets))

    # --- per-planet parameters ---
    for i, pk in enumerate(planet_keys):
        planet = system[pk]
        transit_flag = 1 if planet.get("transit", True) else 0
        lines.append("transit_{:03d}\tFIXED\t{}\n".format(i, transit_flag))

        # simple mapped parameters
        for json_key, (pid, fittable) in PLANET_PARAM_MAP.items():
            value, error, prior, vmin, vmax = parse_value(planet.get(json_key, None))
            param_id = "{}_{:03d}".format(pid, i)
            if value is None:
                if pid == "a":
                    # allow stellar density to define the scaled semi-major axis
                    if parse_value(star.get("density", None))[0] is not None:
                        continue
                if pid == "inc":
                    # allow impact parameter instead of inclination
                    if parse_value(planet.get("impact_parameter", None))[0] is not None:
                        continue
                if pid == "b":
                    continue  # b only written when explicitly provided
                if pid in PLANET_DEFAULTS:
                    default = PLANET_DEFAULTS[pid]
                    if pid in ("tc", "per", "rp") and transit_flag:
                        warnings.warn("Planet '{}': required parameter '{}' not set; using default {}".format(pk, json_key, default))
                    lines.append("{}\tFIXED\t{}\n".format(param_id, _fmt(default)))
                continue
            if pid == "b" and parse_value(planet.get("orbital_inc", None))[0] is not None:
                # inclination takes precedence over impact parameter
                continue
            lines.append(_pars_line(param_id, value, error, prior, vmin, vmax, fittable))

        # limb darkening
        u0, u1 = _get_ld_coeffs(system, planet)
        for name, u in (("u0", u0), ("u1", u1)):
            param_id = "{}_{:03d}".format(name, i)
            if u is None or u[0] is None:
                if transit_flag:
                    warnings.warn("Planet '{}': no limb-darkening coefficients set; using default {} = {}".format(pk, name, PLANET_DEFAULTS[name]))
                lines.append("{}\tFIXED\t{}\n".format(param_id, _fmt(PLANET_DEFAULTS[name])))
            else:
                lines.append(_pars_line(param_id, u[0], u[1], u[2], u[3], u[4], True))

        # eccentricity parameterization
        es, ec = _ecc_omega_to_esinw_ecosw(planet)
        lines.append(_pars_line("esinw_{:03d}".format(i), es[0] if es[0] is not None else 0.0, es[1], es[2], es[3], es[4], True))
        lines.append(_pars_line("ecosw_{:03d}".format(i), ec[0] if ec[0] is not None else 0.0, ec[1], ec[2], ec[3], ec[4], True))

        # Rossiter-McLaughlin parameters: emitted only when the planet has a
        # sky-projected obliquity set (i.e. an RM analysis is intended). The
        # obliquity itself (lambda) is already handled via PLANET_PARAM_MAP.
        lam_value = parse_value(planet.get("spinorbit_obliquity", None))[0]
        if lam_value is not None:
            # vsini is a stellar property but the RM model uses it per planet
            vs_value, vs_err, vs_prior, vs_min, vs_max = parse_value(star.get("vsini", None))
            if vs_value is not None:
                lines.append(_pars_line("vsini_{:03d}".format(i), vs_value, vs_err, vs_prior, vs_min, vs_max, True))
            # RM argument of periastron (defaults to the orbital omega)
            omrm_value, omrm_err, omrm_prior, omrm_min, omrm_max = parse_value(planet.get("rm_omega", None))
            if omrm_value is None:
                omrm_value = parse_value(planet.get("orbital_omega", None))[0]
                if omrm_value is None:
                    omrm_value = 90.0
                lines.append("omega_rm_{:03d}\tFIXED\t{}\n".format(i, _fmt(omrm_value)))
            else:
                lines.append(_pars_line("omega_rm_{:03d}".format(i), omrm_value, omrm_err, omrm_prior, omrm_min, omrm_max, True))
            # RM limb-darkening coefficient (defaults to the linear transit u0)
            ldc_value, ldc_err, ldc_prior, ldc_min, ldc_max = parse_value(planet.get("rm_limb_darkening", None))
            if ldc_value is None:
                u0, _ = _get_ld_coeffs(system, planet)
                if u0 is not None and u0[0] is not None:
                    lines.append(_pars_line("ldc_{:03d}".format(i), u0[0], u0[1], u0[2], u0[3], u0[4], True))
            else:
                lines.append(_pars_line("ldc_{:03d}".format(i), ldc_value, ldc_err, ldc_prior, ldc_min, ldc_max, True))

    return lines


def system_to_pars(system, output):
    """Write a system parameters dict (or JSON file path) to a .pars priors file."""
    if isinstance(system, str):
        system = load_system(system)
    lines = system_to_pars_lines(system)
    with open(output, "w") as f:
        f.writelines(lines)
    return output


# ----------------------------------------------------------------------------
# Posterior (.pars) -> JSON system update
# ----------------------------------------------------------------------------

def _read_posterior_params(posterior_file):
    """Read a posterior/priors .pars file into {id: (value, error, prior_type)}."""
    from exoplanet_analysis import priorslib
    priors = priorslib.read_priors(posterior_file)
    out = {}
    for key in priors.keys():
        if key.endswith("_err") or key.endswith("_pdf"):
            continue
        entry = priors[key]
        if not isinstance(entry, dict) or "object" not in entry:
            continue
        ptype = entry.get("type", "FIXED")
        value = float(entry["object"].value)
        error = None
        err_key = "{}_err".format(key)
        if ptype in ("Normal", "Normal_positive") and err_key in priors:
            error = float(priors[err_key][1])
        elif ptype in ("Uniform", "Jeffreys") and err_key in priors:
            error = abs(float(priors[err_key][1]) - float(priors[err_key][0])) / 2.0
        out[key] = (value, error, ptype)
    return out


def _set_json_value(block, json_key, value, error, ptype, overwrite_fixed=False):
    """Update a [value, error] JSON field from a posterior parameter."""
    current = block.get(json_key, None)
    if ptype == "FIXED" and not overwrite_fixed:
        # fixed parameters were inputs; only fill in missing values
        cur_val = parse_value(current)[0] if not isinstance(current, dict) else current.get("value")
        if cur_val is not None:
            return
        block[json_key] = [value, None]
        return
    block[json_key] = [value, error]


def update_system_from_posterior(system, posterior_files, planet_indexes=None, overwrite_fixed=False):
    """Update a system parameters dict with values from posterior .pars file(s).

    Fitted parameters (Normal posteriors) update the corresponding JSON fields
    with [value, error]; FIXED posterior entries only fill fields that are
    still null. Internal parameters without a JSON mapping (calibration
    coefficients, GP hyperparameters, jitter, etc.) are stored under the
    top-level "fitted_parameters" block so no information is lost.

    Parameters
    ----------
    system : dict or str
        System parameters dict or path to a system JSON file.
    posterior_files : str or list of str
        Posterior .pars file(s) as written by the fitting routines.
    planet_indexes : list of int, optional
        Map from the posterior planet indexes to the system planet order
        (default: identity).
    overwrite_fixed : bool
        If True, FIXED posterior values overwrite existing JSON values.

    Returns
    -------
    dict
        The updated system parameters dict.
    """
    if isinstance(system, str):
        system = load_system(system)
    if isinstance(posterior_files, str):
        posterior_files = [posterior_files]

    star_key = get_star_key(system)
    planet_keys = get_planet_keys(system)

    params = {}
    for pf in posterior_files:
        params.update(_read_posterior_params(pf))

    fitted_ld = {}
    esinw_ecosw = {}

    for pid, (value, error, ptype) in params.items():
        # star-level parameters
        if pid in INTERNAL_TO_STAR_KEY and star_key is not None:
            _set_json_value(system[star_key], INTERNAL_TO_STAR_KEY[pid], value, error, ptype, overwrite_fixed)
            continue
        if pid == "n_planets":
            continue
        # per-planet parameters: name_{index:03d}
        base, sep, idx_str = pid.rpartition("_")
        if sep and idx_str.isdigit() and len(idx_str) == 3:
            idx = int(idx_str)
            if planet_indexes is not None:
                idx = planet_indexes[idx] if idx < len(planet_indexes) else idx
            if idx < len(planet_keys):
                planet = system[planet_keys[idx]]
                if base in INTERNAL_TO_PLANET_KEY:
                    _set_json_value(planet, INTERNAL_TO_PLANET_KEY[base], value, error, ptype, overwrite_fixed)
                    continue
                if base in ("u0", "u1"):
                    fitted_ld.setdefault(idx, {})[base] = (value, error, ptype)
                    continue
                if base in ("esinw", "ecosw"):
                    esinw_ecosw.setdefault(idx, {})[base] = (value, error, ptype)
                    continue
                if base == "transit":
                    planet["transit"] = bool(int(value))
                    continue
        # anything else: preserve in the fitted_parameters block
        if ptype != "FIXED" or overwrite_fixed:
            system.setdefault("fitted_parameters", {})[pid] = [value, error]

    # limb darkening -> planet block (and star block for single-planet systems)
    for idx, ld in fitted_ld.items():
        planet = system[planet_keys[idx]]
        u0 = ld.get("u0", (None, None, "FIXED"))
        u1 = ld.get("u1", (None, None, "FIXED"))
        if any(v[2] != "FIXED" for v in (u0, u1)) or planet.get("limb_darkening_coeffs") is None:
            coeffs = [[u0[0], u0[1]], [u1[0], u1[1]]]
            planet["limb_darkening_coeffs"] = coeffs
            if len(planet_keys) == 1 and star_key is not None:
                system[star_key]["limb_darkening_coeffs"] = coeffs

    # esinw/ecosw -> eccentricity and omega (with error propagation)
    for idx, ee in esinw_ecosw.items():
        planet = system[planet_keys[idx]]
        es = ee.get("esinw", (0.0, None, "FIXED"))
        ec = ee.get("ecosw", (0.0, None, "FIXED"))
        fitted = (es[2] != "FIXED") or (ec[2] != "FIXED")
        if not fitted and not overwrite_fixed:
            continue
        try:
            from uncertainties import ufloat
            from uncertainties import umath
            ues = ufloat(es[0], es[1] if es[1] else 0.0)
            uec = ufloat(ec[0], ec[1] if ec[1] else 0.0)
            uecc = umath.sqrt(ues ** 2 + uec ** 2)
            ecc_val, ecc_err = uecc.nominal_value, uecc.std_dev
            if ecc_val > 0:
                uw = umath.atan2(ues, uec) * 180.0 / np.pi
                w_val, w_err = uw.nominal_value, uw.std_dev
            else:
                w_val, w_err = 90.0, None
        except Exception:
            ecc_val = float(np.sqrt(es[0] ** 2 + ec[0] ** 2))
            ecc_err = None
            w_val = float(np.rad2deg(np.arctan2(es[0], ec[0]))) if ecc_val > 0 else 90.0
            w_err = None
        planet["orbital_ecc"] = [ecc_val, ecc_err if (ecc_err and ecc_err > 0) else None]
        planet["orbital_omega"] = [w_val, w_err if (w_err and w_err > 0) else None]
        planet["esinw"] = [es[0], es[1]]
        planet["ecosw"] = [ec[0], ec[1]]

    system.setdefault("_posterior_files", [])
    for pf in posterior_files:
        if pf not in system["_posterior_files"]:
            system["_posterior_files"].append(pf)

    return system


def export_system_posterior(planet_priors_file, planet_posterior_file, output=None, verbose=False):
    """If the planet priors input was a system JSON file, write an updated
    system JSON with the fitted posterior values (including derived planet
    parameters when possible). No-op for .pars priors inputs.

    This is called automatically by the fitting routines after saving the
    planet posterior .pars file.
    """
    if not str(planet_priors_file).lower().endswith(".json"):
        return None
    try:
        system = load_system(planet_priors_file)
        system = update_system_from_posterior(system, planet_posterior_file)
        try:
            system = compute_derived_parameters(system)
        except Exception as e:
            warnings.warn("Could not compute derived parameters: {}".format(e))
        if output is None:
            from exoplanet_analysis import priorslib
            output = priorslib.derive_filename(planet_priors_file, "_posterior.json")
        save_system(system, output)
        if verbose:
            print("Output SYSTEM posterior (JSON): ", output)
        return output
    except Exception as e:
        warnings.warn("Could not export system JSON posterior: {}".format(e))
        return None


# ----------------------------------------------------------------------------
# Derived physical parameters
# ----------------------------------------------------------------------------

def _ufloat_from_entry(entry):
    """Ufloat from entry.

    Parameters
    ----------
    entry
    """
    from uncertainties import ufloat
    value, error, _, _, _ = parse_value(entry)
    if value is None:
        return None
    return ufloat(value, error if error else 0.0)


def compute_derived_parameters(system, geom_albedo=0.1, f=0.5):
    """Fill in derived planet parameters (radius, mass, density, surface
    gravity, semi-major axis in au, equilibrium temperature, impact parameter,
    transit duration) from the fitted/known parameters, propagating
    uncertainties.

    Reuses the physics functions from the get_derived_parameters module.
    Fields that cannot be computed (missing inputs) are left unchanged.
    """
    from uncertainties import ufloat
    from exoplanet_analysis.scripts import get_derived_parameters as gdp

    if isinstance(system, str):
        system = load_system(system)

    star_key = get_star_key(system)
    star = system.get(star_key, {}) if star_key else {}
    rsun_m = 696.34e6
    au_m = 1.495978707e11

    for pk in get_planet_keys(system):
        planet = system[pk]
        params = {}
        for name, entry in (("teff", star.get("teff")), ("ms", star.get("mass")), ("rs", star.get("radius"))):
            u = _ufloat_from_entry(entry)
            if u is not None:
                params[name] = u
        for name, jkey in (("per", "orbital_period_days"), ("tc", "transit_time_bjd"),
                           ("a", "orbital_sma_rstar"), ("rp", "radius_rstar"),
                           ("inc", "orbital_inc"), ("b", "impact_parameter"),
                           ("k", "rv_semi_amplitude")):
            u = _ufloat_from_entry(planet.get(jkey))
            if u is not None:
                params[name] = u
        ecc = _ufloat_from_entry(planet.get("orbital_ecc"))
        w = _ufloat_from_entry(planet.get("orbital_omega"))
        params["ecc"] = ecc if ecc is not None else ufloat(0.0, 0.0)
        params["w"] = w if w is not None else ufloat(90.0, 0.0)

        # The RV semi-amplitude is stored (and fitted) in m/s, but the derived-
        # parameter physics functions (planet_mass, planet_density) expect the
        # semi-amplitude in km/s (they convert to m/s internally). Convert here
        # so mass and density come out in the correct units.
        if "k" in params:
            params["k"] = params["k"] / 1000.0

        def _store(json_key, u):
            """Store.

            Parameters
            ----------
            json_key
            u
            """
            if u is not None:
                err = u.std_dev if u.std_dev > 0 else None
                planet[json_key] = [u.nominal_value, err]

        def _null(entry):
            """Null.

            Parameters
            ----------
            entry
            """
            return parse_value(planet.get(entry))[0] is None

        try:
            if "rp" in params and "rs" in params:
                if _null("radius_jupiter"):
                    _store("radius_jupiter", gdp.planet_radius(params, units="rjup"))
                if _null("radius_earth"):
                    _store("radius_earth", gdp.planet_radius(params, units="rearth"))
                if _null("transit_depth"):
                    _store("transit_depth", params["rp"] ** 2)
        except Exception as e:
            warnings.warn("Planet '{}': could not derive radius: {}".format(pk, e))

        try:
            if "a" in params and "rs" in params and _null("orbital_sma_au"):
                _store("orbital_sma_au", params["a"] * params["rs"] * rsun_m / au_m)
        except Exception as e:
            warnings.warn("Planet '{}': could not derive sma [au]: {}".format(pk, e))

        try:
            if "a" in params and "inc" in params and _null("impact_parameter"):
                from uncertainties import umath
                _store("impact_parameter", params["a"] * umath.cos(params["inc"] * np.pi / 180.0))
        except Exception as e:
            warnings.warn("Planet '{}': could not derive impact parameter: {}".format(pk, e))

        has_k = "k" in params and params["k"].nominal_value > 0
        try:
            if has_k and "ms" in params and "inc" in params and "per" in params:
                if _null("mass_jupiter"):
                    _store("mass_jupiter", gdp.planet_mass(params, params["k"], units="mjup"))
                if _null("mass_earth"):
                    _store("mass_earth", gdp.planet_mass(params, params["k"], units="mearth"))
                if "rp" in params and "rs" in params:
                    if _null("density_cgs"):
                        _store("density_cgs", gdp.planet_density(params, params["k"]))
                    if _null("surface_gravity"):
                        from scipy import constants as sciconst
                        mjup, rjup = 1.898e27, 69911000.0
                        mp = gdp.planet_mass(params, params["k"], units="mjup") * mjup
                        rp = gdp.planet_radius(params, units="rjup") * rjup
                        _store("surface_gravity", sciconst.G * mp / rp ** 2)
        except Exception as e:
            warnings.warn("Planet '{}': could not derive mass/density: {}".format(pk, e))

        try:
            if "teff" in params and "ms" in params and "per" in params and _null("equilibrium_temperature"):
                _store("equilibrium_temperature", gdp.teq(params, geom_albedo=geom_albedo, f=f))
        except Exception as e:
            warnings.warn("Planet '{}': could not derive equilibrium temperature: {}".format(pk, e))

        try:
            need = ("per", "rp", "a", "inc", "ms")
            if all(n in params for n in need) and _null("transit_duration_hours"):
                _store("transit_duration_hours", gdp.transit_duration(params) * 24.0)
        except Exception as e:
            warnings.warn("Planet '{}': could not derive transit duration: {}".format(pk, e))

    return system
