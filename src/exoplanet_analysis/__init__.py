"""
Exoplanet Analysis Tools
========================

A toolkit for the joint analysis of radial velocity (RV) and photometry data
to obtain the best-fit orbital and physical parameters of exoplanets.

Library modules
---------------
- ``exoplanetlib``   : transit and RV models (batman-based) and Keplerian orbits
- ``priorslib``      : priors and posteriors I/O and handling
- ``fitlib``         : optimization and MCMC (emcee) fitting engine
- ``gp_lib``         : Gaussian Process utilities (george / celerite)
- ``rvutils``        : radial velocity utilities and periodograms
- ``timeseries_lib`` : time series analysis utilities
- ``tess``           : TESS data access (MAST), light curves and limb darkening
- ``systemlib``      : planetary system parameters I/O in JSON format
- ``rmlib``          : Rossiter-McLaughlin effect model and helpers

Command-line tools are installed alongside this package; see the README
or run any of them with ``-h`` (e.g. ``transit_fit -h``, ``rv_fit -h``,
``transit_rv_fit -h``).

@author: Eder Martioli
"""

import importlib

__version__ = "1.6.0"

# Submodules are imported lazily (PEP 562) so that `import exoplanet_analysis`
# remains fast and does not require every optional heavy dependency upfront.
_SUBMODULES = (
    "datasets",
    "exoplanetlib",
    "fitlib",
    "gp_lib",
    "priorslib",
    "rmlib",
    "rvutils",
    "systemlib",
    "tess",
    "timeseries_lib",
)

__all__ = list(_SUBMODULES)


def __getattr__(name):
    """Getattr.

    Parameters
    ----------
    name
    """
    if name in _SUBMODULES:
        module = importlib.import_module("." + name, __name__)
        globals()[name] = module
        return module
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__():
    """Dir.
    """
    return sorted(list(globals().keys()) + list(_SUBMODULES))
