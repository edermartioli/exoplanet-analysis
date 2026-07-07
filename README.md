# Exoplanet Analysis Tools

A Python toolkit for the **joint analysis of radial velocity (RV) and photometry data** to obtain the best-fit orbital and physical parameters of exoplanets.

The package provides a set of library modules for transit and RV modeling, priors handling, MCMC fitting (via `emcee`), and Gaussian Process modeling of stellar activity, together with a suite of ready-to-use command-line tools for common analysis workflows, including direct access to TESS data through MAST.

Developed by Eder Martioli — Laboratório Nacional de Astrofísica (LNA), Brazil & Institut d'Astrophysique de Paris (IAP), France.

## Features

- Joint MCMC fitting of transits and radial velocities with flexible priors
- Rossiter-McLaughlin effect modeling (Ohta et al. 2005) — sky-projected spin-orbit obliquity and stellar v sin i, both as a stand-alone RM fit and within the joint photometry + RV fit
- Transit models based on `batman`, Keplerian RV orbits, and linear ephemerides
- Gaussian Process modeling of stellar activity (`george` / `celerite`), including quasi-periodic kernels
- Generalized Lomb-Scargle (GLS) and BLS periodogram analyses
- Automatic retrieval of TESS light curves and DVT products from MAST (`astroquery` / `lightkurve`)
- TESS limb-darkening coefficients from the Claret (2017) tables (shipped with the package)
- Transit timing variation (TTV) analysis
- Derivation of physical planet parameters (masses, radii, densities, equilibrium temperatures) with uncertainties
- Support for ground-based photometry pipelines (OPD/LNA and SPARC4 instruments) and SOPHIE RV data

## Installation

### From a local clone

```bash
git clone https://github.com/edermartioli/exoplanet-analysis.git
cd exoplanet-analysis
pip install -U .
```

For development (editable install):

```bash
pip install -U -e .
```

All dependencies (`numpy`, `scipy`, `matplotlib`, `astropy`, `astroquery`, `emcee`, `corner`, `batman-package`, `celerite`, `george`, `lightkurve`, `PyAstronomy`, `uncertainties`, `mpmath`, `scikit-learn`, `h5py`) are installed automatically.

> **Note:** `batman-package`, `celerite`, and `george` compile C/C++ extensions on some platforms, so a C compiler may be required if pre-built wheels are not available for your system.

## Package structure

```
src/exoplanet_analysis/
├── exoplanetlib.py     # Transit models (batman), Keplerian RV orbits
├── priorslib.py        # Priors/posteriors definition, I/O, and handling
├── fitlib.py           # Optimization (OLS) and MCMC (emcee) fitting engine
├── gp_lib.py           # Gaussian Process utilities (george / celerite)
├── rvutils.py          # RV utilities, periodograms, phase folding
├── timeseries_lib.py   # Time series analysis utilities
├── tess.py             # TESS/MAST data access, light curves, limb darkening
├── systemlib.py        # Planetary system parameters I/O in JSON format
├── rmlib.py            # Rossiter-McLaughlin effect model (Ohta et al. 2005)
├── data/               # Packaged data (Claret LDC table, TESS objects database)
└── scripts/            # Command-line tools (see below)
```

## Library usage

```python
from exoplanet_analysis import fitlib, priorslib, exoplanetlib, tess, gp_lib

# Load planet priors
planet_priors = priorslib.read_priors("planet.pars")

# Compute a batman transit model from a planet parameters dictionary
flux_model = exoplanetlib.batman_transit_model(time, planet_params, planet_index=0)

# Bin a light curve
bin_time, bin_flux, bin_err = fitlib.bin_data(time, flux, fluxerr, binsize=0.1)

# Retrieve TESS limb-darkening coefficients (Claret 2017)
u0, u1 = tess.get_claret_ld_coeffs(teff, logg=logg, zsun=feh)
```

Submodules are imported lazily, so `import exoplanet_analysis` is fast and heavy optional dependencies are only loaded when the corresponding module is used.

## System parameters in JSON (priors in, posteriors out)

All modules accept a **planetary system JSON file** as priors input and write fitted posteriors back to JSON. The JSON format describes the full system (star + planets) and supports **all system parameters**, including those not determined by the fitted model (see `examples/system_template.json` for the complete template and `examples/WASP-108_system_example.json` for a filled example).

```bash
# create a template with all supported parameters
system_params --create_template=MY-SYSTEM.json --n_planets=2 --name="MY-SYSTEM"
```

Each parameter takes the form `[value, 1-sigma]`. A parameter with `null` or `0` uncertainty is held **FIXED**; a fittable parameter with a non-zero uncertainty gets a **Normal** prior. Full prior control is available with the dict form `{"value": v, "error": e, "prior": "Uniform", "min": a, "max": b}` (supported priors: `FIXED`, `Normal`, `Normal_positive`, `Uniform`, `Jeffreys`). Units: times in BJD, periods in days, angles in degrees, RVs in km/s, stellar mass/radius in solar units.

The JSON file can be passed anywhere a `.pars` priors file is accepted — the conversion is transparent:

```bash
transit_fit --object="WASP-108" --planet_priors=WASP-108_system.json -vp
```

After fitting, three outputs are produced: the usual posterior `.pars` file, a JSON mirror of it, and — when the priors input was a system JSON — a **full system posterior JSON** (`*_posterior.json`) with fitted values `[value, error]`, plus automatically derived physical parameters (planet mass, radius, density, surface gravity, semi-major axis in au, equilibrium temperature, impact parameter, transit duration) with propagated uncertainties. Internal fitted parameters without a system-level meaning (calibration coefficients, GP hyperparameters, jitter) are preserved under a `"fitted_parameters"` block.

Conversions and merging are also available from the command line and the library API:

```bash
# convert a system JSON into a .pars priors file
system_params --input=MY-SYSTEM.json --to_pars=MY-SYSTEM.pars

# merge fitted posteriors into a system file and compute derived parameters
system_params --input=MY-SYSTEM.json --posteriors=MY-SYSTEM_posterior.pars --derive --output=MY-SYSTEM_updated.json
```

```python
from exoplanet_analysis import systemlib

system = systemlib.create_template(n_planets=2, system_name="MY-SYSTEM")
systemlib.save_system(system, "MY-SYSTEM.json")
system = systemlib.update_system_from_posterior(system, "MY-SYSTEM_posterior.pars")
system = systemlib.compute_derived_parameters(system)
```

## Rossiter-McLaughlin effect

The package models the classical (analytical) Rossiter-McLaughlin effect of
Ohta, Taruya & Suto (2005) through the `rmlib` module, which reuses the same Keplerian
and eccentricity machinery as the rest of the package. The RM anomaly adds three
per-planet parameters to the standard set: the sky-projected obliquity `lambda_{iii}`
[deg], the projected stellar rotation velocity `vsini_{iii}` (in the same velocity units
as the input RVs), and, optionally, `omega_rm_{iii}` and `ldc_{iii}`.

An **RM-only fit** (from RVs taken across a transit, with the orbit and geometry fixed)
reuses the RV-fit MCMC infrastructure:

```python
from exoplanet_analysis import fitlib

priors = fitlib.read_rm_priors("WASP-108_rm.pars", n_rvdatasets)
posterior = fitlib.guess_rvcalib(priors, bjds, rvs, prior_type="Normal")
posterior = fitlib.fitRMWithMCMC(bjds, rvs, rverrs, posterior,
                                 nwalkers=32, niter=2000, burnin=600,
                                 samples_filename="WASP-108_rm_samples.h5", verbose=True)
lam = posterior["planet_params"]["lambda_000"]
vsini = posterior["planet_params"]["vsini_000"]
```

or from the command line:

```bash
rm_fit --rvdata="WASP-108_ghost_*.rdb" --planet_priors=WASP-108_rm.pars --rv_units=kmps -v
```

The RM anomaly can also be added to the **joint transit + RV fit** by passing
`include_rm=True` to `fitlib.fitTransitsAndRVsWithMCMC`, so the photometry, orbital RVs,
and RM RVs are modelled together and share the geometry. RM parameters are written to the
output posterior files (both `.pars` and JSON) and round-trip through the JSON
system-parameters format (set a planet's `spinorbit_obliquity` and the star's `vsini`).
See `notebooks/05_rossiter_mclaughlin.ipynb` for a complete worked example.

### Limb darkening (shared with the transit or independent)

The RM effect and the transit are darkened by the same stellar limb darkening, so the
package links them explicitly. The transit uses a quadratic law (`u0`, `u1`); the analytical
RM model uses a linear coefficient. If a planet has **no** `ldc_{iii}` parameter, the RM
model derives its coefficient from the transit `u0/u1` (via `eps = u0 + 2/3 u1`), i.e. the
RM data and the transit are treated as the **same bandpass**. If a planet **has** an
`ldc_{iii}` parameter, the RM data is treated as an **independent bandpass** with its own
coefficient. For different photometric bandpasses (e.g. TESS vs a ground-based I band), the
transit fit's per-instrument `u0_inst`/`u1_inst` coefficients are used, and the RM model can
be tied to a specific bandpass by index. `rmlib.rm_ldc_report(...)` states which case is in
effect, and the fit prints it when `verbose=True`.

## Command-line tools

After installation the following commands are available in your shell (run any of them with `-h` for the full list of options):

| Command | Description |
|---|---|
| `transit_fit` | Fit planetary transit data using MCMC |
| `rv_fit` | Fit the orbit of a planetary system to RV data using MCMC |
| `transit_rv_fit` | Joint fit of transits and RVs using MCMC |
| `transit_fit_gp` | Transit fit with a Gaussian Process baseline |
| `rv_fit_detrend` | RV orbit fit with detrending |
| `transits_analysis` | Transit analysis pipeline |
| `detect_transits` | GP detrending + BLS transit search in TESS DVT light curves |
| `gls_analysis` | Generalized Lomb-Scargle periodogram analysis |
| `gp_analysis` | Gaussian Process analysis of RV/activity time series |
| `mcmc_analysis` | Analysis of a saved MCMC chain (convergence, corner plots) |
| `time_series_analysis` | Time series analysis of RV and activity indicators |
| `time_series_analysis_sophiedrs` | Time series analysis for SOPHIE DRS products |
| `get_derived_parameters` | Derive physical planet parameters from posteriors |
| `tess_ttv` | Transit timing variation analysis of TESS data |
| `tess_ffi_lc` | Extract light curves from TESS full-frame images |
| `check_tess_data` | Check available TESS data for an object |
| `system_params` | Create/convert/update system parameters JSON files |
| `rm_fit` | Fit the Rossiter-McLaughlin effect from RVs around transit |
| `fit_opd_transits` | Fit transits from OPD/LNA photometry |
| `fit_sparc4_transits` | Fit transits from SPARC4 photometry |

### Examples

```bash
# Fit TESS transits of a known object (data retrieved automatically from MAST)
transit_fit --object="HATS-24" -vp

# Fit transits from local TESS light curve files
transit_fit --object="TIC 287256467" --lcdata="TOI-2141/TESS/*lc.fits" --nsteps=5000 --burnin=1000 -vpe

# Fit an RV orbit
rv_fit --rvdata=TOI-1736_sophie.rdb --planet_priors=TOI-1736.pars -v

# Joint transit + RV fit with a GP activity model
transit_rv_fit --object="TOI-1759" --rvdata=lbl_TOI-1759_GL846_drift.rdb --gp_priors=priors/gp-priors.pars -vpa

# GP analysis of an RV time series
gp_analysis --input=TOI-1736_sophie.rdb --gp_priors=gp_rv_priors.pars --nsteps=10000 --burnin=3000 -pvm

# GLS periodogram
gls_analysis --input=TOI-2141_sophie_clean_rv.rdb --min_period=5. --max_period=50. --ofac=1000 -pv

# TTV analysis
tess_ttv --object="HATS-24" --burnin=500 --nsteps=3000 -vpo

# Analyze a saved MCMC chain
mcmc_analysis --input=TOI-1736_mcmc_samples.h5 --output=TOI-1736_pairsplot.png
```

## Data files

- **Claret (2017) TESS limb-darkening coefficients** are shipped with the package and used automatically.
- **TESS objects database** (`tess_objects.json`): a read/write cache of object parameters. On first use, the database shipped with the package is copied to `~/.exoplanet_analysis/tess_objects.json`, where it is subsequently read and updated. This keeps the installed package read-only while preserving the pre-populated entries.

## Tutorial notebooks

The `notebooks/` directory contains a Jupyter tutorial series that introduces the
whole package. Notebooks 01–04 use a real-life example — the characterization of the
hot super-Neptune **TOI-3568 b** ([Martioli et al. 2024](https://ui.adsabs.harvard.edu/abs/2024A%26A...691A.312M/abstract)),
using the actual TESS photometry and MAROON-X + SPIRou radial velocities from the paper —
and notebooks 05-06 demonstrate the Rossiter-McLaughlin analysis and a full joint fit on real photometry and spectroscopy of WASP-108 b:

1. `01_getting_started.ipynb` — package tour, priors files, system parameters in JSON, and the transit/RV forward models.
2. `02_tess_photometry_transit_fit.ipynb` — loading TESS light curves, selecting transit windows, and an MCMC transit fit.
3. `03_radial_velocities.ipynb` — loading RVs, a GLS periodogram, and an MCMC Keplerian orbit fit.
4. `04_joint_fit_derived_parameters.ipynb` — a joint transit+RV MCMC fit and the derived physical parameters of the planet.
5. `05_rossiter_mclaughlin.ipynb` — the Rossiter-McLaughlin effect: fitting the sky-projected obliquity and stellar v sin i from RVs taken across a transit.
6. `06_joint_photometry_rv_rm.ipynb` — a full joint fit of WASP-108 b combining TESS and OPD/LNA 0.6-m photometry, CORALIE orbital RVs, and GHOST Rossiter-McLaughlin RVs in a single MCMC.

Run notebooks 01–04 in order (notebook 04 uses the system JSON file created in notebook 01);
notebooks 05-06 are self-contained. The input data is in `notebooks/data/` (~17 MB); each notebook
writes its products to `notebooks/outputs/`. Every analysis shown in the notebooks is also
available as a command-line tool (see below) — run any of them with `-h` for usage.

The `notebooks/` directory also contains `simulate-exoplanet-rvs.ipynb`, which demonstrates
simulating radial velocity data for exoplanetary systems.

Every function in the package carries a numpydoc-style docstring, so `help(function)`
(or `?` in Jupyter/IPython) documents any part of the API.

## Citation

If you use this package in your research, please cite the relevant publications by E. Martioli et al. and acknowledge the use of the *Exoplanet Analysis Tools* package.

## Releasing / updating the distribution

Maintainers: see [`PUBLISHING.md`](PUBLISHING.md) for the step-by-step workflow to
publish a new version to GitHub from macOS. In short, the version lives only in
`src/exoplanet_analysis/__init__.py`; pushing a matching tag (e.g. `v1.4.2`)
triggers a GitHub Actions workflow that builds the package and creates a GitHub
Release automatically. See [`CHANGELOG.md`](CHANGELOG.md) for the version history.

## License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.
