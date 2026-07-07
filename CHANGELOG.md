# Changelog

All notable changes to **Exoplanet Analysis Tools** are documented here. This
project follows [Semantic Versioning](https://semver.org): `MAJOR.MINOR.PATCH`.

## [1.6.0]

- The tutorial datasets are no longer bundled in the repository, making it much
  lighter. They are hosted on Google Drive and downloaded on demand into
  `notebooks/data/` by the new `exoplanet_analysis.datasets` helper
  (`datasets.ensure(...)` / `datasets.download_all()`), which uses `gdown`.
- Each tutorial notebook now downloads only the data it needs, on first run, and
  reuses the local copy thereafter. Added a `[tutorials]` optional-dependency
  extra (`pip install "exoplanet-analysis-tools[tutorials]"`) and a
  `notebooks/data/README.md` with instructions.

## [1.5.0]

- New tutorial notebook 07: Gaussian-Process modelling of stellar activity and
  RV activity-indicator diagnostics, using the real TOI-1736 data of Martioli et
  al. (2023). It fits a quasi-periodic rotation GP to the TESS photometry
  (recovering Prot ~ 9.8 d), shows GLS periodograms of the RVs and of the FWHM,
  BIS, S-index and H-alpha indicators, and compares RV-vs-activity correlations
  before and after removing the two-planet orbit (illustrating that the RVs are
  not strongly contaminated by activity).
- Added the TOI-1736 SOPHIE RVs (with activity indicators) and TESS light curves
  under `notebooks/data/TOI-1736/`.

## [1.4.1]

- The Rossiter-McLaughlin limb-darkening coefficient is now tied to the transit
  model's quadratic coefficients (`u0`, `u1`) by default (converted to an
  effective linear coefficient), so the RM and transit share the same stellar
  limb darkening when they cover the same bandpass. Include a dedicated `ldc`
  parameter to treat the RM data as an independent bandpass, or tie the RM to a
  specific photometric bandpass via per-instrument coefficients.
- New helpers: `rmlib.quadratic_to_linear_ld`, `rmlib.resolve_rm_ldc`, and
  `rmlib.rm_ldc_report`. The RV/RM and joint fits print the limb-darkening
  configuration when `verbose=True`.

## [1.4.0]

- New tutorial notebook 06: a full joint fit of WASP-108 b combining TESS and
  OPD/LNA 0.6-m photometry, CORALIE orbital RVs, and GHOST Rossiter-McLaughlin
  RVs in a single MCMC.
- New `init_walker_ball`: a per-parameter (prior-scaled) MCMC walker
  initialization, enabled by `scaled_ball=True` in the RV/RM and joint fits.
- Added the CORALIE orbital RVs for WASP-108 (reconstructed from the published
  best-fit orbit of Anderson et al. 2015; see the file header and cite that
  paper).

## [1.3.0]

- Integrated the Rossiter-McLaughlin (RM) effect (Ohta et al. 2005) via the new
  `rmlib` module, sharing all base MCMC machinery with the rest of the package.
- New RM parameters (`lambda`, `vsini`, `omega_rm`, `ldc`), RM-only fitting
  (`read_rm_priors` + `fitRMWithMCMC`), and RM support in the joint fit
  (`fitTransitsAndRVsWithMCMC(..., include_rm=True)`).
- New `rm_fit` command-line tool and tutorial notebook 05. RM parameters are
  written to the output posterior files and round-trip through the JSON
  system-parameters format.

## [1.2.0]

- Full numpydoc-style docstring coverage across the package.
- Four-part tutorial notebook series (01-04) built on real TOI-3568 b data
  (Martioli et al. 2024).
- Physics fixes: corrected the argument-of-periastron convention in
  `get_ecc_omg` (degrees via `atan2`); support for the legacy `ecc`/`w`
  parameterization; fixed the RV-semi-amplitude units in the derived-parameter
  planet mass/density.

## [1.1.0]

- JSON system-parameters I/O: priors in and posteriors out for every module,
  with a complete template, transparent `.pars` <-> JSON conversion, and
  automatically derived physical parameters.
- New `systemlib` module and `system_params` command-line tool.
- numpy 2.x compatibility fixes.

## [1.0.0]

- First packaged release: `src`-layout, pip-installable package with library
  modules and console-script command-line tools, GPL-3.0 license.
