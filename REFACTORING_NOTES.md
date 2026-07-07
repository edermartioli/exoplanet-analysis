# Refactoring Notes

This document describes every change made when restructuring the original flat
collection of scripts into this pip-installable package, so they can be
reviewed and audited. **No algorithm, model, fitting routine, or numerical
code was modified.** A functional equivalence test verified bit-identical
outputs between the original code and this package for transit models, RV
models, data binning, priors file parsing, limb-darkening coefficient lookup,
and Gaussian Process interpolation.

## 1. Package layout

- The 7 library modules (`exoplanetlib`, `fitlib`, `gp_lib`, `priorslib`,
  `rvutils`, `tess`, `timeseries_lib`) moved unchanged (except imports, see
  below) to `src/exoplanet_analysis/`.
- The 18 command-line scripts moved to `src/exoplanet_analysis/scripts/` and
  are installed as console commands (see `[project.scripts]` in
  `pyproject.toml`). `transits.py` installs as `transits_analysis` to avoid an
  overly generic command name; all others keep their original names.
- Data files moved to `src/exoplanet_analysis/data/` and are installed with
  the package (`tess_claret_ldc/table25.dat`, `tess_objects.json`).
- `notebooks/` kept at the repository root; imports inside the notebook were
  updated to the new package imports.
- `rv_calibration_posterior.pars` (a run-time *output* artifact, not package
  data) was moved to `examples/`.
- macOS metadata (`__MACOSX/`, `._*`) and `.ipynb_checkpoints/` were removed
  and are now covered by `.gitignore`.

## 2. Import rewrites (mechanical)

All internal imports were rewritten to package-absolute imports, e.g.

```python
import fitlib, priorslib          ->  from exoplanet_analysis import fitlib, priorslib
import timeseries_lib as tslib    ->  from exoplanet_analysis import timeseries_lib as tslib
```

Module names and all public function/class names are unchanged, so existing
user code only needs the same one-line import change.

## 3. Scripts wrapped into `main()` entry points

Each script's executable body (everything from `parser = OptionParser()` to
the end of file — verified to contain no function definitions in any script)
was indented into a `def main():` function with an
`if __name__ == "__main__": main()` guard. Function definitions preceding the
parser remain at module level, exactly as before.

Three scripts (`fit_opd_transits`, `fit_sparc4_transits`, `tess_ttv`) contain
functions that reference the module-level `options` global; in those scripts
`main()` declares `global options` so the original global semantics are
preserved exactly.

## 4. Run-time file locations (`config.py`)

The original code wrote run-time files into the code directory
(`<repo>/priors/`, `<repo>/ttvs/`, and the `tess_objects.json` cache next to
`tess.py`). An installed package must remain read-only, so a new
`exoplanet_analysis/config.py` defines per-user locations under
`~/.exoplanet_analysis/`:

- `priors_dir` -> `~/.exoplanet_analysis/priors/` (auto-generated priors)
- `ttvs_dir` -> `~/.exoplanet_analysis/ttvs/` (TTV outputs); the directory is
  now created before writing (previously the code crashed if it didn't exist)
- `tess_objects.json` -> `~/.exoplanet_analysis/tess_objects.json`; on first
  use it is **seeded from the database shipped with the package** (108
  pre-populated objects), then read and updated there.

The Claret limb-darkening table is read directly from the installed package
data (read-only).

## 5. Bug fixes (behavior-affecting, all in error/help handling only)

- **`--help` exit status**: every script wrapped `parser.parse_args()` in a
  bare `except:` which also swallowed optparse's clean `SystemExit(0)` from
  `-h/--help`, causing help requests to print a spurious error and exit with
  status 1. Clean exits are now re-raised; genuine parse errors behave as
  before (custom message + exit 1).
- **`tess_ttv` output**: `ttvs_dir` is created if missing before writing
  (previously an unhandled `FileNotFoundError`).
- One block-comment string in `mcmc_analysis.py` was made a raw string to
  silence a pre-existing `SyntaxWarning: invalid escape sequence`.
- `detect_transits.py` originally had a hardcoded absolute path to a local
  file and no CLI. It was reorganized into two documented functions
  (`load_detrended_lc`, `detect_transits_bls`) plus a CLI with `--input`,
  `--period_ranges`, `--binsize` and `--nperiods` options. The GP detrending
  and BLS search logic is identical; the previously hardcoded values
  (binsize 0.2, period ranges 1:10 and 7:15) are the defaults.
- Usage examples in docstrings and error messages were updated from
  `python <script>.py ...` to the installed command names.
- Dead code removed: unused `ExoplanetAnalysis_dir` assignments that remained
  from an earlier `sys.path`-based layout (the two genuinely used cases,
  `priors_dir` and `ttvs_dir`, are handled by `config.py` as described above).

## 6. Packaging

- `pyproject.toml` (setuptools, src layout) with all dependencies pinned by
  name, package data, console entry points, and GPL-3.0-or-later metadata.
- `README.md`, `LICENSE` (full GNU GPL v3 text), `.gitignore`, `MANIFEST.in`.
- `exoplanet_analysis/__init__.py` exposes submodules lazily (PEP 562), so
  `import exoplanet_analysis` is fast and heavy optional dependencies load
  only when the corresponding module is first used.

## 7. Verification performed

- All 26 Python files compile cleanly.
- `pip install -U .` succeeds; all 7 library modules import.
- All 18 console commands run and display help correctly (exit 0).
- Functional equivalence test: identical numerical outputs vs the original
  code for `exoplanetlib.batman_transit_model`, `exoplanetlib.rv_model`,
  `fitlib.bin_data`, `priorslib.read_priors`, `tess.get_claret_ld_coeffs`,
  and `gp_lib.interp_gp`.
- End-to-end run of `gls_analysis` on synthetic RV data (GLS + stacked BGLS
  periodograms) completes successfully.

## 8. Version 1.1.0 — JSON system parameters I/O

- New `systemlib.py`: full planetary-system JSON schema (star + planets, all
  parameters including ones not determined by fits), template creation,
  JSON -> priors conversion, posterior -> JSON updates with error propagation
  (including esinw/ecosw -> eccentricity/omega), and derived physical
  parameters (reusing the get_derived_parameters physics with the
  `uncertainties` package).
- `priorslib.read_priors` transparently accepts system JSON files anywhere a
  `.pars` priors file was accepted (all modules funnel through it). The line
  parser was extracted into `read_priors_from_lines` with identical behavior
  (plus tolerance to blank lines).
- `priorslib.save_posterior` now also writes a JSON mirror of every posterior
  file (disable with `save_json=False`).
- `fitlib` exports a full system posterior JSON automatically after saving the
  planet posterior whenever the priors input was a system JSON file.
- New `priorslib.derive_filename`: all 39 `x.replace(".pars", ...)` output
  filename derivations were replaced by this extension-safe helper. This was
  a latent hazard: with a non-.pars input the old pattern returned the input
  path unchanged, which would have caused outputs to OVERWRITE the input file.
- New console command `system_params` (create template, convert to .pars,
  merge posteriors, compute derived parameters).
- numpy 2.x compatibility fixes in original code: `np.float` (3 sites in
  priorslib) and `np.RankWarning` (fitlib.fit_continuum) were removed in
  numpy >= 2.0 and crashed at run time; both fixed compatibly.
- Verified end-to-end: an MCMC transit fit of synthetic WASP-108 b data using
  the system JSON directly as priors recovered all injected parameters and
  produced the .pars posterior, its JSON mirror, and the full system posterior
  JSON with derived parameters, leaving the input file untouched.

### Circular-orbit prior bug in original code (FIXED in v1.2.0)

`tess.save_planet_prior` wrote `ecosw FIXED 90.` for circular orbits (the
value of omega had been written in place of ecosw = e*cos(omega), which must
be 0). Downstream, `exoplanetlib.get_ecc_omg(esinw=0, ecosw=90)` yielded
ecc = 90, corrupting the transit model for any priors generated with
`circular_orbit=True`. Corrected to `ecosw FIXED 0.`; a circular-orbit prior
now reads back as ecc = 0, omega = 90 deg. This changes the content of
auto-generated priors files (only the previously-wrong circular case).

## 9. Version 1.2.0 — Docstrings, tutorial notebooks, and physics fixes

### Documentation
- Every function in the package now has a numpydoc-style docstring
  (306 -> 307 functions, 100% coverage). Old-style "Description / -----"
  docstrings were normalized. Key public-API functions received curated
  one-line summaries; parameters are described from a shared glossary.
- Docstring insertion was purely additive to source text; it does not change
  any runtime behavior (the circular-orbit equivalence check remains
  bit-identical).

### Tutorial notebooks (notebooks/)
- A four-part Jupyter tutorial series introduces the whole package using the
  real TOI-3568 b data from Martioli et al. (2024): TESS photometry
  (sectors 15, 55, 56), MAROON-X blue/red RVs, and binned SPIRou RVs.
    01_getting_started            - package tour, priors, system JSON, models
    02_tess_photometry_transit_fit - load LCs, select transits, MCMC transit fit
    03_radial_velocities          - load RVs, GLS periodogram, MCMC orbit fit
    04_joint_fit_derived_parameters - joint MCMC fit, derived physical parameters
- All notebooks are executed and produce the expected figures. Notebook 04
  reproduces the published planet parameters (Rp = 5.3 R_Earth,
  Mp = 26 M_Earth, rho = 0.96 g/cm^3, K = 12 m/s) with short demo chains.
- Data lives in notebooks/data/TOI-3568/ (~17 MB); outputs are written to
  notebooks/outputs/ (git-ignored). Notebook 04 depends on notebook 01
  having been run (it updates the system JSON that notebook 01 creates).

### Physics / behavior fixes (these CHANGE numerical results)
- get_ecc_omg (exoplanetlib): previously returned the argument of periastron
  as arcsin(esinw/ecc) in RADIANS, losing the quadrant, while every caller
  passes the result on in DEGREES (e.g. to the batman transit model). It now
  returns np.degrees(np.arctan2(esinw, ecosw)). Circular orbits are
  unaffected (omega defaults to 90); eccentric-orbit transit and RV models
  now differ from the pre-1.2.0 output ON PURPOSE. The equivalence test was
  split into a circular check (still bit-identical) and an eccentric check
  (flagged as an intentional difference).
- New exoplanetlib.get_esinw_ecosw helper: reads either the esinw/ecosw
  parameterization or the legacy ecc/w parameterization (w in degrees) found
  in older priors files such as TOI-3568.pars. All nine esinw/ecosw lookups
  in exoplanetlib and fitlib now go through it, so legacy priors no longer
  raise KeyError: 'esinw_000'.
- systemlib.compute_derived_parameters: the RV semi-amplitude is stored and
  fitted in m/s, but get_derived_parameters.planet_mass / planet_density
  expect it in km/s (they multiply by 1000 internally). The derived planet
  mass and density were therefore 1000x too large. Fixed by converting k from
  m/s to km/s before the mass/density/gravity calls; results now match the
  published values.

### numpy 2.x compatibility (already in 1.1.0, restated)
- np.float and np.RankWarning were removed in numpy >= 2.0; both are handled
  compatibly.

## 10. Version 1.2.0 follow-up fixes

- `tess.save_planet_prior` circular-orbit bug fixed: `ecosw` for a circular
  orbit is now written as `0.` (was `90.`, omega's value in the wrong slot).
  See the updated §8 for details. This corrects auto-generated priors files.
- Tutorial notebook 02: the corner (pairs) plot cell now reads the MCMC
  samples through emcee's own `HDFBackend.get_chain(discard=..., flat=True)`
  reader and calls `plt.show()`, so the corner plot renders reliably. The
  transit-only fit does not expose the samples in the returned posterior dict
  (unlike the RV and joint fits), so the plot is built from the saved
  `_samples.h5` file.

## 11. Version 1.3.0 — Rossiter-McLaughlin (RM) integration

The pyRM package (classical analytical Rossiter-McLaughlin model of Ohta, Taruya &
Suto 2005) was integrated into the framework, sharing all base MCMC resources with
the rest of the package rather than duplicating them.

### New module: rmlib.py
- Ports the RM physics (`_gfunction`, `rm_anomaly`) adapted to the package's
  `_{iii}`-indexed per-planet parameter convention, reusing
  `exoplanetlib.get_esinw_ecosw`, `get_ecc_omg`, `true_anomaly` and
  `timetrans_to_timeperi`.
- `rm_rv_anomaly(time, planet_params, planet_index)` returns the RM velocity
  anomaly; `rm_model(...)` returns the full model (Keplerian orbit + RM anomaly).
- `has_rm_parameters(...)` detects RM parameters; `RM_PLANET_PARAM_IDS` lists them.
- New per-planet parameters: `lambda_{iii}` (sky-projected obliquity, deg),
  `vsini_{iii}` (projected rotation velocity, in the RV velocity units),
  `omega_rm_{iii}` (RM argument of periastron, deg; defaults to the orbital omega),
  and `ldc_{iii}` (RM linear limb-darkening; defaults to the transit `u0_{iii}`).
- Validation: the ported model matches the original pyRM `rv_model` to machine
  precision (~1e-16) for circular orbits when both use a single consistent epoch.
  The original referenced the RV phase to a separate `phi0` epoch distinct from the
  transit time; the integrated version references everything self-consistently to
  `tc`, which is the package-wide convention.

### Shared fitting infrastructure (fitlib.py)
- `calculate_rv_model_new` gained an `include_rm` flag: when set, the RM anomaly is
  added for any planet that has RM parameters. Default False, so orbit-only fits are
  unchanged.
- `include_rm` is threaded through `lnlikelihood_rv`/`lnprob_rv` (RV-only) and
  `lnlikelihood_tr_rv`/`lnprob_tr_rv` (joint), and through `fitRVsWithMCMC` and
  `fitTransitsAndRVsWithMCMC`. The same emcee machinery, priors, calibration, and
  output routines are reused.
- New convenience wrappers: `read_rm_priors` (alias of `read_rv_priors`, since the RM
  parameters are read automatically when present) and `fitRMWithMCMC` (calls
  `fitRVsWithMCMC` with `include_rm=True`). The RV/RM posterior now also exposes
  `planet_theta_fit/labels/err` so callers can save the posterior regardless of the
  verbose flag.

### Priors and parameter I/O (priorslib.py)
- `read_exoplanet_rv_params` and `read_exoplanet_transit_rv_params` now pick up the
  optional RM parameters (and, for the RV path, the transit-geometry parameters a, rp,
  inc, u0, u1 needed by the RM model) whenever they are present in the priors file.
- Free RM parameters flow through `get_theta_from_rv_priors` /
  `get_theta_from_transit_rv_priors` automatically, and the output posterior files
  (.pars and their JSON mirror) include the fitted RM parameters — this is the change
  to the output parameter-file format requested for RM fits.

### System JSON format (systemlib.py)
- `spinorbit_obliquity` (planet) maps to `lambda`; the star's `vsini` is emitted as the
  planet's `vsini_{iii}`; optional `rm_omega` and `rm_limb_darkening` planet fields map
  to `omega_rm` and `ldc`. RM parameters are emitted into the generated `.pars` only
  when a planet has an obliquity set, so non-RM systems are unaffected. RM parameters
  round-trip through the JSON <-> pars conversion.

### Command-line tool
- New `rm_fit` console command fits the RM effect from RVs around transit (RM-only),
  with a `--rv_units` option to keep km/s (vsini units) consistent with the RVs.

### Tutorial notebook
- New `notebooks/05_rossiter_mclaughlin.ipynb` walks through the RM model, an RM-only
  fit of real WASP-108 b GHOST RVs (recovering lambda ~ 8 deg and v sin i ~ 5.8 km/s,
  consistent with the published aligned-orbit result), building RM priors from a system
  JSON file, and the joint photometry+RV+RM fit. Data in `notebooks/data/WASP-108/`.

All previous regression checks still pass (equivalence PASS, all commands respond to
-h, clean compile); the RM additions are backward-compatible and off by default.

## 12. Version 1.4.0 — Joint photometry + RV + RM example and robust walker ball

### New tutorial notebook 06
- `notebooks/06_joint_photometry_rv_rm.ipynb` demonstrates a full joint fit of WASP-108 b
  combining four TESS sectors + one OPD/LNA 0.6-m ground-based transit (photometry),
  CORALIE orbital RVs, and GHOST Rossiter-McLaughlin RVs, all in a single MCMC with
  `include_rm=True`. The fit recovers the published parameters
  (K ~ 111 m/s, a/Rs ~ 7.5, i ~ 88.7 deg, Rp/Rs ~ 0.113) and lambda ~ 0 (aligned).
- Data added under `notebooks/data/WASP-108/`: the OPD light curve
  (`WASP-108_opd_bc06_lc.fits`), four TESS 2-min light curves (TIC 404340025, sectors
  11/37/38/64), the CORALIE RVs (`WASP-108_coralie.rdb`), and the joint priors file
  (`WASP-108_joint.pars`).

### CORALIE RV data provenance
- `WASP-108_coralie.rdb` provides the orbital RVs reconstructed from the published
  best-fit circular Keplerian orbit of Anderson et al. (2015), MNRAS 448, 1952
  (arXiv:1410.3449), as shown in their Figure 2, sampled at representative CORALIE epochs
  using the Table 6 parameters (P, Tc, K = 117.8 m/s, gamma = 47.07 km/s, e = 0). The
  file header documents this and asks users to cite Anderson et al. (2015). The individual
  measurements are available in full via the CDS.

### Robust per-parameter walker ball (fitlib.py)
- New `init_walker_ball(theta, labels, theta_priors, nwalkers, amp)` spreads each walker
  coordinate by a fraction of that parameter's prior width, rather than using a single
  global amplitude. This is important for RM/joint fits where an obliquity (+/-180 deg)
  and a transit time (~1e-4 d) coexist: a global amp is either too tight (walkers stuck
  in lambda) or too wide (geometry thrown into bad regions).
- `fitRVsWithMCMC`, `fitRMWithMCMC` and `fitTransitsAndRVsWithMCMC` gained a
  `scaled_ball=True` option (default) that uses this initialization; the transit-only and
  GP fits are unchanged, so the equivalence test still passes.

## 13. Version 1.4.1 — RM limb darkening tied to the transit model

Previously the RM limb-darkening coefficient (ldc) was an independent linear parameter,
unrelated to the quadratic u0/u1 of the transit model even though both describe the same
star. This is now handled coherently, with the sharing made explicit and bandpass-aware.

### rmlib.py
- `quadratic_to_linear_ld(u0, u1)` converts the transit quadratic limb-darkening pair to
  the effective linear coefficient used by the Ohta RM model: `eps = u0 + (2/3) u1`.
- `resolve_rm_ldc(planet_params, planet_index, instrum_params, instrum_index)` resolves the
  RM coefficient with a clear precedence: (1) a dedicated `ldc_{iii}` if present (RM treated
  as an independent bandpass); otherwise (2) tie to the transit limb darkening of the
  matching bandpass — per-instrument `u0_inst`/`u1_inst` when a bandpass index is given, or
  the planet-level `u0`/`u1` otherwise; else (3) 0. It also returns whether the coefficient
  was tied.
- `rm_ldc_report(...)` returns a human-readable description of the configuration, so it is
  always clear to the user whether the RM and transit limb darkening are shared (same
  bandpass) or independent (different bandpass).
- `rm_rv_anomaly` and `rm_model` now accept `instrum_params`/`instrum_index` to tie the RM
  limb darkening to a specific photometric bandpass.

### fitlib.py
- `calculate_rv_model_new` gained `rm_instrum_params`/`rm_instrum_index`; the RV and joint
  likelihoods thread a per-RV-dataset `rm_instrument_indexes` mapping through to the RM
  model, so each RV dataset can be tied to the transit LD of its own bandpass.
- `fitTransitsAndRVsWithMCMC` gained an `rm_instrument_indexes` argument and passes the
  transit `instrum_params` through to the RM model. The RV/RM and joint fits print the LD
  report (`rmlib.rm_ldc_report`) when `verbose=True`.

### Behavior and usage
- Omitting `ldc_{iii}` from the priors now ties the RM limb darkening to the fitted transit
  `u0/u1` (same bandpass). Including `ldc_{iii}` keeps it independent (different bandpass).
  The WASP-108 example priors (notebooks 05 and 06) were updated to tie the LD by default,
  with comments explaining the TESS / OPD I-band / GHOST bandpass choice, and the notebooks
  gained a short section on limb darkening across bandpasses.
- Backward compatible: priors that still include `ldc_{iii}` behave exactly as before, and
  the equivalence test still passes.
