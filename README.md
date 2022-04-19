# Exoplanet-analysis
Toolkit to perform the analysis of photometry and radial velocity data for the detection and characterization of exoplanets.

### To install dependencies:
```
pip install emcee uncertainties PyAstronomy george batman-package corner astroquery
```

For a quick start try to run one of the main codes below:

## To fit TESS transits only:

```
python transit_fit.py --object="HATS-24" -vp
```

## To fit TESS transits and RV data simultaneously:

```
python transit_rv_fit.py --object="TOI-1759" --rvdata=example/lbl_TOI-1759_GL846_drift.rdb --nsteps=10000 --burnin=2000 -vp
```

* Note: the 'object' ID must be identified by SIMBAD.

If you're using this code, please cite [Martioli et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022arXiv220201259M/abstract)


## TTV Tool: fit TESS transits globally and then fit each transit individually

```
python tess_ttv.py --object="TOI-1759" --calib_order=3 --burnin=200 --nsteps=1000 -vpo
```

## To calculate lightcurves from FFI TESS data and then fit the transits to these data using GP

```
python tess_ffi_lc.py --object="TIC 160390955" --fold_period=4.4199503 --epoch_time=2458730.234903 --output=TOI-3568_tess_lc.fits --mask_threshold=8 -vp

python transit_fit_gp.py --input=TOI-3568_tess_lc.fits --epoch_time=1712.5551018 --fold_period=4.4199503 --transit_duration=2.232 --planet_priors=priors/TOI-3568.pars --calib_order=2 --nsteps=1000 --burnin=300 --gp_priors=priors/TOI-3568_gp_priors_phot.pars -vpmal
```
