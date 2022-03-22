# Exoplanet-analysis
Toolkit to perform the analysis of photometry and radial velocity data for the detection and characterization of exoplanets.

### Dependencies:
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
