# Exoplanet-analysis
Toolkit to perform the analysis of photometry and radial velocity data for the detection and characterization of exoplanets.

For a quick start try to run one of the two main codes below:

## To fit TESS transits only:

```
python transit_fit.py --object="55 Cnc" -vp
```

## To fit TESS transits and RV data simultaneously:

```
python transit_rv_fit.py --object="TOI-1759" --rvdata=example/lbl_TOI-1759_GL846_drift.rdb --gp_priors=priors/gp-priors.pars -vp
```

If you're using this code, please cite [Martioli et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022arXiv220201259M/abstract)
