# Tutorial datasets

The example data used by the tutorial notebooks (TESS light curves, radial
velocities with activity indicators, and priors files for **TOI-3568**,
**WASP-108**, and **TOI-1736**) is **not stored in this repository**, to keep the
repo lightweight. It lives in a shared Google Drive folder and is downloaded into
this directory on demand.

## Automatic download (recommended)

The notebooks download what they need automatically on first run, using the
`exoplanet_analysis.datasets` helper. This requires the `gdown` package:

```bash
pip install gdown
```

Then just run the notebooks — the first cell that needs data calls
`datasets.ensure("<target>")`, which downloads the shared folder into
`notebooks/data/` only if the files are not already present. Subsequent runs use
the local copy and do not re-download.

You can also fetch everything up front from Python:

```python
from exoplanet_analysis import datasets
datasets.download_all()      # downloads TOI-3568, WASP-108 and TOI-1736
```

## Manual download

If you prefer, download the `data` folder directly from Google Drive:

<https://drive.google.com/drive/folders/1jKAL85m5OLMiFhnLyU3xjAWgbJzlTPXS?usp=sharing>

and place its sub-folders here so that the layout is:

```
notebooks/data/
├── TOI-3568/
├── WASP-108/
└── TOI-1736/
```

## Notes

- The downloaded files are git-ignored, so they will never be accidentally
  committed back into the repository.
- The data are the observations used in the corresponding papers
  (Martioli et al. 2024 for TOI-3568; Anderson et al. 2015 and the GHOST/OPD
  observations for WASP-108; Martioli et al. 2023 for TOI-1736). Please cite the
  relevant papers when using them.
