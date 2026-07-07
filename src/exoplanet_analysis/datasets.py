# -*- coding: utf-8 -*-
"""
    Description: Helper to download the tutorial datasets.

    The example data used by the tutorial notebooks (TESS light curves, radial
    velocities, priors files) is not bundled with the repository to keep it
    lightweight. Instead it lives in a shared Google Drive folder and is
    downloaded on demand into ``notebooks/data/`` the first time a notebook
    needs it.

    Typical use, at the top of a notebook::

        from exoplanet_analysis import datasets
        datasets.ensure("TOI-1736")     # downloads only if missing
        DATA = datasets.data_dir("TOI-1736") + "/"

    The download uses the ``gdown`` package. Install it if needed::

        pip install gdown

    @author: Eder Martioli
"""

import os
import sys

# Shared Google Drive folder holding the sub-folders TOI-3568, WASP-108 and
# TOI-1736. Kept here as the single place to update if the location changes.
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1jKAL85m5OLMiFhnLyU3xjAWgbJzlTPXS?usp=sharing"

# Known dataset sub-folders (for messages and validation).
DATASETS = ("TOI-3568", "WASP-108", "TOI-1736")


def default_data_root():
    """Return the default local root for the tutorial data.

    When run from inside the ``notebooks/`` directory (as the tutorials are),
    this is ``./data``. Otherwise it falls back to ``notebooks/data`` relative
    to the current directory.

    Returns
    -------
    str
        Path to the local data root.
    """
    if os.path.isdir("data") or os.path.basename(os.getcwd()) == "notebooks":
        return os.path.abspath("data")
    if os.path.isdir(os.path.join("notebooks", "data")):
        return os.path.abspath(os.path.join("notebooks", "data"))
    return os.path.abspath("data")


def data_dir(dataset, data_root=None):
    """Return the local directory for a given dataset.

    Parameters
    ----------
    dataset : str
        Dataset name, e.g. "TOI-1736".
    data_root : str, optional
        Local data root. Defaults to :func:`default_data_root`.

    Returns
    -------
    str
        Path to the dataset directory (may not exist yet).
    """
    root = data_root if data_root is not None else default_data_root()
    return os.path.join(root, dataset)


def is_present(dataset, data_root=None, min_files=1):
    """Return True if the dataset directory exists and is non-empty.

    Parameters
    ----------
    dataset : str
        Dataset name, e.g. "TOI-1736".
    data_root : str, optional
        Local data root. Defaults to :func:`default_data_root`.
    min_files : int, optional (default: 1)
        Minimum number of files required to consider the dataset present.

    Returns
    -------
    bool
    """
    d = data_dir(dataset, data_root)
    if not os.path.isdir(d):
        return False
    n = sum(len(files) for _, _, files in os.walk(d))
    return n >= min_files


def _import_gdown():
    try:
        import gdown
        return gdown
    except ImportError:
        raise ImportError(
            "The 'gdown' package is required to download the tutorial data.\n"
            "Install it with:  pip install gdown\n"
            "Or download the 'data' folder manually from:\n    {0}\n"
            "and place its sub-folders under your local notebooks/data/ directory."
            .format(GDRIVE_FOLDER_URL)
        )


def download_all(data_root=None, quiet=False):
    """Download the entire shared data folder into the local data root.

    This mirrors the shared Google Drive ``data`` folder (with its TOI-3568,
    WASP-108 and TOI-1736 sub-folders) into ``data_root``.

    Parameters
    ----------
    data_root : str, optional
        Local data root. Defaults to :func:`default_data_root`.
    quiet : bool, optional (default: False)
        Suppress gdown progress output.

    Returns
    -------
    str
        The local data root the files were downloaded into.
    """
    gdown = _import_gdown()
    root = data_root if data_root is not None else default_data_root()
    os.makedirs(root, exist_ok=True)
    if not quiet:
        print("Downloading tutorial data into {0} ...".format(root))
    # gdown recreates the shared folder (named 'data') inside output; download
    # into the parent of 'data' so the sub-folders land directly under root.
    parent = os.path.dirname(root)
    gdown.download_folder(GDRIVE_FOLDER_URL, output=os.path.join(parent, "data"),
                          quiet=quiet, use_cookies=False)
    return root


def ensure(dataset, data_root=None, quiet=False):
    """Ensure a dataset is available locally, downloading it if necessary.

    If the dataset directory is missing or empty, the shared data folder is
    downloaded. If it is already present, nothing happens.

    Parameters
    ----------
    dataset : str
        Dataset name, e.g. "TOI-1736".
    data_root : str, optional
        Local data root. Defaults to :func:`default_data_root`.
    quiet : bool, optional (default: False)
        Suppress gdown progress output.

    Returns
    -------
    str
        Path to the (now present) dataset directory.
    """
    if dataset not in DATASETS:
        print("Warning: '{0}' is not a known dataset (known: {1}). "
              "Attempting download anyway.".format(dataset, ", ".join(DATASETS)),
              file=sys.stderr)

    d = data_dir(dataset, data_root)
    if is_present(dataset, data_root):
        if not quiet:
            print("Dataset '{0}' already present at {1}".format(dataset, d))
        return d

    download_all(data_root=data_root, quiet=quiet)

    if not is_present(dataset, data_root):
        raise RuntimeError(
            "Downloaded the shared folder but dataset '{0}' was not found at {1}.\n"
            "Check the folder is shared as 'Anyone with the link' and contains a "
            "sub-folder named '{0}', or download it manually from:\n    {2}"
            .format(dataset, d, GDRIVE_FOLDER_URL))
    return d
