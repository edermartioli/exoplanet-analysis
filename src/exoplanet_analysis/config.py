"""
    Description: Package configuration and user-writable data locations.

    The installed package is read-only, so all files generated or updated at
    run time (auto-generated priors, the TESS objects database cache) live in
    a per-user data directory.
    """

import os

# Per-user data directory for run-time generated/updated files
user_data_dir = os.path.join(os.path.expanduser('~'), '.exoplanet_analysis')

# Directory where auto-generated priors files (e.g. from TESS DVT products)
# are saved and re-read by the analysis scripts
priors_dir = os.path.join(user_data_dir, 'priors/')


def ensure_user_data_dir():
    """Create the user data directory if it does not exist and return it."""
    os.makedirs(user_data_dir, exist_ok=True)
    return user_data_dir

# Directory where TTV analysis output files are saved
ttvs_dir = os.path.join(user_data_dir, 'ttvs/')
