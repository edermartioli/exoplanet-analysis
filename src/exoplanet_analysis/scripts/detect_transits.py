"""
    Description: This routine detects planetary transits in a TESS DVT light
    curve. It first removes long-term trends using a Gaussian Process fit to
    binned data, then runs a Box Least Squares (BLS) periodogram search over
    one or more period ranges, reporting the period, epoch and duration of the
    strongest signal in each range.

    @author: Eder Martioli

    Simple usage examples:

    detect_transits --input=tess2019199201929-s0014-s0050-0000000224298134-00611_dvt.fits
    detect_transits --input=lc_dvt.fits --period_ranges="1:10,7:15" --binsize=0.2 -p
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import sys
from optparse import OptionParser

import matplotlib.pyplot as plt
import numpy as np

import astropy.io.fits as fits
import lightkurve as lk

from exoplanet_analysis import fitlib, gp_lib


def load_detrended_lc(input_file, binsize=0.2, gap_size=1.0, min_npoints=30, verbose=False):
    """Load a TESS DVT light curve and remove long-term trends with a GP.

    Parameters
    ----------
    input_file : str
        Path to a TESS DVT FITS file.
    binsize : float
        Bin size (in days) used to bin the data before the GP fit.
    gap_size : float
        Minimum gap (in days) used to split the time series into windows.
    min_npoints : int
        Minimum number of valid binned points required to fit a window.
    verbose : bool
        Print progress information.

    Returns
    -------
    lightkurve.LightCurve
        The GP-detrended light curve.
    """
    data = fits.getdata(input_file, 1)
    keep = np.isfinite(data['LC_DETREND'])
    time, flux, fluxerr = data['TIME'][keep], data['LC_DETREND'][keep], data['LC_INIT_ERR'][keep]

    dt = np.abs(time[1:] - time[:-1])
    gaps = dt > gap_size
    tis = np.array([time[0]])
    tis = np.append(tis, time[:-1][gaps])
    tfs = time[1:][gaps]
    tfs = np.append(tfs, time[-1])

    t, f, ef = np.array([]), np.array([]), np.array([])
    for i in range(len(tis)):
        if verbose:
            print("Processing range {}/{} -> ti={} tf={}".format(i + 1, len(tis), tis[i], tfs[i]))
        window = (time > tis[i]) & (time < tfs[i])
        bin_time, bin_flux, bin_fluxerr = fitlib.bin_data(time[window], flux[window], fluxerr[window], median=False, binsize=binsize)
        keep = (np.isfinite(bin_flux)) & (np.isfinite(bin_fluxerr))
        if len(bin_time[keep]) > min_npoints:
            good_windows = [[tis[i], tfs[i]]]
            gp_flux, gp_fluxerr = gp_lib.interp_gp(time[window], bin_time[keep], bin_flux[keep], bin_fluxerr[keep], good_windows, verbose=False, plot=False)
            t = np.append(t, time[window])
            f = np.append(f, flux[window] - gp_flux)
            ef = np.append(ef, fluxerr[window])

    return lk.LightCurve(time=t, flux=f, flux_err=ef)


def detect_transits_bls(lc, min_period, max_period, nperiods=10000, frequency_factor=500, label="planet", plot=False, verbose=False):
    """Run a BLS periodogram search over a period range and return the best model.

    Parameters
    ----------
    lc : lightkurve.LightCurve
        The (detrended) light curve to search.
    min_period, max_period : float
        Period search range in days.
    nperiods : int
        Number of trial periods.
    frequency_factor : float
        BLS frequency oversampling factor.
    label : str
        Label used in printed output and plots.
    plot : bool
        Show the periodogram and folded light curve.
    verbose : bool
        Print the detected ephemeris.

    Returns
    -------
    (period, t0, duration, transit_model)
    """
    period = np.linspace(min_period, max_period, nperiods)
    bls = lc.to_periodogram(method='bls', period=period, frequency_factor=frequency_factor)
    if plot:
        bls.plot()
        plt.show()

    planet_period = bls.period_at_max_power
    planet_t0 = bls.transit_time_at_max_power
    planet_dur = bls.duration_at_max_power

    if verbose:
        print("{}: period={} t0={} dur={}".format(label, planet_period, planet_t0, planet_dur))

    if plot:
        ax = lc.fold(period=planet_period, epoch_time=planet_t0).scatter()
        lc.fold(planet_period, planet_t0).bin(.1).plot(ax=ax, c='r', lw=2, label='Binned Flux')
        ax.set_xlim(-5, 5)
        plt.show()

    planet_model = bls.get_transit_model(period=planet_period,
                                         transit_time=planet_t0,
                                         duration=planet_dur)

    return planet_period, planet_t0, planet_dur, planet_model


def main() :

    """Main.
    """
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", help="Input TESS DVT light curve FITS file", type='string', default="")
    parser.add_option("-r", "--period_ranges", dest="period_ranges", help='Period search ranges in days, e.g. "1:10,7:15"', type='string', default="1:10,7:15")
    parser.add_option("-b", "--binsize", dest="binsize", help="Bin size (days) for GP detrending", type='float', default=0.2)
    parser.add_option("-n", "--nperiods", dest="nperiods", help="Number of trial periods in BLS search", type='int', default=10000)
    parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
    parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

    try:
        options, args = parser.parse_args(sys.argv[1:])
    except SystemExit as e :
        # allow clean exits from optparse (e.g. --help)
        if e.code == 0 or e.code is None :
            raise
        print("Error: check usage with detect_transits -h")
        sys.exit(1)

    if options.input == "":
        print("Error: input file is required, check usage with detect_transits -h")
        sys.exit(1)

    if options.verbose:
        print('Input TESS DVT file: ', options.input)
        print('Period search ranges: ', options.period_ranges)

    lc = load_detrended_lc(options.input, binsize=options.binsize, verbose=options.verbose)

    models = []
    planet_labels = "bcdefghijklmnopqrstuvwxyz"
    for j, prange in enumerate(options.period_ranges.split(",")):
        pmin, pmax = (float(v) for v in prange.split(":"))
        label = "Planet {}".format(planet_labels[j % len(planet_labels)])
        period, t0, dur, model = detect_transits_bls(lc, pmin, pmax, nperiods=options.nperiods, label=label, plot=options.plot, verbose=True)
        models.append((label, model))

    if options.plot and len(models):
        ax = lc.scatter()
        colors = ['dodgerblue', 'r', 'g', 'orange', 'purple']
        for j, (label, model) in enumerate(models):
            model.plot(ax=ax, c=colors[j % len(colors)], label='{} Transit Model'.format(label))
        plt.show()


if __name__ == "__main__" :
    main()
