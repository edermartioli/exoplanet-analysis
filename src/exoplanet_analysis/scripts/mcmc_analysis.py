"""
    Created on Jan 22 2021
    
    Description: This routine performs the analysis of an MCMC chain
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    mcmc_analysis --input=aumic_b_mcmc_samples.h5


    mcmc_analysis --input=/Volumes/Samsung_T5/Science/TOI-1736/RV+TRANSITS_ANALYSIS/TOI-1736_mcmc_samples.h5 --output=/Volumes/Samsung_T5/Science/TOI-1736/RV+TRANSITS_ANALYSIS/TOI-1736_transitrvfit_TESS+SOPHIE_pairsplot.png
    mcmc_analysis --input=/Volumes/Samsung_T5/Science/TOI-2141/RV+TRANSITS_ANALYSIS/TOI-2141_mcmc_samples.h5 --output=/Volumes/Samsung_T5/Science/TOI-2141/RV+TRANSITS_ANALYSIS/TOI-2141_transitrvfit_TESS+SOPHIE_pairsplot.png

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import sys

from optparse import OptionParser

import emcee
import corner

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def next_pow_two(n):
    """Next pow two.

    Parameters
    ----------
    n
    """
    i = 1
    while i < n:
        i = i << 1
    return i
    
def autocorr_func_1d(x, norm=True):
    """Autocorr func 1d.

    Parameters
    ----------
    x
        Array of x values.
    norm : bool, optional (default: True)
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    """Auto window.

    Parameters
    ----------
    taus
    c
    """
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    """Autocorr gw2010.

    Parameters
    ----------
    y
        Array of y values.
    c : float, optional (default: 5.0)
    """
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_new(y, c=5.0):
    """Autocorr new.

    Parameters
    ----------
    y
        Array of y values.
    c : float, optional (default: 5.0)
    """
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]





def pairs_plot_emcee_new(samples, labels, calib_params, planet_params, output='', timelabel='BJD', addlabels=True, rvcalib_params={}) :
    """Pairs plot emcee new.

    Parameters
    ----------
    samples
    labels
        Labels of the free parameters.
    calib_params
        Dictionary of photometric calibration parameters.
    planet_params
        Dictionary of planet parameters (internal ids, e.g. per_000).
    output : str, optional (default: '')
        Output file path.
    timelabel : str, optional (default: 'BJD')
        Label of the time axis (for plots).
    addlabels : bool, optional (default: True)
    rvcalib_params : dict, optional (default: {})
        Dictionary of RV calibration parameters.
    """
    truths=[]
    font = {'size': 12}
    matplotlib.rc('font', **font)
    
    pl_suffixes = {0:'b',1:'c',2:'d',3:'e'}
    n_planets = int(planet_params["n_planets"])
    
    newlabels = []
    for lab in labels :
        if lab in calib_params.keys():
            truths.append(calib_params[lab])
        
        if lab in rvcalib_params.keys():
            truths.append(rvcalib_params[lab])

        elif lab in planet_params.keys():
            truths.append(planet_params[lab])
  
        if lab == 'rhos':
            newlabels.append(r"$\rho_{\star}$")
        elif lab == 'b_000':
            newlabels.append(r"b$_b$")
        elif lab == 'rp_000':
            newlabels.append(r"R$_b$/R$_{\star}$")
        elif lab == 'a_000':
            newlabels.append(r"a$_b$/R$_{\star}$")
        elif lab == 'tc_000':
            newlabels.append(r"T$_b$ [{}]".format(timelabel))
        elif lab == 'per_000':
            newlabels.append(r"P$_b$ [d]")
        elif lab == 'inc_000':
            newlabels.append(r"$i_b$ [$^{\circ}$]")
        elif lab == 'u0_000':
            newlabels.append(r"u0$_b$")
        elif lab == 'u1_000':
            newlabels.append(r"u1$_b$")
        elif lab == 'esinw_000':
            newlabels.append(r"$e_b\sin{\omega_b}$")
        elif lab == 'ecosw_000':
            newlabels.append(r"$e_b\cos{\omega_b}$")
        elif lab == 'k_000':
            newlabels.append(r"K$_b$ [m/s]")
        elif lab == 'rvsys_000':
            newlabels.append(r"$\gamma_b$ [m/s]")
        elif lab == 'trend_000':
            newlabels.append(r"$\alpha_b$ [m/s/d]")
        elif lab == 'quadtrend_000':
            newlabels.append(r"$\beta_b$ [m/s/yr$^2$]")
        elif lab == 'b_001':
            newlabels.append(r"b$_c$")
        elif lab == 'rp_001':
            newlabels.append(r"R$_c$/R$_{\star}$")
        elif lab == 'a_001':
            newlabels.append(r"a$_c$/R$_{\star}$")
        elif lab == 'tc_001':
            newlabels.append(r"T$_c$ [{}]".format(timelabel))
        elif lab == 'per_001':
            newlabels.append(r"P$_c$ [d]")
        elif lab == 'inc_001':
            newlabels.append(r"$i_c$ [$^{\circ}$]")
        elif lab == 'u0_001':
            newlabels.append(r"u0$_c$")
        elif lab == 'u1_001':
            newlabels.append(r"u1$_c$")
        elif lab == 'esinw_001':
            newlabels.append(r"$e_c\sin{\omega_c}$")
        elif lab == 'ecosw_001':
            newlabels.append(r"$e_c\cos{\omega_c}$")
        elif lab == 'k_001':
            newlabels.append(r"K$_c$ [m/s]")
        elif lab == 'rvsys_001':
            newlabels.append(r"$\gamma_c$ [m/s]")
        elif lab == 'trend_001':
            newlabels.append(r"$\alpha_c$ [m/s/d]")
        elif lab == 'quadtrend_001':
            newlabels.append(r"$\beta_c$ [m/s/yr$^2$]")
        elif lab == 'rv_d00':
            newlabels.append(r"$\gamma$ [m/s]")
        else :
            newlabels.append(lab)

    if addlabels :
        fig = corner.corner(samples, labels = newlabels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], truths=truths, labelsize=15, label_kwargs={"fontsize": 15}, show_titles=True)
        #fig.set_size_inches(40, 45)
        #fig = marginals.corner(samples, labels = newlabels, quantiles=[0.16, 0.5, 0.84], truths=truths)
    else :
        fig = corner.corner(samples, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], labelpad=2.0,truths=truths, labelsize=10, show_titles=False)

    for ax in fig.get_axes():
        plt.setp(ax.get_xticklabels(), ha="left", rotation=60)
        plt.setp(ax.get_yticklabels(), ha="right", rotation=60)
        ax.tick_params(axis='both', labelsize=12)

    if output != '' :
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
    else :
        plt.show()



def bin_data(x, y, median=False, binsize = 100) :

    """Bin data.

    Parameters
    ----------
    x
        Array of x values.
    y
        Array of y values.
    median : bool, optional (default: False)
        Use the median instead of the weighted mean.
    binsize : int, optional (default: 100)
        Bin size in time units [d].
    """
    xi, xf = np.min(x), np.max(x)
    
    bins = np.arange(xi, xf, binsize)
    digitized = np.digitize(x, bins)
    
    bin_y = []
    bin_x = []
    
    for i in range(len(bins)+1):
        if len(x[digitized == i]) :
            if median :
                mean_y = np.median(y[digitized == i])
            else :
                mean_y = np.mean(y[digitized == i])
            
            bin_x.append(np.mean(x[digitized == i]))
            bin_y.append(mean_y)

    bin_y = np.array(bin_y)
    bin_x = np.array(bin_x)

    return bin_x, bin_y


def main() :

    """Main.
    """
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", help='Input MCMC samples file (*.h5)',type='string',default="")
    parser.add_option("-o", "--output", dest="output", help='Output pairsplot file',type='string',default="")
    parser.add_option("-p", action="store_true", dest="plot", help="verbose",default=False)
    parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

    try:
        options,args = parser.parse_args(sys.argv[1:])
    except SystemExit as e :
        # allow clean exits from optparse (e.g. --help)
        if e.code == 0 or e.code is None :
            raise
        print("Error: check usage with mcmc_analysis -h "); sys.exit(1);

    if options.verbose:
        print('Input MCMC samples file (*.h5): ', options.input)


    # read input MCMC samples into emcee sampler class
    sampler = emcee.backends.HDFBackend(options.input)
    samples = sampler.get_chain(flat=True)



    # Compute the estimators for a few different chain lengths
    N = np.exp(np.linspace(np.log(100), np.log(samples.shape[1]), 10)).astype(int)
    gw2010 = np.empty(len(N))
    new = np.empty(len(N))
    for i, n in enumerate(N):
        gw2010[i] = autocorr_gw2010(samples[:, :n])
        new[i] = autocorr_new(samples[:, :n])

    # Plot the comparisons
    plt.loglog(N, gw2010, "o-", label="G&W 2010")
    plt.loglog(N, new, "o-", label="new")
    ylim = plt.gca().get_ylim()
    plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
    #plt.axhline(true_tau, color="k", label="truth", zorder=-100)
    plt.ylim(ylim)
    plt.xlabel("number of samples, $N$")
    plt.ylabel(r"$\tau$ estimates")
    plt.legend(fontsize=14);
    plt.show()
    exit()

    """
    # calculate autocorrelation time
    tau = sampler.get_autocorr_time()

    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))

    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)

    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
    log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)

    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))
    print("flat chain shape: {0}".format(samples.shape))
    print("flat log prob shape: {0}".format(log_prob_samples.shape))
    print("flat log prior shape: {0}".format(log_prior_samples.shape))

    """

    usemode = True
    nbins = 30
    r"""
    #labels=[r"T$_c$ [TBJD]", r"P [d]", r"a/R$_{\star}$", r"R$_{p}$/R$_{\star}$", r"$i$ [$^{\circ}$]", r"u$_{0}$", r"u$_{1}$"]
    labels=[r"a$_b$/R$_{\star}$",
            r"R$_b$/R$_{\star}$",
            r"$i_b$ [deg]",
            r"u$_{0,b}$",
            r"u$_{1,b}$",
            r"K$_b$ [m/s]",
            r"T$_b$ [BJD]",
            r"P$_b$ [d]",
            r"K$_c$ [m/s]",
            r"T$_c$ [BJD]",
            r"P$_c$ [d]",
            r"$e_c$",
            r"$\omega_c$ [deg]",
            r"$\alpha_c$ [m/s/d]"]
    preci=[2, 5, 2, 2, 2, 2, 8, 8, 2, 1, 1, 4, 2, 4]
    """
    labels=[r"a$_b$/R$_{\star}$",
            r"R$_b$/R$_{\star}$",
            r"$i_b$ [deg]",
            r"u$_{0,b}$",
            r"u$_{1,b}$",
            r"K$_b$ [m/s]",
            r"T$_b$ [BJD]",
            r"P$_b$ [d]",
            r"$\gamma$ [m/s]"]
    preci=[2, 5, 2, 2, 2, 2, 8, 8, 1]

    #npars = len(samples[0])
    npars = len(labels)

    fig, axs = plt.subplots(npars, sharex=True)

    # set mcmc parameters
    nwalkers = 32
    ndim= 14
    burnin = 3000

    truths=[]
    font = {'size': 10}
    matplotlib.rc('font', **font)


    if usemode :
        func = lambda v: (v[1], v[2]-v[1], v[1]-v[0])
        percents = np.percentile(samples, [16, 50, 84], axis=0)
        seq = list(zip(*percents))
        values = list(map(func, seq))
        max_values = []
        for i in range(len(values)) :
            hist, bin_edges = np.histogram(samples[:,i], bins=nbins, range=(values[i][0]-5*values[i][1],values[i][0]+5*values[i][2]), density=True)
            xcen = (bin_edges[:-1] + bin_edges[1:])/2
            mode = xcen[np.argmax(hist)]
            max_values.append(mode)
        max_values = np.array(max_values)


    #Get the marginalized density in each dimension.
    for i in range(npars) :
        #chain = sampler.get_chain()[:, :, i].T
        chain = samples[:,i]
        x = np.arange(len(chain)) / nwalkers
        good = x > burnin
        discard = x <= burnin
    
        axs[i].plot(x[discard], chain[discard], '.', color='r', alpha=0.1, lw=0.05, markersize=0.05)
        axs[i].plot(x[good], chain[good], '.', color='k', alpha=0.1, lw=0.05, markersize=0.05)
    
        percents = np.percentile(chain[good], [16, 50, 84], axis=0)
        err_high = np.round(percents[2]-percents[1],preci[i])
        err_low = np.round(percents[1]-percents[0],preci[i])
        cen = np.round(percents[1],preci[i])
    
        if usemode :
            cen = max_values[i]
    
        truths.append(cen)
    
        axs[i].set_ylim(cen-5*err_low,cen+5*err_low)
    
        xmin, xmax = np.min(x), np.max(x)
    
        fit_par='{0}={1:f}+{2:f}-{3:f}'.format(labels[i], cen, err_high, err_low)
        print(fit_par)
        axs[i].hlines(percents, xmin, xmax, colors="darkblue", linestyles=['dashed','solid','dashed'], label=fit_par)
    
        xbin_discard, ybin_discard = bin_data(x[discard], chain[discard], median=True, binsize = 1000)
        axs[i].plot(xbin_discard, ybin_discard, '-', color='brown', lw=2)

        xbin, ybin = bin_data(x[good], chain[good], median=True, binsize = 1000)
        axs[i].plot(xbin, ybin, '-', lw=2)

        axs[i].set_ylabel(labels[i])
        #axs[i].legend()
        if i == npars - 1 :
            axs[i].set_xlabel("step number")

    plt.show()



    font = {'size': 8}
    matplotlib.rc('font', **font)

    #plt.figure(figsize=(100,60))

    figure = corner.corner(samples[burnin:,:npars], labels = labels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], truths=truths, labelsize=15, label_kwargs={"fontsize": 14}, show_titles=True, figsize=(100, 100), labelpad=0.02)

    figure.set_figwidth(18)
    figure.set_figheight(24)
    #figure = corner.corner(samples[burnin:,:npars], quantiles=[0.16, 0.5, 0.84], truths=truths)
    plt.show()

    if options.output != "" :
        figure.savefig(options.output)
        plt.close(figure)
    """
    # Initialize the walkers
    coords = np.random.randn(32, 5)
    nwalkers, ndim = coords.shape

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(coords, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
    
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1
        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
    """


if __name__ == "__main__" :
    main()
