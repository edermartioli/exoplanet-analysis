"""
    Created on April 13 2022
    
    Description: This routine calculates the lightcurve from FFI TESS data. It's useful for objects which has not reduced light curves available.
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    tess_ffi_lc --object="TIC 160390955" --fold_period=4.417960456092522 --epoch_time=2458712.5639538825487 --cutout_size=10 --mask_threshold=7 -vp --output=/Volumes/Samsung_T5/Science/TOI-3568/FFI_EXTRACTION/TOI-3568_ffi_lc.fits
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os, sys

from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt

from exoplanet_analysis import tess

import lightkurve as lk

#ttvs_dir = os.path.join(ExoplanetAnalysis_dir, 'ttvs/')


def main() :

    """Main.
    """
    parser = OptionParser()
    parser.add_option("-j", "--object", dest="object", help='Object ID',type='string',default="")
    parser.add_option("-i", "--input", dest="input", help='Input Targe Pixel File FITS',type='string',default="")
    parser.add_option("-o", "--output", dest="output", help='Output lightcurve file name',type='string',default="")
    parser.add_option("-c", "--cutout_size", dest="cutout_size", help='TESS cut off size',type='int',default=20)
    parser.add_option("-t", "--mask_threshold", dest="mask_threshold", help='Mask threshold',type='int',default=10)
    parser.add_option("-b", "--bkg_threshold", dest="bkg_threshold", help='Background threshold',type='float',default=0.001)
    parser.add_option("-f", "--fold_period", dest="fold_period", help='Fold period (d)',type='float',default=0)
    parser.add_option("-e", "--epoch_time", dest="epoch_time", help='Epoch time (BJD)',type='float',default=0)
    parser.add_option("-r", "--reference_pixel", dest="reference_pixel", help='Reference pixel',type='string',default="center")
    parser.add_option("-p", action="store_true", dest="plot", help="verbose",default=False)
    parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

    try:
        options,args = parser.parse_args(sys.argv[1:])
    except SystemExit as e :
        # allow clean exits from optparse (e.g. --help)
        if e.code == 0 or e.code is None :
            raise
        print("Error: check usage with tess_ffi_lc -h "); sys.exit(1);

    if options.verbose:
        print('Object ID: ', options.object)
        print('Input Targe Pixel File FITS: ', options.input)
        print('Output lightcurve file name: ', options.output)
        print('TESS cut off size: ', options.cutout_size)
        print('Mask threshold: ', options.mask_threshold)
        print('Reference pixel: ', options.reference_pixel)
        print('Background threshold: ', options.bkg_threshold)
        print('Fold period (d): ', options.fold_period)
        print('Epoch time (BJD): ', options.epoch_time)

    lk.log.setLevel('INFO')
    
    if options.input != "" :
        tpf = lk.read(options.input)
    else :
        pixelfile = lk.search_targetpixelfile(options.object)
    
        search_result = lk.search_tesscut(options.object)

        if options.verbose:
            print(search_result)
        
        tpf = search_result.download(cutout_size=options.cutout_size)

    target_mask = tpf.create_threshold_mask(threshold=options.mask_threshold, reference_pixel=options.reference_pixel)

    n_target_pixels = target_mask.sum()

    if options.verbose:
        print("N target pixels:", n_target_pixels)

    if options.plot :
        tpf.plot(aperture_mask=target_mask, mask_color='k')
        plt.show()
    
    target_lc = tpf.to_lightcurve(aperture_mask=target_mask)
    
    background_mask = ~tpf.create_threshold_mask(threshold=options.bkg_threshold, reference_pixel=None)

    if options.plot :
        tpf.plot(aperture_mask=background_mask, mask_color='w')
        plt.show()
    
    n_background_pixels = background_mask.sum()

    if options.verbose :
        print("N background pixels:",n_background_pixels)
    
    
    background_lc_per_pixel = tpf.to_lightcurve(aperture_mask=background_mask) / n_background_pixels
    background_estimate_lc = background_lc_per_pixel * n_target_pixels
    common_normalization = np.nanpercentile(target_lc.flux, 10)
    norm_lc = target_lc / common_normalization
    norm_background = background_estimate_lc / common_normalization

    if options.plot :
        ax = norm_lc.plot(normalize=False, label='Target + Background', lw=1)
        (norm_background +1).plot(ax=ax, normalize=False, label='Background',ylabel='Normalized, shifted flux')
        plt.show()
    
    corrected_lc = target_lc - background_estimate_lc.flux

    if options.plot :
        if options.verbose :
            print("Plotting corrected lc")

        corrected_lc.plot()
        plt.show()

    flatten_lc = corrected_lc.flatten(101)

    if options.plot :

        if options.verbose :
            print("plotting flatten lc")
        ax = flatten_lc.scatter()
        plt.show()
 
    if options.fold_period and options.plot :
        epoch_time = options.epoch_time - 2457000
        ax = flatten_lc.fold(options.fold_period, epoch_time=epoch_time).scatter()
        plt.show()

    #lk.show_citation_instructions()

    if options.output != "" :
        if options.verbose :
            print("Saving flatten corrected lightcurve to file: ",options.output)
        flatten_lc.to_fits(path=options.output, overwrite=True)


if __name__ == "__main__" :
    main()
