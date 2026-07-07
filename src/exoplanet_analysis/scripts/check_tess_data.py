"""
    Created on Feb 14 2022
    
    Description: This routine gets TESS data products from the MAST archive and plot it as a check
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brasil.
    Institut d'Astrophysique de Paris, France.

    Simple usage examples:
    
    check_tess_data --object="HATS-24"
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import sys, os

from optparse import OptionParser
from exoplanet_analysis import tess

from exoplanet_analysis.config import priors_dir


def main() :

    """Main.
    """
    parser = OptionParser()
    parser.add_option("-o", "--object", dest="object", help='Object ID',type='string',default="HATS-24")
    parser.add_option("-s", "--sector", dest="sector", help='Select TESS sector',type='int',default=0)
    parser.add_option("-i", action="store_true", dest="individual_sectors", help="get individual sectors",default=False)
    parser.add_option("-p", action="store_true", dest="plot", help="verbose",default=False)
    parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

    try:
        options,args = parser.parse_args(sys.argv[1:])
    except SystemExit as e :
        # allow clean exits from optparse (e.g. --help)
        if e.code == 0 or e.code is None :
            raise
        print("Error: check usage with check_tess_data -h "); sys.exit(1);


    if options.verbose:
        print('Object ID: ', options.object)
        if options.sector :
            print('TESS sector selected: ', options.sector)
        
    dvt_filenames = tess.retrieve_tess_data_files(options.object, sector=options.sector, products_wanted_keys = ["DVT"], individual_sectors=options.individual_sectors, verbose=options.verbose)

    tesslc = tess.load_dvt_files(options.object, priors_dir=priors_dir, save_priors=True, plot=options.plot, verbose=options.verbose)


if __name__ == "__main__" :
    main()
