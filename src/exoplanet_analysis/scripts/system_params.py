# -*- coding: utf-8 -*-
"""
    Description: Tool to manage planetary system parameters in JSON format:
    create a template, convert a system JSON file into a .pars priors file,
    update a system JSON file with fitted posteriors, and compute derived
    planet parameters.

    @author: Eder Martioli

    Simple usage examples:

    # create a template with all supported parameters for a 2-planet system
    system_params --create_template=MY-SYSTEM.json --n_planets=2 --name="MY-SYSTEM"

    # convert a system JSON file into a .pars priors file
    system_params --input=TEST_SYSTEM.json --to_pars=TEST_SYSTEM.pars

    # update a system JSON file with fitted posteriors and derive parameters
    system_params --input=TEST_SYSTEM.json --posteriors="WASP-108_posterior.pars" --output=TEST_SYSTEM_posterior.json --derive -v
    """

__version__ = "1.0"

import sys
from optparse import OptionParser

from exoplanet_analysis import systemlib


def main() :

    """Main.
    """
    parser = OptionParser()
    parser.add_option("-c", "--create_template", dest="create_template", help="Create a system parameters JSON template with this file name", type='string', default="")
    parser.add_option("-n", "--n_planets", dest="n_planets", help="Number of planets in the template", type='int', default=1)
    parser.add_option("-s", "--name", dest="name", help="System name for the template", type='string', default="MY-SYSTEM")
    parser.add_option("-i", "--input", dest="input", help="Input system parameters JSON file", type='string', default="")
    parser.add_option("-t", "--to_pars", dest="to_pars", help="Output .pars priors file converted from the input JSON", type='string', default="")
    parser.add_option("-p", "--posteriors", dest="posteriors", help="Comma-separated posterior .pars file(s) to merge into the system", type='string', default="")
    parser.add_option("-o", "--output", dest="output", help="Output system parameters JSON file", type='string', default="")
    parser.add_option("-d", "--derive", action="store_true", dest="derive", help="Compute derived planet parameters", default=False)
    parser.add_option("-f", "--overwrite_fixed", action="store_true", dest="overwrite_fixed", help="Let FIXED posterior values overwrite existing values", default=False)
    parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

    try:
        options, args = parser.parse_args(sys.argv[1:])
    except SystemExit as e :
        # allow clean exits from optparse (e.g. --help)
        if e.code == 0 or e.code is None :
            raise
        print("Error: check usage with system_params -h")
        sys.exit(1)

    if options.create_template != "" :
        system = systemlib.create_template(n_planets=options.n_planets, system_name=options.name)
        systemlib.save_system(system, options.create_template)
        if options.verbose :
            print("Saved system parameters template ({} planet(s)): {}".format(options.n_planets, options.create_template))
        if options.input == "" :
            return

    if options.input == "" :
        print("Error: an --input system JSON file is required, check usage with system_params -h")
        sys.exit(1)

    system = systemlib.load_system(options.input)
    if options.verbose :
        print("Loaded system parameters: {} ({} component(s))".format(options.input, len(system.get("components", []))))

    if options.to_pars != "" :
        systemlib.system_to_pars(system, options.to_pars)
        if options.verbose :
            print("Saved priors file: {}".format(options.to_pars))

    updated = False
    if options.posteriors != "" :
        posterior_files = [p.strip() for p in options.posteriors.split(",") if p.strip() != ""]
        system = systemlib.update_system_from_posterior(system, posterior_files, overwrite_fixed=options.overwrite_fixed)
        updated = True
        if options.verbose :
            print("Updated system parameters from posterior(s): {}".format(", ".join(posterior_files)))

    if options.derive :
        system = systemlib.compute_derived_parameters(system)
        updated = True
        if options.verbose :
            print("Computed derived planet parameters")

    if options.output != "" :
        systemlib.save_system(system, options.output)
        if options.verbose :
            print("Saved system parameters: {}".format(options.output))
    elif updated :
        print("Warning: system parameters were updated but no --output file was given; nothing saved.")


if __name__ == "__main__" :
    main()
