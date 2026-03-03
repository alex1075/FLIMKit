#!/usr/bin/env python
import numpy as np
import warnings
from pathlib import Path
import matplotlib
import argparse
from pyflim.PTU.reader import PTUFile
from pyflim.FLIM.irf_tools import gaussian_irf_from_fwhm, irf_from_scatter_ptu, irf_from_xlsx, irf_from_xlsx_analytical, estimate_irf_from_decay_parametric, estimate_irf_from_decay_raw, compare_irfs
from pyflim.FLIM.fitters import fit_summed, fit_per_pixel, MIN_PHOTONS_PERPIX
from pyflim.utils.plotting import plot_summed, plot_pixel_maps, plot_lifetime_histogram
from pyflim.utils.misc import print_summary
from pyflim.utils.xlsx_tools import load_xlsx
from pyflim.FLIM.fit_tools import find_irf_peak_bin
from pyflim.configs import *

warnings.filterwarnings("ignore")



def main():
    print('Come back later. Use fit_cli.py instead for FLIM fitting from the command line interface (CLI).')
    
if __name__ == "__main__":
    main()