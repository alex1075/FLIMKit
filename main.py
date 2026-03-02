#!/usr/bin/env python
import numpy as np
import warnings
from pathlib import Path
import matplotlib
import argparse
from code.PTU.reader import PTUFile
from code.FLIM.irf_tools import gaussian_irf_from_fwhm, irf_from_scatter_ptu, irf_from_xlsx, irf_from_xlsx_analytical, estimate_irf_from_decay_parametric, estimate_irf_from_decay_raw, compare_irfs
from code.FLIM.fitters import fit_summed, fit_per_pixel, MIN_PHOTONS_PERPIX
from code.utils.plotting import plot_summed, plot_pixel_maps, plot_lifetime_histogram
from code.utils.misc import print_summary
from code.utils.xlsx_tools import load_xlsx
from code.FLIM.fit_tools import find_irf_peak_bin
from code.configs import *

warnings.filterwarnings("ignore")



def main():
    print('Come back later. Use fit_cli.py instead for FLIM fitting from the command line interface (CLI).')
    
if __name__ == "__main__":
    main()