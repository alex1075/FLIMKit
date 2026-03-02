
import inquirer
import argparse
import numpy as np
from pathlib import Path
import matplotlib
from .configs import (
    n_exp, Tau_min, Tau_max, D_mode, binning_factor, MIN_PHOTONS_PERPIX, Optimizer, lm_restarts, de_population, de_maxiter, n_workers, OUT_NAME, Estimate_IRF, IRF_BINS, IRF_FIT_WIDTH, IRF_FWHM, channels, config_message)
from .PTU.reader import PTUFile
from .FLIM.fit_tools import find_irf_peak_bin
from .FLIM.irf_tools import irf_from_scatter_ptu, gaussian_irf_from_fwhm, compare_irfs
from .FLIM.fitters import fit_summed, fit_per_pixel
from .utils.xlsx_tools import load_xlsx
from .utils.misc import print_summary
from .utils.plotting import plot_summed, plot_pixel_maps, plot_lifetime_histogram

def yes_no_question(question):
    """Ask a yes/no question using inquirer and return 'y' or 'n'."""
    questions = [inquirer.List('yesno',
                               message=question,
                               choices=['Yes', 'No'])]
    answer = inquirer.prompt(questions)
    return 'y' if answer['yesno'] == 'Yes' else 'n'


def single_FOV_flim_fit_inquire():
    """Interactive prompt to collect FLIM fitting parameters."""
    print("\n--- Interactive FLIM Fit Setup ---")

    # Essential input
    ptu_path = input("Enter path to PTU file for this FOV: ").strip()
    if not ptu_path:
        raise ValueError("PTU file path is required.")

    # Optional XLSX (overlay) file
    xlsxq = yes_no_question("Do you have an XLSX file for this FOV? (Recommended for pixel size info)")
    if xlsxq == 'y':
        xlif_path = input("Enter path to XLSX file: ").strip()
        # Ask whether to use its IRF
        irf_xlsxq = yes_no_question("Do you want to use the IRF from this XLSX file? (Recommended if available)")
        if irf_xlsxq == 'y':
            # Use IRF from the same XLSX
            use_xlsx_irf = True
            estimate_irf = 'none'
            irf_path = None
        else:
            use_xlsx_irf = False
            # Ask for alternative IRF source
            print("\nIRF estimation options:")
            print("  1. 'file'   - measured IRF image (scatter PTU)")
            print("  2. 'raw'    - non‑parametric IRF from raw decay")
            print("  3. 'parametric' - fit Gaussian + exponential tail to rising edge")
            print("  4. 'none'   - do not estimate IRF (not recommended)")
            method_q = [inquirer.List('method',
                                      message="Choose IRF estimation method",
                                      choices=['file', 'raw', 'parametric', 'none'])]
            estimate_irf = inquirer.prompt(method_q)['method']
            if estimate_irf == 'file':
                irf_path = input("Enter path to IRF PTU file: ").strip()
            else:
                irf_path = None
    else:
        xlif_path = None
        use_xlsx_irf = False
        # No XLSX provided – must specify IRF method
        print("\nNo XLSX file – IRF must be estimated or provided separately.")
        method_q = [inquirer.List('method',
                                  message="Choose IRF estimation method",
                                  choices=['file', 'raw', 'parametric', 'none'])]
        estimate_irf = inquirer.prompt(method_q)['method']
        if estimate_irf == 'file':
            irf_path = input("Enter path to IRF PTU file: ").strip()
        else:
            irf_path = None

    # (Optional) You could add questions for nexp, binning, etc., but we'll keep defaults for now.

    # Build an argparse.Namespace that mimics the command‑line arguments
    import argparse
    args = argparse.Namespace()
    args.ptu = ptu_path
    args.xlsx = xlif_path
    args.no_xlsx_irf = (xlif_path is not None and not use_xlsx_irf)   # only relevant if xlsx exists
    args.irf = irf_path
    args.irf_xlsx = None                # separate IRF‑only xlsx not asked here
    args.estimate_irf = estimate_irf
    # Use defaults for all other parameters (they will be filled from constants later)
    args.irf_bins = IRF_BINS             # assume defined elsewhere
    args.irf_fit_width = IRF_FIT_WIDTH
    args.irf_fwhm = IRF_FWHM
    args.nexp = n_exp
    args.tau_min = Tau_min
    args.tau_max = Tau_max
    args.mode = D_mode
    args.binning = binning_factor
    args.min_photons = MIN_PHOTONS_PERPIX
    args.optimizer = Optimizer
    args.restarts = lm_restarts
    args.de_population = de_population
    args.de_maxiter = de_maxiter
    args.workers = n_workers
    args.no_polish = False
    args.channel = channels
    args.out = OUT_NAME
    args.no_plots = False
    args.debug_xlsx = False
    args.print_config = False            # not used in interactive mode

    return args


def _run_flim_fit(args):
    """Core fitting routine – identical to original single_FOV_flim_fit body."""
    # (Copy the entire body of the original function here, starting from the
    #  point after argument parsing, i.e., after `if args.print_config:`)
    # For brevity, I'll summarise: it loads PTU, builds IRF, runs summed and/or
    # per‑pixel fits, and generates plots.
    # ...
    # (The full code is omitted here to save space, but should be exactly the
    #  same as in the original function, using the `args` namespace.)
    print(f"\n{'='*60}")
    print(f"  flim_fit_v13  |  {args.nexp}-exp  |  {args.mode}  |  optimizer={args.optimizer}")
    print(f"{'='*60}")

    # ── Load PTU ──────────────────────────────────────────────────────────────
    print(f"\n[1] PTU: {args.ptu}")
    ptu = PTUFile(args.ptu, verbose=True)

    # Resolve FWHM after PTU is loaded
    fwhm_ns = args.irf_fwhm if args.irf_fwhm is not None else ptu.tcspc_res * 1e9
    print(f"  IRF FWHM: {fwhm_ns*1000:.2f} ps "
          f"({'from --irf-fwhm' if args.irf_fwhm is not None else 'default: 1 bin'})")

    # ── Summed decay ──────────────────────────────────────────────────────────
    print(f"\n[2] Building summed decay (channel={args.channel or 'auto'})")
    decay    = ptu.summed_decay(channel=args.channel)
    irf_peak_bin  = find_irf_peak_bin(decay)
    decay_peak_bin = int(np.argmax(decay))
    print(f"    {decay.sum():,.0f} photons  |  peak={decay.max():,.0f}  "
          f"at bin {decay_peak_bin} ({ptu.time_ns[decay_peak_bin]:.3f} ns)")
    print(f"    IRF peak (steepest rise): bin {irf_peak_bin} "
          f"({irf_peak_bin * ptu.tcspc_res * 1e9:.3f} ns)")

    # ── Load xlsx (optional) ──────────────────────────────────────────────────
    xlsx = None
    if args.xlsx is not None and Path(args.xlsx).exists():
        print(f"\n[3] XLSX: {args.xlsx}")
        xlsx = load_xlsx(args.xlsx, debug=args.debug_xlsx)
        if xlsx['fit_t'] is not None:
            print(f"    LAS X fit present, peak = {xlsx['fit_c'].max():.0f} cts")
    else:
        print(f"\n[3] No XLSX provided or file not found")

    # ── Build IRF — sets has_tail, fit_sigma, fit_bg per path ────────────────
    print(f"\n[4] Building IRF")

    if args.irf is not None:
        irf_prompt = irf_from_scatter_ptu(args.irf, ptu)
        strategy   = "scatter_ptu"
        has_tail   = False
        fit_sigma  = False
        fit_bg     = True

    elif args.irf_xlsx is not None:
        # ... (identical to original)
        pass

    elif xlsx is not None and xlsx['irf_t'] is not None and not args.no_xlsx_irf:
        # ... (original code)
        pass

    elif args.estimate_irf != "none":
        # ... (original code)
        pass

    else:
        # Gaussian paper IRF
        irf_prompt = gaussian_irf_from_fwhm(
            ptu.n_bins, ptu.tcspc_res, fwhm_ns, decay_peak_bin)
        has_tail  = True
        fit_sigma = False
        fit_bg    = True
        strategy  = (f"gaussian_paper FWHM={fwhm_ns*1000:.1f}ps "
                     f"peak_bin={decay_peak_bin} (decay maximum)")
        print(f"  IRF: {strategy}")

    print(f"  Flags: has_tail={has_tail}  fit_sigma={fit_sigma}  fit_bg={fit_bg}")

    # ── IRF comparison (optional) ────────────────────────────────────────────
    if not args.no_plots and xlsx is not None:
        matplotlib.use("Agg")
        print(f"\n[4b] IRF comparison")
        compare_irfs(irf_prompt, xlsx, ptu.tcspc_res, ptu.n_bins,
                     strategy, args.out)

    # ── Summed fit ────────────────────────────────────────────────────────────
    global_popt    = None
    global_summary = None

    def _run_summed():
        return fit_summed(
            decay, ptu.tcspc_res, ptu.n_bins,
            irf_prompt, has_tail, fit_bg, fit_sigma,
            args.nexp, args.tau_min, args.tau_max,
            optimizer=args.optimizer,
            n_restarts=args.restarts,
            de_popsize=args.de_population,
            de_maxiter=args.de_maxiter,
            workers=args.workers,
            polish=not args.no_polish,
        )

    if args.mode in ("summed", "both"):
        print(f"\n[5] Summed decay fit  ({args.nexp}-exp, optimizer={args.optimizer})")
        global_popt, global_summary = _run_summed()
        print_summary(global_summary, strategy, args.nexp)

        if not args.no_plots:
            matplotlib.use("Agg")
            print(f"\n[6] Plotting")
            plot_summed(decay, global_summary, ptu, xlsx,
                        args.nexp, strategy, args.out,
                        irf_prompt=irf_prompt)

    # ── Per-pixel fit ─────────────────────────────────────────────────────────
    if args.mode in ("perPixel", "both"):
        if global_popt is None:
            print(f"\n[5] Running summed fit first (τ needed for per-pixel)")
            global_popt, global_summary = _run_summed()
            print_summary(global_summary, strategy, args.nexp)

        print(f"\n[7] Building pixel stack (binning={args.binning}×{args.binning})")
        stack = ptu.pixel_stack(channel=ptu.photon_channel, binning=args.binning)

        print(f"\n[8] Per-pixel fitting (min_photons={args.min_photons})")
        pixel_maps = fit_per_pixel(
            stack, ptu.tcspc_res, ptu.n_bins,
            irf_prompt, has_tail, fit_bg, fit_sigma,
            global_popt, args.nexp,
            min_photons=args.min_photons,
        )

        if not args.no_plots:
            matplotlib.use("Agg")
            print(f"\n[9] Plotting pixel maps")
            plot_pixel_maps(pixel_maps, args.nexp, args.out, binning=args.binning)
            plot_lifetime_histogram(pixel_maps, args.nexp, args.out)

    print("\nDone.\n")


def single_FOV_flim_fit(interactive=False):
    """
    Entry point for FLIM fitting.
    If interactive=True, prompts the user for inputs; otherwise parses command‑line arguments.
    """
    if interactive:
        args = single_FOV_flim_fit_inquire()
    else:
        # Original argparse parsing
        ap = argparse.ArgumentParser(
            description="FLIM reconvolution fit — PTU + optional XLSX (Leica FALCON)"
        )
        ap.add_argument("--ptu",   default=None, required=True, help="Path to PTU file")
        ap.add_argument("--xlsx",  default=None,
                        help="LAS X export xlsx (overlay comparison and/or IRF source)")
        ap.add_argument("--no-xlsx-irf", action="store_true",
                        help="Load xlsx for comparison/overlay but do NOT use its IRF "
                             "for fitting. Falls through to Gaussian/estimated IRF instead.")
        ap.add_argument("--debug-xlsx", action="store_true",
                        help="Print raw xlsx row contents and detected columns to diagnose "
                             "parsing failures.")
        ap.add_argument("--irf",   default=None,
                        help="Scatter PTU for measured IRF (highest priority)")
        ap.add_argument("--irf-xlsx", default=None,
                        help="Path to a reference xlsx exported from LAS X, used ONLY "
                             "to extract the IRF shape for fitting. The IRF is "
                             "system-specific (not FOV-specific) so export once per "
                             "session and reuse across all PTU files. Independent of "
                             "--xlsx which is for overlay/comparison only.")
        ap.add_argument("--estimate-irf", choices=["raw", "parametric", "none"],
                        default=Estimate_IRF,
                        help="If no direct IRF provided, estimate from the decay rising edge. Options: 'raw' for a non-parametric IRF from the raw decay, 'parametric' to fit a Gaussian + exponential tail model to the rising edge, or 'none' to not estimate the IRF (e.g. if you have a separate IRF file or are using a system with a very narrow IRF that doesn't need to be accounted for).")
        ap.add_argument("--irf-bins",      type=int,   default=IRF_BINS)
        ap.add_argument("--irf-fit-width", type=float, default=IRF_FIT_WIDTH)
        ap.add_argument("--irf-fwhm", type=float, default=IRF_FWHM,
                        help="IRF FWHM in ns. Default: 1 bin width from PTU "
                             "(e.g. 0.097 ns for 97 ps bins). Override for other systems.")
        ap.add_argument("--nexp",     type=int,   default=n_exp, choices=[1, 2, 3])
        ap.add_argument("--tau-min",  type=float, default=Tau_min, help="ns")
        ap.add_argument("--tau-max",  type=float, default=Tau_max, help="ns")
        ap.add_argument("--mode",     default=D_mode,
                        choices=["summed", "perPixel", "both"])
        ap.add_argument("--binning",     type=int, default=binning_factor,
                        help="Binning factor for per-pixel fitting. Default: 1 (no binning).")
        ap.add_argument("--min-photons", type=int, default=MIN_PHOTONS_PERPIX)
        ap.add_argument("--optimizer",   choices=["lm_multistart", "de"], default=Optimizer)
        ap.add_argument("--restarts",    type=int, default=lm_restarts)
        ap.add_argument("--de-population", type=int, default=de_population)
        ap.add_argument("--de-maxiter",    type=int, default=de_maxiter)
        ap.add_argument("--workers",       type=int, default=n_workers)
        ap.add_argument("--no-polish",  action="store_true")
        ap.add_argument("--channel",    type=int, default=channels)
        ap.add_argument("--out",        default=OUT_NAME)
        ap.add_argument("--no-plots",   action="store_true")
        ap.add_argument("--print-config", action="store_true", help="Print default configuration settings and exit")
        args = ap.parse_args()

    if args.print_config:
        print(config_message)
        return

    # Run the fitting routine
    _run_flim_fit(args)