#!/usr/bin/env python
import numpy as np
import warnings
from pathlib import Path
import matplotlib
import argparse
from flimkit.PTU.reader import PTUFile
from flimkit.FLIM.irf_tools import gaussian_irf_from_fwhm, irf_from_scatter_ptu, irf_from_xlsx, irf_from_xlsx_analytical, estimate_irf_from_decay_parametric, estimate_irf_from_decay_raw, reconstruct_irf_from_decay, compare_irfs
from flimkit.FLIM.fitters import fit_summed, fit_per_pixel, MIN_PHOTONS_PERPIX
from flimkit.utils.plotting import plot_summed, plot_pixel_maps, plot_lifetime_histogram
from flimkit.utils.misc import print_summary
from flimkit.utils.xlsx_tools import load_xlsx
from flimkit.FLIM.fit_tools import find_irf_peak_bin
from flimkit.image.tools import make_intensity_image, apply_intensity_threshold, pick_intensity_threshold
from flimkit.utils.enhanced_outputs import save_weighted_tau_images, save_individual_tau_maps
from flimkit.configs import *
from flimkit.interactive import _load_machine_irf_prompt

warnings.filterwarnings("ignore")


def single_FOV_flim_fit_cli():
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
    ap.add_argument("--estimate-irf", choices=["raw", "parametric", "machine_irf", "machine_irf_sigma_full", "machine_irf_sigma_half", "none"],
                    default=Estimate_IRF,
                    help="IRF estimation method. 'machine_irf' uses prebuilt .npy. "
                         "'machine_irf_sigma_full' adds full σ broadening (σ≤3.0). "
                         "'machine_irf_sigma_half' adds capped broadening (σ≤0.5, recommended).")
    ap.add_argument("--machine-irf", default=None,
                    help="Path to machine IRF .npy file (default: built-in). "
                         "Used with --estimate-irf machine_irf variants.")
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
    ap.add_argument("--cost-function", choices=["poisson", "chi2"],
                    default=Cost_function,
                    help="Cost function for summed fit. 'poisson' (default, recommended) "
                         "uses Poisson deviance on raw counts. 'chi2' (legacy) normalises "
                         "by peak and underweights the tail.")
    ap.add_argument("--intensity-threshold", default=INTENSITY_THRESHOLD,
                    help="Min photon-count per pixel. Pixels below this are "
                         "excluded from both summed and per-pixel fits. "
                         "Pass an integer, or 'interactive' to choose visually "
                         "with a slider on the intensity image.")
    ap.add_argument("--tau-display-min", type=float, default=TAU_DISPLAY_MIN,
                    help="Min lifetime (ns) for exported tau images. "
                         "Out-of-range pixels are clipped to this value (LAS X style).")
    ap.add_argument("--tau-display-max", type=float, default=TAU_DISPLAY_MAX,
                    help="Max lifetime (ns) for exported tau images. "
                         "Out-of-range pixels are clipped to this value (LAS X style).")
    ap.add_argument("--intensity-display-min", type=float, default=INTENSITY_DISPLAY_MIN,
                    help="Min intensity for exported intensity images. "
                         "Out-of-range pixels are clipped to this value (LAS X style).")
    ap.add_argument("--intensity-display-max", type=float, default=INTENSITY_DISPLAY_MAX,
                    help="Max intensity for exported intensity images. "
                         "Out-of-range pixels are clipped to this value (LAS X style).")
    ap.add_argument("--print-config", action="store_true", help="Print default configuration settings and exit")
    args = ap.parse_args()

    if args.print_config:
        print(config_message)
        return

    print(f"\n{'='*60}")
    print(f"  flim_fit_v13  |  {args.nexp}-exp  |  {args.mode}  |  optimizer={args.optimizer}")
    print(f"{'='*60}")

    #  Load PTU
    print(f"\n[1] PTU: {args.ptu}")
    ptu = PTUFile(args.ptu, verbose=True)

    # Resolve FWHM after PTU is loaded
    fwhm_ns = args.irf_fwhm if args.irf_fwhm is not None else ptu.tcspc_res * 1e9
    print(f"  IRF FWHM: {fwhm_ns*1000:.2f} ps "
          f"({'from --irf-fwhm' if args.irf_fwhm is not None else 'default: 1 bin'})")

    #  Intensity threshold (optional) 
    intensity_mask = None
    _int_thr = getattr(args, 'intensity_threshold', None)
    if _int_thr is not None:
        print(f"\n[1b] Intensity threshold")
        intensity_img = make_intensity_image(args.ptu, rotate_90_cw=False, save_image=False)
        if str(_int_thr).lower() == 'interactive':
            _int_thr = pick_intensity_threshold(intensity_img)
        else:
            _int_thr = int(_int_thr)
        intensity_mask = apply_intensity_threshold(intensity_img, _int_thr)
        n_kept = int(intensity_mask.sum())
        n_total = intensity_mask.size
        print(f"    Threshold: {_int_thr} photons  →  "
              f"{n_kept:,}/{n_total:,} pixels kept ({100*n_kept/n_total:.1f}%)")

    #  Summed decay 
    print(f"\n[2] Building summed decay (channel={args.channel or 'auto'})")
    if intensity_mask is not None:
        stack_tmp = ptu.pixel_stack(channel=args.channel, binning=1)
        stack_tmp[~intensity_mask] = 0
        decay = stack_tmp.sum(axis=(0, 1))
        del stack_tmp
        print(f"    (Using intensity-masked photons only)")
    else:
        decay = ptu.summed_decay(channel=args.channel)
    # IRF peak from steepest rise of the decay, not from the decay maximum.
    # np.argmax(decay) is the fluorescence convolution peak — shifted right
    # of the true IRF peak by ~1-2 bins depending on the shortest lifetime.
    irf_peak_bin  = find_irf_peak_bin(decay)
    decay_peak_bin = int(np.argmax(decay))
    print(f"    {decay.sum():,.0f} photons  |  peak={decay.max():,.0f}  "
          f"at bin {decay_peak_bin} ({ptu.time_ns[decay_peak_bin]:.3f} ns)")
    print(f"    IRF peak (steepest rise): bin {irf_peak_bin} "
          f"({irf_peak_bin * ptu.tcspc_res * 1e9:.3f} ns)")

    #  Load xlsx (optional) 
    xlsx = None
    if args.xlsx is not None and Path(args.xlsx).exists():
        print(f"\n[3] XLSX: {args.xlsx}")
        xlsx = load_xlsx(args.xlsx, debug=args.debug_xlsx)
        if xlsx['fit_t'] is not None and xlsx['fit_c'] is not None:
            print(f"    LAS X fit present, peak = {xlsx['fit_c'].max():.0f} cts")
    else:
        print(f"\n[3] No XLSX provided or file not found")

    #  Build IRF — sets has_tail, fit_sigma, fit_bg per path 
    print(f"\n[4] Building IRF")

    sigma_max = MACHINE_IRF_SIGMA_MAX_FULL   # default; overridden by sigma variants


    if args.irf is not None:
        # Scatter PTU: IRF fully measured, no tail or sigma needed
        irf_prompt = irf_from_scatter_ptu(args.irf, ptu)
        strategy   = "scatter_ptu"
        has_tail   = False
        fit_sigma  = False
        fit_bg     = True

    elif args.irf_xlsx is not None:
        print(f"  IRF: fitting Leica analytical model to: {args.irf_xlsx}")
        if not Path(args.irf_xlsx).exists():
            raise FileNotFoundError(f"--irf-xlsx file not found: {args.irf_xlsx}")
        irf_ref = load_xlsx(args.irf_xlsx, debug=False)
        if irf_ref['irf_t'] is None or irf_ref['irf_c'] is None:
            raise ValueError(f"No IRF columns found in --irf-xlsx: {args.irf_xlsx}")
        irf_prompt, irf_params = irf_from_xlsx_analytical(
            irf_ref, ptu.n_bins, ptu.tcspc_res, verbose=True)

        # The analytical IRF is centred at t0 from the Gaussian fit (~bin 29).
        # The optimizer consistently finds shift ≈ -1.26 bins because the true
        # IRF onset is at the steepest-rise bin (~27), not the decay peak (29).
        # Pre-shift to irf_peak_bin so the free shift starts near 0.
        irf_current_peak = int(np.argmax(irf_prompt))
        pre_shift = irf_peak_bin - irf_current_peak
        if pre_shift != 0:
            x          = np.arange(ptu.n_bins, dtype=float)
            irf_prompt = np.interp(x - pre_shift, x, irf_prompt,
                                   left=0.0, right=0.0)
            s = irf_prompt.sum()
            if s > 0:
                irf_prompt /= s
            print(f"  Pre-shifted IRF by {pre_shift:+d} bins "
                  f"(from bin {irf_current_peak} → {irf_peak_bin})")

        strategy  = (f"irf_xlsx_analytical ({Path(args.irf_xlsx).name})  "
                     f"FWHM={irf_params['fwhm_ns']*1000:.1f}ps  "
                     f"tail_amp={irf_params['tail_amp']:.3f}  "
                     f"tail_tau={irf_params['tail_tau_ns']:.3f}ns")
        # IRF shape fully determined by analytical fit — tail & sigma not free
        has_tail  = False
        fit_sigma = False
        fit_bg    = True
        print(f"  IRF peak bin after pre-shift = {np.argmax(irf_prompt)}")

    elif xlsx is not None and xlsx['irf_t'] is not None and not args.no_xlsx_irf:
        # xlsx IRF: sparse rising-edge, tail and sigma needed
        irf_prompt = irf_from_xlsx(xlsx, ptu.n_bins, ptu.tcspc_res)
        above      = np.where(irf_prompt >= irf_prompt.max() / 2)[0]
        fwhm_xlsx  = (above[-1] - above[0]) * ptu.tcspc_res * 1e9 if len(above) > 1 else 0
        print(f"  IRF: xlsx prompt  peak bin={int(np.argmax(irf_prompt))}  "
              f"FWHM={fwhm_xlsx:.3f} ns  + tail + σ as free params")
        strategy  = "xlsx"
        has_tail  = True
        fit_sigma = True
        fit_bg    = True

    elif args.estimate_irf not in ("none",) and not args.estimate_irf.startswith("machine_irf"):
        # Rising-edge estimation: tail and sigma still needed
        if args.estimate_irf == "raw":
            irf_prompt = estimate_irf_from_decay_raw(
                decay, ptu.tcspc_res, ptu.n_bins, n_irf_bins=args.irf_bins)
            strategy = "estimated_raw"
        else:
            irf_prompt = estimate_irf_from_decay_parametric(
                decay, ptu.tcspc_res, ptu.n_bins,
                fit_window_width_ns=args.irf_fit_width)
            strategy = "estimated_parametric"
        has_tail  = True
        fit_sigma = True
        fit_bg    = True
        print(f"  IRF: {strategy} + tail + σ as free params")

    elif args.estimate_irf.startswith("machine_irf"):
        irf_prompt, strategy = _load_machine_irf_prompt(
            getattr(args, 'machine_irf', None), ptu.n_bins, decay_peak_bin)
        has_tail  = MACHINE_IRF_FIT_TAIL
        fit_bg    = MACHINE_IRF_FIT_BG
        if args.estimate_irf == "machine_irf":
            fit_sigma = MACHINE_IRF_FIT_SIGMA
        elif args.estimate_irf == "machine_irf_sigma_full":
            fit_sigma = True
            sigma_max = MACHINE_IRF_SIGMA_MAX_FULL
            strategy += " + σ≤{:.1f}".format(sigma_max)
        elif args.estimate_irf == "machine_irf_sigma_half":
            fit_sigma = True
            sigma_max = MACHINE_IRF_SIGMA_MAX_HALF
            strategy += " + σ≤{:.1f}".format(sigma_max)
        print(f"  IRF: {strategy}")

    else:
        # Gaussian (paper equation): FWHM known, σ fixed at 0.
        # has_tail=True — real HyD/SPAD IRF is asymmetric (fast rise, slower
        # fall). A pure symmetric Gaussian cannot fit this. The exponential
        # tail captures detector afterpulsing / slow component.
        # Peak position: Leica places its synthetic IRF at the decay maximum,
        # not at the steepest rise. np.argmax(decay) is correct here.
        # find_irf_peak_bin() is only useful for measured scatter PTU IRFs.
        irf_prompt = gaussian_irf_from_fwhm(
            ptu.n_bins, ptu.tcspc_res, fwhm_ns, decay_peak_bin)
        has_tail  = True    # asymmetric tail needed for real detector IRF
        fit_sigma = False   # FWHM is known — no extra broadening
        fit_bg    = True    # bg free — matches Leica "Tail Offset"
        strategy  = (f"gaussian_paper FWHM={fwhm_ns*1000:.1f}ps "
                     f"peak_bin={decay_peak_bin} (decay maximum)")
        print(f"  IRF: {strategy}")

    print(f"  Flags: has_tail={has_tail}  fit_sigma={fit_sigma}  fit_bg={fit_bg}")

    #  IRF comparison (always run if xlsx present, regardless of IRF path) 
    if not args.no_plots and xlsx is not None:
        matplotlib.use("Agg")
        print(f"\n[4b] IRF comparison")
        compare_irfs(irf_prompt, xlsx, ptu.tcspc_res, ptu.n_bins,
                     strategy, args.out)

    #  Summed fit 
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
            cost_function=args.cost_function,
            sigma_max=sigma_max,
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

    #  Per-pixel fit ─
    if args.mode in ("perPixel", "both"):
        if global_popt is None:
            print(f"\n[5] Running summed fit first (τ needed for per-pixel)")
            global_popt, global_summary = _run_summed()
            print_summary(global_summary, strategy, args.nexp)

        print(f"\n[7] Building pixel stack (binning={args.binning}×{args.binning})")
        stack = ptu.pixel_stack(channel=ptu.photon_channel, binning=args.binning)

        # Apply intensity mask to per-pixel stack
        if intensity_mask is not None:
            import cv2
            sy, sx = stack.shape[:2]
            if intensity_mask.shape != (sy, sx):
                mask_resized = cv2.resize(intensity_mask.astype(np.uint8),
                                          (sx, sy),
                                          interpolation=cv2.INTER_NEAREST) > 0
            else:
                mask_resized = intensity_mask
            stack[~mask_resized] = 0
            print(f"    Applied intensity threshold mask to pixel stack")

        print(f"\n[8] Per-pixel fitting (min_photons={args.min_photons})")
        pixel_maps = fit_per_pixel(
            stack, ptu.tcspc_res, ptu.n_bins,
            irf_prompt, has_tail, fit_bg, fit_sigma,
            global_popt, args.nexp,
            min_photons=args.min_photons,
        )

        # Save weighted tau and intensity images
        roi_name = Path(args.ptu).stem
        save_weighted_tau_images(
            pixel_maps,
            Path(args.out),
            roi_name=roi_name,
            n_exp=args.nexp,
            save_intensity=True,
            save_amplitude=True,
            tau_display_min=getattr(args, "tau_display_min", None),
            tau_display_max=getattr(args, "tau_display_max", None),
            intensity_display_min=getattr(args, "intensity_display_min", None),
            intensity_display_max=getattr(args, "intensity_display_max", None),
        )

        if not args.no_plots:
            matplotlib.use("Agg")
            print(f"\n[9] Plotting pixel maps")
            plot_pixel_maps(pixel_maps, args.nexp, args.out, binning=args.binning)
            plot_lifetime_histogram(pixel_maps, args.nexp, args.out)

    print("\nDone.\n")

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
    ap.add_argument("--estimate-irf", choices=["raw", "parametric", "machine_irf", "machine_irf_sigma_full", "machine_irf_sigma_half", "none"],
                    default=Estimate_IRF,
                    help="IRF estimation method. 'machine_irf' uses prebuilt .npy. "
                         "'machine_irf_sigma_full' adds full σ broadening (σ≤3.0). "
                         "'machine_irf_sigma_half' adds capped broadening (σ≤0.5, recommended).")
    ap.add_argument("--machine-irf", default=None,
                    help="Path to machine IRF .npy file (default: built-in). "
                         "Used with --estimate-irf machine_irf variants.")
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
    ap.add_argument("--cost-function", choices=["poisson", "chi2"],
                    default=Cost_function,
                    help="Cost function for summed fit. 'poisson' (default, recommended) "
                         "uses Poisson deviance on raw counts. 'chi2' (legacy) normalises "
                         "by peak and underweights the tail.")
    ap.add_argument("--intensity-threshold", default=INTENSITY_THRESHOLD,
                    help="Min photon-count per pixel. Pixels below this are "
                         "excluded from both summed and per-pixel fits. "
                         "Pass an integer, or 'interactive' to choose visually "
                         "with a slider on the intensity image.")
    ap.add_argument("--tau-display-min", type=float, default=TAU_DISPLAY_MIN,
                    help="Min lifetime (ns) for exported tau images. "
                         "Out-of-range pixels are clipped to this value (LAS X style).")
    ap.add_argument("--tau-display-max", type=float, default=TAU_DISPLAY_MAX,
                    help="Max lifetime (ns) for exported tau images. "
                         "Out-of-range pixels are clipped to this value (LAS X style).")
    ap.add_argument("--intensity-display-min", type=float, default=INTENSITY_DISPLAY_MIN,
                    help="Min intensity for exported intensity images. "
                         "Out-of-range pixels are clipped to this value (LAS X style).")
    ap.add_argument("--intensity-display-max", type=float, default=INTENSITY_DISPLAY_MAX,
                    help="Max intensity for exported intensity images. "
                         "Out-of-range pixels are clipped to this value (LAS X style).")
    ap.add_argument("--print-config", action="store_true", help="Print default configuration settings and exit")
    args = ap.parse_args()

    if args.print_config:
        print(config_message)
        return

    print(f"\n{'='*60}")
    print(f"  flim_fit_v13  |  {args.nexp}-exp  |  {args.mode}  |  optimizer={args.optimizer}")
    print(f"{'='*60}")

    #  Load PTU 
    print(f"\n[1] PTU: {args.ptu}")
    ptu = PTUFile(args.ptu, verbose=True)

    # Resolve FWHM after PTU is loaded
    fwhm_ns = args.irf_fwhm if args.irf_fwhm is not None else ptu.tcspc_res * 1e9
    print(f"  IRF FWHM: {fwhm_ns*1000:.2f} ps "
          f"({'from --irf-fwhm' if args.irf_fwhm is not None else 'default: 1 bin'})")

    #  Intensity threshold (optional) 
    intensity_mask = None
    _int_thr = getattr(args, 'intensity_threshold', None)
    if _int_thr is not None:
        print(f"\n[1b] Intensity threshold")
        intensity_img = make_intensity_image(args.ptu, rotate_90_cw=False, save_image=False)
        if str(_int_thr).lower() == 'interactive':
            _int_thr = pick_intensity_threshold(intensity_img)
        else:
            _int_thr = int(_int_thr)
        intensity_mask = apply_intensity_threshold(intensity_img, _int_thr)
        n_kept = int(intensity_mask.sum())
        n_total = intensity_mask.size
        print(f"    Threshold: {_int_thr} photons  →  "
              f"{n_kept:,}/{n_total:,} pixels kept ({100*n_kept/n_total:.1f}%)")

    #  Summed decay 
    print(f"\n[2] Building summed decay (channel={args.channel or 'auto'})")
    if intensity_mask is not None:
        stack_tmp = ptu.pixel_stack(channel=args.channel, binning=1)
        stack_tmp[~intensity_mask] = 0
        decay = stack_tmp.sum(axis=(0, 1))
        del stack_tmp
        print(f"    (Using intensity-masked photons only)")
    else:
        decay    = ptu.summed_decay(channel=args.channel)
    # IRF peak from steepest rise of the decay, not from the decay maximum.
    # np.argmax(decay) is the fluorescence convolution peak — shifted right
    # of the true IRF peak by ~1-2 bins depending on the shortest lifetime.
    irf_peak_bin  = find_irf_peak_bin(decay)
    decay_peak_bin = int(np.argmax(decay))
    print(f"    {decay.sum():,.0f} photons  |  peak={decay.max():,.0f}  "
          f"at bin {decay_peak_bin} ({ptu.time_ns[decay_peak_bin]:.3f} ns)")
    print(f"    IRF peak (steepest rise): bin {irf_peak_bin} "
          f"({irf_peak_bin * ptu.tcspc_res * 1e9:.3f} ns)")

    #  Load xlsx (optional) 
    xlsx = None
    if args.xlsx is not None and Path(args.xlsx).exists():
        print(f"\n[3] XLSX: {args.xlsx}")
        xlsx = load_xlsx(args.xlsx, debug=args.debug_xlsx)
        if xlsx['fit_t'] is not None and xlsx['fit_c'] is not None:
            print(f"    LAS X fit present, peak = {xlsx['fit_c'].max():.0f} cts")
    else:
        print(f"\n[3] No XLSX provided or file not found")

    #  Build IRF — sets has_tail, fit_sigma, fit_bg per path 
    print(f"\n[4] Building IRF")

    sigma_max = MACHINE_IRF_SIGMA_MAX_FULL   # default; overridden by sigma variants


    if args.irf is not None:
        # Scatter PTU: IRF fully measured, no tail or sigma needed
        irf_prompt = irf_from_scatter_ptu(args.irf, ptu)
        strategy   = "scatter_ptu"
        has_tail   = False
        fit_sigma  = False
        fit_bg     = True

    elif args.irf_xlsx is not None:
        print(f"  IRF: fitting Leica analytical model to: {args.irf_xlsx}")
        if not Path(args.irf_xlsx).exists():
            raise FileNotFoundError(f"--irf-xlsx file not found: {args.irf_xlsx}")
        irf_ref = load_xlsx(args.irf_xlsx, debug=False)
        if irf_ref['irf_t'] is None or irf_ref['irf_c'] is None:
            raise ValueError(f"No IRF columns found in --irf-xlsx: {args.irf_xlsx}")
        irf_prompt, irf_params = irf_from_xlsx_analytical(
            irf_ref, ptu.n_bins, ptu.tcspc_res, verbose=True)

        # The analytical IRF is centred at t0 from the Gaussian fit (~bin 29).
        # The optimizer consistently finds shift ≈ -1.26 bins because the true
        # IRF onset is at the steepest-rise bin (~27), not the decay peak (29).
        # Pre-shift to irf_peak_bin so the free shift starts near 0.
        irf_current_peak = int(np.argmax(irf_prompt))
        pre_shift = irf_peak_bin - irf_current_peak
        if pre_shift != 0:
            x          = np.arange(ptu.n_bins, dtype=float)
            irf_prompt = np.interp(x - pre_shift, x, irf_prompt,
                                   left=0.0, right=0.0)
            s = irf_prompt.sum()
            if s > 0:
                irf_prompt /= s
            print(f"  Pre-shifted IRF by {pre_shift:+d} bins "
                  f"(from bin {irf_current_peak} → {irf_peak_bin})")

        strategy  = (f"irf_xlsx_analytical ({Path(args.irf_xlsx).name})  "
                     f"FWHM={irf_params['fwhm_ns']*1000:.1f}ps  "
                     f"tail_amp={irf_params['tail_amp']:.3f}  "
                     f"tail_tau={irf_params['tail_tau_ns']:.3f}ns")
        # IRF shape fully determined by analytical fit — tail & sigma not free
        has_tail  = False
        fit_sigma = False
        fit_bg    = True
        print(f"  IRF peak bin after pre-shift = {np.argmax(irf_prompt)}")

    elif xlsx is not None and xlsx['irf_t'] is not None and not args.no_xlsx_irf:
        # xlsx IRF: sparse rising-edge, tail and sigma needed
        irf_prompt = irf_from_xlsx(xlsx, ptu.n_bins, ptu.tcspc_res)
        above      = np.where(irf_prompt >= irf_prompt.max() / 2)[0]
        fwhm_xlsx  = (above[-1] - above[0]) * ptu.tcspc_res * 1e9 if len(above) > 1 else 0
        print(f"  IRF: xlsx prompt  peak bin={int(np.argmax(irf_prompt))}  "
              f"FWHM={fwhm_xlsx:.3f} ns  + tail + σ as free params")
        strategy  = "xlsx"
        has_tail  = True
        fit_sigma = True
        fit_bg    = True

    elif args.estimate_irf not in ("none",) and not args.estimate_irf.startswith("machine_irf"):
        # Rising-edge estimation: tail and sigma still needed
        if args.estimate_irf == "raw":
            irf_prompt = estimate_irf_from_decay_raw(
                decay, ptu.tcspc_res, ptu.n_bins, n_irf_bins=args.irf_bins)
            strategy = "estimated_raw"
        else:
            irf_prompt = estimate_irf_from_decay_parametric(
                decay, ptu.tcspc_res, ptu.n_bins,
                fit_window_width_ns=args.irf_fit_width)
            strategy = "estimated_parametric"
        has_tail  = True
        fit_sigma = True
        fit_bg    = True
        print(f"  IRF: {strategy} + tail + σ as free params")

    elif args.estimate_irf.startswith("machine_irf"):
        irf_prompt, strategy = _load_machine_irf_prompt(
            getattr(args, 'machine_irf', None), ptu.n_bins, decay_peak_bin)
        has_tail  = MACHINE_IRF_FIT_TAIL
        fit_bg    = MACHINE_IRF_FIT_BG
        if args.estimate_irf == "machine_irf":
            fit_sigma = MACHINE_IRF_FIT_SIGMA
        elif args.estimate_irf == "machine_irf_sigma_full":
            fit_sigma = True
            sigma_max = MACHINE_IRF_SIGMA_MAX_FULL
            strategy += " + σ≤{:.1f}".format(sigma_max)
        elif args.estimate_irf == "machine_irf_sigma_half":
            fit_sigma = True
            sigma_max = MACHINE_IRF_SIGMA_MAX_HALF
            strategy += " + σ≤{:.1f}".format(sigma_max)
        print(f"  IRF: {strategy}")

    else:
        # Gaussian (paper equation): FWHM known, σ fixed at 0.
        # has_tail=True — real HyD/SPAD IRF is asymmetric (fast rise, slower
        # fall). A pure symmetric Gaussian cannot fit this. The exponential
        # tail captures detector afterpulsing / slow component.
        # Peak position: Leica places its synthetic IRF at the decay maximum,
        # not at the steepest rise. np.argmax(decay) is correct here.
        # find_irf_peak_bin() is only useful for measured scatter PTU IRFs.
        irf_prompt = gaussian_irf_from_fwhm(
            ptu.n_bins, ptu.tcspc_res, fwhm_ns, decay_peak_bin)
        has_tail  = True    # asymmetric tail needed for real detector IRF
        fit_sigma = False   # FWHM is known — no extra broadening
        fit_bg    = True    # bg free — matches Leica "Tail Offset"
        strategy  = (f"gaussian_paper FWHM={fwhm_ns*1000:.1f}ps "
                     f"peak_bin={decay_peak_bin} (decay maximum)")
        print(f"  IRF: {strategy}")

    print(f"  Flags: has_tail={has_tail}  fit_sigma={fit_sigma}  fit_bg={fit_bg}")

    #  IRF comparison (always run if xlsx present, regardless of IRF path) 
    if not args.no_plots and xlsx is not None:
        matplotlib.use("Agg")
        print(f"\n[4b] IRF comparison")
        compare_irfs(irf_prompt, xlsx, ptu.tcspc_res, ptu.n_bins,
                     strategy, args.out)

    #  Summed fit 
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
            cost_function=args.cost_function,
            sigma_max=sigma_max,
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

    #  Per-pixel fit ─
    if args.mode in ("perPixel", "both"):
        if global_popt is None:
            print(f"\n[5] Running summed fit first (τ needed for per-pixel)")
            global_popt, global_summary = _run_summed()
            print_summary(global_summary, strategy, args.nexp)

        print(f"\n[7] Building pixel stack (binning={args.binning}×{args.binning})")
        stack = ptu.pixel_stack(channel=ptu.photon_channel, binning=args.binning)

        # Apply intensity mask to per-pixel stack
        if intensity_mask is not None:
            import cv2
            sy, sx = stack.shape[:2]
            if intensity_mask.shape != (sy, sx):
                mask_resized = cv2.resize(intensity_mask.astype(np.uint8),
                                          (sx, sy),
                                          interpolation=cv2.INTER_NEAREST) > 0
            else:
                mask_resized = intensity_mask
            stack[~mask_resized] = 0
            print(f"    Applied intensity threshold mask to pixel stack")

        print(f"\n[8] Per-pixel fitting (min_photons={args.min_photons})")
        pixel_maps = fit_per_pixel(
            stack, ptu.tcspc_res, ptu.n_bins,
            irf_prompt, has_tail, fit_bg, fit_sigma,
            global_popt, args.nexp,
            min_photons=args.min_photons,
        )

        # Save weighted tau and intensity images
        roi_name = Path(args.ptu).stem
        save_weighted_tau_images(
            pixel_maps,
            Path(args.out),
            roi_name=roi_name,
            n_exp=args.nexp,
            save_intensity=True,
            save_amplitude=True,
            tau_display_min=getattr(args, "tau_display_min", None),
            tau_display_max=getattr(args, "tau_display_max", None),
            intensity_display_min=getattr(args, "intensity_display_min", None),
            intensity_display_max=getattr(args, "intensity_display_max", None),
        )

        if not args.no_plots:
            matplotlib.use("Agg")
            print(f"\n[9] Plotting pixel maps")
            plot_pixel_maps(pixel_maps, args.nexp, args.out, binning=args.binning)
            plot_lifetime_histogram(pixel_maps, args.nexp, args.out)

    print("\nDone.\n")

if __name__ == "__main__":
    single_FOV_flim_fit_cli()    