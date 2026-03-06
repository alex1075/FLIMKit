import inquirer
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
from .configs import (
    n_exp, Tau_min, Tau_max, D_mode, binning_factor, MIN_PHOTONS_PERPIX, Optimizer, lm_restarts, de_population, de_maxiter, n_workers, OUT_NAME, Estimate_IRF, IRF_BINS, IRF_FIT_WIDTH, IRF_FWHM, channels, config_message, INTENSITY_THRESHOLD)
from .PTU.reader import PTUFile
from .PTU.stitch import stitch_flim_tiles, load_flim_for_fitting  
from .utils.xml_utils import parse_xlif_tile_positions 
from .FLIM.fit_tools import find_irf_peak_bin
from .FLIM.irf_tools import irf_from_scatter_ptu, gaussian_irf_from_fwhm, compare_irfs, estimate_irf_from_decay_parametric, estimate_irf_from_decay_raw, irf_from_xlsx, irf_from_xlsx_analytical
from .FLIM.fitters import fit_summed, fit_per_pixel
from .utils.xlsx_tools import load_xlsx
from .utils.misc import print_summary
from .utils.plotting import plot_summed, plot_pixel_maps, plot_lifetime_histogram
from .utils.enhanced_outputs import (
    save_fit_summary_txt,
    save_weighted_tau_images,
    save_individual_tau_maps,
    create_complete_output_package
)
from .image.tools import make_intensity_image, make_cell_mask, apply_intensity_threshold, pick_intensity_threshold

def yes_no_question(question):
    """Ask a yes/no question using inquirer and return 'y' or 'n'."""
    questions = [inquirer.List('yesno',
                               message=question,
                               choices=['Yes', 'No'])]
    answer = inquirer.prompt(questions)
    return 'y' if answer['yesno'] == 'Yes' else 'n'


def stitch_tiles_inquire():
    """Interactive prompt for tile stitching parameters."""
    print("\n--- ROI Reconstruction: Tile Stitching ---")
    
    # XLIF metadata file
    xlif_path = input("Enter path to XLIF metadata file: ").strip()
    if not xlif_path or not Path(xlif_path).exists():
        raise ValueError(f"XLIF file not found: {xlif_path}")
    
    # PTU directory
    ptu_dir = input("Enter directory containing PTU tiles: ").strip()
    if not ptu_dir or not Path(ptu_dir).exists():
        raise ValueError(f"PTU directory not found: {ptu_dir}")
    
    # Output directory – now we automatically create a subdirectory named after the ROI
    base_output = input("Enter base output directory (a subfolder named after the ROI will be created inside it): ").strip()
    if not base_output:
        raise ValueError("Output directory is required")
    
    # Extract PTU basename from XLIF filename (e.g., "R 2.xlif" -> "R 2")
    ptu_basename = Path(xlif_path).stem
    roi_clean = ptu_basename.replace(' ', '_')
    output_dir = str(Path(base_output) / roi_clean)
    print(f"  Data will be saved in: {output_dir}")
    
    # Ask about rotation
    rotate_q = yes_no_question("Apply 90° clockwise rotation to tiles? (Recommended for Leica data)")
    rotate_tiles = (rotate_q == 'y')
    
    # Build namespace
    args = argparse.Namespace()
    args.xlif = xlif_path
    args.ptu_dir = ptu_dir
    args.output_dir = output_dir
    args.ptu_basename = ptu_basename
    args.rotate_tiles = rotate_tiles
    
    return args


def stitch_and_fit_inquire():
    """Interactive prompt for combined stitch + fit workflow."""
    print("\n--- ROI Reconstruction + FLIM Fitting ---")
    
    # First get stitching parameters
    print("\nStep 1: Tile Stitching Setup")
    stitch_args = stitch_tiles_inquire()   # this already creates the ROI subdirectory
    
    # Then get fitting parameters
    print("\nStep 2: FLIM Fitting Setup")
    
    # IRF method
    print("\nIRF estimation options:")
    print("  1. 'irf_xlsx'   - analytical Gaussian+tail fit from XLSX (recommended)")
    print("  2. 'file'       - measured IRF image (scatter PTU)")
    print("  3. 'raw'        - non-parametric IRF from raw decay")
    print("  4. 'parametric' - fit Gaussian + exponential tail")
    print("  5. 'gaussian'   - simple Gaussian IRF (fastest)")
    method_q = [inquirer.List('method',
                              message="Choose IRF estimation method",
                              choices=['irf_xlsx', 'file', 'raw', 'parametric', 'gaussian'])]
    estimate_irf = inquirer.prompt(method_q)['method']
    
    irf_path = None
    irf_xlsx_path = None
    if estimate_irf == 'irf_xlsx':
        irf_xlsx_path = input("Enter path to IRF XLSX file: ").strip()
        if not irf_xlsx_path or not Path(irf_xlsx_path).exists():
            print("  Warning: XLSX file not found, falling back to Gaussian")
            estimate_irf = 'gaussian'
            irf_xlsx_path = None
    elif estimate_irf == 'file':
        irf_path = input("Enter path to IRF PTU file: ").strip()
        if not irf_path or not Path(irf_path).exists():
            print("  Warning: IRF file not found, falling back to Gaussian")
            estimate_irf = 'gaussian'
            irf_path = None
    
    # Number of exponentials
    nexp_q = [inquirer.List('nexp',
                            message="Number of exponential components",
                            choices=['1', '2', '3'],
                            default='2')]
    nexp = int(inquirer.prompt(nexp_q)['nexp'])
    
    # Per-pixel fitting
    perpixel_q = yes_no_question("Perform per-pixel fitting? (slower but gives lifetime maps)")
    fit_per_pixel_mode = (perpixel_q == 'y')
    
    # Ask about saving individual component maps
    save_individual_q = yes_no_question("Save individual component maps? (tau1, tau2, a1, a2) (default: No)")
    save_individual = (save_individual_q == 'y')
    
    # Auto-select optimizer: DE with log-tau reparameterisation for robust
    # global search.  The optimizer only affects the summed (global) fit —
    # per-pixel fitting always uses NNLS — so there is no speed penalty.
    optimizer_choice = 'de'
    print(f"\n[Auto] Using optimizer: de (log-tau + Sobol, robust global search)")
    
    # Lifetime display range for weighted tau images
    if fit_per_pixel_mode:
        tau_range_q = yes_no_question("Set a lifetime display range for tau images? (e.g. 0-5 ns)")
        if tau_range_q == 'y':
            tau_min_display = input("  Min lifetime (ns, press Enter for 0): ").strip()
            tau_max_display = input("  Max lifetime (ns, press Enter for no limit): ").strip()
            tau_min_display = float(tau_min_display) if tau_min_display else 0.0
            tau_max_display = float(tau_max_display) if tau_max_display else None
        else:
            tau_min_display = None
            tau_max_display = None
    else:
        tau_min_display = None
        tau_max_display = None
    
    # Intensity threshold
    thresh_q = yes_no_question("Apply an intensity threshold to exclude low-signal pixels?")
    if thresh_q == 'y':
        thresh_mode_q = [inquirer.List('tmode',
                                       message="Choose threshold method",
                                       choices=['Enter a value', 'Pick interactively (slider)'],
                                       default='Pick interactively (slider)')]
        thresh_mode = inquirer.prompt(thresh_mode_q)['tmode']
        if thresh_mode == 'Enter a value':
            val = input("  Min photon intensity per pixel: ").strip()
            intensity_threshold = int(val) if val else None
            intensity_threshold_interactive = False
        else:
            intensity_threshold = 'interactive'
            intensity_threshold_interactive = True
    else:
        intensity_threshold = None
        intensity_threshold_interactive = False

    # Build combined namespace
    args = argparse.Namespace()
    
    # Stitching parameters
    args.xlif = stitch_args.xlif
    args.ptu_dir = stitch_args.ptu_dir
    args.output_dir = stitch_args.output_dir    # already includes ROI subfolder
    args.ptu_basename = stitch_args.ptu_basename
    args.rotate_tiles = stitch_args.rotate_tiles
    
    # Fitting parameters
    args.irf = irf_path
    args.irf_xlsx = irf_xlsx_path
    args.estimate_irf = estimate_irf
    args.nexp = nexp
    args.tau_min = Tau_min
    args.tau_max = Tau_max
    args.mode = 'both' if fit_per_pixel_mode else 'summed'
    args.binning = binning_factor
    args.min_photons = MIN_PHOTONS_PERPIX
    args.optimizer = optimizer_choice
    args.restarts = lm_restarts
    args.de_population = de_population
    args.de_maxiter = de_maxiter
    args.workers = n_workers
    args.no_polish = False
    args.channel = channels
    args.irf_fwhm = IRF_FWHM
    args.irf_bins = IRF_BINS
    args.irf_fit_width = IRF_FIT_WIDTH
    args.no_plots = False
    args.intensity_threshold = intensity_threshold
    
    # Output control flags
    args.save_individual = save_individual
    args.save_weighted = True   # always save weighted tau images
    args.tau_display_min = tau_min_display
    args.tau_display_max = tau_max_display
    
    return args


def _run_tile_stitch(args):
    """Execute tile stitching workflow."""
    print(f"\n{'='*60}")
    print(f"  TILE STITCHING")
    print(f"{'='*60}")
    
    result = stitch_flim_tiles(
        xlif_path=Path(args.xlif),
        ptu_dir=Path(args.ptu_dir),
        output_dir=Path(args.output_dir),
        ptu_basename=args.ptu_basename,
        rotate_tiles=args.rotate_tiles,
        verbose=True,
    )
    
    if result['tiles_processed'] == 0:
        raise RuntimeError("No tiles were successfully stitched!")
    
    print(f"\n{'='*60}")
    print(f"  STITCHING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {result['tiles_processed']}/{result['tiles_processed'] + result['tiles_skipped']} tiles")
    print(f"\nOutputs:")
    print(f"  Intensity: {result['intensity_path']}")
    print(f"  FLIM data: {result['flim_path']}")
    print(f"  Time axis: {result['time_axis_path']}")
    print(f"  Metadata:  {result['metadata_path']}")
    
    return result


def _run_stitch_and_fit(args):
    """Execute combined stitch + fit workflow."""
    
    # Step 1: Stitch tiles
    print(f"\n{'='*60}")
    print(f"  STEP 1: TILE STITCHING")
    print(f"{'='*60}")
    
    stitch_result = stitch_flim_tiles(
        xlif_path=Path(args.xlif),
        ptu_dir=Path(args.ptu_dir),
        output_dir=Path(args.output_dir),
        ptu_basename=args.ptu_basename,
        rotate_tiles=args.rotate_tiles,
        verbose=True,
    )
    
    if stitch_result['tiles_processed'] == 0:
        raise RuntimeError("No tiles were successfully stitched!")
    
    print(f"\nStitching complete: {stitch_result['tiles_processed']} tiles processed")
    
    # Derive ROI name from PTU basename (replace spaces with underscores)
    roi_name = args.ptu_basename.replace(' ', '_')
    
    # Step 2: Load stitched data
    print(f"\n{'='*60}")
    print(f"  STEP 2: LOADING STITCHED DATA")
    print(f"{'='*60}")
    
    stack, tcspc_res, n_bins = load_flim_for_fitting(
        Path(args.output_dir),
        load_to_memory=True  # Load to RAM for faster fitting
    )
    
    print(f"  Loaded: {stack.shape} @ {tcspc_res*1e12:.2f} ps/bin")
    print(f"  Total photons: {stack.sum():,.0f}")
    
    # Step 3: Prepare for fitting (create mock PTU object with necessary attributes)
    class StitchedData:
        """Mock PTU object for stitched data."""
        def __init__(self, stack, tcspc_res, n_bins):
            self.n_bins = n_bins
            self.tcspc_res = tcspc_res
            self.time_ns = np.arange(n_bins) * tcspc_res * 1e9
            self.photon_channel = None  # Already channel-selected
            self._stack = stack
        
        def summed_decay(self, channel=None):
            """Return summed decay."""
            return self._stack.sum(axis=(0, 1))
        
        def pixel_stack(self, channel=None, binning=1):
            """Return pixel stack."""
            if binning == 1:
                return self._stack
            else:
                # Simple binning
                ny, nx, nt = self._stack.shape
                new_ny = ny // binning
                new_nx = nx // binning
                binned = np.zeros((new_ny, new_nx, nt), dtype=self._stack.dtype)
                for i in tqdm(range(new_ny), desc='  Binning rows'):
                    for j in range(new_nx):
                        binned[i, j, :] = self._stack[
                            i*binning:(i+1)*binning,
                            j*binning:(j+1)*binning,
                            :
                        ].sum(axis=(0, 1))
                return binned
    
    ptu = StitchedData(stack, tcspc_res, n_bins)
    
    # Build tissue mask from stitched intensity to exclude background pixels
    print(f"\n  Building tissue mask from stitched intensity...")
    intensity_2d = stack.sum(axis=2)   # raw photon-count intensity
    tissue_mask = make_cell_mask(intensity_2d,
                                 save_mask=True,
                                 path=str(Path(args.output_dir) / roi_name))
    n_tissue = int(tissue_mask.sum())
    n_total  = tissue_mask.size
    print(f"    Tissue mask: {n_tissue:,} / {n_total:,} pixels "
          f"({100*n_tissue/n_total:.1f}%) are tissue")

    # Apply intensity threshold (optional)
    _int_thr = getattr(args, 'intensity_threshold', None)
    if _int_thr is not None:
        print(f"\n  Applying intensity threshold...")
        if _int_thr == 'interactive':
            _int_thr = pick_intensity_threshold(intensity_2d)
        else:
            _int_thr = int(_int_thr)
        int_mask = apply_intensity_threshold(intensity_2d, _int_thr)
        tissue_mask = tissue_mask & int_mask
        n_after = int(tissue_mask.sum())
        print(f"    Intensity threshold: {_int_thr} photons  →  "
              f"{n_after:,}/{n_total:,} pixels kept ({100*n_after/n_total:.1f}%)")
    
    # Zero out background in the stack for summed-decay fitting
    stack_masked = stack.copy()
    stack_masked[~tissue_mask] = 0
    
    # Step 4: Fit using existing fitting code
    print(f"\n{'='*60}")
    print(f"  STEP 3: FLIM FITTING")
    print(f"{'='*60}")
    print(f"  {args.nexp}-exp  |  {args.mode}  |  optimizer={args.optimizer}")
    
    # Get summed decay (tissue-only)
    decay = stack_masked.sum(axis=(0, 1))
    irf_peak_bin = find_irf_peak_bin(decay)
    decay_peak_bin = int(np.argmax(decay))
    
    print(f"\nSummed decay (tissue-masked):")
    print(f"  {decay.sum():,.0f} photons  |  peak={decay.max():,.0f} at bin {decay_peak_bin}")
    print(f"  IRF peak (steepest rise): bin {irf_peak_bin} ({irf_peak_bin * tcspc_res * 1e9:.3f} ns)")
    
    # Build IRF
    print(f"\nBuilding IRF (method: {args.estimate_irf})...")
    
    if args.irf is not None and Path(args.irf).exists():
        # Load scatter IRF
        irf_prompt = irf_from_scatter_ptu(args.irf, ptu)
        strategy = "scatter_ptu"
        has_tail = False
        fit_sigma = False
        fit_bg = True
        print(f"  IRF: scatter PTU from {args.irf}")

    elif getattr(args, 'irf_xlsx', None) is not None:
        print(f"  IRF: fitting Leica analytical model to: {args.irf_xlsx}")
        if not Path(args.irf_xlsx).exists():
            raise FileNotFoundError(f"IRF XLSX file not found: {args.irf_xlsx}")
        irf_ref = load_xlsx(args.irf_xlsx, debug=False)
        if irf_ref['irf_t'] is None or irf_ref['irf_c'] is None:
            raise ValueError(f"No IRF columns found in XLSX: {args.irf_xlsx}")
        irf_prompt, irf_params = irf_from_xlsx_analytical(
            irf_ref, n_bins, tcspc_res, verbose=True)

        # Pre-shift analytical IRF to align with steepest-rise bin
        irf_current_peak = int(np.argmax(irf_prompt))
        pre_shift = irf_peak_bin - irf_current_peak
        if pre_shift != 0:
            x          = np.arange(n_bins, dtype=float)
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
        has_tail  = False
        fit_sigma = False
        fit_bg    = True
        print(f"  IRF peak bin after pre-shift = {np.argmax(irf_prompt)}")

    elif args.estimate_irf == "raw":
        from .FLIM.irf_tools import estimate_irf_from_decay_raw
        irf_prompt = estimate_irf_from_decay_raw(
            decay, ptu.tcspc_res, args.irf_bins, args.irf_fit_width, irf_peak_bin
        )
        strategy = f"estimated_raw (bins={args.irf_bins}, width={args.irf_fit_width})"
        has_tail = True
        fit_sigma = True
        fit_bg = True
        print(f"  IRF: {strategy}")
    
    elif args.estimate_irf == "parametric":
        from .FLIM.irf_tools import estimate_irf_from_decay_parametric
        irf_prompt = estimate_irf_from_decay_parametric(
            decay, ptu.tcspc_res, args.irf_bins, args.irf_fit_width, irf_peak_bin
        )
        strategy = f"estimated_parametric (bins={args.irf_bins}, width={args.irf_fit_width})"
        has_tail = True
        fit_sigma = True
        fit_bg = True
        print(f"  IRF: {strategy}")
    
    else:
        # Gaussian IRF (default)
        fwhm_ns = args.irf_fwhm if args.irf_fwhm is not None else tcspc_res * 1e9
        irf_prompt = gaussian_irf_from_fwhm(
            n_bins, tcspc_res, fwhm_ns, decay_peak_bin
        )
        strategy = f"gaussian FWHM={fwhm_ns*1000:.1f}ps"
        has_tail = False
        fit_sigma = False
        fit_bg = True
        print(f"  IRF: {strategy}")
    
    print(f"  Flags: has_tail={has_tail}  fit_sigma={fit_sigma}  fit_bg={fit_bg}")
    
    # Fit summed decay
    print(f"\nFitting summed decay ({args.nexp}-exp, optimizer={args.optimizer})...")
    
    global_popt, global_summary = fit_summed(
        decay, tcspc_res, n_bins,
        irf_prompt, has_tail, fit_bg, fit_sigma,
        args.nexp, args.tau_min, args.tau_max,
        optimizer=args.optimizer,
        n_restarts=args.restarts,
        de_popsize=args.de_population,
        de_maxiter=args.de_maxiter,
        workers=args.workers,
        polish=not args.no_polish,
        cost_function=getattr(args, 'cost_function', 'poisson'),
    )
    
    print_summary(global_summary, strategy, args.nexp)
    
    # --- Save fit summary text file ---
    metadata = {
        'canvas_shape': stack.shape[:2],
        'total_photons': int(stack.sum()),
        'tiles_processed': stitch_result['tiles_processed'],
    }
    save_fit_summary_txt(
        global_summary,
        Path(args.output_dir) / f"{roi_name}_fit_summary.txt",
        n_exp=args.nexp,
        strategy=strategy,
        metadata=metadata
    )
    
    # Plot summed fit
    if not args.no_plots:
        matplotlib.use("Agg")
        output_name = Path(args.output_dir) / "fit_results"
        print(f"\nGenerating plots...")
        plot_summed(
            decay, global_summary, ptu, None,  # No xlsx
            args.nexp, strategy, str(output_name),
            irf_prompt=irf_prompt
        )
    
    # Per-pixel fitting (if requested)
    pixel_maps = None
    if args.mode in ("perPixel", "both"):
        print(f"\nBuilding pixel stack (binning={args.binning}×{args.binning})...")
        pixel_stack = ptu.pixel_stack(channel=None, binning=args.binning)
        
        # Apply tissue mask to per-pixel stack
        if tissue_mask is not None:
            sy, sx, _ = pixel_stack.shape
            if tissue_mask.shape != (sy, sx):
                import cv2
                mask_resized = cv2.resize(tissue_mask.astype(np.uint8), (sx, sy),
                                          interpolation=cv2.INTER_NEAREST) > 0
            else:
                mask_resized = tissue_mask
            pixel_stack[~mask_resized] = 0
            print(f"  Applied tissue mask to pixel stack")
        
        print(f"Per-pixel fitting (min_photons={args.min_photons})...")
        pixel_maps = fit_per_pixel(
            pixel_stack, tcspc_res, n_bins,
            irf_prompt, has_tail, fit_bg, fit_sigma,
            global_popt, args.nexp,
            min_photons=args.min_photons,
        )
        
        # Save weighted tau images
        if getattr(args, 'save_weighted', True):
            save_weighted_tau_images(
                pixel_maps,
                Path(args.output_dir),
                roi_name=roi_name,
                n_exp=args.nexp,
                save_intensity=True,
                save_amplitude=True,
                tau_display_min=getattr(args, 'tau_display_min', None),
                tau_display_max=getattr(args, 'tau_display_max', None),
            )
        
        # Save individual component maps if requested
        if getattr(args, 'save_individual', False):
            save_individual_tau_maps(
                pixel_maps,
                Path(args.output_dir),
                roi_name=roi_name,
                n_exp=args.nexp
            )
        
        if not args.no_plots:
            matplotlib.use("Agg")
            print(f"Generating pixel maps...")
            plot_pixel_maps(pixel_maps, args.nexp, str(output_name), binning=args.binning)
            plot_lifetime_histogram(pixel_maps, args.nexp, str(output_name))
    
    print(f"\n{'='*60}")
    print(f"  WORKFLOW COMPLETE")
    print(f"{'='*60}")
    print(f"Stitched data: {args.output_dir}")
    print(f"Fit results:   {output_name}_*")
    print("\nDone!\n")


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
            estimate_irf = 'none'        # not needed
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

    # Build an argparse.Namespace that mimics the command‑line arguments
    import argparse
    args = argparse.Namespace()
    args.ptu = ptu_path
    args.xlsx = xlif_path
    args.no_xlsx_irf = (xlif_path is not None and not use_xlsx_irf)   # only relevant if xlsx exists
    args.irf = irf_path
    # When user chose to use the xlsx IRF, route through the analytical path
    args.irf_xlsx = xlif_path if use_xlsx_irf else None
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

    # Cell mask option
    mask_q = yes_no_question("Apply cell mask to filter background before fitting?")
    args.cell_mask = (mask_q == 'y')

    # Intensity threshold option
    thresh_q = yes_no_question("Apply an intensity threshold to exclude low-signal pixels?")
    if thresh_q == 'y':
        thresh_mode_q = [inquirer.List('tmode',
                                       message="Choose threshold method",
                                       choices=['Enter a value', 'Pick interactively (slider)'],
                                       default='Pick interactively (slider)')]
        thresh_mode = inquirer.prompt(thresh_mode_q)['tmode']
        if thresh_mode == 'Enter a value':
            val = input("  Min photon intensity per pixel: ").strip()
            args.intensity_threshold = int(val) if val else None
        else:
            args.intensity_threshold = 'interactive'
    else:
        args.intensity_threshold = None

    return args


def _run_flim_fit(args):
    """Core fitting routine – identical to original single_FOV_flim_fit body."""
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

    #  Cell mask (optional) 
    cell_mask = None
    if getattr(args, 'cell_mask', False):
        print(f"\n[1b] Building cell mask")
        intensity_img = make_intensity_image(args.ptu, rotate_90_cw=False, save_image=False)
        cell_mask = make_cell_mask(intensity_img, save_mask=True, path=args.out)
        n_cell_px = int(cell_mask.sum())
        n_total_px = cell_mask.size
        print(f"    Cell mask: {n_cell_px:,} / {n_total_px:,} pixels "
              f"({100*n_cell_px/n_total_px:.1f}% of FOV)")

    #  Intensity threshold (optional) 
    intensity_mask = None
    _int_thr = getattr(args, 'intensity_threshold', None)
    if _int_thr is not None:
        print(f"\n[1c] Intensity threshold")
        # Build intensity image if we don't already have one
        if cell_mask is not None:
            # reuse intensity_img computed above
            pass
        else:
            intensity_img = make_intensity_image(args.ptu, rotate_90_cw=False, save_image=False)
        if _int_thr == 'interactive':
            _int_thr = pick_intensity_threshold(intensity_img)
        else:
            _int_thr = int(_int_thr)
        intensity_mask = apply_intensity_threshold(intensity_img, _int_thr)
        n_kept = int(intensity_mask.sum())
        n_total_px = intensity_mask.size
        print(f"    Intensity threshold: {_int_thr} photons  →  "
              f"{n_kept:,}/{n_total_px:,} pixels kept ({100*n_kept/n_total_px:.1f}%)")
        # Combine with cell mask if both present
        if cell_mask is not None:
            cell_mask = cell_mask & intensity_mask
            print(f"    Combined with cell mask: {int(cell_mask.sum()):,} pixels kept")
        else:
            cell_mask = intensity_mask

    #  Summed decay 
    print(f"\n[2] Building summed decay (channel={args.channel or 'auto'})")
    if cell_mask is not None:
        # Build decay only from cell pixels
        stack_for_decay = ptu.raw_pixel_stack(channel=args.channel)  # (Y, X, H)
        stack_for_decay[~cell_mask] = 0  # zero out background
        decay = stack_for_decay.sum(axis=(0, 1))  # sum → 1-D histogram
        del stack_for_decay
        print(f"    (Using cell-masked photons only)")
    else:
        decay = ptu.summed_decay(channel=args.channel)
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
        if xlsx['fit_t'] is not None:
            print(f"    LAS X fit present, peak = {xlsx['fit_c'].max():.0f} cts")
    else:
        print(f"\n[3] No XLSX provided or file not found")

    #  Build IRF — sets has_tail, fit_sigma, fit_bg per path 
    print(f"\n[4] Building IRF")

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

    elif args.estimate_irf != "none":
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

    #  IRF comparison (optional) 
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
            cost_function=getattr(args, 'cost_function', 'poisson'),
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

    # Per-pixel fit ─
    if args.mode in ("perPixel", "both"):
        if global_popt is None:
            print(f"\n[5] Running summed fit first (τ needed for per-pixel)")
            global_popt, global_summary = _run_summed()
            print_summary(global_summary, strategy, args.nexp)

        print(f"\n[7] Building pixel stack (binning={args.binning}×{args.binning})")
        stack = ptu.pixel_stack(channel=ptu.photon_channel, binning=args.binning)

        # Apply cell mask to per-pixel stack if requested
        if cell_mask is not None:
            # Resize mask if binning changed the spatial dimensions
            import cv2
            sy, sx = stack.shape[:2]
            if cell_mask.shape != (sy, sx):
                mask_resized = cv2.resize(cell_mask.astype(np.uint8), (sx, sy),
                                          interpolation=cv2.INTER_NEAREST) > 0
            else:
                mask_resized = cell_mask
            stack[~mask_resized] = 0
            print(f"    Applied cell mask to pixel stack")

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
                        help="If no direct IRF provided, estimate from the decay rising edge.")
        ap.add_argument("--irf-bins",      type=int,   default=IRF_BINS)
        ap.add_argument("--irf-fit-width", type=float, default=IRF_FIT_WIDTH)
        ap.add_argument("--irf-fwhm", type=float, default=IRF_FWHM,
                        help="IRF FWHM in ns. Default: 1 bin width from PTU")
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
        ap.add_argument("--intensity-threshold", default=INTENSITY_THRESHOLD,
                        help="Min photon-count per pixel. Pixels below this are "
                             "excluded from both summed and per-pixel fits. "
                             "Pass an integer, or 'interactive' to choose visually "
                             "with a slider on the intensity image.")
        ap.add_argument("--print-config", action="store_true", help="Print default configuration settings and exit")
        args = ap.parse_args()

    if args.print_config:
        print(config_message)
        return

    # Run the fitting routine
    _run_flim_fit(args)


def stitch_tiles(interactive=False):
    """
    Entry point for tile stitching (no fitting).
    """
    if interactive:
        args = stitch_tiles_inquire()
        _run_tile_stitch(args)
    else:
        # Command-line argument parsing
        ap = argparse.ArgumentParser(
            description="Stitch FLIM PTU tiles into mosaic using XLIF metadata"
        )
        ap.add_argument("--xlif", required=True, help="Path to XLIF metadata file")
        ap.add_argument("--ptu-dir", required=True, help="Directory containing PTU tiles")
        ap.add_argument("--output-dir", required=True, 
                        help="Output directory for stitched data. "
                             "To avoid overwriting when processing multiple ROIs, "
                             "use a separate directory per ROI (e.g., 'results/R_2/').")
        ap.add_argument("--ptu-basename", default=None, help="PTU basename (default: from XLIF filename)")
        ap.add_argument("--rotate-tiles", action="store_true", default=True, 
                       help="Apply 90° CW rotation (default: True)")
        ap.add_argument("--no-rotate", action="store_true", help="Disable tile rotation")
        args = ap.parse_args()
        
        if args.no_rotate:
            args.rotate_tiles = False
        
        if args.ptu_basename is None:
            args.ptu_basename = Path(args.xlif).stem
        
        _run_tile_stitch(args)


def stitch_and_fit(interactive=False):
    """
    Entry point for combined stitch + fit workflow.
    """
    if interactive:
        args = stitch_and_fit_inquire()
        _run_stitch_and_fit(args)
    else:
        # Command-line argument parsing
        ap = argparse.ArgumentParser(
            description="Stitch FLIM tiles and perform fitting"
        )
        # Stitching args
        ap.add_argument("--xlif", required=True, help="Path to XLIF metadata file")
        ap.add_argument("--ptu-dir", required=True, help="Directory containing PTU tiles")
        ap.add_argument("--output-dir", required=True, 
                        help="Output directory. To avoid overwriting, use a separate "
                             "directory per ROI (e.g., 'results/R_2/').")
        ap.add_argument("--ptu-basename", default=None, help="PTU basename (default: from XLIF)")
        ap.add_argument("--rotate-tiles", action="store_true", default=True)
        ap.add_argument("--no-rotate", action="store_true")
        
        # Fitting args
        ap.add_argument("--irf", default=None, help="Scatter PTU for IRF")
        ap.add_argument("--estimate-irf", choices=["raw", "parametric", "gaussian"], 
                       default="gaussian")
        ap.add_argument("--nexp", type=int, default=n_exp, choices=[1, 2, 3])
        ap.add_argument("--tau-min", type=float, default=Tau_min)
        ap.add_argument("--tau-max", type=float, default=Tau_max)
        ap.add_argument("--mode", default=D_mode, choices=["summed", "perPixel", "both"])
        ap.add_argument("--binning", type=int, default=binning_factor)
        ap.add_argument("--min-photons", type=int, default=MIN_PHOTONS_PERPIX)
        ap.add_argument("--optimizer", choices=["lm_multistart", "de"], default=Optimizer)
        ap.add_argument("--restarts", type=int, default=lm_restarts)
        ap.add_argument("--de-population", type=int, default=de_population)
        ap.add_argument("--de-maxiter", type=int, default=de_maxiter)
        ap.add_argument("--workers", type=int, default=n_workers)
        ap.add_argument("--no-polish", action="store_true")
        ap.add_argument("--channel", type=int, default=channels)
        ap.add_argument("--irf-fwhm", type=float, default=IRF_FWHM)
        ap.add_argument("--irf-bins", type=int, default=IRF_BINS)
        ap.add_argument("--irf-fit-width", type=float, default=IRF_FIT_WIDTH)
        ap.add_argument("--no-plots", action="store_true")
        
        # Output control flags
        ap.add_argument("--save-individual", action="store_true",
                        help="Save individual tau/amplitude component maps")
        ap.add_argument("--no-save-weighted", action="store_true",
                        help="Disable saving of weighted tau images (default: enabled)")
        ap.add_argument("--intensity-threshold", default=INTENSITY_THRESHOLD,
                        help="Min photon-count per pixel. Pixels below this are "
                             "excluded from both summed and per-pixel fits. "
                             "Pass an integer, or 'interactive' to choose visually "
                             "with a slider on the intensity image.")
        
        args = ap.parse_args()
        
        if args.no_rotate:
            args.rotate_tiles = False
        
        if args.ptu_basename is None:
            args.ptu_basename = Path(args.xlif).stem
        
        # Set weighted flag opposite of no-save-weighted
        args.save_weighted = not args.no_save_weighted
        
        _run_stitch_and_fit(args)