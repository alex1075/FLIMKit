import inquirer
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib

# Disable tqdm globally – all progress is shown via progress windows instead
tqdm.disable = True

from .configs import (
    n_exp, Tau_min, Tau_max, D_mode, binning_factor, MIN_PHOTONS_PERPIX, Optimizer, lm_restarts, de_population, de_maxiter, n_workers, OUT_NAME, Estimate_IRF, IRF_BINS, IRF_FIT_WIDTH, IRF_FWHM, channels, config_message, INTENSITY_THRESHOLD,
    TAU_DISPLAY_MIN, TAU_DISPLAY_MAX, INTENSITY_DISPLAY_MIN, INTENSITY_DISPLAY_MAX,
    MACHINE_IRF_DEFAULT_PATH, MACHINE_IRF_FIT_BG, MACHINE_IRF_FIT_SIGMA, MACHINE_IRF_FIT_TAIL)
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
from .utils.lifetime_image import make_lifetime_image
from .image.tools import make_intensity_image, make_cell_mask, apply_intensity_threshold, pick_intensity_threshold


def _make_operation_progress_callback(operation_name, progress_window_manager):
    """Create a progress callback that shows a progress window for an operation."""
    if progress_window_manager is None:
        return None
    
    # Create progress window for this operation
    window_id = progress_window_manager.create_progress_window(task_name=operation_name)
    
    def callback(current, total):
        progress_window_manager.update_progress(window_id, current, total)
    
    return callback


def _load_machine_irf_prompt(machine_irf_path, n_bins, decay_peak_bin):
    """Load a machine IRF, map it to PTU bins, and align its peak to decay peak."""
    irf_path = Path(machine_irf_path) if machine_irf_path else Path(MACHINE_IRF_DEFAULT_PATH)
    if not irf_path.exists():
        raise FileNotFoundError(f"Machine IRF file not found: {irf_path}")

    irf_prompt = np.asarray(np.load(str(irf_path)), dtype=float).ravel()
    if irf_prompt.size == 0:
        raise ValueError(f"Machine IRF file is empty: {irf_path}")

    irf_prompt = np.maximum(irf_prompt, 0.0)
    if irf_prompt.size > n_bins:
        irf_prompt = irf_prompt[:n_bins]
    elif irf_prompt.size < n_bins:
        padded = np.zeros(n_bins, dtype=float)
        padded[:irf_prompt.size] = irf_prompt
        irf_prompt = padded

    s = irf_prompt.sum()
    if s <= 0:
        raise ValueError(f"Machine IRF has zero total intensity after processing: {irf_path}")
    irf_prompt /= s

    current_peak = int(np.argmax(irf_prompt))
    shift = int(decay_peak_bin) - current_peak
    if shift != 0:
        irf_prompt = np.roll(irf_prompt, shift)

    strategy = f"machine_irf ({irf_path.name}) peak_aligned_to_decay"
    return irf_prompt, strategy

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
    print("  3. 'machine_irf' - prebuilt machine IRF (.npy), peak-aligned to decay")
    print("  4. 'raw'        - non-parametric IRF from raw decay")
    print("  5. 'parametric' - fit Gaussian + exponential tail")
    print("  6. 'gaussian'   - simple Gaussian IRF (fastest)")
    method_q = [inquirer.List('method',
                              message="Choose IRF estimation method",
                              choices=['irf_xlsx', 'file', 'machine_irf', 'raw', 'parametric', 'gaussian'])]
    estimate_irf = inquirer.prompt(method_q)['method']
    
    irf_path = None
    irf_xlsx_path = None
    machine_irf_path = None
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
    elif estimate_irf == 'machine_irf':
        _default_machine = str(MACHINE_IRF_DEFAULT_PATH)
        machine_irf_path = input(
            f"Enter path to machine IRF .npy (Enter for default: {_default_machine}): "
        ).strip()
        machine_irf_path = machine_irf_path if machine_irf_path else _default_machine
        if not Path(machine_irf_path).exists():
            print("  Warning: machine IRF file not found, falling back to Gaussian")
            estimate_irf = 'gaussian'
            machine_irf_path = None
    
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
    
    # Auto-select optimizer
    optimizer_choice = 'de'
    print(f"\n[Auto] Using optimizer: de (log-tau + Sobol, robust global search)")
    
    # Lifetime display range for weighted tau images
    if fit_per_pixel_mode:
        tau_range_q = yes_no_question("Set a lifetime display range for tau images? (e.g. 0-5 ns)")
        if tau_range_q == 'y':
            _def_lo = TAU_DISPLAY_MIN
            _def_hi = TAU_DISPLAY_MAX
            tau_min_display = input(f"  Min lifetime (ns, Enter={'0' if _def_lo is None else _def_lo}): ").strip()
            tau_max_display = input(f"  Max lifetime (ns, Enter={'no limit' if _def_hi is None else _def_hi}): ").strip()
            tau_min_display = float(tau_min_display) if tau_min_display else (_def_lo if _def_lo is not None else 0.0)
            tau_max_display = float(tau_max_display) if tau_max_display else _def_hi
        else:
            tau_min_display = TAU_DISPLAY_MIN
            tau_max_display = TAU_DISPLAY_MAX

        # Intensity display range
        int_range_q = yes_no_question("Set an intensity display range for exported images?")
        if int_range_q == 'y':
            _idef_lo = INTENSITY_DISPLAY_MIN
            _idef_hi = INTENSITY_DISPLAY_MAX
            int_min_display = input(f"  Min intensity (Enter={'no limit' if _idef_lo is None else _idef_lo}): ").strip()
            int_max_display = input(f"  Max intensity (Enter={'no limit' if _idef_hi is None else _idef_hi}): ").strip()
            int_min_display = float(int_min_display) if int_min_display else _idef_lo
            int_max_display = float(int_max_display) if int_max_display else _idef_hi
        else:
            int_min_display = INTENSITY_DISPLAY_MIN
            int_max_display = INTENSITY_DISPLAY_MAX
    else:
        tau_min_display = None
        tau_max_display = None
        int_min_display = None
        int_max_display = None
    
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
    args.output_dir = stitch_args.output_dir
    args.ptu_basename = stitch_args.ptu_basename
    args.rotate_tiles = stitch_args.rotate_tiles
    
    # Fitting parameters
    args.irf = irf_path
    args.irf_xlsx = irf_xlsx_path
    args.machine_irf = machine_irf_path
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
    args.save_weighted = True
    args.tau_display_min = tau_min_display
    args.tau_display_max = tau_max_display
    args.intensity_display_min = int_min_display
    args.intensity_display_max = int_max_display
    
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
        register_tiles=getattr(args, 'register_tiles', True),
        reg_max_shift_px=getattr(args, 'reg_max_shift_px', 80),
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


def _run_stitch_and_fit(args, progress_callback=None, cancel_event=None, progress_window_manager=None):
    """Execute combined stitch + fit workflow."""
    print(f"\n{'='*60}")
    print(f"  STEP 1: TILE STITCHING")
    print(f"{'='*60}")
    
    # Create progress window for tile stitching
    stitch_progress_cb = _make_operation_progress_callback(
        "Tile Stitching", progress_window_manager) or progress_callback
    
    stitch_result = stitch_flim_tiles(
        xlif_path=Path(args.xlif),
        ptu_dir=Path(args.ptu_dir),
        output_dir=Path(args.output_dir),
        ptu_basename=args.ptu_basename,
        rotate_tiles=args.rotate_tiles,
        register_tiles=getattr(args, 'register_tiles', True),
        reg_max_shift_px=getattr(args, 'reg_max_shift_px', 80),
        verbose=True,
        progress_callback=stitch_progress_cb,
        cancel_event=cancel_event,
    )
    
    if stitch_result['tiles_processed'] == 0:
        raise RuntimeError("No tiles were successfully stitched!")
    
    print(f"\nStitching complete: {stitch_result['tiles_processed']} tiles processed")
    
    roi_name = args.ptu_basename.replace(' ', '_')
    
    print(f"\n{'='*60}")
    print(f"  STEP 2: LOADING STITCHED DATA")
    print(f"{'='*60}")

    stack, tcspc_res, n_bins = load_flim_for_fitting(
        Path(args.output_dir),
        load_to_memory=False
    )

    print(f"  Mapped: {stack.shape} @ {tcspc_res*1e12:.2f} ps/bin")

    weight_candidates = sorted(Path(args.output_dir).glob("*_weight_map.npy"))
    if weight_candidates:
        weight_map = np.load(str(weight_candidates[0])).astype(np.float32)
        weight_map = np.clip(weight_map, 1.0, None)
        print(f"  Loaded weight map: max overlap = {int(weight_map.max())} tiles")
    else:
        weight_map = None
        print("  No weight map found — overlap normalisation skipped")

    class StitchedData:
        """Mock PTU object for stitched data."""
        def __init__(self, stack, tcspc_res, n_bins, weight_map=None):
            self.n_bins = n_bins
            self.tcspc_res = tcspc_res
            self.time_ns = np.arange(n_bins) * tcspc_res * 1e9
            self.photon_channel = None
            self._stack = stack
            self._weight_map = weight_map

        def summed_decay(self, channel=None):
            return self._stack.sum(axis=(0, 1))

        def pixel_stack(self, channel=None, binning=1):
            ny, nx, nt = self._stack.shape

            if binning == 1:
                if self._weight_map is not None:
                    out = np.empty((ny, nx, nt), dtype=np.float32)
                    CHUNK = 64
                    for r0 in tqdm(range(0, ny, CHUNK),
                                   desc='  Normalising overlap (pixel stack)',
                                   unit='chunk', disable=True):
                        r1 = min(r0 + CHUNK, ny)
                        w  = self._weight_map[r0:r1, :, np.newaxis]
                        out[r0:r1] = self._stack[r0:r1].astype(np.float32) / w
                    return out
                else:
                    return self._stack
            else:
                new_ny = ny // binning
                new_nx = nx // binning
                binned = np.zeros((new_ny, new_nx, nt), dtype=np.float32)
                for i in tqdm(range(new_ny), desc='  Binning rows', disable=True):
                    for j in range(new_nx):
                        patch = self._stack[
                            i*binning:(i+1)*binning,
                            j*binning:(j+1)*binning, :
                        ].astype(np.float32)
                        binned[i, j, :] = patch.sum(axis=(0, 1))
                        if self._weight_map is not None:
                            w = self._weight_map[
                                i*binning:(i+1)*binning,
                                j*binning:(j+1)*binning
                            ].mean()
                            if w > 0:
                                binned[i, j, :] /= w
                return binned

    ptu = StitchedData(stack, tcspc_res, n_bins, weight_map=weight_map)

    CHUNK_ROWS = 64
    ny, nx = stack.shape[:2]

    print(f"\n  Building tissue mask from stitched intensity (chunked)...")
    intensity_2d = np.zeros((ny, nx), dtype=np.float64)
    for r0 in tqdm(range(0, ny, CHUNK_ROWS), desc='  Building intensity', unit='chunk', disable=True):
        r1 = min(r0 + CHUNK_ROWS, ny)
        intensity_2d[r0:r1] = stack[r0:r1].sum(axis=2)

    tissue_mask = make_cell_mask(intensity_2d,
                                 save_mask=True,
                                 path=str(Path(args.output_dir) / roi_name))
    n_tissue = int(tissue_mask.sum())
    n_total  = tissue_mask.size
    print(f"    Tissue mask: {n_tissue:,} / {n_total:,} pixels "
          f"({100*n_tissue/n_total:.1f}%) are tissue")

    _int_thr = getattr(args, 'intensity_threshold', None)
    if _int_thr is not None:
        print(f"\n  Applying intensity threshold...")
        if _int_thr == 'interactive':
            _int_thr = pick_intensity_threshold(intensity_2d)
        else:
            _int_thr = int(_int_thr)
        int_mask    = apply_intensity_threshold(intensity_2d, _int_thr)
        tissue_mask = tissue_mask & int_mask
        n_after = int(tissue_mask.sum())
        print(f"    Intensity threshold: {_int_thr} photons  →  "
              f"{n_after:,}/{n_total:,} pixels kept ({100*n_after/n_total:.1f}%)")

    print(f"\n  Building masked summed decay (chunked)...")
    decay = np.zeros(n_bins, dtype=np.float64)
    for r0 in tqdm(range(0, ny, CHUNK_ROWS), desc='  Building decay', unit='chunk', disable=True):
        r1       = min(r0 + CHUNK_ROWS, ny)
        chunk    = stack[r0:r1].astype(np.float32)
        row_mask = tissue_mask[r0:r1]
        chunk[~row_mask] = 0
        decay   += chunk.sum(axis=(0, 1))
    del chunk

    print(f"  Total photons (tissue): {decay.sum():,.0f}")

    if decay.sum() == 0:
        print("  \u26a0 WARNING: tissue mask produced zero photons "
              "(mask may be inverted or canvas gaps only). "
              "Falling back to full-canvas summed decay.")
        decay = np.zeros(n_bins, dtype=np.float64)
        for r0 in range(0, ny, CHUNK_ROWS):
            r1 = min(r0 + CHUNK_ROWS, ny)
            decay += stack[r0:r1].astype(np.float64).sum(axis=(0, 1))
        print(f"  Total photons (full canvas): {decay.sum():,.0f}")

    print(f"\n{'='*60}")
    print(f"  STEP 3: FLIM FITTING")
    print(f"{'='*60}")
    print(f"  {args.nexp}-exp  |  {args.mode}  |  optimizer={args.optimizer}")
    
    irf_peak_bin = find_irf_peak_bin(decay)
    decay_peak_bin = int(np.argmax(decay))
    
    print(f"\nSummed decay (tissue-masked):")
    print(f"  {decay.sum():,.0f} photons  |  peak={decay.max():,.0f} at bin {decay_peak_bin}")
    print(f"  IRF peak (steepest rise): bin {irf_peak_bin} ({irf_peak_bin * tcspc_res * 1e9:.3f} ns)")
    
    print(f"\nBuilding IRF (method: {args.estimate_irf})...")
    
    if args.irf is not None and Path(args.irf).exists():
        irf_prompt = irf_from_scatter_ptu(args.irf, ptu)
        strategy = "scatter_ptu"
        has_tail = False
        fit_sigma = False
        fit_bg = True

    elif getattr(args, 'irf_xlsx', None) is not None:
        print(f"  IRF: fitting Leica analytical model to: {args.irf_xlsx}")
        if not Path(args.irf_xlsx).exists():
            raise FileNotFoundError(f"IRF XLSX file not found: {args.irf_xlsx}")
        irf_ref = load_xlsx(args.irf_xlsx, debug=False)
        if irf_ref['irf_t'] is None or irf_ref['irf_c'] is None:
            raise ValueError(f"No IRF columns found in XLSX: {args.irf_xlsx}")
        irf_prompt, irf_params = irf_from_xlsx_analytical(
            irf_ref, n_bins, tcspc_res, verbose=True)
        irf_current_peak = int(np.argmax(irf_prompt))
        pre_shift = irf_peak_bin - irf_current_peak
        if pre_shift != 0:
            x          = np.arange(n_bins, dtype=float)
            irf_prompt = np.interp(x - pre_shift, x, irf_prompt, left=0.0, right=0.0)
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

    elif args.estimate_irf == "machine_irf":
        irf_prompt, strategy = _load_machine_irf_prompt(
            getattr(args, 'machine_irf', None), n_bins, decay_peak_bin)
        has_tail  = MACHINE_IRF_FIT_TAIL
        fit_sigma = MACHINE_IRF_FIT_SIGMA
        fit_bg    = MACHINE_IRF_FIT_BG
        print(f"  IRF: {strategy}")

    elif args.estimate_irf == "raw":
        from .FLIM.irf_tools import estimate_irf_from_decay_raw
        irf_prompt = estimate_irf_from_decay_raw(
            decay, tcspc_res, n_bins, n_irf_bins=args.irf_bins)
        strategy  = f"estimated_raw (bins={args.irf_bins})"
        has_tail  = True
        fit_sigma = True
        fit_bg    = True
        print(f"  IRF: {strategy}")
    
    elif args.estimate_irf == "parametric":
        from .FLIM.irf_tools import estimate_irf_from_decay_parametric
        irf_prompt = estimate_irf_from_decay_parametric(
            decay, tcspc_res, n_bins, fit_window_width_ns=args.irf_fit_width)
        strategy  = f"estimated_parametric (width={args.irf_fit_width}ns)"
        has_tail  = True
        fit_sigma = True
        fit_bg    = True
        print(f"  IRF: {strategy}")
    
    else:
        fwhm_ns = args.irf_fwhm if args.irf_fwhm is not None else tcspc_res * 1e9
        irf_prompt = gaussian_irf_from_fwhm(n_bins, tcspc_res, fwhm_ns, decay_peak_bin)
        strategy  = f"gaussian FWHM={fwhm_ns*1000:.1f}ps"
        has_tail  = False
        fit_sigma = False
        fit_bg    = True
        print(f"  IRF: {strategy}")
    
    print(f"  Flags: has_tail={has_tail}  fit_sigma={fit_sigma}  fit_bg={fit_bg}")
    
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
    
    metadata = {
        'canvas_shape': stack.shape[:2],
        'total_photons': int(decay.sum()),
        'tiles_processed': stitch_result['tiles_processed'],
    }
    save_fit_summary_txt(
        global_summary,
        Path(args.output_dir) / f"{roi_name}_fit_summary.txt",
        n_exp=args.nexp,
        strategy=strategy,
        metadata=metadata
    )
    
    if not args.no_plots:
        matplotlib.use("Agg")
        output_name = Path(args.output_dir) / "fit_results"
        print(f"\nGenerating plots...")
        plot_summed(
            decay, global_summary, ptu, None,
            args.nexp, strategy, str(output_name),
            irf_prompt=irf_prompt
        )
    
    pixel_maps = None
    if args.mode in ("perPixel", "both"):
        print(f"\nBuilding pixel stack (binning={args.binning}×{args.binning})...")
        pixel_stack = ptu.pixel_stack(channel=None, binning=args.binning)

        if tissue_mask is not None:
            print(f"  Background pixels will be skipped via min_photons threshold")
        
        print(f"Per-pixel fitting (min_photons={args.min_photons})...")
        # Create progress window for per-pixel fitting
        perpixel_progress_cb = _make_operation_progress_callback(
            "Per-pixel fitting", progress_window_manager) or progress_callback
        
        pixel_maps = fit_per_pixel(
            pixel_stack, tcspc_res, n_bins,
            irf_prompt, has_tail, fit_bg, fit_sigma,
            global_popt, args.nexp,
            min_photons=args.min_photons,
            progress_callback=perpixel_progress_cb,
        )
        
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
                intensity_display_min=getattr(args, 'intensity_display_min', None),
                intensity_display_max=getattr(args, 'intensity_display_max', None),
                target_shape=(ny, nx),  # full stitched canvas size
            )
        
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
    print("\nDone!\n")


def single_FOV_flim_fit_inquire():
    """Interactive prompt to collect FLIM fitting parameters."""
    print("\n--- Interactive FLIM Fit Setup ---")

    ptu_path = input("Enter path to PTU file for this FOV: ").strip()
    if not ptu_path:
        raise ValueError("PTU file path is required.")

    xlsxq = yes_no_question("Do you have an XLSX file for this FOV? (Recommended for pixel size info)")
    if xlsxq == 'y':
        xlif_path = input("Enter path to XLSX file: ").strip()
        irf_xlsxq = yes_no_question("Do you want to use the IRF from this XLSX file? (Recommended if available)")
        if irf_xlsxq == 'y':
            use_xlsx_irf = True
            estimate_irf = 'none'
            irf_path = None
        else:
            use_xlsx_irf = False
            print("\nIRF estimation options:")
            print("  1. 'file'   - measured IRF image (scatter PTU)")
            print("  2. 'machine_irf' - prebuilt machine IRF (.npy)")
            print("  3. 'raw'    - non‑parametric IRF from raw decay")
            print("  4. 'parametric' - fit Gaussian + exponential tail to rising edge")
            print("  5. 'none'   - do not estimate IRF (not recommended)")
            method_q = [inquirer.List('method',
                                      message="Choose IRF estimation method",
                                      choices=['file', 'machine_irf', 'raw', 'parametric', 'none'])]
            estimate_irf = inquirer.prompt(method_q)['method']
            if estimate_irf == 'file':
                irf_path = input("Enter path to IRF PTU file: ").strip()
                machine_irf_path = None
            elif estimate_irf == 'machine_irf':
                _default_machine = str(MACHINE_IRF_DEFAULT_PATH)
                machine_irf_path = input(
                    f"Enter path to machine IRF .npy (Enter for default: {_default_machine}): "
                ).strip()
                machine_irf_path = machine_irf_path if machine_irf_path else _default_machine
                irf_path = None
            else:
                irf_path = None
                machine_irf_path = None
    else:
        xlif_path = None
        use_xlsx_irf = False
        print("\nNo XLSX file – IRF must be estimated or provided separately.")
        method_q = [inquirer.List('method',
                                  message="Choose IRF estimation method",
                                  choices=['file', 'machine_irf', 'raw', 'parametric', 'none'])]
        estimate_irf = inquirer.prompt(method_q)['method']
        if estimate_irf == 'file':
            irf_path = input("Enter path to IRF PTU file: ").strip()
            machine_irf_path = None
        elif estimate_irf == 'machine_irf':
            _default_machine = str(MACHINE_IRF_DEFAULT_PATH)
            machine_irf_path = input(
                f"Enter path to machine IRF .npy (Enter for default: {_default_machine}): "
            ).strip()
            machine_irf_path = machine_irf_path if machine_irf_path else _default_machine
            irf_path = None
        else:
            irf_path = None
            machine_irf_path = None

    import argparse
    args = argparse.Namespace()
    args.ptu = ptu_path
    args.xlsx = xlif_path
    args.no_xlsx_irf = (xlif_path is not None and not use_xlsx_irf)
    args.irf = irf_path
    args.machine_irf = machine_irf_path if 'machine_irf_path' in locals() else None
    args.irf_xlsx = xlif_path if use_xlsx_irf else None
    args.estimate_irf = estimate_irf
    args.irf_bins = IRF_BINS
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
    args.print_config = False

    mask_q = yes_no_question("Apply cell mask to filter background before fitting?")
    args.cell_mask = (mask_q == 'y')

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


def _run_flim_fit(args, progress_callback=None, cancel_event=None, progress_window_manager=None):
    """Core fitting routine – identical to original single_FOV_flim_fit body."""
    print(f"\n{'='*60}")
    print(f"  flim_fit_v13  |  {args.nexp}-exp  |  {args.mode}  |  optimizer={args.optimizer}")
    print(f"{'='*60}")

    print(f"\n[1] PTU: {args.ptu}")
    ptu = PTUFile(args.ptu, verbose=True)

    fwhm_ns = args.irf_fwhm if args.irf_fwhm is not None else ptu.tcspc_res * 1e9
    print(f"  IRF FWHM: {fwhm_ns*1000:.2f} ps "
          f"({'from --irf-fwhm' if args.irf_fwhm is not None else 'default: 1 bin'})")

    cell_mask = None
    if getattr(args, 'cell_mask', False):
        print(f"\n[1b] Building cell mask")
        intensity_img = make_intensity_image(args.ptu, rotate_90_cw=False, save_image=False)
        cell_mask = make_cell_mask(intensity_img, save_mask=True, path=args.out)
        n_cell_px = int(cell_mask.sum())
        n_total_px = cell_mask.size
        print(f"    Cell mask: {n_cell_px:,} / {n_total_px:,} pixels "
              f"({100*n_cell_px/n_total_px:.1f}% of FOV)")

    intensity_mask = None
    _int_thr = getattr(args, 'intensity_threshold', None)
    if _int_thr is not None:
        print(f"\n[1c] Intensity threshold")
        if cell_mask is None:
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
        if cell_mask is not None:
            cell_mask = cell_mask & intensity_mask
            print(f"    Combined with cell mask: {int(cell_mask.sum()):,} pixels kept")
        else:
            cell_mask = intensity_mask

    print(f"\n[2] Building summed decay (channel={args.channel or 'auto'})")
    if cell_mask is not None:
        stack_for_decay = ptu.raw_pixel_stack(channel=args.channel)
        stack_for_decay[~cell_mask] = 0
        decay = stack_for_decay.sum(axis=(0, 1))
        del stack_for_decay
        print(f"    (Using cell-masked photons only)")
    else:
        decay = ptu.summed_decay(channel=args.channel)
    irf_peak_bin   = find_irf_peak_bin(decay)
    decay_peak_bin = int(np.argmax(decay))
    print(f"    {decay.sum():,.0f} photons  |  peak={decay.max():,.0f}  "
          f"at bin {decay_peak_bin} ({ptu.time_ns[decay_peak_bin]:.3f} ns)")
    print(f"    IRF peak (steepest rise): bin {irf_peak_bin} "
          f"({irf_peak_bin * ptu.tcspc_res * 1e9:.3f} ns)")

    xlsx = None
    if args.xlsx is not None and Path(args.xlsx).exists():
        print(f"\n[3] XLSX: {args.xlsx}")
        xlsx = load_xlsx(args.xlsx, debug=args.debug_xlsx)
        if xlsx['fit_t'] is not None and xlsx['fit_c'] is not None:
            print(f"    LAS X fit present, peak = {xlsx['fit_c'].max():.0f} cts")
        elif xlsx['fit_t'] is not None:
            print(f"    LAS X fit_t present but fit_c absent")
    else:
        print(f"\n[3] No XLSX provided or file not found")

    print(f"\n[4] Building IRF")

    if args.irf is not None:
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
        irf_current_peak = int(np.argmax(irf_prompt))
        pre_shift = irf_peak_bin - irf_current_peak
        if pre_shift != 0:
            x          = np.arange(ptu.n_bins, dtype=float)
            irf_prompt = np.interp(x - pre_shift, x, irf_prompt, left=0.0, right=0.0)
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

    elif args.estimate_irf == "machine_irf":
        irf_prompt, strategy = _load_machine_irf_prompt(
            getattr(args, 'machine_irf', None), ptu.n_bins, decay_peak_bin)
        has_tail  = MACHINE_IRF_FIT_TAIL
        fit_sigma = MACHINE_IRF_FIT_SIGMA
        fit_bg    = MACHINE_IRF_FIT_BG
        print(f"  IRF: {strategy}")

    elif xlsx is not None and xlsx['irf_t'] is not None and not args.no_xlsx_irf:
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

    if not args.no_plots and xlsx is not None:
        matplotlib.use("Agg")
        print(f"\n[4b] IRF comparison")
        compare_irfs(irf_prompt, xlsx, ptu.tcspc_res, ptu.n_bins, strategy, args.out)

    global_popt    = None
    global_summary = None
    pixel_maps     = None

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

    if args.mode in ("perPixel", "both"):
        if global_popt is None:
            print(f"\n[5] Running summed fit first (τ needed for per-pixel)")
            global_popt, global_summary = _run_summed()
            print_summary(global_summary, strategy, args.nexp)

        print(f"\n[7] Building pixel stack (binning={args.binning}×{args.binning})")
        stack = ptu.pixel_stack(channel=ptu.photon_channel, binning=args.binning)

        if cell_mask is not None:
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
        # Create progress window for per-pixel fitting
        perpixel_progress_cb = _make_operation_progress_callback(
            "Per-pixel fitting", progress_window_manager) or progress_callback
        
        pixel_maps = fit_per_pixel(
            stack, ptu.tcspc_res, ptu.n_bins,
            irf_prompt, has_tail, fit_bg, fit_sigma,
            global_popt, args.nexp,
            min_photons=args.min_photons,
            progress_callback=perpixel_progress_cb,
        )

        if not args.no_plots:
            matplotlib.use("Agg")
            print(f"\n[9] Plotting pixel maps")
            plot_pixel_maps(pixel_maps, args.nexp, args.out, binning=args.binning)
            plot_lifetime_histogram(pixel_maps, args.nexp, args.out)

    print("\nDone.\n")

    return {
        'pixel_maps':     pixel_maps     if args.mode in ("perPixel", "both") else None,
        'global_summary': global_summary,
        'global_popt':    global_popt,
        'tcspc_res':      ptu.tcspc_res,
        'n_bins':         ptu.n_bins,
        'strategy':       strategy,
        'irf_prompt':     irf_prompt,
        'time_ns':        ptu.time_ns,
        'decay':          decay,
    }


def single_FOV_flim_fit(interactive=False):
    """Entry point for FLIM fitting."""
    if interactive:
        args = single_FOV_flim_fit_inquire()
    else:
        ap = argparse.ArgumentParser(
            description="FLIM reconvolution fit — PTU + optional XLSX (Leica FALCON)"
        )
        ap.add_argument("--ptu",   default=None, required=True)
        ap.add_argument("--xlsx",  default=None)
        ap.add_argument("--no-xlsx-irf", action="store_true")
        ap.add_argument("--debug-xlsx", action="store_true")
        ap.add_argument("--irf",   default=None)
        ap.add_argument("--irf-xlsx", default=None)
        ap.add_argument("--estimate-irf", choices=["raw", "parametric", "machine_irf", "none"],
                        default=Estimate_IRF)
        ap.add_argument("--machine-irf", default=str(MACHINE_IRF_DEFAULT_PATH))
        ap.add_argument("--irf-bins",      type=int,   default=IRF_BINS)
        ap.add_argument("--irf-fit-width", type=float, default=IRF_FIT_WIDTH)
        ap.add_argument("--irf-fwhm", type=float, default=IRF_FWHM)
        ap.add_argument("--nexp",     type=int,   default=n_exp, choices=[1, 2, 3])
        ap.add_argument("--tau-min",  type=float, default=Tau_min)
        ap.add_argument("--tau-max",  type=float, default=Tau_max)
        ap.add_argument("--mode",     default=D_mode, choices=["summed", "perPixel", "both"])
        ap.add_argument("--binning",     type=int, default=binning_factor)
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
        ap.add_argument("--intensity-threshold", default=INTENSITY_THRESHOLD)
        ap.add_argument("--print-config", action="store_true")
        args = ap.parse_args()

    if args.print_config:
        print(config_message)
        return

    _run_flim_fit(args)


def stitch_tiles(interactive=False):
    """Entry point for tile stitching (no fitting)."""
    if interactive:
        args = stitch_tiles_inquire()
        _run_tile_stitch(args)
    else:
        ap = argparse.ArgumentParser(
            description="Stitch FLIM PTU tiles into mosaic using XLIF metadata"
        )
        ap.add_argument("--xlif", required=True)
        ap.add_argument("--ptu-dir", required=True)
        ap.add_argument("--output-dir", required=True)
        ap.add_argument("--ptu-basename", default=None)
        ap.add_argument("--rotate-tiles", action="store_true", default=True)
        ap.add_argument("--no-rotate", action="store_true")
        args = ap.parse_args()
        
        if args.no_rotate:
            args.rotate_tiles = False
        if args.ptu_basename is None:
            args.ptu_basename = Path(args.xlif).stem
        
        _run_tile_stitch(args)


def stitch_and_fit(interactive=False):
    """Entry point for combined stitch + fit workflow."""
    if interactive:
        print("\n--- Stitched ROI Fitting ---")
        pipeline_q = [inquirer.List(
            'pipeline',
            message="Which fitting pipeline?",
            choices=[
                'Per-tile (recommended) — fit each tile independently, no IRF mismatch',
                'Global (legacy)        — stitch then fit summed decay',
            ]
        )]
        pipeline = inquirer.prompt(pipeline_q)['pipeline']

        if pipeline.startswith('Per-tile'):
            args = tile_fit_inquire()
            _run_tile_fit(args)
        else:
            args = stitch_and_fit_inquire()
            _run_stitch_and_fit(args)
    else:
        ap = argparse.ArgumentParser(description="Stitch FLIM tiles and perform fitting")
        ap.add_argument("--xlif", required=True)
        ap.add_argument("--ptu-dir", required=True)
        ap.add_argument("--output-dir", required=True)
        ap.add_argument("--ptu-basename", default=None)
        ap.add_argument("--rotate-tiles", action="store_true", default=True)
        ap.add_argument("--no-rotate", action="store_true")
        ap.add_argument("--irf", default=None)
        ap.add_argument("--estimate-irf", choices=["raw", "parametric", "machine_irf", "gaussian"],
                       default="gaussian")
        ap.add_argument("--machine-irf", default=str(MACHINE_IRF_DEFAULT_PATH))
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
        ap.add_argument("--save-individual", action="store_true")
        ap.add_argument("--no-save-weighted", action="store_true")
        ap.add_argument("--intensity-threshold", default=INTENSITY_THRESHOLD)
        args = ap.parse_args()
        
        if args.no_rotate:
            args.rotate_tiles = False
        if args.ptu_basename is None:
            args.ptu_basename = Path(args.xlif).stem
        args.save_weighted = not args.no_save_weighted
        
        _run_stitch_and_fit(args)


# ══════════════════════════════════════════════════════════════════════════════
#  Per-tile fitting pipeline
# ══════════════════════════════════════════════════════════════════════════════

def tile_fit_inquire():
    """Interactive prompt for per-tile fitting parameters."""
    print("\n--- Per-Tile FLIM Fitting ---")
    print("Fits each tile independently with its own IRF, then assembles")
    print("the results into a full-ROI canvas.  Avoids inter-tile IRF mismatch.\n")

    stitch_args = stitch_tiles_inquire()

    print("\nFitting Parameters")

    print("\nIRF estimation options:")
    print("  1. 'parametric' - fit Gaussian + exponential tail to rising edge (default)")
    print("  2. 'raw'        - non-parametric IRF from raw decay")
    print("  3. 'machine_irf' - prebuilt machine IRF (.npy), reused for all tiles")
    print("  4. 'irf_xlsx_dir' - per-tile xlsx directory (fill in once format is known)")
    method_q = [inquirer.List('method',
                              message="Choose IRF method",
                              choices=['parametric', 'raw', 'machine_irf', 'irf_xlsx_dir'])]
    estimate_irf = inquirer.prompt(method_q)['method']

    irf_xlsx_dir = None
    machine_irf_path = None
    if estimate_irf == 'machine_irf':
        _default_machine = str(MACHINE_IRF_DEFAULT_PATH)
        machine_irf_path = input(
            f"Enter path to machine IRF .npy (Enter for default: {_default_machine}): "
        ).strip()
        machine_irf_path = machine_irf_path if machine_irf_path else _default_machine
        if not Path(machine_irf_path).exists():
            print("  Warning: machine IRF file not found, falling back to parametric")
            estimate_irf = 'parametric'
            machine_irf_path = None
    if estimate_irf == 'irf_xlsx_dir':
        irf_xlsx_dir = input("Enter path to directory containing per-tile IRF xlsx files: ").strip()
        if not irf_xlsx_dir or not Path(irf_xlsx_dir).exists():
            print("  Warning: directory not found, falling back to parametric")
            estimate_irf = 'parametric'
            irf_xlsx_dir = None

    nexp_q = [inquirer.List('nexp',
                            message="Number of exponential components",
                            choices=['1', '2', '3'],
                            default='2')]
    nexp = int(inquirer.prompt(nexp_q)['nexp'])

    thresh_q = yes_no_question("Apply intensity threshold to exclude low-signal pixels?")
    if thresh_q == 'y':
        val = input("  Min photon intensity per pixel (Enter for none): ").strip()
        intensity_threshold = int(val) if val else None
    else:
        intensity_threshold = None

    tau_range_q = yes_no_question("Set lifetime display range for tau images?")
    if tau_range_q == 'y':
        tau_min_display = input(f"  Min tau (ns, Enter=auto): ").strip()
        tau_max_display = input(f"  Max tau (ns, Enter=auto): ").strip()
        tau_min_display = float(tau_min_display) if tau_min_display else None
        tau_max_display = float(tau_max_display) if tau_max_display else None
    else:
        tau_min_display = TAU_DISPLAY_MIN
        tau_max_display = TAU_DISPLAY_MAX

    args = argparse.Namespace()

    args.xlif         = stitch_args.xlif
    args.ptu_dir      = stitch_args.ptu_dir
    args.output_dir   = stitch_args.output_dir
    args.ptu_basename = stitch_args.ptu_basename
    args.rotate_tiles    = stitch_args.rotate_tiles
    args.register_tiles  = True    # phase-corr Y registration (recommended)
    args.reg_max_shift_px = 120

    args.estimate_irf  = estimate_irf
    args.irf_xlsx_dir  = irf_xlsx_dir
    args.machine_irf   = machine_irf_path
    args.irf           = None
    args.irf_xlsx      = None
    args.irf_bins      = IRF_BINS
    args.irf_fit_width = IRF_FIT_WIDTH
    args.irf_fwhm      = IRF_FWHM

    args.nexp        = nexp
    args.tau_min     = Tau_min
    args.tau_max     = Tau_max
    args.mode        = 'both'
    args.binning     = binning_factor
    args.min_photons = MIN_PHOTONS_PERPIX
    args.optimizer   = 'de'
    args.restarts    = lm_restarts
    args.de_population = de_population
    args.de_maxiter    = de_maxiter
    args.workers       = n_workers
    args.no_polish     = False
    args.channel       = channels
    args.no_plots      = True
    args.cell_mask     = False
    args.intensity_threshold = intensity_threshold
    args.no_xlsx_irf   = True
    args.debug_xlsx    = False
    args.print_config  = False
    args.xlsx          = None
    args.out           = None

    args.tau_display_min = tau_min_display
    args.tau_display_max = tau_max_display

    return args


def _run_tile_fit(args, progress_callback=None, cancel_event=None, progress_window_manager=None):
    """
    Execute per-tile fitting pipeline:
        1. Fit each tile via fit_flim_tiles() (pooled machine IRF)
        2. Assemble pixel maps onto canvas via assemble_tile_maps()
        3. Derive global tau summary via derive_global_tau()
        4. Save outputs via save_assembled_maps()
        5. Generate intensity-weighted lifetime colour image
    """
    from .PTU.stitch import fit_flim_tiles
    from .FLIM.assemble import assemble_tile_maps, derive_global_tau, save_assembled_maps

    roi_name = args.ptu_basename.replace(' ', '_')

    print(f"\n{'='*60}")
    print(f"  STEP 1: PER-TILE FITTING")
    print(f"{'='*60}")

    (tile_results, canvas_height, canvas_width, corrected_positions,
     pooled_decay, pooled_irf, tcspc_ref, global_popt, global_summary) = fit_flim_tiles(
        xlif_path     = Path(args.xlif),
        ptu_dir       = Path(args.ptu_dir),
        output_dir    = Path(args.output_dir),
        args          = args,
        ptu_basename  = args.ptu_basename,
        rotate_tiles  = args.rotate_tiles,
        irf_xlsx_dir  = getattr(args, 'irf_xlsx_dir', None),
        irf_xlsx_map  = getattr(args, 'irf_xlsx_map', None),
        verbose       = True,
        progress_callback = progress_callback,
        cancel_event  = cancel_event,
    )

    if not tile_results:
        raise RuntimeError("No tiles were successfully fitted.")

    print(f"\n{'='*60}")
    print(f"  STEP 2: ASSEMBLING CANVAS")
    print(f"{'='*60}")

    canvas = assemble_tile_maps(
        tile_results   = tile_results,
        canvas_height  = canvas_height,
        canvas_width   = canvas_width,
        n_exp          = args.nexp,
    )

    # ── Upsample canvas to full tile resolution when binning > 1 ──────
    # fit_flim_tiles computes the canvas at (H//binning, W//binning)
    # using effective_pixel_size_m = pixel_size_m * binning.  Batch
    # hardcodes binning=1 so its canvas is always full-res; single-ROI
    # tile-fit uses args.binning (default 4) giving a 4x-shrunken canvas
    # whose TIFFs appear as big blocky pixels next to full-res intensity.
    # Nearest-neighbour upsample restores native resolution before saving.
    _binning = getattr(args, 'binning', 1)
    if _binning > 1:
        try:
            import cv2 as _cv2
            _th = canvas_height * _binning
            _tw = canvas_width  * _binning
            def _up(arr):
                if arr is None or not hasattr(arr, 'ndim'): return arr
                return _cv2.resize(arr.astype('float32'), (_tw, _th),
                                   interpolation=_cv2.INTER_NEAREST)
            canvas = {k: _up(v) for k, v in canvas.items()}
            canvas_height = _th
            canvas_width  = _tw
            print(f"  ↑ Canvas upsampled ×{_binning} → {_th}×{_tw} px (full tile resolution)")
        except Exception as _upe:
            print(f"  ⚠ Canvas upsample failed: {_upe} — TIFFs will be at binned resolution")

    print(f"\n{'='*60}")
    print(f"  STEP 3: GLOBAL TAU SUMMARY")
    print(f"{'='*60}")

    # Preserve fields from the consensus (pooled) fit before derive_global_tau
    # replaces global_summary.  'model' is needed by display_fit_results to draw
    # the fitted curve on the decay panel; the chi2 values go into the fit summary.
    _consensus_summary = global_summary  # from fit_flim_tiles
    global_summary = derive_global_tau(canvas, n_exp=args.nexp)
    for _key in ('model', 'reduced_chi2', 'reduced_chi2_tail',
                 'reduced_chi2_tail_pearson', 'reduced_chi2_pearson'):
        if _key in _consensus_summary:
            global_summary[_key] = _consensus_summary[_key]

    n_px     = global_summary.get('n_pixels_fitted', 0)
    tau_mean = global_summary.get('tau_mean_amp_global_ns', float('nan'))
    tau_std  = global_summary.get('tau_std_amp_global_ns',  float('nan'))
    tau_med  = global_summary.get('tau_median_amp_global_ns', float('nan'))
    print(f"\n{'─'*60}")
    print(f"  Per-tile fit: {args.nexp}-exp | {n_px:,} pixels")
    print(f"{'─'*60}")
    for k in range(1, args.nexp + 1):
        tau_k = global_summary.get(f'tau{k}_mean_ns', float('nan'))
        a_k   = global_summary.get(f'a{k}_mean_frac', float('nan'))
        print(f"  τ{k} = {tau_k:8.4f} ns   α{k} = {a_k:.3e}   f{k} = {a_k:.4f}")
    print(f"  τ_mean (amplitude-weighted)  = {tau_mean:.4f} ns")
    print(f"  τ_mean (median, amp-wtd)     = {tau_med:.4f} ns")
    print(f"  τ σ (pixel distribution)     = {tau_std:.4f} ns")
    print(f"  n pixels fitted              = {n_px}")
    print(f"  ✓ Optimizer: per-pixel (per-tile fit)")

    print(f"\n{'='*60}")
    print(f"  STEP 4: SAVING OUTPUTS")
    print(f"{'='*60}")

    save_assembled_maps(
        canvas          = canvas,
        global_summary  = global_summary,
        output_dir      = Path(args.output_dir),
        roi_name        = roi_name,
        n_exp           = args.nexp,
        tau_display_min = getattr(args, 'tau_display_min', None),
        tau_display_max = getattr(args, 'tau_display_max', None),
    )

    # Intensity-weighted lifetime colour image
    print(f"\n{'='*60}")
    print(f"  STEP 5: LIFETIME IMAGE")
    print(f"{'='*60}")

    tau_min_img = getattr(args, 'tau_display_min', None) or 0.0
    tau_max_img = getattr(args, 'tau_display_max', None) or 5.0

    make_lifetime_image(
        canvas          = canvas,
        output_dir      = Path(args.output_dir),
        roi_name        = roi_name,
        tau_min_ns      = tau_min_img,
        tau_max_ns      = tau_max_img,
        smooth_sigma_px = 2.0,
        verbose         = True,
    )

    print(f"\n{'='*60}")
    print(f"  PER-TILE FITTING COMPLETE")
    print(f"{'='*60}\n")

    # Build time_ns array from tcspc resolution and number of bins
    import numpy as np
    n_bins_pooled = len(pooled_decay)
    time_ns = np.arange(n_bins_pooled) * tcspc_ref * 1e9
    
    return {
        'canvas': canvas,
        'global_summary': global_summary,
        'global_popt': global_popt,
        'irf_prompt': pooled_irf,
        'time_ns': time_ns,
        'decay': pooled_decay,
        'tcspc_res': tcspc_ref,
        'n_bins': n_bins_pooled,
    }


def tile_fit(interactive=False):
    """Entry point for per-tile fitting pipeline."""
    if interactive:
        args = tile_fit_inquire()
        _run_tile_fit(args)
    else:
        ap = argparse.ArgumentParser(
            description="Per-tile FLIM fitting — fits each tile independently, "
                        "assembles canvas maps, derives global tau."
        )
        ap.add_argument("--xlif",         required=True)
        ap.add_argument("--ptu-dir",      required=True)
        ap.add_argument("--output-dir",   required=True)
        ap.add_argument("--ptu-basename", default=None)
        ap.add_argument("--rotate-tiles", action="store_true", default=True)
        ap.add_argument("--no-rotate",    action="store_true")
        ap.add_argument("--no-register",  action="store_true",
                        help="Disable phase-correlation Y registration between tile columns")
        ap.add_argument("--reg-max-shift", type=int, default=120,
                        help="Max Y search range for registration (px)")
        ap.add_argument("--irf-xlsx-dir", default=None)
        ap.add_argument("--estimate-irf", default="parametric",
                        choices=["parametric", "raw", "machine_irf"])
        ap.add_argument("--machine-irf", default=str(MACHINE_IRF_DEFAULT_PATH))
        ap.add_argument("--nexp",         type=int, default=n_exp, choices=[1, 2, 3])
        ap.add_argument("--tau-min",      type=float, default=Tau_min)
        ap.add_argument("--tau-max",      type=float, default=Tau_max)
        ap.add_argument("--binning",      type=int, default=binning_factor)
        ap.add_argument("--min-photons",  type=int, default=MIN_PHOTONS_PERPIX)
        ap.add_argument("--optimizer",    default='de', choices=["lm_multistart", "de"])
        ap.add_argument("--restarts",     type=int, default=lm_restarts)
        ap.add_argument("--de-population",type=int, default=de_population)
        ap.add_argument("--de-maxiter",   type=int, default=de_maxiter)
        ap.add_argument("--workers",      type=int, default=n_workers)
        ap.add_argument("--no-polish",    action="store_true")
        ap.add_argument("--channel",      type=int, default=channels)
        ap.add_argument("--irf-fwhm",     type=float, default=IRF_FWHM)
        ap.add_argument("--irf-bins",     type=int,   default=IRF_BINS)
        ap.add_argument("--irf-fit-width",type=float, default=IRF_FIT_WIDTH)
        ap.add_argument("--intensity-threshold", default=None)
        ap.add_argument("--tau-display-min", type=float, default=None)
        ap.add_argument("--tau-display-max", type=float, default=None)

        args = ap.parse_args()

        if args.no_rotate:
            args.rotate_tiles = False
        if args.ptu_basename is None:
            args.ptu_basename = Path(args.xlif).stem

        args.register_tiles   = not args.no_register
        args.reg_max_shift_px = args.reg_max_shift
        args.irf          = None
        args.irf_xlsx     = None
        args.irf_xlsx_map = None
        args.irf_xlsx_dir = args.irf_xlsx_dir
        args.machine_irf  = args.machine_irf
        args.mode         = 'both'
        args.no_plots     = True
        args.cell_mask    = False
        args.no_xlsx_irf  = True
        args.debug_xlsx   = False
        args.print_config = False
        args.xlsx         = None
        args.out          = None
        args.no_polish    = args.no_polish

        _run_tile_fit(args)