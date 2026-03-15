"""
flimkit/PTU/stitch.py
=====================
FLIM tile stitching and per-tile fitting pipeline.

fit_flim_tiles() uses a two-pass pooled machine IRF strategy:

  Pass 1 — summed_decay() only (no pixel stacks). Accumulates a pooled
            decay across all tiles, runs fit_summed once to get consensus
            τ values and a single pooled_irf.

  Pass 2 — raw_pixel_stack() one tile at a time. All tiles use the same
            consensus τ and pooled_irf → identical convolution basis →
            smooth amplitude maps with no inter-tile seams.

This replaces the previous per-tile independent fitting approach which
called _run_flim_fit per tile, giving different τ per tile and causing
visible seams in assembled lifetime maps.
"""

from __future__ import annotations

import json
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, Any, Optional, List

from ..utils.xml_utils import (
    parse_xlif_tile_positions,
    get_pixel_size_from_xlif,
    compute_tile_pixel_positions,
)
from .decode import get_flim_histogram_from_ptufile, create_time_axis


# ══════════════════════════════════════════════════════════════════════════════
# Tile stitching (intensity + FLIM cube)
# ══════════════════════════════════════════════════════════════════════════════

def stitch_flim_tiles(
    xlif_path: Path,
    ptu_dir: Path,
    output_dir: Path,
    ptu_basename: str = "R 2",
    rotate_tiles: bool = True,
    verbose: bool = True,
    progress_callback=None,
    cancel_event=None,
) -> Dict[str, Any]:
    """
    Stitch FLIM PTU tiles into a mosaic using your existing PTUFile class.

    Creates:
        - Intensity image (TIFF)
        - FLIM histogram cube (NPY memmap)
        - Time axis (NPY)
        - Weight map (NPY)
        - Metadata (JSON)

    Args:
        xlif_path: Path to XLIF metadata file
        ptu_dir: Directory with PTU files
        output_dir: Output directory
        ptu_basename: PTU filename base (e.g., "R 2" → "R 2_s1.ptu")
        rotate_tiles: Apply 90° CW rotation
        verbose: Print progress

    Returns:
        Dict with output paths and metadata

    Example:
        >>> result = stitch_flim_tiles(
        ...     xlif_path=Path("metadata/R 2.xlif"),
        ...     ptu_dir=Path("PTUs/"),
        ...     output_dir=Path("stitched/R_002/"),
        ...     ptu_basename="R 2",
        ... )
        >>> # Load for fitting:
        >>> flim = np.load(result['flim_path'], mmap_mode='r')
        >>> time = np.load(result['time_axis_path'])
        >>> decay = flim.sum(axis=(0,1))
    """
    xlif_path  = Path(xlif_path)
    ptu_dir    = Path(ptu_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    roi_prefix = ptu_basename.replace(' ', '_')

    output_intensity = output_dir / f"{roi_prefix}_stitched_intensity.tif"
    output_flim      = output_dir / f"{roi_prefix}_stitched_flim_counts.npy"
    output_time      = output_dir / f"{roi_prefix}_time_axis_ns.npy"
    output_weight    = output_dir / f"{roi_prefix}_weight_map.npy"
    output_meta      = output_dir / f"{roi_prefix}_metadata.json"

    if verbose:
        print(f"{'='*60}")
        print(f"FLIM TILE STITCHING")
        print(f"{'='*60}")
        print(f"XLIF: {xlif_path}")
        print(f"PTUs: {ptu_dir}")
        print(f"Output: {output_dir}")
        print()
        print("Parsing XLIF metadata...")

    tile_positions = parse_xlif_tile_positions(xlif_path, ptu_basename)
    pixel_size_m, n_pixels = get_pixel_size_from_xlif(xlif_path)

    if verbose:
        print(f"  Found {len(tile_positions)} tiles")
        print(f"  Pixel size: {pixel_size_m * 1e6:.4f} µm")

    first_tile_path = ptu_dir / tile_positions[0]["file"]
    if not first_tile_path.exists():
        raise FileNotFoundError(f"First tile not found: {first_tile_path}")

    if verbose:
        print(f"  Loading first tile: {first_tile_path.name}")

    first_hist, first_meta = get_flim_histogram_from_ptufile(
        first_tile_path, rotate_cw=rotate_tiles, binning=1, channel=None)

    tile_y, tile_x    = first_meta['tile_shape']
    n_time_bins       = first_meta['n_time_bins']
    tcspc_resolution  = first_meta['tcspc_resolution']
    time_axis_ns      = create_time_axis(n_time_bins, tcspc_resolution)

    if verbose:
        print(f"  Tile shape: ({tile_y}, {tile_x}, {n_time_bins})")
        print(f"  TCSPC: {tcspc_resolution * 1e12:.2f} ps/bin")
        print(f"  Time range: 0 - {time_axis_ns[-1]:.2f} ns")

    tile_positions, canvas_width, canvas_height = compute_tile_pixel_positions(
        tile_positions, pixel_size_m, tile_x)

    if verbose:
        print(f"  Canvas: {canvas_height} × {canvas_width} pixels")
        print()
        print("Allocating arrays...")

    intensity_canvas = np.zeros((canvas_height, canvas_width), dtype=np.float64)
    flim_canvas = np.memmap(
        str(output_flim), dtype=np.uint32, mode='w+',
        shape=(canvas_height, canvas_width, n_time_bins))
    weight_canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    if verbose:
        print(f"Stitching {len(tile_positions)} tiles...")
        print()

    tiles_processed = tiles_skipped = 0
    total_tiles = len(tile_positions)

    for i, t in enumerate(tqdm(tile_positions, desc='  Loading tiles')):
        if cancel_event is not None and cancel_event.is_set():
            if verbose:
                print("\nStitching cancelled by user.")
            break
        if progress_callback is not None:
            progress_callback(i, total_tiles)

        tile_path = ptu_dir / t["file"]
        if not tile_path.exists():
            if verbose:
                print(f"  [{i+1:3d}/{len(tile_positions)}] MISSING: {t['file']}")
            tiles_skipped += 1
            continue

        try:
            hist, meta = get_flim_histogram_from_ptufile(
                tile_path, rotate_cw=rotate_tiles, binning=1, channel=None)

            if hist.shape[2] != n_time_bins:
                if hist.shape[2] < n_time_bins:
                    padded = np.zeros(
                        (hist.shape[0], hist.shape[1], n_time_bins), dtype=hist.dtype)
                    padded[:, :, :hist.shape[2]] = hist
                    hist = padded
                else:
                    hist = hist[:, :, :n_time_bins]

            y0, x0 = t["pixel_y"], t["pixel_x"]
            y1, x1 = y0 + tile_y, x0 + tile_x

            flim_canvas[y0:y1, x0:x1, :]   += hist.astype(np.uint32)
            intensity_canvas[y0:y1, x0:x1]  += hist.sum(axis=2).astype(np.float64)
            weight_canvas[y0:y1, x0:x1]     += 1.0
            tiles_processed += 1

        except Exception as e:
            if verbose:
                print(f"  [{i+1:3d}/{len(tile_positions)}] ERROR: {t['file']}: {e}")
            tiles_skipped += 1
            continue

    # Normalise intensity image for display (FLIM cube intentionally NOT normalised)
    if verbose:
        print()
        print("Normalising intensity image for display...")

    mask = weight_canvas > 0
    intensity_canvas[mask] /= weight_canvas[mask]

    n_overlap = int((weight_canvas > 1).sum())
    if verbose and n_overlap > 0:
        print(f"  {n_overlap:,} overlap pixels "
              f"({100*n_overlap/mask.sum():.1f}% of canvas) — "
              f"photon counts summed across tiles")

    if verbose:
        print("Saving outputs...")

    max_val = intensity_canvas.max()
    intensity_scaled = (
        (intensity_canvas / max_val * 65535).astype(np.uint16)
        if max_val > 0 else
        np.zeros_like(intensity_canvas, dtype=np.uint16)
    )
    tifffile.imwrite(str(output_intensity), intensity_scaled)
    np.save(str(output_time), time_axis_ns)
    np.save(str(output_weight), weight_canvas)
    flim_canvas.flush()

    metadata = {
        'canvas_shape':        (canvas_height, canvas_width),
        'n_time_bins':         int(n_time_bins),
        'time_range_ns':       (0.0, float(time_axis_ns[-1])),
        'tcspc_resolution_ps': float(tcspc_resolution * 1e12),
        'pixel_size_um':       float(pixel_size_m * 1e6),
        'tiles_processed':     tiles_processed,
        'tiles_skipped':       tiles_skipped,
        'ptu_basename':        ptu_basename,
    }
    with open(output_meta, 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        for name in (output_intensity, output_flim, output_time,
                     output_weight, output_meta):
            print(f"  ✓ {name.name}")
        print()
        print(f"{'='*60}")
        print(f"STITCHING COMPLETE")
        print(f"{'='*60}")
        print(f"Processed: {tiles_processed}/{len(tile_positions)} tiles")
        print(f"Canvas: {canvas_height} × {canvas_width} × {n_time_bins}")
        print(f"Time: 0 - {time_axis_ns[-1]:.2f} ns")

    return {
        'intensity_path':  output_intensity,
        'flim_path':       output_flim,
        'time_axis_path':  output_time,
        'weight_map_path': output_weight,
        'metadata_path':   output_meta,
        'canvas_shape':    (canvas_height, canvas_width),
        'n_time_bins':     n_time_bins,
        'tiles_processed': tiles_processed,
        'tiles_skipped':   tiles_skipped,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Load helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_stitched_flim(
    output_dir: Path,
    mode: str = 'r',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load previously stitched FLIM data.

    Returns
    -------
    (flim_cube, time_axis, intensity, metadata)
    """
    output_dir = Path(output_dir)

    meta_candidates = sorted(output_dir.glob("*_metadata.json"))
    if meta_candidates:
        meta_path  = meta_candidates[0]
        roi_prefix = meta_path.name.replace('_metadata.json', '')
    elif (output_dir / "metadata.json").exists():
        meta_path  = output_dir / "metadata.json"
        roi_prefix = None
    else:
        raise FileNotFoundError(f"No metadata.json found in {output_dir}")

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    canvas_shape = tuple(metadata['canvas_shape'])
    n_time_bins  = metadata['n_time_bins']

    def _find(prefixed, generic):
        p = output_dir / prefixed
        return p if p.exists() else output_dir / generic

    if roi_prefix:
        time_path = _find(f"{roi_prefix}_time_axis_ns.npy",       "time_axis_ns.npy")
        int_path  = _find(f"{roi_prefix}_stitched_intensity.tif",  "stitched_intensity.tif")
        flim_path = _find(f"{roi_prefix}_stitched_flim_counts.npy","stitched_flim_counts.npy")
    else:
        time_path = output_dir / "time_axis_ns.npy"
        int_path  = output_dir / "stitched_intensity.tif"
        flim_path = output_dir / "stitched_flim_counts.npy"

    time_axis = np.load(str(time_path))
    intensity  = tifffile.imread(str(int_path))
    flim = np.memmap(str(flim_path), dtype=np.uint32, mode=mode,
                     shape=(canvas_shape[0], canvas_shape[1], n_time_bins))

    return flim, time_axis, intensity, metadata


def load_flim_for_fitting(
    source_dir: Path,
    load_to_memory: bool = False,
) -> Tuple[np.ndarray, float, int]:
    """
    Load stitched FLIM data ready for the fitting pipeline.

    Returns
    -------
    (stack, tcspc_res_s, n_bins)
    """
    flim_memmap, _, _, metadata = load_stitched_flim(source_dir)
    tcspc_res = metadata['tcspc_resolution_ps'] * 1e-12
    n_bins    = metadata['n_time_bins']
    stack = np.array(flim_memmap, dtype=np.float32) if load_to_memory else flim_memmap
    return stack, tcspc_res, n_bins


# ══════════════════════════════════════════════════════════════════════════════
# Per-tile fitting pipeline — pooled machine IRF
# ══════════════════════════════════════════════════════════════════════════════

def _peek_tile_width(ptu_dir, tile_positions, rotate_tiles) -> int:
    """Load the first available tile just to get its pixel width."""
    for t in tile_positions:
        p = Path(ptu_dir) / t['file']
        if p.exists():
            _, meta = get_flim_histogram_from_ptufile(
                p, rotate_cw=rotate_tiles, binning=1, channel=None)
            return meta['tile_shape'][1]
    raise FileNotFoundError("No tile PTU files found to determine tile width")


def _resolve_tile_irf(ptu_name: str, irf_xlsx_dir=None, irf_xlsx_map=None):
    """
    Return the xlsx path to use as IRF for this tile, or None.
    Kept for API compatibility — not used in the pooled pipeline.
    """
    stem = Path(ptu_name).stem
    if irf_xlsx_map:
        if ptu_name in irf_xlsx_map:
            return irf_xlsx_map[ptu_name]
        if stem in irf_xlsx_map:
            return irf_xlsx_map[stem]
    if irf_xlsx_dir is not None:
        candidate = Path(irf_xlsx_dir) / f"{stem}.xlsx"
        if candidate.exists():
            return candidate
    return None


def _load_machine_irf(path: str | Path) -> tuple[np.ndarray, int]:
    """Load and normalise machine IRF. Returns (irf_norm, peak_bin)."""
    irf = np.asarray(np.load(str(path)), dtype=float).ravel()
    irf = np.maximum(irf, 0.0)
    s   = irf.sum()
    if s <= 0:
        raise ValueError(f"Machine IRF is all-zero: {path}")
    irf /= s
    return irf, int(np.argmax(irf))


def _get_tile_irf(machine_irf: np.ndarray, pi_machine: int,
                  tile_peak_bin: int, n_bins: int) -> np.ndarray:
    """Shift machine IRF to tile_peak_bin, clipped/padded to n_bins."""
    irf = machine_irf.copy()
    if irf.size > n_bins:
        irf = irf[:n_bins]
    elif irf.size < n_bins:
        irf = np.pad(irf, (0, n_bins - irf.size))
    shift = tile_peak_bin - pi_machine
    if shift != 0:
        irf = np.roll(irf, shift)
    s = irf.sum()
    return irf / s if s > 0 else irf


def _adapt_pixel_maps(pixel_maps: dict, n_exp: int,
                      taus_ns: np.ndarray) -> dict:
    """
    Remap fit_per_pixel output keys → assemble_tile_maps format.

    fit_per_pixel:       intensity, tau_mean_amp, tau_mean_int, chi2_r,
                         alpha_N, frac_N
    assemble_tile_maps:  intensity, tau_mean_amp, chi2, tauN, aN
    """
    adapted = {
        'intensity':    pixel_maps['intensity'],
        'tau_mean_amp': pixel_maps['tau_mean_amp'],
        'chi2':         pixel_maps['chi2_r'],
    }
    ny, nx = pixel_maps['intensity'].shape
    for k in range(1, n_exp + 1):
        adapted[f'tau{k}'] = np.full((ny, nx), taus_ns[k - 1], dtype=np.float32)
        adapted[f'a{k}']   = pixel_maps.get(
            f'alpha_{k}', np.full((ny, nx), np.nan, dtype=np.float32))
    return adapted


def fit_flim_tiles(
    xlif_path:     Path,
    ptu_dir:       Path,
    output_dir:    Path,
    args,
    ptu_basename:  str  = "R 2",
    rotate_tiles:  bool = True,
    irf_xlsx_dir          = None,   # kept for API compat — not used
    irf_xlsx_map          = None,   # kept for API compat — not used
    verbose:       bool = True,
    progress_callback     = None,
    cancel_event          = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Per-tile FLIM fitting with pooled machine IRF.

    Two-pass strategy — eliminates inter-tile seams in lifetime maps:

    Pass 1  summed_decay() only (no pixel stacks, fast).
            Accumulates a pooled decay across all tiles, runs fit_summed
            once → consensus τ values + pooled_irf shared by all tiles.

    Pass 2  raw_pixel_stack() one tile at a time (memory-bounded).
            Every tile uses the same consensus τ and pooled_irf, giving an
            identical convolution basis → smooth amplitude maps with no
            tile boundary artefacts.

    Args
    ----
    xlif_path:       XLIF metadata file (tile positions)
    ptu_dir:         directory containing PTU tile files
    output_dir:      where assembled outputs are saved
    args:            argparse.Namespace with fitting parameters:
                       .nexp, .tau_min, .tau_max, .optimizer, .restarts,
                       .de_population, .de_maxiter, .workers, .binning,
                       .min_photons, .cost_function, .machine_irf
    ptu_basename:    PTU filename base (e.g. "R 2")
    rotate_tiles:    apply 90° CW rotation to each tile stack
    irf_xlsx_dir:    ignored (kept for API compat)
    irf_xlsx_map:    ignored (kept for API compat)
    verbose:         print progress
    progress_callback: optional callable(step, total_steps)
    cancel_event:    optional threading.Event to abort

    Returns
    -------
    tile_results   : list of dicts ready for assemble_tile_maps()
    canvas_height  : int
    canvas_width   : int
    """
    from ..PTU.reader import PTUFile
    from ..FLIM.fitters import fit_summed, fit_per_pixel
    from ..configs import (
        MACHINE_IRF_DEFAULT_PATH,
        MACHINE_IRF_FIT_BG, MACHINE_IRF_FIT_SIGMA, MACHINE_IRF_FIT_TAIL,
        MIN_PHOTONS_PERPIX,
        Tau_min, Tau_max, n_exp as _cfg_nexp,
        Cost_function, Optimizer, lm_restarts, n_workers,
        binning_factor,
    )

    xlif_path  = Path(xlif_path)
    ptu_dir    = Path(ptu_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve parameters from args ──────────────────────────────────────
    n_exp_      = getattr(args, 'nexp',          _cfg_nexp)
    tau_min_ns  = getattr(args, 'tau_min',       Tau_min)
    tau_max_ns  = getattr(args, 'tau_max',       Tau_max)
    cost_fn     = getattr(args, 'cost_function', Cost_function)
    optimizer   = getattr(args, 'optimizer',     Optimizer)
    restarts    = getattr(args, 'restarts',      lm_restarts)
    workers     = getattr(args, 'workers',       n_workers)
    binning     = getattr(args, 'binning',       binning_factor)
    min_photons = getattr(args, 'min_photons',   MIN_PHOTONS_PERPIX)
    fit_bg      = MACHINE_IRF_FIT_BG
    fit_sigma   = MACHINE_IRF_FIT_SIGMA
    has_tail    = MACHINE_IRF_FIT_TAIL
    mach_path   = getattr(args, 'machine_irf',  str(MACHINE_IRF_DEFAULT_PATH))

    machine_irf, pi_machine = _load_machine_irf(mach_path)

    # ── Parse tile positions ───────────────────────────────────────────────
    tile_positions = parse_xlif_tile_positions(xlif_path, ptu_basename)
    pixel_size_m, _ = get_pixel_size_from_xlif(xlif_path)
    tile_positions, canvas_w, canvas_h = compute_tile_pixel_positions(
        tile_positions, pixel_size_m,
        _peek_tile_width(ptu_dir, tile_positions, rotate_tiles))

    if verbose:
        print(f"\n{'='*60}")
        print(f"  PER-TILE FLIM FITTING — POOLED MACHINE IRF")
        print(f"{'='*60}")
        print(f"  XLIF:        {xlif_path}")
        print(f"  PTUs:        {ptu_dir}")
        print(f"  Tiles:       {len(tile_positions)}")
        print(f"  Canvas:      {canvas_h} × {canvas_w} px")
        print(f"  Machine IRF: {mach_path}  (peak bin {pi_machine})\n")

    total_steps = 2 * len(tile_positions)

    # ══════════════════════════════════════════════════════════════════════
    # PASS 1 — pool summed decays, fit consensus τ
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print("Pass 1: accumulating pooled decay (summed_decay only)...")

    tile_meta    = []
    pooled_decay = None
    n_bins_ref   = None
    tcspc_ref    = None

    for i, t in enumerate(tqdm(tile_positions,
                                desc='  Pass 1', disable=not verbose)):
        if cancel_event is not None and cancel_event.is_set():
            break
        if progress_callback is not None:
            progress_callback(i, total_steps)

        ptu_path = ptu_dir / t['file']
        if not ptu_path.exists():
            continue

        ptu    = PTUFile(str(ptu_path), verbose=False)
        decay  = ptu.summed_decay()
        n_bins = ptu.n_bins
        tcspc  = ptu.tcspc_res

        if pooled_decay is None:
            pooled_decay = decay.copy()
            n_bins_ref   = n_bins
            tcspc_ref    = tcspc
        else:
            # Expand pooled array if this tile has more bins
            if n_bins > n_bins_ref:
                pooled_decay = np.pad(pooled_decay, (0, n_bins - n_bins_ref))
                n_bins_ref   = n_bins
            if len(decay) < len(pooled_decay):
                decay = np.pad(decay, (0, len(pooled_decay) - len(decay)))
            pooled_decay[:len(decay)] += decay[:len(pooled_decay)]

        tile_meta.append({
            't':        t,
            'n_bins':   n_bins,
            'tcspc':    tcspc,
            'peak_bin': int(np.argmax(decay)),
        })

    if pooled_decay is None:
        raise RuntimeError("No tiles found — check PTU_DIR and PTU_BASENAME.")

    pooled_peak = int(np.argmax(pooled_decay))
    pooled_irf  = _get_tile_irf(machine_irf, pi_machine, pooled_peak, n_bins_ref)

    if verbose:
        print(f"\n  Pooled: {len(tile_meta)} tiles  "
              f"{pooled_decay.sum():,.0f} photons  peak bin {pooled_peak}")
        print("\n  Running consensus fit_summed on pooled decay...")

    global_popt, global_summary = fit_summed(
        pooled_decay, tcspc_ref, n_bins_ref, pooled_irf,
        has_tail      = has_tail,
        fit_bg        = fit_bg,
        fit_sigma     = fit_sigma,
        n_exp         = n_exp_,
        tau_min_ns    = tau_min_ns,
        tau_max_ns    = tau_max_ns,
        optimizer     = optimizer,
        cost_function = cost_fn,
        n_restarts    = restarts,
        workers       = workers,
    )
    consensus_taus_ns = global_summary['taus_ns']

    if verbose:
        print(f"\n  Consensus τ = {[f'{t:.3f}' for t in consensus_taus_ns]} ns")
        print(f"  χ²_r (tail) = {global_summary['reduced_chi2_tail']:.4f}")

    # Zero irf_shift_bins — pooled_irf is already peak-aligned
    popt_for_px = global_popt.copy()
    popt_for_px[2 * n_exp_] = 0.0

    # ══════════════════════════════════════════════════════════════════════
    # PASS 2 — per-pixel fit, one tile at a time
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\nPass 2: per-pixel fit ({len(tile_meta)} tiles)...")
        print(f"  Fixed τ   = {[f'{t:.3f}' for t in consensus_taus_ns]} ns")
        print(f"  Fixed IRF = pooled_irf (peak bin {pooled_peak})\n")

    tile_results  = []
    tiles_skipped = 0

    for i, tc in enumerate(tqdm(tile_meta,
                                 desc='  Pass 2', disable=not verbose)):
        if cancel_event is not None and cancel_event.is_set():
            break
        if progress_callback is not None:
            progress_callback(len(tile_meta) + i, total_steps)

        ptu_path = ptu_dir / tc['t']['file']
        n_bins   = tc['n_bins']
        tcspc    = tc['tcspc']

        # Pad/crop pooled_irf to this tile's n_bins, renormalise
        if len(pooled_irf) < n_bins:
            irf_tile = np.pad(pooled_irf, (0, n_bins - len(pooled_irf)))
        else:
            irf_tile = pooled_irf[:n_bins]
        irf_tile = irf_tile / irf_tile.sum()

        try:
            ptu = PTUFile(str(ptu_path), verbose=False)
            ptu.summed_decay()                               # sets photon_channel
            stack = ptu.raw_pixel_stack(
                channel=ptu.photon_channel, binning=binning)
            if rotate_tiles:
                stack = np.rot90(stack, k=-1, axes=(0, 1))
            tile_h, tile_w = stack.shape[:2]

            pixel_maps_raw = fit_per_pixel(
                stack.astype(float),
                tcspc, n_bins, irf_tile,
                has_tail    = has_tail,
                fit_bg      = fit_bg,
                fit_sigma   = fit_sigma,
                global_popt = popt_for_px,
                n_exp       = n_exp_,
                min_photons = min_photons,
            )
            del stack

            pixel_maps = _adapt_pixel_maps(pixel_maps_raw, n_exp_, consensus_taus_ns)
            n_fitted   = int(np.isfinite(pixel_maps['tau_mean_amp']).sum())

            tile_results.append({
                'pixel_maps':     pixel_maps,
                'global_summary': global_summary,
                'pixel_y':        tc['t']['pixel_y'],
                'pixel_x':        tc['t']['pixel_x'],
                'tile_h':         tile_h,
                'tile_w':         tile_w,
                'peak_bin':       tc['peak_bin'],
                'ptu_name':       tc['t']['file'],
            })

            if verbose:
                tqdm.write(
                    f"    {tc['t']['file']:<30}  "
                    f"{pixel_maps['intensity'].sum():>10,.0f} ph  "
                    f"fitted={n_fitted}")

        except Exception as e:
            import traceback, sys
            if verbose:
                tqdm.write(f"  ERROR: {tc['t']['file']}: {e}", file=sys.stderr)
                tqdm.write(traceback.format_exc(), file=sys.stderr)
            tiles_skipped += 1
            continue

    if verbose:
        print(f"\n  {len(tile_results)}/{len(tile_meta)} tiles fitted "
              f"({tiles_skipped} errors)")

    return tile_results, canvas_h, canvas_w
