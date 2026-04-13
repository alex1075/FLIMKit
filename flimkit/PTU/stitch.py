"""
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

# Disable tqdm globally – all progress is shown via progress windows instead
tqdm.disable = True

from typing import Tuple, Dict, Any, Optional, List

from ..utils.xml_utils import (
    parse_xlif_tile_positions,
    get_pixel_size_from_xlif,
    compute_tile_pixel_positions,
)
from .decode import get_flim_histogram_from_ptufile, create_time_axis

# Try to import GUI_MODE flag (set by gui.py if running in GUI mode)
try:
    from ..UI.gui import GUI_MODE
except (ImportError, AttributeError):
    GUI_MODE = False



# Tile stitching (intensity + FLIM cube)


def stitch_flim_tiles(
    xlif_path: Path,
    ptu_dir: Path,
    output_dir: Path,
    ptu_basename: str = "R 2",
    rotate_tiles: bool = True,
    register_tiles: bool = True,
    reg_max_shift_px: int = 120,
    tile_positions: Optional[list] = None,
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

    # Accept pre-computed (and optionally registered) tile positions
    # from a prior fit_flim_tiles call — avoids re-running registration.
    if tile_positions is None:
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

    # Only recompute pixel positions if they weren't pre-supplied
    _positions_precomputed = ('pixel_x' in tile_positions[0] and
                              'pixel_y' in tile_positions[0])
    if not _positions_precomputed:
        tile_positions, canvas_width, canvas_height = compute_tile_pixel_positions(
            tile_positions, pixel_size_m, tile_x)
    else:
        canvas_width  = max(t['pixel_x'] for t in tile_positions) + tile_x
        canvas_height = max(t['pixel_y'] for t in tile_positions) + tile_y

    # Registration note for stitch_flim_tiles:
    # If tile_positions are pre-supplied (from a prior fit_flim_tiles call
    # that already ran _register_tile_columns), they are used directly.
    # For standalone stitch_only runs, intensity maps are extracted from PTU files
    # and registration runs inline via _register_tile_columns if register_tiles=True.

    if verbose:
        print(f"  Canvas: {canvas_height} × {canvas_width} pixels")
        print()
        print("Allocating arrays...")

    intensity_canvas = np.zeros((canvas_height, canvas_width), dtype=np.float64)
    flim_canvas = np.memmap(
        str(output_flim), dtype=np.uint32, mode='w+',
        shape=(canvas_height, canvas_width, n_time_bins))
    # Nearest-centre ownership: each pixel is owned by whichever tile centre
    # is closest — no blending of overlapping tiles, so overlaps stay sharp.
    _owner     = np.full((canvas_height, canvas_width), -1,     dtype=np.int32)
    _min_dist2 = np.full((canvas_height, canvas_width), np.inf, dtype=np.float64)
    _hists     = []   # (ti, y0, x0, hist) deferred until ownership is known

    if verbose:
        print(f"Stitching {len(tile_positions)} tiles...")
        print()

    tiles_processed = tiles_skipped = 0
    total_tiles = len(tile_positions)
    tile_results = []  # For registration: [{pixel_maps: {intensity}, pixel_y, pixel_x, tile_h, tile_w}, ...]

    for i, t in enumerate(tqdm(tile_positions, desc='  Loading tiles', disable=True)):
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
            y1 = min(y0 + tile_y, canvas_height)
            x1 = min(x0 + tile_x, canvas_width)
            dy, dx = y1 - y0, x1 - x0

            ti = len(_hists)
            _hists.append((ti, y0, x0, hist[:dy, :dx, :]))
            
            # Extract intensity map for registration (sum over time bins)
            intensity_map = hist[:dy, :dx, :].sum(axis=2).astype(np.float32)
            tile_results.append({
                'pixel_maps':     {'intensity': intensity_map},
                'pixel_y':        y0,
                'pixel_x':        x0,
                'tile_h':         dy,
                'tile_w':         dx,
                'ptu_name':       t['file'],
            })

            # Claim ownership of pixels closer to this tile's centre
            cy = y0 + tile_y / 2.0
            cx = x0 + tile_x / 2.0
            rows = np.arange(y0, y1, dtype=np.float64)
            cols = np.arange(x0, x1, dtype=np.float64)
            dist2 = (rows - cy)[:, np.newaxis] ** 2 + (cols - cx) ** 2
            region  = _min_dist2[y0:y1, x0:x1]
            closer  = dist2 < region
            _min_dist2[y0:y1, x0:x1] = np.where(closer, dist2, region)
            _owner[y0:y1, x0:x1]     = np.where(closer, ti, _owner[y0:y1, x0:x1])

            tiles_processed += 1

        except Exception as e:
            if verbose:
                print(f"  [{i+1:3d}/{len(tile_positions)}] ERROR: {t['file']}: {e}")
            tiles_skipped += 1
            continue

    #  Optional registration using intensity maps
    if register_tiles and tiles_processed > 1 and tile_results:
        if verbose:
            print(f"\nRunning tile registration (phase correlation)...")
        tile_results = _register_tile_columns(
            tile_results,
            max_shift_px=reg_max_shift_px,
            verbose=verbose,
        )
        # Update tile_positions with corrected coordinates
        for i, tr in enumerate(tile_results):
            if i < len(tile_positions):
                tile_positions[i]["pixel_y"] = tr["pixel_y"]
                tile_positions[i]["pixel_x"] = tr["pixel_x"]
        # Recalculate ownership map with corrected positions
        _owner[:] = -1
        _min_dist2[:] = np.inf
        for ti, (hist_ti, y0_old, x0_old, h) in enumerate(_hists):
            # Get corrected position
            y0 = tile_positions[ti]["pixel_y"] if ti < len(tile_positions) else y0_old
            x0 = tile_positions[ti]["pixel_x"] if ti < len(tile_positions) else x0_old
            y1 = min(y0 + h.shape[0], canvas_height)
            x1 = min(x0 + h.shape[1], canvas_width)
            dy, dx = y1 - y0, x1 - x0
            if dy <= 0 or dx <= 0:
                continue
            # Reclaim ownership with corrected positions
            cy = y0 + h.shape[0] / 2.0
            cx = x0 + h.shape[1] / 2.0
            rows = np.arange(y0, y1, dtype=np.float64)
            cols = np.arange(x0, x1, dtype=np.float64)
            dist2 = (rows - cy)[:, np.newaxis] ** 2 + (cols - cx) ** 2
            region  = _min_dist2[y0:y1, x0:x1]
            closer  = dist2 < region
            _min_dist2[y0:y1, x0:x1] = np.where(closer, dist2, region)
            _owner[y0:y1, x0:x1]     = np.where(closer, ti, _owner[y0:y1, x0:x1])
            # Update _hists with corrected position
            _hists[ti] = (hist_ti, y0, x0, h)

        # Recompute canvas size from corrected positions so tiles shifted
        # beyond the original XLIF bounds are not clipped.
        new_canvas_height = max(y0_ + h_.shape[0] for _, y0_, x0_, h_ in _hists)
        new_canvas_width  = max(x0_ + h_.shape[1] for _, y0_, x0_, h_ in _hists)
        if new_canvas_height > canvas_height or new_canvas_width > canvas_width:
            if verbose:
                print(
                    f"  Registration expanded canvas: "
                    f"{canvas_height}×{canvas_width} → "
                    f"{new_canvas_height}×{new_canvas_width} px"
                )
            # Reallocate in-memory canvas; re-open memmap at new shape.
            intensity_canvas = np.zeros(
                (new_canvas_height, new_canvas_width), dtype=np.float64)
            flim_canvas._mmap.close()
            flim_canvas = np.memmap(
                str(output_flim), dtype=np.uint32, mode="w+",
                shape=(new_canvas_height, new_canvas_width, n_time_bins))
            # Rebuild ownership map at the new size.
            _owner     = np.full(
                (new_canvas_height, new_canvas_width), -1,     dtype=np.int32)
            _min_dist2 = np.full(
                (new_canvas_height, new_canvas_width), np.inf, dtype=np.float64)
            for ti_, (_, y0_, x0_, h_) in enumerate(_hists):
                y1_ = min(y0_ + h_.shape[0], new_canvas_height)
                x1_ = min(x0_ + h_.shape[1], new_canvas_width)
                cy_ = y0_ + h_.shape[0] / 2.0
                cx_ = x0_ + h_.shape[1] / 2.0
                rows_ = np.arange(y0_, y1_, dtype=np.float64)
                cols_ = np.arange(x0_, x1_, dtype=np.float64)
                d2_   = (rows_ - cy_)[:, np.newaxis] ** 2 + (cols_ - cx_) ** 2
                reg_  = _min_dist2[y0_:y1_, x0_:x1_]
                cl_   = d2_ < reg_
                _min_dist2[y0_:y1_, x0_:x1_] = np.where(cl_, d2_, reg_)
                _owner[y0_:y1_, x0_:x1_]     = np.where(cl_, ti_, _owner[y0_:y1_, x0_:x1_])
            canvas_height = new_canvas_height
            canvas_width  = new_canvas_width

    # Write each tile's data only for pixels it owns
    if verbose:
        blending_mode = "with registration" if (register_tiles and tiles_processed > 1) else "no blending"
        print(f"  Writing canvas (nearest-centre, {blending_mode})...")
    for ti, y0, x0, h in _hists:
        y1 = y0 + h.shape[0]
        x1 = x0 + h.shape[1]
        owned_r, owned_c = np.where(_owner[y0:y1, x0:x1] == ti)
        if owned_r.size > 0:
            flim_canvas[y0 + owned_r, x0 + owned_c, :] = h[owned_r, owned_c, :]
            intensity_canvas[y0 + owned_r, x0 + owned_c] = \
                h[owned_r, owned_c, :].sum(axis=1).astype(np.float64)
    del _min_dist2

    # Coverage report
    n_covered = int((_owner >= 0).sum())
    if verbose:
        print(f"  {n_covered:,} pixels covered  "
              f"({100*n_covered/(canvas_height*canvas_width):.1f}% of canvas)  "
              f"nearest-centre selection, no blending")
        print("Saving outputs...")

    max_val = intensity_canvas.max()
    intensity_scaled = (
        (intensity_canvas / max_val * 65535).astype(np.uint16)
        if max_val > 0 else
        np.zeros_like(intensity_canvas, dtype=np.uint16)
    )
    tifffile.imwrite(str(output_intensity), intensity_scaled)
    np.save(str(output_time), time_axis_ns)
    # Save ownership map: each value = tile index + 1 (0 = uncovered)
    np.save(str(output_weight), (_owner + 1).astype(np.uint16))
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



# Load helpers
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



# Per-tile fitting pipeline — pooled machine IRF
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








def _phase_corr_2d(patch_a, patch_b, max_shift_y=120, max_shift_x=30):
    """
    2D normalised phase correlation (Kuglin & Hines 1975).
    Returns (dy, dx, confidence):
        dy, dx      : sub-pixel shift — patch_b is shifted (dy, dx) vs patch_a
        confidence  : peak / mean of search region (higher = more reliable)
    Uses Hann window and Gaussian sub-pixel fitting (Preibisch et al. 2009).
    """
    h = min(patch_a.shape[0], patch_b.shape[0])
    w = min(patch_a.shape[1], patch_b.shape[1])
    pa = patch_a[:h, :w].astype(np.float64)
    pb = patch_b[:h, :w].astype(np.float64)
    wy  = np.hanning(h)
    wx  = np.hanning(w)
    win = wy[:, np.newaxis] * wx[np.newaxis, :]
    pa  = (pa - pa.mean()) * win
    pb  = (pb - pb.mean()) * win
    Fa  = np.fft.fft2(pa)
    Fb  = np.fft.fft2(pb)
    cross = Fa * np.conj(Fb)
    denom = np.abs(cross)
    denom[denom < 1e-10] = 1e-10
    corr  = np.real(np.fft.ifft2(cross / denom))
    corr_s = np.fft.fftshift(corr)
    cy, cx = h // 2, w // 2
    mask   = np.zeros_like(corr_s)
    y_lo   = max(0, cy - max_shift_y);  y_hi = min(h, cy + max_shift_y + 1)
    x_lo   = max(0, cx - max_shift_x);  x_hi = min(w, cx + max_shift_x + 1)
    mask[y_lo:y_hi, x_lo:x_hi] = 1
    corr_s *= mask
    pk_y, pk_x = np.unravel_index(np.argmax(corr_s), corr_s.shape)
    peak_val   = corr_s[pk_y, pk_x]
    confidence = peak_val / (corr_s[y_lo:y_hi, x_lo:x_hi].mean() + 1e-10)
    def _sub(arr, pk, lo, hi):
        if lo < pk < hi - 1:
            a, b, c = arr[pk-1], arr[pk], arr[pk+1]
            if a > 0 and b > 0 and c > 0:
                try:
                    la, lb, lc = np.log(a), np.log(b), np.log(c)
                    return pk + (la - lc) / (2 * (la - 2*lb + lc))
                except Exception:
                    pass
        return float(pk)
    sub_y = _sub(corr_s[:, pk_x], pk_y, y_lo, y_hi) - cy
    sub_x = _sub(corr_s[pk_y, :], pk_x, x_lo, x_hi) - cx
    return sub_y, sub_x, confidence


def _register_tile_columns(
    tile_results: list,
    max_shift_px: int = 120,
    verbose: bool = True,
) -> list:
    """
    Three-pass tile registration (Preibisch et al. 2009 approach):

    Pass A — column Y drift:  ~8px/col stage encoder drift measured from
              horizontal (col) overlap zones. Cumulative per-column Y correction.

    Pass B — row Y residual:  remaining Y mismatch at each row boundary measured
              from vertical (row) overlap zones. Whole-row Y shift.

    Pass C — row X residual:  X drift between rows (bidirectional scan backlash)
              measured from vertical overlap zones. Whole-row X shift.

    Key design: tiles are indexed by ORIGINAL (row_idx, col_idx) so that
    Pass A's per-column Y corrections don't fragment the row grouping in
    Passes B and C.

    Uses 2D phase correlation with Hann windowing and Gaussian sub-pixel
    refinement. MAD-based outlier rejection per group. Tissue-fraction
    filter discards overlap strips that are mostly background.
    """
    REG_MAX_SHIFT_Y = max_shift_px
    REG_MAX_SHIFT_X = 30
    MIN_CONF        = 5.0
    MAD_THRESHOLD   = 3.0
    MIN_TISSUE_FRAC = 0.05

    if not tile_results:
        return tile_results

    #  geometry from original XLIF positions ─
    orig_col_xs = sorted(set(int(round(tr['pixel_x']/10)*10) for tr in tile_results))
    orig_row_ys = sorted(set(int(round(tr['pixel_y']/10)*10) for tr in tile_results))
    tile_w = max(tr['tile_w'] for tr in tile_results)
    tile_h = max(tr['tile_h'] for tr in tile_results)
    col_pitch   = int(np.median(np.diff(orig_col_xs))) if len(orig_col_xs)>1 else tile_w
    row_pitch   = int(np.median(np.diff(orig_row_ys))) if len(orig_row_ys)>1 else tile_h
    col_overlap = tile_w - col_pitch
    row_overlap = tile_h - row_pitch
    N_rows = len(orig_row_ys)
    N_cols = len(orig_col_xs)

    if col_overlap < 4:
        if verbose:
            print(f'  Registration: col_overlap={col_overlap}px too small — skipping')
        return tile_results

    if verbose:
        print(f'  Registration: {N_rows}r×{N_cols}c  '
              f'col_overlap={col_overlap}px  row_overlap={row_overlap}px')

    # Index tiles by original (row_idx, col_idx) — stable across all passes
    orig_grid = {}
    for i, tr in enumerate(tile_results):
        try:
            ci = orig_col_xs.index(int(round(tr['pixel_x']/10)*10))
        except ValueError:
            ci = min(range(N_cols), key=lambda c: abs(orig_col_xs[c]-tr['pixel_x']))
        try:
            ri = orig_row_ys.index(int(round(tr['pixel_y']/10)*10))
        except ValueError:
            ri = min(range(N_rows), key=lambda r: abs(orig_row_ys[r]-tr['pixel_y']))
        tile_results[i]['_orig_row_idx'] = ri
        tile_results[i]['_orig_col_idx'] = ci
        orig_grid[(ri, ci)] = i

    def _prep(strip, gamma=0.5):
        s = strip.astype(np.float64)
        if s.max() > 0: s = (s/s.max())**gamma * s.max()
        return s

    def _mad_wmean(vals, wts, thr):
        vals = np.array(vals, dtype=float)
        wts  = np.array(wts,  dtype=float)
        med  = np.median(vals)
        mad  = max(np.median(np.abs(vals - med)), 0.5)
        keep = np.abs(vals - med) <= thr * mad
        if not keep.any():
            return float(med), 0, len(vals)
        return (float(np.average(vals[keep], weights=wts[keep])),
                int((~keep).sum()), len(vals))

    #  Pass A: column Y drift ─
    if verbose:
        print('  Pass A: column Y drift')
    col_shift = {}
    for ci in range(N_cols-1):
        dys, confs = [], []
        for ri in range(N_rows):
            ti = orig_grid.get((ri, ci))
            tj = orig_grid.get((ri, ci+1))
            if ti is None or tj is None: continue
            Ii = np.asarray(tile_results[ti]['pixel_maps']['intensity'], dtype=float)
            Ij = np.asarray(tile_results[tj]['pixel_maps']['intensity'], dtype=float)
            sa = _prep(Ii[:, col_pitch:col_pitch+col_overlap])
            sb = _prep(Ij[:, :col_overlap])
            mr = min(sa.shape[0], sb.shape[0])
            if mr<20 or sa[:mr].max()<0.5 or sb[:mr].max()<0.5: continue
            dy, dx, conf = _phase_corr_2d(sa[:mr], sb[:mr],
                                           max_shift_y=REG_MAX_SHIFT_Y,
                                           max_shift_x=max(4, col_overlap//4))
            if conf >= MIN_CONF:
                dys.append(dy); confs.append(conf)
        if not dys:
            col_shift[ci] = 0.0
            continue
        s, _, _ = _mad_wmean(dys, confs, MAD_THRESHOLD)
        col_shift[ci] = s
        if verbose:
            print(f'    col {orig_col_xs[ci]:5d}→{orig_col_xs[ci+1]:5d}: {s:+.2f}px')

    cum_col_y = np.zeros(N_cols)
    for ci in range(1, N_cols):
        cum_col_y[ci] = cum_col_y[ci-1] + col_shift.get(ci-1, 0.0)
    if verbose:
        print(f'    Cumulative: {[round(v,1) for v in cum_col_y]}')

    for i, tr in enumerate(tile_results):
        ci   = tr['_orig_col_idx']
        corr = int(round(float(cum_col_y[ci])))
        if corr:
            tile_results[i]['pixel_y'] = max(0, tr['pixel_y'] + corr)

    #  Pass B: row Y residual ─
    if verbose:
        print('  Pass B: row Y residual')
    row_shift_y = {}
    for ri in range(N_rows-1):
        dys, confs = [], []
        for ci in range(N_cols):
            ti = orig_grid.get((ri,   ci))
            tj = orig_grid.get((ri+1, ci))
            if ti is None or tj is None: continue
            Ii = np.asarray(tile_results[ti]['pixel_maps']['intensity'], dtype=float)
            Ij = np.asarray(tile_results[tj]['pixel_maps']['intensity'], dtype=float)
            sa = _prep(Ii[row_pitch:row_pitch+row_overlap, :])
            sb = _prep(Ij[:row_overlap, :])
            mr = min(sa.shape[0], sb.shape[0])
            mc = min(sa.shape[1], sb.shape[1])
            if mr<20 or sa[:mr,:mc].max()<0.5 or sb[:mr,:mc].max()<0.5: continue
            dy, dx, conf = _phase_corr_2d(sa[:mr,:mc], sb[:mr,:mc],
                                           max_shift_y=max(4, row_overlap//4),
                                           max_shift_x=REG_MAX_SHIFT_X)
            tf = min((sa[:mr,:mc]>1).mean(), (sb[:mr,:mc]>1).mean())
            if conf >= MIN_CONF and tf >= MIN_TISSUE_FRAC:
                dys.append(dy); confs.append(conf)
        if not dys:
            row_shift_y[ri] = 0.0
            continue
        s, _, _ = _mad_wmean(dys, confs, MAD_THRESHOLD)
        row_shift_y[ri] = s
        if verbose:
            print(f'    row {ri} ({orig_row_ys[ri]}→{orig_row_ys[ri+1]}): {s:+.2f}px')

    cum_row_y = np.zeros(N_rows)
    for ri in range(1, N_rows):
        cum_row_y[ri] = cum_row_y[ri-1] + row_shift_y.get(ri-1, 0.0)
    if verbose:
        print(f'    Cumulative: {[round(v,1) for v in cum_row_y]}')

    for i, tr in enumerate(tile_results):
        ri   = tr['_orig_row_idx']
        corr = int(round(float(cum_row_y[ri])))
        if corr:
            tile_results[i]['pixel_y'] = max(0, tr['pixel_y'] + corr)

    #  Pass C: row X residual ─
    if verbose:
        print('  Pass C: row X residual')
    row_shift_x = {}
    for ri in range(N_rows-1):
        dxs, confs = [], []
        for ci in range(N_cols):
            ti = orig_grid.get((ri,   ci))
            tj = orig_grid.get((ri+1, ci))
            if ti is None or tj is None: continue
            Ii = np.asarray(tile_results[ti]['pixel_maps']['intensity'], dtype=float)
            Ij = np.asarray(tile_results[tj]['pixel_maps']['intensity'], dtype=float)
            sa = _prep(Ii[row_pitch:row_pitch+row_overlap, :])
            sb = _prep(Ij[:row_overlap, :])
            mr = min(sa.shape[0], sb.shape[0])
            mc = min(sa.shape[1], sb.shape[1])
            if mr<20 or sa[:mr,:mc].max()<0.5 or sb[:mr,:mc].max()<0.5: continue
            dy, dx, conf = _phase_corr_2d(sa[:mr,:mc], sb[:mr,:mc],
                                           max_shift_y=max(4, row_overlap//4),
                                           max_shift_x=REG_MAX_SHIFT_X)
            tf = min((sa[:mr,:mc]>1).mean(), (sb[:mr,:mc]>1).mean())
            if conf >= MIN_CONF and tf >= MIN_TISSUE_FRAC:
                dxs.append(dx); confs.append(conf)
        if not dxs:
            row_shift_x[ri] = 0.0
            continue
        s, _, _ = _mad_wmean(dxs, confs, MAD_THRESHOLD)
        row_shift_x[ri] = s
        if verbose:
            print(f'    row {ri} (y={orig_row_ys[ri]}): dx={s:+.2f}px')

    cum_row_x = np.zeros(N_rows)
    for ri in range(1, N_rows):
        cum_row_x[ri] = cum_row_x[ri-1] + row_shift_x.get(ri-1, 0.0)
    if verbose:
        print(f'    X cumulative: {[round(v,1) for v in cum_row_x]}')

    for i, tr in enumerate(tile_results):
        ri   = tr['_orig_row_idx']
        corr = int(round(float(cum_row_x[ri])))
        if corr:
            tile_results[i]['pixel_x'] = max(0, tr['pixel_x'] + corr)

    n_corrected = sum(
        1 for tr in tile_results
        if tr.get('_orig_row_idx',0)>0 or tr.get('_orig_col_idx',0)>0
    )
    if verbose:
        canvas_h = max(tr['pixel_y']+tr['tile_h'] for tr in tile_results)
        canvas_w = max(tr['pixel_x']+tr['tile_w'] for tr in tile_results)
        print(f'  Registration complete. Canvas: {canvas_h}×{canvas_w}px')

    return tile_results


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
) -> tuple[list[dict[str, Any]], int, int, list[dict], np.ndarray, np.ndarray, float, np.ndarray, dict]:
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

    #  Resolve parameters from args 
    n_exp_      = getattr(args, 'nexp',          _cfg_nexp)
    tau_min_ns  = getattr(args, 'tau_min',       Tau_min)
    tau_max_ns  = getattr(args, 'tau_max',       Tau_max)
    cost_fn     = getattr(args, 'cost_function', Cost_function)
    optimizer   = getattr(args, 'optimizer',     Optimizer)
    restarts    = getattr(args, 'restarts',      lm_restarts)
    workers     = getattr(args, 'workers',       n_workers)
    binning     = getattr(args, 'binning',       binning_factor)
    min_photons       = getattr(args, 'min_photons',         MIN_PHOTONS_PERPIX)
    intensity_thr     = getattr(args, 'intensity_threshold', None)  # photons/px; None = no background masking
    register_tiles    = getattr(args, 'register_tiles',      True)   # phase-corr Y registration
    reg_max_shift_px  = getattr(args, 'reg_max_shift_px',    120)    # max search range (px)
    fit_bg      = MACHINE_IRF_FIT_BG
    fit_sigma   = MACHINE_IRF_FIT_SIGMA
    has_tail    = MACHINE_IRF_FIT_TAIL
    mach_path   = getattr(args, 'machine_irf',  str(MACHINE_IRF_DEFAULT_PATH))

    machine_irf, pi_machine = _load_machine_irf(mach_path)

    #  Parse tile positions ─
    tile_positions = parse_xlif_tile_positions(xlif_path, ptu_basename)
    pixel_size_m, _ = get_pixel_size_from_xlif(xlif_path)
    # When binning > 1, each output pixel represents binning × binning raw pixels.
    # Scale canvas positions by effective pixel size so the canvas and
    # pixel maps have matching dimensions.
    effective_pixel_size_m = pixel_size_m * binning
    tile_positions, canvas_w, canvas_h = compute_tile_pixel_positions(
        tile_positions, effective_pixel_size_m,
        _peek_tile_width(ptu_dir, tile_positions, rotate_tiles) // binning)

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

    
    # PASS 1 — pool summed decays, fit consensus τ
    
    if verbose:
        print("Pass 1: accumulating pooled decay (summed_decay only)...")

    tile_meta    = []
    pooled_decay = None
    n_bins_ref   = None
    tcspc_ref    = None

    for i, t in enumerate(tqdm(tile_positions,
                                desc='  Pass 1', disable=True)):
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

        # Background mask on pooled decay — zero out sub-threshold pixels so
        # glass/empty regions don't bias the consensus τ from fit_summed.
        if intensity_thr is not None:
            stack_p1 = ptu.raw_pixel_stack(channel=ptu.photon_channel)
            px_int   = stack_p1.sum(axis=-1)            # (Y, X) photon count
            mask_p1  = px_int >= intensity_thr
            stack_p1[~mask_p1] = 0
            decay = stack_p1.sum(axis=(0, 1))
            del stack_p1, px_int, mask_p1

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

    
    # PASS 2 — per-pixel fit, one tile at a time
    

    tile_results  = []
    tiles_skipped = 0

    for i, tc in enumerate(tqdm(tile_meta,
                                 desc='  Pass 2', disable=True, leave=False)):
        # Print info header on first iteration
        if i == 0 and verbose:
            tqdm.write(f"Pass 2: per-pixel fit ({len(tile_meta)} tiles)...")
            tqdm.write(f"  Fixed τ   = {[f'{t:.3f}' for t in consensus_taus_ns]} ns")
            tqdm.write(f"  Fixed IRF = pooled_irf (peak bin {pooled_peak})\n")
        
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

            # Apply intensity threshold: zero out background pixels.
            # This is separate from min_photons — min_photons is the NNLS
            # quality gate inside fit_per_pixel; intensity_threshold is an
            # explicit background mask applied before fitting.
            if intensity_thr is not None:
                px_int = stack.sum(axis=-1)
                stack[px_int < intensity_thr] = 0
                del px_int

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

    #  Optional Y registration using computed intensity maps 
    # Uses already-computed pixel_maps['intensity'] — no extra PTU reads.
    # Corrects pixel_y in tile_results before assembly.
    if register_tiles and len(tile_results) > 1:
        tile_results = _register_tile_columns(
            tile_results,
            max_shift_px=reg_max_shift_px,
            verbose=verbose,
        )
        # Recompute canvas size after position corrections
        canvas_h = max(tr['pixel_y'] + tr['tile_h'] for tr in tile_results)
        canvas_w = max(tr['pixel_x'] + tr['tile_w'] for tr in tile_results)
        if verbose:
            print(f'  Canvas after registration: {canvas_h}×{canvas_w} px')

    # Build corrected_positions keyed by PTU filename so that any tiles
    # skipped in Pass 2 (tile_results shorter than tile_meta) don't corrupt
    # the file→position mapping via a silent zip truncation.
    _pos_by_file = {tr['ptu_name']: tr for tr in tile_results}
    corrected_positions = [
        {
            **tc['t'],
            'pixel_y': _pos_by_file[tc['t']['file']]['pixel_y']
                       if tc['t']['file'] in _pos_by_file
                       else tc['t']['pixel_y'],
            'pixel_x': _pos_by_file[tc['t']['file']]['pixel_x']
                       if tc['t']['file'] in _pos_by_file
                       else tc['t']['pixel_x'],
        }
        for tc in tile_meta
    ]
    return tile_results, canvas_h, canvas_w, corrected_positions, pooled_decay, pooled_irf, tcspc_ref, global_popt, global_summary