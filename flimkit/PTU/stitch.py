import json
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, Any, Optional
from ..utils.xml_utils import (
    parse_xlif_tile_positions,
    get_pixel_size_from_xlif,
    compute_tile_pixel_positions,
)
from .decode import get_flim_histogram_from_ptufile, create_time_axis


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
    xlif_path = Path(xlif_path)
    ptu_dir = Path(ptu_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Derive ROI prefix from ptu_basename (e.g. "R 2" -> "R_2")
    roi_prefix = ptu_basename.replace(' ', '_')
    
    # Output paths — prefixed with ROI name for clarity
    output_intensity = output_dir / f"{roi_prefix}_stitched_intensity.tif"
    output_flim = output_dir / f"{roi_prefix}_stitched_flim_counts.npy"
    output_time = output_dir / f"{roi_prefix}_time_axis_ns.npy"
    output_weight = output_dir / f"{roi_prefix}_weight_map.npy"
    output_meta = output_dir / f"{roi_prefix}_metadata.json"
    
    if verbose:
        print(f"{'='*60}")
        print(f"FLIM TILE STITCHING")
        print(f"{'='*60}")
        print(f"XLIF: {xlif_path}")
        print(f"PTUs: {ptu_dir}")
        print(f"Output: {output_dir}")
        print()
    
    # Parse XLIF for tile positions
    if verbose:
        print("Parsing XLIF metadata...")
    
    tile_positions = parse_xlif_tile_positions(xlif_path, ptu_basename)
    pixel_size_m, n_pixels = get_pixel_size_from_xlif(xlif_path)
    
    if verbose:
        print(f"  Found {len(tile_positions)} tiles")
        print(f"  Pixel size: {pixel_size_m * 1e6:.4f} µm")
    
    # Load first tile to get dimensions
    first_tile_path = ptu_dir / tile_positions[0]["file"]
    
    if not first_tile_path.exists():
        raise FileNotFoundError(f"First tile not found: {first_tile_path}")
    
    if verbose:
        print(f"  Loading first tile: {first_tile_path.name}")
    
    first_hist, first_meta = get_flim_histogram_from_ptufile(
        first_tile_path,
        rotate_cw=rotate_tiles,
        binning=1,
        channel=None
    )
    
    tile_y, tile_x = first_meta['tile_shape']
    n_time_bins = first_meta['n_time_bins']
    tcspc_resolution = first_meta['tcspc_resolution']
    
    if verbose:
        print(f"  Tile shape: ({tile_y}, {tile_x}, {n_time_bins})")
        print(f"  TCSPC: {tcspc_resolution * 1e12:.2f} ps/bin")
    
    # Create time axis
    time_axis_ns = create_time_axis(n_time_bins, tcspc_resolution)
    
    if verbose:
        print(f"  Time range: 0 - {time_axis_ns[-1]:.2f} ns")
    
    # Compute canvas size
    tile_positions, canvas_width, canvas_height = compute_tile_pixel_positions(
        tile_positions, pixel_size_m, tile_x
    )
    
    if verbose:
        print(f"  Canvas: {canvas_height} × {canvas_width} pixels")
        print()
        print("Allocating arrays...")
    
    # Allocate output arrays
    intensity_canvas = np.zeros((canvas_height, canvas_width), dtype=np.float64)
    
    # Use memmap for FLIM cube (memory efficient for large mosaics)
    flim_canvas = np.memmap(
        str(output_flim),
        dtype=np.uint32,
        mode='w+',
        shape=(canvas_height, canvas_width, n_time_bins)
    )
    
    weight_canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    
    # Stitch tiles
    if verbose:
        print(f"Stitching {len(tile_positions)} tiles...")
        print()
    
    tiles_processed = 0
    tiles_skipped = 0
    
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
            # Load tile using your PTUFile class
            hist, meta = get_flim_histogram_from_ptufile(
                tile_path,
                rotate_cw=rotate_tiles,
                binning=1,
                channel=None
            )
            
            # Handle time bin mismatch (pad or crop)
            if hist.shape[2] != n_time_bins:
                if hist.shape[2] < n_time_bins:
                    # Pad with zeros
                    padded = np.zeros((hist.shape[0], hist.shape[1], n_time_bins),
                                     dtype=hist.dtype)
                    padded[:, :, :hist.shape[2]] = hist
                    hist = padded
                else:
                    # Crop
                    hist = hist[:, :, :n_time_bins]
            
            # Place tile on canvas
            y0 = t["pixel_y"]
            x0 = t["pixel_x"]
            y1 = y0 + tile_y
            x1 = x0 + tile_x
            
            # Accumulate — overlaps are summed, giving more photons where
            # tiles overlap (which is correct for FLIM fitting statistics).
            flim_canvas[y0:y1, x0:x1, :] += hist.astype(np.uint32)
            intensity_canvas[y0:y1, x0:x1] += hist.sum(axis=2).astype(np.float64)
            weight_canvas[y0:y1, x0:x1] += 1.0
            
            tiles_processed += 1
            
        except Exception as e:
            if verbose:
                print(f"  [{i+1:3d}/{len(tile_positions)}] ERROR: {t['file']}: {e}")
            tiles_skipped += 1
            continue
    
    # Normalise intensity image for display only.
    # The FLIM cube is intentionally NOT normalised — photon counts in overlap
    # regions are summed across tiles, which improves fitting statistics there.
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

    # Save outputs
    if verbose:
        print("Saving outputs...")
    
    # Scale intensity to full 16-bit range for clean display
    max_val = intensity_canvas.max()
    if max_val > 0:
        intensity_scaled = (intensity_canvas / max_val * 65535).astype(np.uint16)
    else:
        intensity_scaled = np.zeros_like(intensity_canvas, dtype=np.uint16)
    tifffile.imwrite(str(output_intensity), intensity_scaled)
    np.save(str(output_time), time_axis_ns)
    np.save(str(output_weight), weight_canvas)
    
    flim_canvas.flush()  # Flush memmap to disk
    
    # Save metadata
    metadata = {
        'canvas_shape': (canvas_height, canvas_width),
        'n_time_bins': int(n_time_bins),
        'time_range_ns': (0.0, float(time_axis_ns[-1])),
        'tcspc_resolution_ps': float(tcspc_resolution * 1e12),
        'pixel_size_um': float(pixel_size_m * 1e6),
        'tiles_processed': tiles_processed,
        'tiles_skipped': tiles_skipped,
        'ptu_basename': ptu_basename,
    }
    
    with open(output_meta, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        print(f"  ✓ {output_intensity.name}")
        print(f"  ✓ {output_flim.name}")
        print(f"  ✓ {output_time.name}")
        print(f"  ✓ {output_weight.name}")
        print(f"  ✓ {output_meta.name}")
        print()
        print(f"{'='*60}")
        print(f"STITCHING COMPLETE")
        print(f"{'='*60}")
        print(f"Processed: {tiles_processed}/{len(tile_positions)} tiles")
        print(f"Canvas: {canvas_height} × {canvas_width} × {n_time_bins}")
        print(f"Time: 0 - {time_axis_ns[-1]:.2f} ns")
    
    return {
        'intensity_path': output_intensity,
        'flim_path': output_flim,
        'time_axis_path': output_time,
        'weight_map_path': output_weight,
        'metadata_path': output_meta,
        'canvas_shape': (canvas_height, canvas_width),
        'n_time_bins': n_time_bins,
        'tiles_processed': tiles_processed,
        'tiles_skipped': tiles_skipped,
    }


def load_stitched_flim(
    output_dir: Path,
    mode: str = 'r'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load previously stitched FLIM data.
    
    Args:
        output_dir: Directory with stitched outputs
        mode: 'r' (read-only) or 'r+' (read-write)
    
    Returns:
        Tuple of (flim_cube, time_axis, intensity, metadata)
    
    Example:
        >>> flim, time, intensity, meta = load_stitched_flim(Path("stitched/R_002/"))
        >>> decay = flim.sum(axis=(0,1))  # Sum for global fit
    """
    output_dir = Path(output_dir)
    
    # Find metadata file — try ROI-prefixed first, fall back to generic
    meta_candidates = sorted(output_dir.glob("*_metadata.json"))
    if meta_candidates:
        meta_path = meta_candidates[0]
        roi_prefix = meta_path.name.replace('_metadata.json', '')
    elif (output_dir / "metadata.json").exists():
        meta_path = output_dir / "metadata.json"
        roi_prefix = None
    else:
        raise FileNotFoundError(f"No metadata.json found in {output_dir}")
    
    # Load metadata
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    canvas_shape = tuple(metadata['canvas_shape'])
    n_time_bins = metadata['n_time_bins']
    
    # Resolve filenames (ROI-prefixed or generic)
    def _find(prefixed, generic):
        p = output_dir / prefixed
        return p if p.exists() else output_dir / generic
    
    if roi_prefix:
        time_path = _find(f"{roi_prefix}_time_axis_ns.npy", "time_axis_ns.npy")
        int_path  = _find(f"{roi_prefix}_stitched_intensity.tif", "stitched_intensity.tif")
        flim_path = _find(f"{roi_prefix}_stitched_flim_counts.npy", "stitched_flim_counts.npy")
    else:
        time_path = output_dir / "time_axis_ns.npy"
        int_path  = output_dir / "stitched_intensity.tif"
        flim_path = output_dir / "stitched_flim_counts.npy"
    
    # Load arrays
    time_axis = np.load(str(time_path))
    intensity = tifffile.imread(str(int_path))
    
    # Load FLIM cube as memmap
    flim = np.memmap(
        str(flim_path),
        dtype=np.uint32,
        mode=mode,
        shape=(canvas_shape[0], canvas_shape[1], n_time_bins)
    )
    
    return flim, time_axis, intensity, metadata


def load_flim_for_fitting(
    source_dir: Path,
    load_to_memory: bool = False
) -> Tuple[np.ndarray, float, int]:
    """
    Load stitched FLIM data ready for your fitting pipeline.
    
    Args:
        source_dir: Directory with stitched outputs
        load_to_memory: If True, load full array to RAM; if False, use memmap
    
    Returns:
        Tuple of (stack, tcspc_res, n_bins) ready for fit_summed/fit_per_pixel
    
    Example:
        >>> from code.FLIM.fitters import fit_summed
        >>> from code.FLIM.irf_tools import gaussian_irf_from_fwhm
        >>> 
        >>> # Load stitched data
        >>> stack, tcspc_res, n_bins = load_flim_for_fitting(Path("stitched/R_002/"))
        >>> 
        >>> # Fit summed decay
        >>> decay = stack.sum(axis=(0,1))
        >>> irf = gaussian_irf_from_fwhm(n_bins, fwhm_ns=0.3, center_bin=n_bins//10)
        >>> popt, summary = fit_summed(decay, tcspc_res, n_bins, irf, ...)
    """
    flim_memmap, time_axis, intensity, metadata = load_stitched_flim(source_dir)
    
    tcspc_res = metadata['tcspc_resolution_ps'] * 1e-12  # ps → s
    n_bins = metadata['n_time_bins']

    if load_to_memory:
        # Convert memmap to full array in RAM
        stack = np.array(flim_memmap, dtype=np.float32)
    else:
        # Return the memmap directly — do NOT call .astype() here as that
        # materialises the entire array in RAM, defeating the purpose.
        stack = flim_memmap
    
    return stack, tcspc_res, n_bins


# ── Per-tile fitting pipeline ──────────────────────────────────────────────────

def _resolve_tile_irf(ptu_name: str,
                      irf_xlsx_dir=None,
                      irf_xlsx_map=None):
    """
    Return the xlsx path to use as IRF for this tile, or None to fall back
    to rising-edge estimation.

    Args:
        ptu_name:     tile filename, e.g. "R_2_s43.ptu"
        irf_xlsx_dir: directory containing per-tile xlsx files named to match
                      the PTU stem, e.g. irf_xlsx_dir/R_2_s43.xlsx
                      (populate once the per-tile IRF xlsx format is known)
        irf_xlsx_map: explicit dict {ptu_name: Path} — takes priority over dir

    Returns:
        Path or None
    """
    stem = Path(ptu_name).stem

    # Explicit map takes priority
    if irf_xlsx_map and ptu_name in irf_xlsx_map:
        return irf_xlsx_map[ptu_name]
    if irf_xlsx_map and stem in irf_xlsx_map:
        return irf_xlsx_map[stem]

    # Directory lookup
    if irf_xlsx_dir is not None:
        candidate = Path(irf_xlsx_dir) / f"{stem}.xlsx"
        if candidate.exists():
            return candidate

    # TODO: add additional lookup strategies here once the xlsx structure
    # is known (e.g. single multi-sheet workbook, naming convention, etc.)
    return None


def fit_flim_tiles(
    xlif_path: Path,
    ptu_dir: Path,
    output_dir: Path,
    args,                       # argparse.Namespace with fitting parameters
    ptu_basename: str = "R 2",
    rotate_tiles: bool = True,
    irf_xlsx_dir=None,          # per-tile IRF xlsx directory (optional)
    irf_xlsx_map=None,          # explicit {ptu_name: xlsx_path} dict (optional)
    verbose: bool = True,
    progress_callback=None,
    cancel_event=None,
) -> list:
    """
    Fit each FLIM tile independently and return results for canvas assembly.

    Rather than stitching photon counts and fitting the ensemble (which suffers
    from inter-tile IRF mismatch), this function fits each tile with its own
    IRF — either from a matched xlsx or estimated from the tile's own rising
    edge — then returns the per-tile pixel_maps ready for assemble_tile_maps().

    No intermediate 65 GB memmap is created.  Peak RAM is ~one tile at a time
    plus the assembled 2D canvas maps.

    Args:
        xlif_path:      XLIF metadata file (tile positions)
        ptu_dir:        directory containing PTU tile files
        output_dir:     where to write assembled outputs
        args:           argparse.Namespace — same fitting args as _run_flim_fit.
                        args.ptu and args.irf_xlsx will be overwritten per tile.
        ptu_basename:   PTU filename base (e.g. "R 2")
        rotate_tiles:   apply 90° CW rotation to each tile
        irf_xlsx_dir:   optional directory of per-tile IRF xlsx files
        irf_xlsx_map:   optional explicit {ptu_name: Path} IRF lookup
        verbose:        print progress
        progress_callback: optional callable(i, total)
        cancel_event:   optional threading.Event to abort

    Returns:
        List of tile result dicts, each with:
            'pixel_maps', 'global_summary', 'pixel_y', 'pixel_x',
            'tile_h', 'tile_w', 'ptu_name'
    """
    from ..utils.xml_utils import (
        parse_xlif_tile_positions,
        get_pixel_size_from_xlif,
        compute_tile_pixel_positions,
    )
    from .decode import get_flim_histogram_from_ptufile
    import argparse, copy

    xlif_path  = Path(xlif_path)
    ptu_dir    = Path(ptu_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  PER-TILE FLIM FITTING")
        print(f"{'='*60}")
        print(f"  XLIF:   {xlif_path}")
        print(f"  PTUs:   {ptu_dir}")
        print(f"  Output: {output_dir}")

    # Parse tile positions from XLIF
    tile_positions = parse_xlif_tile_positions(xlif_path, ptu_basename)
    pixel_size_m, _ = get_pixel_size_from_xlif(xlif_path)
    tile_positions, canvas_width, canvas_height = compute_tile_pixel_positions(
        tile_positions, pixel_size_m,
        # tile pixel width — peek at first tile
        _peek_tile_width(ptu_dir, tile_positions, rotate_tiles)
    )

    if verbose:
        print(f"  Tiles:  {len(tile_positions)}")
        print(f"  Canvas: {canvas_height} × {canvas_width} px\n")

    tile_results   = []
    tiles_skipped  = 0
    total          = len(tile_positions)

    for i, t in enumerate(tqdm(tile_positions,
                                desc='  Fitting tiles',
                                disable=not verbose)):
        if cancel_event is not None and cancel_event.is_set():
            print("\n  Cancelled by user.")
            break
        if progress_callback is not None:
            progress_callback(i, total)

        ptu_path = ptu_dir / t['file']
        if not ptu_path.exists():
            if verbose:
                tqdm.write(f"  [{i+1}/{total}] MISSING: {t['file']} — skipping")
            tiles_skipped += 1
            continue

        # Build per-tile args — deep copy so we don't mutate the caller's namespace
        tile_args = copy.copy(args)
        tile_args.ptu  = str(ptu_path)
        tile_args.out  = str(output_dir / Path(t['file']).stem)
        tile_args.no_plots = True   # individual tile plots would be overwhelming
        tile_args.no_polish = True  # LM polish crashes when DE lands on a bound

        # IRF resolution — xlsx if available, otherwise rising-edge estimate
        irf_xlsx = _resolve_tile_irf(t['file'], irf_xlsx_dir, irf_xlsx_map)
        if irf_xlsx is not None:
            tile_args.irf_xlsx    = str(irf_xlsx)
            tile_args.estimate_irf = 'none'
            if verbose:
                tqdm.write(f"  [{i+1}/{total}] {t['file']}  IRF: {irf_xlsx.name}")
        else:
            tile_args.irf_xlsx    = None
            # Fall back to parametric rising-edge estimate — works well per tile
            if not hasattr(tile_args, 'estimate_irf') or tile_args.estimate_irf == 'none':
                tile_args.estimate_irf = 'parametric'
            if verbose:
                tqdm.write(f"  [{i+1}/{total}] {t['file']}  IRF: {tile_args.estimate_irf}")

        try:
            # Quick photon count check — skip empty/background-only tiles before
            # attempting IRF estimation (which fails on flat noise decays).
            from .reader import PTUFile as _PTUFile
            _ptu_check = _PTUFile(str(ptu_path), verbose=False)
            _total_photons = int(_ptu_check.summed_decay(channel=None).sum())
            _min_photons_tile = getattr(args, 'min_photons_tile',
                                        getattr(args, 'min_photons', 50) * 100)
            if _total_photons < _min_photons_tile:
                if verbose:
                    tqdm.write(f"  [{i+1}/{total}] SKIP {t['file']} "
                               f"— only {_total_photons:,} photons (empty tile?)")
                tiles_skipped += 1
                continue

            # _run_flim_fit now returns a dict — import here to avoid circular import
            from ..interactive import _run_flim_fit
            import contextlib, io
            # Suppress all per-tile output — stdout and stderr (includes tqdm bars).
            # Errors are caught by the except block below and printed via tqdm.write.
            _buf = io.StringIO()
            with contextlib.redirect_stdout(_buf),                  contextlib.redirect_stderr(_buf):
                result = _run_flim_fit(tile_args)

            # Peek tile spatial dimensions from pixel_maps or PTU
            pm = result.get('pixel_maps')
            if pm is not None and 'intensity' in pm:
                th, tw = pm['intensity'].shape[:2]
            else:
                th = tw = _peek_tile_width(ptu_dir, [t], rotate_tiles)

            tile_results.append({
                'ptu_name':     t['file'],
                'pixel_y':      t['pixel_y'],
                'pixel_x':      t['pixel_x'],
                'tile_h':       th,
                'tile_w':       tw,
                'pixel_maps':   pm,
                'global_summary': result.get('global_summary'),
                'strategy':     result.get('strategy'),
            })

        except Exception as e:
            import traceback, sys
            if verbose:
                # tqdm.write keeps the progress bar stable
                tqdm.write(f"  [{i+1}/{total}] ERROR: {t['file']}: {e}",
                           file=sys.stderr)
                tqdm.write(traceback.format_exc(), file=sys.stderr)
            tiles_skipped += 1
            continue

    if verbose:
        print(f"\n  Fitted {len(tile_results)}/{total} tiles "
              f"({tiles_skipped} skipped)")

    return tile_results, canvas_height, canvas_width


def _peek_tile_width(ptu_dir, tile_positions, rotate_tiles):
    """Load the first available tile just to get its pixel width."""
    from .decode import get_flim_histogram_from_ptufile
    for t in tile_positions:
        p = Path(ptu_dir) / t['file']
        if p.exists():
            _, meta = get_flim_histogram_from_ptufile(p, rotate_cw=rotate_tiles,
                                                       binning=1, channel=None)
            return meta['tile_shape'][1]   # width
    raise FileNotFoundError("No tile PTU files found to determine tile width")