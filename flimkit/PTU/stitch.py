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
    
    for i, t in enumerate(tqdm(tile_positions, desc='  Loading tiles')):
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
            
            # Accumulate (overlaps get summed)
            flim_canvas[y0:y1, x0:x1, :] += hist.astype(np.uint32)
            intensity_canvas[y0:y1, x0:x1] += hist.sum(axis=2).astype(np.float64)
            weight_canvas[y0:y1, x0:x1] += 1.0
            
            tiles_processed += 1
            
        except Exception as e:
            if verbose:
                print(f"  [{i+1:3d}/{len(tile_positions)}] ERROR: {t['file']}: {e}")
            tiles_skipped += 1
            continue
    
    # Normalize overlaps — average FLIM counts so overlap regions
    # have the same per-pixel count level as non-overlap regions
    if verbose:
        print()
        print("Normalizing overlaps...")
    
    mask = weight_canvas > 0
    intensity_canvas[mask] /= weight_canvas[mask]
    
    # Average FLIM histogram cube in overlap regions
    # Vectorised: chunk by spatial pixels (all time bins at once) instead of
    # looping over thousands of time bins, cutting I/O by ~10×.
    overlap_mask = weight_canvas > 1
    n_overlap = int(overlap_mask.sum())
    if n_overlap > 0:
        if verbose:
            print(f"  Averaging {n_overlap:,} overlap pixels "
                  f"({100*n_overlap/mask.sum():.1f}% of canvas)")
        overlap_ys, overlap_xs = np.where(overlap_mask)
        weights = weight_canvas[overlap_ys, overlap_xs].astype(np.float64)
        CHUNK = 2000                       # overlap pixels per batch (~48 MB at 3k bins)
        for i in tqdm(range(0, n_overlap, CHUNK),
                      desc='  Normalising overlaps',
                      total=(n_overlap + CHUNK - 1) // CHUNK):
            sl = slice(i, min(i + CHUNK, n_overlap))
            ys, xs, w = overlap_ys[sl], overlap_xs[sl], weights[sl, np.newaxis]
            data = flim_canvas[ys, xs, :].astype(np.float64)   # (chunk, T)
            data /= w
            flim_canvas[ys, xs, :] = np.round(data).astype(np.uint32)
    
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
    
    if load_to_memory:
        # Convert memmap to full array in RAM
        stack = np.array(flim_memmap, dtype=np.float32)
    else:
        # Keep as memmap (memory efficient but slower)
        stack = flim_memmap.astype(np.float32)
    
    tcspc_res = metadata['tcspc_resolution_ps'] * 1e-12  # ps → s
    n_bins = metadata['n_time_bins']
    
    return stack, tcspc_res, n_bins
