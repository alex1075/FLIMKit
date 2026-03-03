
"""stitch.py - FLIM tile stitching module.

Provides functions for stitching PTU tiles using XLIF metadata
to produce intensity images and FLIM histogram cubes for fitting.
"""

import numpy as np
import tifffile
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from ..utils.misc import setup_loggers
from ..utils.xml_utils import (parse_xlif_tile_positions,
    get_pixel_size_from_xlif,
    compute_tile_pixel_positions,
)
from ..PTU.decode import get_flim_histogram, create_time_axis


def stitch_flim_tiles(
    xlif_path: Path,
    ptu_dir: Path,
    output_dir: Path,
    ptu_basename: str = "R 2",
    rotate_tiles: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Stitch FLIM PTU tiles into a single mosaic using XLIF metadata.
    
    Produces:
    - Intensity image (TIFF)
    - FLIM histogram cube (NPY memmap) for fitting
    - Time axis (NPY) in nanoseconds
    - Weight map (NPY) showing tile overlap counts
    
    Args:
        xlif_path: Path to XLIF metadata file
        ptu_dir: Directory containing PTU tile files
        output_dir: Directory for output files
        ptu_basename: Base name for PTU files (e.g., "R 2" -> "R 2_s1.ptu")
        rotate_tiles: If True, rotate each tile 90° clockwise
        verbose: Print progress messages
    
    Returns:
        Dict with output file paths and metadata
    """
    loggers = setup_loggers(str(output_dir))
    run_logger = loggers['run']
    
    xlif_path = Path(xlif_path)
    ptu_dir = Path(ptu_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Output paths
    output_intensity = output_dir / "stitched_intensity.tif"
    output_flim = output_dir / "stitched_flim_counts.npy"
    output_time = output_dir / "time_axis_ns.npy"
    output_weight = output_dir / "weight_map.npy"
    
    # --- PARSE XML FOR TILE POSITIONS ---
    if verbose:
        print(f"Parsing XLIF: {xlif_path}")
    tile_positions = parse_xlif_tile_positions(xlif_path, ptu_basename)
    if verbose:
        print(f"Found {len(tile_positions)} tiles in XML.")
    run_logger.info(f"Found {len(tile_positions)} tiles in {xlif_path}")
    
    # --- GET PIXEL SIZE ---
    pixel_size_m, n_pixels = get_pixel_size_from_xlif(xlif_path)
    if verbose:
        print(f"Pixel size: {pixel_size_m * 1e6:.4f} µm ({pixel_size_m * 1e9:.2f} nm)")
    
    # --- LOAD FIRST TILE TO GET SHAPE AND TIME RESOLUTION ---
    first_tile_path = ptu_dir / tile_positions[0]["file"]
    if verbose:
        print(f"Loading first tile: {first_tile_path}")
    
    first_hist, first_meta = get_flim_histogram(first_tile_path, rotate_cw=rotate_tiles)
    tile_y, tile_x = first_meta['tile_shape']
    n_time_bins = first_meta['n_time_bins']
    tcspc_resolution = first_meta['tcspc_resolution']
    
    if verbose:
        print(f"Tile shape (Y, X, H): {tile_y}, {tile_x}, {n_time_bins}")
        print(f"TCSPC resolution: {tcspc_resolution * 1e12:.2f} ps/bin")
    
    # Create time axis
    time_axis_ns = create_time_axis(n_time_bins, tcspc_resolution)
    if verbose:
        print(f"Time range: 0 - {time_axis_ns[-1]:.2f} ns ({n_time_bins} bins)")
    
    # --- COMPUTE CANVAS SIZE ---
    tile_positions, canvas_width, canvas_height = compute_tile_pixel_positions(
        tile_positions, pixel_size_m, tile_x
    )
    if verbose:
        print(f"Stitched canvas size: {canvas_height} x {canvas_width} pixels")
    
    # --- ALLOCATE ARRAYS ---
    if verbose:
        print("Allocating arrays...")
    
    intensity_canvas = np.zeros((canvas_height, canvas_width), dtype=np.float64)
    flim_canvas = np.memmap(
        str(output_flim),
        dtype=np.uint32,
        mode='w+',
        shape=(canvas_height, canvas_width, n_time_bins)
    )
    weight_canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    
    # --- STITCH TILES ---
    if verbose:
        print("Stitching tiles...")
    
    tiles_processed = 0
    tiles_skipped = 0
    
    for i, t in enumerate(tile_positions):
        tile_path = ptu_dir / t["file"]
        if not tile_path.exists():
            if verbose:
                print(f"  [{i+1}/{len(tile_positions)}] MISSING: {t['file']}")
            tiles_skipped += 1
            continue
        
        try:
            hist, meta = get_flim_histogram(tile_path, rotate_cw=rotate_tiles)
            
            # Handle time bin mismatch (pad/crop)
            if hist.shape[2] != n_time_bins:
                if verbose:
                    print(f"  [{i+1}/{len(tile_positions)}] {t['file']}: time bins {hist.shape[2]} != {n_time_bins}, padding/cropping")
                if hist.shape[2] < n_time_bins:
                    padded = np.zeros((hist.shape[0], hist.shape[1], n_time_bins), dtype=hist.dtype)
                    padded[:, :, :hist.shape[2]] = hist
                    hist = padded
                else:
                    hist = hist[:, :, :n_time_bins]
            
            # Compute placement
            y0 = t["pixel_y"]
            x0 = t["pixel_x"]
            y1 = y0 + tile_y
            x1 = x0 + tile_x
            
            # Add to canvases
            flim_canvas[y0:y1, x0:x1, :] += hist.astype(np.uint32)
            intensity_canvas[y0:y1, x0:x1] += hist.sum(axis=2).astype(np.float64)
            weight_canvas[y0:y1, x0:x1] += 1.0
            
            if verbose:
                print(f"  [{i+1}/{len(tile_positions)}] {t['file']} -> ({y0}:{y1}, {x0}:{x1})")
            
            tiles_processed += 1
            
        except Exception as e:
            if verbose:
                print(f"  [{i+1}/{len(tile_positions)}] ERROR {t['file']}: {e}")
            run_logger.error(f"Error processing {t['file']}: {e}")
            tiles_skipped += 1
            continue
    
    # --- NORMALIZE OVERLAPS ---
    if verbose:
        print("Normalizing overlaps...")
    mask = weight_canvas > 0
    intensity_canvas[mask] /= weight_canvas[mask]
    
    # --- SAVE OUTPUTS ---
    if verbose:
        print("Saving outputs...")
    
    # Intensity image
    tifffile.imwrite(str(output_intensity), intensity_canvas.astype(np.float32))
    if verbose:
        print(f"  Intensity image: {output_intensity}")
    
    # Time axis
    np.save(str(output_time), time_axis_ns)
    if verbose:
        print(f"  Time axis: {output_time}")
    
    # Weight map
    np.save(str(output_weight), weight_canvas)
    
    # FLIM counts (flush memmap)
    flim_canvas.flush()
    if verbose:
        print(f"  FLIM counts: {output_flim}")
        print(f"    Shape: {flim_canvas.shape}")
        print(f"    Dtype: {flim_canvas.dtype}")
    
    run_logger.info(f"Stitching complete: {tiles_processed} tiles, {tiles_skipped} skipped")
    
    # --- SUMMARY ---
    result = {
        'intensity_path': output_intensity,
        'flim_path': output_flim,
        'time_axis_path': output_time,
        'weight_map_path': output_weight,
        'canvas_shape': (canvas_height, canvas_width),
        'n_time_bins': n_time_bins,
        'time_range_ns': (0, time_axis_ns[-1]),
        'tcspc_resolution_ps': tcspc_resolution * 1e12,
        'pixel_size_um': pixel_size_m * 1e6,
        'tiles_processed': tiles_processed,
        'tiles_skipped': tiles_skipped,
    }
    
    if verbose:
        print(f"\n=== SUMMARY ===")
        print(f"Tiles processed: {tiles_processed}")
        print(f"Tiles skipped: {tiles_skipped}")
        print(f"Canvas size: {canvas_height} x {canvas_width} pixels")
        print(f"Time bins: {n_time_bins}")
        print(f"Time range: 0 - {time_axis_ns[-1]:.2f} ns")
        print(f"TCSPC resolution: {tcspc_resolution * 1e12:.2f} ps/bin")
        print(f"\nOutputs:")
        print(f"  {output_intensity}")
        print(f"  {output_flim}")
        print(f"  {output_time}")
        print(f"  {output_weight}")
        print("\nDone!")
    
    return result


def load_stitched_flim(
    output_dir: Path,
    canvas_shape: Optional[Tuple[int, int]] = None,
    n_time_bins: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load previously stitched FLIM data for fitting.
    
    Args:
        output_dir: Directory containing output files
        canvas_shape: (height, width) of canvas (required if not inferrable)
        n_time_bins: Number of time bins (required if not inferrable)
    
    Returns:
        Tuple of (flim_cube, time_axis_ns, intensity_image)
    """
    output_dir = Path(output_dir)
    
    # Load time axis
    time_axis = np.load(output_dir / "time_axis_ns.npy")
    if n_time_bins is None:
        n_time_bins = len(time_axis)
    
    # Load intensity to get shape
    intensity = tifffile.imread(output_dir / "stitched_intensity.tif")
    if canvas_shape is None:
        canvas_shape = intensity.shape
    
    # Load FLIM cube via memmap (read-only)
    flim = np.memmap(
        str(output_dir / "stitched_flim_counts.npy"),
        dtype=np.uint32,
        mode='r',
        shape=(canvas_shape[0], canvas_shape[1], n_time_bins)
    )
    
    return flim, time_axis, intensity
