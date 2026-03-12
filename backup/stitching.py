#!/usr/bin/env python3
"""
Single-file module to rebuild an ROI and intensity image/NPY from a Leica FLIM project.

Contains all necessary functions to:
- Discover ROIs in a Leica project folder
- Parse XLIF metadata for tile positions and pixel sizes
- Load PTU files and extract FLIM histogram cubes
- Stitch multiple PTU tiles into a mosaic using the XLIF positions
- Save the stitched intensity image as TIFF and the FLIM cube as NPY
- Optionally stitch intensity from LOF files (Leica's pre‑processed intensity)

Dependencies:
    numpy, tifffile, ptufile, liffile, xml.etree.ElementTree, pathlib, json, re
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import tifffile
from liffile import LifFile
from ptufile import PtuFile


# ============================================================================
# Utility functions (from utils.py)
# ============================================================================

def setup_loggers(log_dir: str = '.', log_prefix: str = 'run'):
    """
    Set up loggers for run, error, and warning logs.
    Logs are written to run.log, error.log, and warning.log in the specified directory.
    """
    import logging
    import os
    os.makedirs(log_dir, exist_ok=True)
    loggers = {}
    
    # Main run logger
    run_logger = logging.getLogger('run')
    run_logger.setLevel(logging.INFO)
    run_fh = logging.FileHandler(os.path.join(log_dir, f'{log_prefix}.log'))
    run_fh.setLevel(logging.INFO)
    run_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    run_fh.setFormatter(run_formatter)
    run_logger.handlers = [run_fh]
    loggers['run'] = run_logger

    # Error logger
    error_logger = logging.getLogger('error')
    error_logger.setLevel(logging.ERROR)
    error_fh = logging.FileHandler(os.path.join(log_dir, 'error.log'))
    error_fh.setLevel(logging.ERROR)
    error_fh.setFormatter(run_formatter)
    error_logger.handlers = [error_fh]
    loggers['error'] = error_logger

    # Warning logger
    warning_logger = logging.getLogger('warning')
    warning_logger.setLevel(logging.WARNING)
    warning_fh = logging.FileHandler(os.path.join(log_dir, 'warning.log'))
    warning_fh.setLevel(logging.WARNING)
    warning_fh.setFormatter(run_formatter)
    warning_logger.handlers = [warning_fh]
    loggers['warning'] = warning_logger

    return loggers


def match_xml_ptu_sets(ptu_dir: Path) -> List[Dict[str, Any]]:
    """
    Match XML metadata files in ptu_dir/Metadata/ to PTU tile files in ptu_dir/.
    Returns a list of dicts with R_number, xml_files, ptu_files, counts, and status.
    """
    metadata_dir = ptu_dir / 'Metadata'
    xml_files = list(metadata_dir.glob('*.xlif')) + list(metadata_dir.glob('*.xlof')) + list(metadata_dir.glob('*.xml'))
    ptu_files = list(ptu_dir.glob('*.ptu'))
    # Extract R numbers from filenames
    r_pattern = re.compile(r'R\s*\d+')
    xml_r_map = {}
    for xml in xml_files:
        m = r_pattern.search(xml.name)
        if m:
            r = m.group().replace(' ', '')
            xml_r_map.setdefault(r, []).append(str(xml))
    ptu_r_map = {}
    for ptu in ptu_files:
        m = r_pattern.search(ptu.name)
        if m:
            r = m.group().replace(' ', '')
            ptu_r_map.setdefault(r, []).append(str(ptu))
    # Match sets
    results = []
    for r in sorted(set(xml_r_map) | set(ptu_r_map)):
        xmls = xml_r_map.get(r, [])
        ptus = ptu_r_map.get(r, [])
        status = 'MATCHED' if xmls and ptus else 'MISSING_XML' if ptus else 'MISSING_PTU'
        results.append({
            'R_number': r,
            'xml_files': xmls,
            'ptu_files': ptus,
            'xml_count': len(xmls),
            'ptu_count': len(ptus),
            'status': status
        })
    return results


def parse_xlif_tile_positions(xlif_path: Path, ptu_basename: str = "R 2") -> List[Dict[str, Any]]:
    """
    Parse tile positions from a Leica XLIF file.
    
    Args:
        xlif_path: Path to the XLIF metadata file
        ptu_basename: Base name for PTU files (e.g., "R 2" -> "R 2_s1.ptu")
    
    Returns:
        List of dicts with keys: file, field_x, pos_x, pos_y (positions in meters)
    """
    tree = ET.parse(xlif_path)
    root = tree.getroot()
    
    tile_scan_info = root.find(".//Attachment[@Name='TileScanInfo']")
    if tile_scan_info is None:
        raise RuntimeError(f"No TileScanInfo found in {xlif_path}")
    
    tile_positions = []
    for tile_elem in tile_scan_info.findall("Tile"):
        field_x = int(tile_elem.attrib.get("FieldX", 0))
        pos_x = float(tile_elem.attrib.get("PosX", 0))
        pos_y = float(tile_elem.attrib.get("PosY", 0))
        filename = f"{ptu_basename}_s{field_x + 1}.ptu"
        tile_positions.append({
            "file": filename,
            "field_x": field_x,
            "pos_x": pos_x,
            "pos_y": pos_y,
        })
    
    return tile_positions


def get_pixel_size_from_xlif(xlif_path: Path) -> Tuple[float, int]:
    """
    Extract pixel size from XLIF DimensionDescription.
    
    Args:
        xlif_path: Path to the XLIF metadata file
    
    Returns:
        Tuple of (pixel_size_meters, n_pixels)
    """
    tree = ET.parse(xlif_path)
    root = tree.getroot()
    
    dim_desc = root.find(".//DimensionDescription[@DimID='1']")
    if dim_desc is not None:
        n_pixels = int(dim_desc.attrib.get("NumberOfElements", 512))
        length_m = float(dim_desc.attrib.get("Length", 1.5377e-4))
        pixel_size_m = length_m / n_pixels
        return pixel_size_m, n_pixels
    
    # Fallback defaults
    return 1.5377e-4 / 512, 512


def compute_tile_pixel_positions(
    tile_positions: List[Dict[str, Any]],
    pixel_size_m: float,
    tile_size: int
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Convert physical tile positions (meters) to pixel coordinates.
    
    Args:
        tile_positions: List of tile dicts with pos_x, pos_y in meters
        pixel_size_m: Pixel size in meters
        tile_size: Size of each tile in pixels (assumes square tiles)
    
    Returns:
        Tuple of (updated tile_positions with pixel_x/pixel_y, canvas_width, canvas_height)
    """
    pos_x_list = [t["pos_x"] for t in tile_positions]
    pos_y_list = [t["pos_y"] for t in tile_positions]
    
    min_pos_x = min(pos_x_list)
    min_pos_y = min(pos_y_list)
    
    for t in tile_positions:
        t["pixel_x"] = int(round((t["pos_x"] - min_pos_x) / pixel_size_m))
        t["pixel_y"] = int(round((t["pos_y"] - min_pos_y) / pixel_size_m))
    
    canvas_width = max(t["pixel_x"] for t in tile_positions) + tile_size
    canvas_height = max(t["pixel_y"] for t in tile_positions) + tile_size
    
    return tile_positions, canvas_width, canvas_height


# ============================================================================
# Decode functions (from decode.py)
# ============================================================================

def get_flim_histogram(ptu_path: Path, rotate_cw: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load FLIM histogram data from a PTU file with proper shape handling.
    
    Args:
        ptu_path: Path to PTU file
        rotate_cw: If True, rotate tile 90° clockwise (for Leica data)
    
    Returns:
        hist: numpy array with shape (Y, X, H)
        metadata: dict with tcspc_resolution, n_time_bins, tile_shape, etc.
    """
    ptu = PtuFile(str(ptu_path))
    data = ptu[:]  # Full data, may be (T, Y, X, C, H) or similar
    
    # Get dimension names
    dims = ptu.dims if hasattr(ptu, 'dims') else None
    
    # Sum over time frames (T) and channels (C), keep Y, X, H
    # Typical dims: ('T', 'Y', 'X', 'C', 'H') or ('Y', 'X', 'H')
    if dims is not None and len(data.shape) > 3:
        # Find which axes are T, C (to sum) vs Y, X, H (to keep)
        sum_axes = []
        for i, d in enumerate(dims):
            if d in ('T', 'C'):  # Sum over time frames and channels
                sum_axes.append(i)
        
        if sum_axes:
            data = data.sum(axis=tuple(sum_axes))
    
    # Squeeze any remaining singleton dimensions
    data = data.squeeze()
    
    if data.ndim != 3:
        raise ValueError(f"Unexpected data shape after processing: {data.shape}, expected (Y, X, H)")
    
    hist = data  # Should be (Y, X, H)
    
    # Rotate 90 degrees clockwise for Leica data
    if rotate_cw:
        hist = np.rot90(hist, k=-1, axes=(0, 1))
    
    metadata = {
        'tcspc_resolution': ptu.tcspc_resolution,
        'n_time_bins': hist.shape[2],
        'tile_shape': (hist.shape[0], hist.shape[1]),
        'frequency': ptu.frequency,
        'dims': ptu.dims,
        'shape': ptu.shape,
    }
    
    ptu.close()
    return hist, metadata


def create_time_axis(n_bins: int, tcspc_resolution: float) -> np.ndarray:
    """
    Create time axis in nanoseconds for FLIM fitting.
    
    Args:
        n_bins: Number of time bins
        tcspc_resolution: Time per bin in seconds
    
    Returns:
        time_axis_ns: Array of time values in nanoseconds
    """
    return np.arange(n_bins) * tcspc_resolution * 1e9


def get_intensity_image(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Get a 2D intensity image from a PTU file.
    Args:
        path: Path to PTU file
    Returns:
        img: 2D numpy array (Y, X) with photon counts per pixel
        metadata: dict with file metadata
    """
    data, metadata = get_flim_histogram(path, rotate_cw=True)
    intensity = data.sum(axis=2)  # sum over time bins
    return intensity, metadata


# ============================================================================
# Stitch functions (from stitch.py)
# ============================================================================

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


# ============================================================================
# Project functions (from project.py)
# ============================================================================

def discover_rois(project_path: Path) -> List[Dict[str, Any]]:
    """
    Discover all ROIs in a Leica project.
    
    Scans PTUs.sptw for PTU files and matches them to LOF intensity
    files and XLIF metadata in TileScan folders.
    
    Args:
        project_path: Path to Leica project root
    
    Returns:
        List of dicts with:
            - roi_num: int (e.g., 2 for R 2)
            - roi_name: str (e.g., "R 2")
            - tilescan: str (e.g., "TileScan 3")
            - lof_path: Path to intensity LOF file
            - xlif_path: Path to XLIF metadata
            - ptu_dir: Path to PTUs.sptw folder
            - n_tiles: int
            - is_single_fov: bool
    """
    rois = []
    ptu_dir = project_path / "PTUs.sptw"
    
    if not ptu_dir.exists():
        print(f"Warning: PTUs.sptw folder not found in {project_path}")
        return rois
    
    # Find all unique ROIs from PTU files
    ptu_files = list(ptu_dir.glob("R *_s*.ptu"))
    roi_pattern = re.compile(r"R (\d+)_s(\d+)\.ptu")
    
    roi_tiles: Dict[int, List[int]] = {}
    for ptu_file in ptu_files:
        match = roi_pattern.match(ptu_file.name)
        if match:
            roi_num = int(match.group(1))
            tile_num = int(match.group(2))
            if roi_num not in roi_tiles:
                roi_tiles[roi_num] = []
            roi_tiles[roi_num].append(tile_num)
    
    # Also check for single-FOV PTU files (R N.ptu without _sY)
    single_ptu_files = list(ptu_dir.glob("R *.ptu"))
    single_pattern = re.compile(r"R (\d+)\.ptu$")
    for ptu_file in single_ptu_files:
        match = single_pattern.match(ptu_file.name)
        if match:
            roi_num = int(match.group(1))
            if roi_num not in roi_tiles:
                roi_tiles[roi_num] = [1]  # Single tile
    
    # Find corresponding LOF and XLIF files
    for roi_num in sorted(roi_tiles.keys()):
        roi_name = f"R {roi_num}"
        n_tiles = len(roi_tiles[roi_num])
        
        # Search for LOF file in TileScan folders
        lof_path = None
        xlif_path = None
        tilescan = None
        
        for ts_folder in sorted(project_path.glob("TileScan *")):
            candidate_lof = ts_folder / f"{roi_name}.lof"
            if candidate_lof.exists():
                lof_path = candidate_lof
                tilescan = ts_folder.name
                
                # Look for XLIF in PTUs.sptw/Metadata or TileScan/Metadata
                xlif_candidates = [
                    ptu_dir / "Metadata" / f"{roi_name}.xlif",
                    ts_folder / "Metadata" / f"{roi_name}.xlif",
                ]
                for xlif_candidate in xlif_candidates:
                    if xlif_candidate.exists():
                        xlif_path = xlif_candidate
                        break
                break
        
        # If no LOF found, still add the ROI (PTU-only processing)
        if xlif_path is None:
            # Try PTUs.sptw/Metadata as fallback
            xlif_path = ptu_dir / "Metadata" / f"{roi_name}.xlif"
            if not xlif_path.exists():
                xlif_path = None
        
        rois.append({
            'roi_num': roi_num,
            'roi_name': roi_name,
            'tilescan': tilescan,
            'lof_path': lof_path,
            'xlif_path': xlif_path,
            'ptu_dir': ptu_dir,
            'n_tiles': n_tiles,
            'is_single_fov': n_tiles == 1 and (ptu_dir / f"{roi_name}.ptu").exists(),
        })
    
    return rois


def stitch_intensity_from_lof(
    lof_path: Path,
    xlif_path: Path,
    ptu_basename: str,
    output_path: Path,
    verbose: bool = True,
) -> Optional[np.ndarray]:
    """
    Stitch intensity image from a LOF file using XLIF tile positions.
    
    Uses the same tile positions from XLIF metadata as PTU stitching
    to ensure alignment between intensity and FLIM cube outputs.
    
    Args:
        lof_path: Path to R N.lof file
        xlif_path: Path to XLIF metadata (same as used for PTU stitching)
        ptu_basename: ROI name (e.g., "R 2")
        output_path: Path to save stitched TIFF
        verbose: Print progress
    
    Note: 
        LOF tiles are already in correct orientation (Leica processed output),
        so no rotation is applied. Only raw PTU data needs 90° rotation.
    
    Returns:
        Stitched intensity array or None if failed
    """
    try:
        # Parse XLIF for tile positions (same as PTU stitching)
        tile_positions = parse_xlif_tile_positions(xlif_path, ptu_basename)
        pixel_size_m, n_pixels = get_pixel_size_from_xlif(xlif_path)
        
        with LifFile(str(lof_path)) as lif:
            img = lif.images[0]
            data = img.asarray()
            
            # Get dimensions
            dims = img.dims
            shape = data.shape
            
            if verbose:
                print(f"    LOF shape: {shape}, dims: {dims}")
            
            # Handle different dimension orders
            # Typical: (M, C, Y, X) or (M, Y, X) where M=tiles
            if len(shape) == 4:
                # (tiles, channels, Y, X) - sum channels
                intensity = data.sum(axis=1).astype(np.float32)
            elif len(shape) == 3:
                if dims and 'C' in dims:
                    # (channels, Y, X) - single tile with channels
                    intensity = data.sum(axis=0).astype(np.float32)
                else:
                    # (tiles, Y, X) - already single channel
                    intensity = data.astype(np.float32)
            else:
                intensity = data.astype(np.float32)
            
            # Get tile size (LOF tiles already in correct orientation)
            tile_h, tile_w = intensity.shape[-2:]
            
            # Compute canvas using same method as PTU stitching
            tile_positions, canvas_width, canvas_height = compute_tile_pixel_positions(
                tile_positions, pixel_size_m, tile_w
            )
            
            if verbose:
                print(f"    Canvas size: {canvas_height} x {canvas_width}")
                print(f"    Tiles: {len(tile_positions)} positions, {intensity.shape[0]} in LOF")
            
            # Stitch using XLIF positions
            stitched = np.zeros((canvas_height, canvas_width), dtype=np.float32)
            weight = np.zeros((canvas_height, canvas_width), dtype=np.float32)
            
            for i, t in enumerate(tile_positions):
                if i >= intensity.shape[0]:
                    break
                
                tile = intensity[i]
                
                # LOF tiles are already in correct orientation (Leica processed)
                # No rotation needed - only raw PTU data needs rotation
                
                th, tw = tile.shape
                y0 = t["pixel_y"]
                x0 = t["pixel_x"]
                y1 = min(y0 + th, canvas_height)
                x1 = min(x0 + tw, canvas_width)
                
                # Handle edge clipping
                tile_slice = tile[:y1-y0, :x1-x0]
                
                stitched[y0:y1, x0:x1] += tile_slice
                weight[y0:y1, x0:x1] += 1
            
            # Average overlaps
            weight[weight == 0] = 1
            stitched /= weight
        
        # Save
        output_path.parent.mkdir(exist_ok=True, parents=True)
        tifffile.imwrite(str(output_path), stitched.astype(np.float32))
        
        if verbose:
            print(f"    Saved: {output_path}")
        
        return stitched
        
    except Exception as e:
        if verbose:
            print(f"    Error stitching LOF: {e}")
            import traceback
            traceback.print_exc()
        return None


def process_roi(
    roi: Dict[str, Any],
    output_stitched: Path,
    output_cube: Path,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Process a single ROI: stitch intensity and generate FLIM cube.
    
    Args:
        roi: ROI info dict from discover_rois()
        output_stitched: Directory for stitched intensity images
        output_cube: Directory for FLIM cubes
        verbose: Print progress
    
    Returns:
        Dict with processing results:
            - roi_name: str
            - roi_id: str
            - intensity_path: Path or None
            - flim_path: Path or None
            - time_axis_path: Path or None
            - success: bool
            - error: str or None
    """
    roi_name = roi['roi_name']
    roi_id = f"R_{roi['roi_num']:03d}"
    
    result = {
        'roi_name': roi_name,
        'roi_id': roi_id,
        'intensity_path': None,
        'flim_path': None,
        'time_axis_path': None,
        'success': False,
        'error': None,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing {roi_name} ({roi['n_tiles']} tiles)")
        if roi['tilescan']:
            print(f"  TileScan: {roi['tilescan']}")
        print(f"{'='*60}")
    
    # 1. Stitch intensity from LOF if available (requires XLIF for proper alignment)
    if roi['lof_path'] and roi['lof_path'].exists() and roi['xlif_path'] and roi['xlif_path'].exists():
        if verbose:
            print(f"  Stitching intensity from LOF...")
        intensity_path = output_stitched / f"{roi_id}_intensity.tif"
        stitch_intensity_from_lof(
            lof_path=roi['lof_path'],
            xlif_path=roi['xlif_path'],
            ptu_basename=roi['roi_name'],
            output_path=intensity_path,
            verbose=verbose,
        )
        result['intensity_path'] = intensity_path
    elif roi['lof_path'] and roi['lof_path'].exists():
        if verbose:
            print(f"  No XLIF metadata, skipping LOF intensity stitch (would misalign)")
    else:
        if verbose:
            print(f"  No LOF file found, skipping intensity stitch")
    
    # 2. Generate FLIM cube from PTU files
    if roi['xlif_path'] and roi['xlif_path'].exists():
        if verbose:
            print(f"  Generating FLIM cube from PTU files...")
        
        try:
            # Use existing stitch module
            cube_result = stitch_flim_tiles(
                xlif_path=roi['xlif_path'],
                ptu_dir=roi['ptu_dir'],
                output_dir=output_cube / roi_id,
                ptu_basename=roi['roi_name'],
                rotate_tiles=True,
                verbose=verbose,
            )
            
            # Rename outputs to match convention
            src_flim = output_cube / roi_id / "stitched_flim_counts.npy"
            src_time = output_cube / roi_id / "time_axis_ns.npy"
            dst_flim = output_cube / f"{roi_id}_flim.npy"
            dst_time = output_cube / f"{roi_id}_time_axis_ns.npy"
            
            if src_flim.exists():
                src_flim.rename(dst_flim)
                result['flim_path'] = dst_flim
            if src_time.exists():
                src_time.rename(dst_time)
                result['time_axis_path'] = dst_time
            
            # Save metadata JSON for easy loading later
            meta_path = output_cube / f"{roi_id}_meta.json"
            meta_info = {
                'roi_name': roi_name,
                'roi_id': roi_id,
                'shape': list(cube_result['canvas_shape']) + [cube_result['n_time_bins']],
                'dtype': 'uint32',
                'n_tiles': roi['n_tiles'],
                'tiles_processed': cube_result['tiles_processed'],
                'time_bins': cube_result['n_time_bins'],
                'time_range_ns': list(cube_result['time_range_ns']),
                'tcspc_resolution_ps': cube_result['tcspc_resolution_ps'],
                'pixel_size_um': cube_result['pixel_size_um'],
            }
            with open(meta_path, 'w') as f:
                json.dump(meta_info, f, indent=2)
            result['meta_path'] = meta_path
            
            # Clean up temp folder
            import shutil
            temp_dir = output_cube / roi_id
            if temp_dir.exists():
                for f in temp_dir.iterdir():
                    if f.suffix in ['.tif', '.npy', '.log']:
                        f.unlink()
                try:
                    temp_dir.rmdir()
                except:
                    pass
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            if verbose:
                print(f"    Error: {e}")
    
    elif roi['is_single_fov']:
        # Single FOV - just copy/process the single PTU
        if verbose:
            print(f"  Processing single-FOV PTU...")
        
        try:
            ptu_path = roi['ptu_dir'] / f"{roi['roi_name']}.ptu"
            hist, meta = get_flim_histogram(ptu_path, rotate_cw=True)
            
            # Save FLIM cube
            output_cube.mkdir(exist_ok=True, parents=True)
            flim_path = output_cube / f"{roi_id}_flim.npy"
            time_path = output_cube / f"{roi_id}_time_axis_ns.npy"
            
            np.save(str(flim_path), hist)
            time_axis = create_time_axis(meta['n_time_bins'], meta['tcspc_resolution'])
            np.save(str(time_path), time_axis)
            
            result['flim_path'] = flim_path
            result['time_axis_path'] = time_path
            result['success'] = True
            
            if verbose:
                print(f"    Saved: {flim_path}")
            
        except Exception as e:
            result['error'] = str(e)
            if verbose:
                print(f"    Error: {e}")
    else:
        if verbose:
            print(f"  No XLIF metadata found, skipping FLIM cube generation")
    
    return result


def process_project(
    project_path: Path,
    roi_filter: Optional[List[int]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Process an entire Leica FLIM project.
    
    Discovers all ROIs, stitches intensity images from LOF files,
    and generates FLIM histogram cubes from PTU files.
    
    Args:
        project_path: Path to project folder (containing PTUs.sptw, TileScan folders)
        roi_filter: Optional list of ROI numbers to process (e.g., [2, 3, 4])
        verbose: Print progress
    
    Returns:
        Dict with:
            - success: bool
            - rois_found: int
            - rois_processed: int
            - results: List of per-ROI result dicts
            - output_stitched: Path to stitched/ folder
            - output_cube: Path to cube/ folder
    
    Output structure created:
        {project}/stitched/           # Stitched intensity images
            R_002_intensity.tif
            R_003_intensity.tif
            ...
        {project}/cube/               # FLIM histogram cubes
            R_002_flim.npy            # Raw memmap (Y, X, H)
            R_002_time_axis_ns.npy
            R_002_meta.json           # Metadata for loading
            ...
    """
    project_path = Path(project_path)
    
    if verbose:
        print(f"{'='*60}")
        print(f"LEICA FLIM PROJECT PROCESSOR")
        print(f"{'='*60}")
        print(f"Project: {project_path}")
    
    # Create output directories
    output_stitched = project_path / "stitched"
    output_cube = project_path / "cube"
    output_stitched.mkdir(exist_ok=True)
    output_cube.mkdir(exist_ok=True)
    
    if verbose:
        print(f"Output folders:")
        print(f"  Stitched: {output_stitched}")
        print(f"  FLIM Cube: {output_cube}")
    
    # Discover ROIs
    if verbose:
        print(f"\nDiscovering ROIs...")
    rois = discover_rois(project_path)
    
    if not rois:
        print("No ROIs found in project!")
        return {'success': False, 'rois_found': 0, 'rois_processed': 0}
    
    if verbose:
        print(f"Found {len(rois)} ROIs:")
        for roi in rois:
            status = []
            if roi['lof_path']:
                status.append("LOF")
            if roi['xlif_path']:
                status.append("XLIF")
            if roi['is_single_fov']:
                status.append("single-FOV")
            print(f"  {roi['roi_name']}: {roi['n_tiles']} tiles [{', '.join(status)}]")
    
    # Filter ROIs if specified
    if roi_filter:
        rois = [r for r in rois if r['roi_num'] in roi_filter]
        if verbose:
            print(f"\nFiltered to {len(rois)} ROIs: {roi_filter}")
    
    # Process each ROI
    results = []
    for roi in rois:
        result = process_roi(roi, output_stitched, output_cube, verbose=verbose)
        results.append(result)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"ROIs found: {len(rois)}")
        print(f"Successfully processed: {successful}")
        print(f"\nOutput locations:")
        print(f"  Intensity images: {output_stitched}/")
        print(f"  FLIM cubes:       {output_cube}/")
    
    return {
        'success': True,
        'rois_found': len(rois),
        'rois_processed': successful,
        'results': results,
        'output_stitched': output_stitched,
        'output_cube': output_cube,
    }


def load_flim_cube(cube_dir: Path, roi_id: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load a processed FLIM cube by ROI ID.
    
    Args:
        cube_dir: Path to cube/ folder
        roi_id: ROI identifier (e.g., 'R_002')
    
    Returns:
        Tuple of (flim_cube, time_axis_ns, metadata)
        - flim_cube: np.memmap of shape (Y, X, H)
        - time_axis_ns: np.array of time bins in nanoseconds
        - metadata: dict with shape, dtype, pixel_size_um, etc.
    
    Example:
        flim, time_ns, meta = load_flim_cube('/path/to/cube', 'R_002')
        print(f"Shape: {meta['shape']}")
        decay = flim[100, 100, :]  # Get decay at pixel (100, 100)
    """
    cube_dir = Path(cube_dir)
    
    # Load metadata
    meta_path = cube_dir / f"{roi_id}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    # Load time axis
    time_path = cube_dir / f"{roi_id}_time_axis_ns.npy"
    time_axis = np.load(str(time_path))
    
    # Load FLIM cube as memmap
    flim_path = cube_dir / f"{roi_id}_flim.npy"
    shape = tuple(meta['shape'])
    flim = np.memmap(str(flim_path), dtype=meta['dtype'], mode='r', shape=shape)
    
    return flim, time_axis, meta


# ============================================================================
# Example usage (if run as a script)
# ============================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        project_path = Path(sys.argv[1])
        process_project(project_path, verbose=True)
    else:
        print("Usage: python roi_builder.py /path/to/leica/project")
