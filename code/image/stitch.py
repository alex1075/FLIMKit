import re
import json
from liffile import LifFile
import numpy as np
import tifffile
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from ..utils import (
    setup_loggers,
    parse_xlif_tile_positions,
    get_pixel_size_from_xlif,
    compute_tile_pixel_positions,
)
from ..PTU.reader import get_flim_histogram, create_time_axis
from typing import Dict, List, Tuple, Optional, Any

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
        import tifffile
        
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