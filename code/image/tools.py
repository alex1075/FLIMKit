import cv2
import numpy as np
from pathlib import Path
from ..PTU import reader as ptufile
from .stitch import discover_rois, process_roi
from typing import Dict, List, Tuple, Optional, Any

def make_intensity_image(ptu_path, rotate_90_cw=True, save_image=False):
    """Read a PTU file and create an intensity image by summing all time bins."""
    ptu = ptufile.PtuFile(ptu_path)
    data = ptu[:]                     # shape like (T, Y, X, C, H) or (Y, X, H)
    dims = ptu.dims 
    if 'Y' in dims:
        y_axis = dims.index('Y')
    else:
        return None  # No Y dimension, can't make an image
    if 'X' in dims:
        x_axis = dims.index('X')
    else:
        return None  # No X dimension, can't make an image
    axes_to_sum = [i for i in range(data.ndim) if i not in (y_axis, x_axis)]
    if axes_to_sum:
        intensity = data.sum(axis=tuple(axes_to_sum))
    else:
        intensity = data  # already 2D
    if y_axis > x_axis:
        intensity = intensity.T
    if rotate_90_cw:
        intensity = np.rot90(intensity, k=-1)  # 90° clockwise
        print("Rotated 90° clockwise.")
    ptu.close()
    if save_image:
        cv2.imwrite(Path(ptu_path).stem + "_intensity.png", intensity)
    return intensity

def make_cell_mask(intensity_image, save_mask=False, path=None, name=None):
    """Create a binary mask of cells from the intensity image."""
    try:
        img = cv2.imread(intensity_image, cv2.IMREAD_GRAYSCALE)
    except:
        try:
            img = intensity_image.astype(np.uint8)
        except:
            print("Error: Could not read intensity image.")
            return None
    img = cv2. normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = np.clip(img * 20, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if save_mask:
        # black where mask is not
        mask = np.zeros_like(img)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        cv2.imwrite(Path(path).stem + "_mask.png", mask)
    return contours

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
