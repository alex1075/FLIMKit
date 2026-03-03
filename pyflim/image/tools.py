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