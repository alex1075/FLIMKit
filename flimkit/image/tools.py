import cv2
import numpy as np
from pathlib import Path
from ..PTU import reader as ptufile
from typing import Dict, List, Tuple, Optional, Any


def make_intensity_image(ptu_path, rotate_90_cw=True, save_image=False):
    """Read a PTU file and create an intensity image by summing all time bins.

    Uses the custom PTUFile reader with raw_pixel_stack (overflow‑corrected).
    Returns a 2‑D numpy array (Y, X) of total photon counts per pixel.
    """
    ptu = ptufile.PTUFile(ptu_path, verbose=False)
    stack = ptu.raw_pixel_stack(channel=ptu.photon_channel)  # (Y, X, H)
    intensity = stack.sum(axis=-1)  # sum over histogram bins → (Y, X)
    if rotate_90_cw:
        intensity = np.rot90(intensity, k=-1)  # 90° clockwise
        print("  Rotated 90° clockwise.")
    if save_image:
        out_path = Path(ptu_path).stem + "_intensity.png"
        normed = cv2.normalize(intensity.astype(np.float32), None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(out_path, normed)
        print(f"  Saved intensity image: {out_path}")
    return intensity


def make_cell_mask(intensity_image, save_mask=False, path=None, name=None):
    """Create a binary cell mask from an intensity image.

    Parameters
    ----------
    intensity_image : str, Path, or np.ndarray
        Either a file path to a grayscale image or a 2‑D numpy array
        (e.g. from ``make_intensity_image``).
    save_mask : bool
        If True, save the mask as a PNG beside the source file.
    path : str or Path, optional
        Base path used for the saved PNG filename.
    name : str, optional
        Optional label (unused, kept for API compat).

    Returns
    -------
    mask : np.ndarray
        Boolean 2‑D array — True where cells are detected, False for background.
    """
    # Accept both file paths and numpy arrays
    if isinstance(intensity_image, (str, Path)):
        img = cv2.imread(str(intensity_image), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {intensity_image}")
    elif isinstance(intensity_image, np.ndarray):
        # Normalise to uint8 for OpenCV processing
        img = intensity_image.astype(np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        raise TypeError(f"Expected file path or ndarray, got {type(intensity_image)}")

    # Contrast stretch + boost
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = np.clip(img * 20, 0, 255).astype(np.uint8)

    # Smooth + threshold  (THRESH_BINARY_INV → background is bright after boost)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY_INV)

    # Fill contours to get a solid mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_uint8 = np.zeros_like(img)
    cv2.drawContours(mask_uint8, contours, -1, 255, thickness=cv2.FILLED)

    if save_mask and path is not None:
        out_path = str(Path(path).with_suffix('')) + "_cell_mask.png"
        cv2.imwrite(out_path, mask_uint8)
        print(f"  Saved cell mask: {out_path}")

    return mask_uint8 > 0  # boolean mask