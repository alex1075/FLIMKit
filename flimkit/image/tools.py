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

    # Contrast stretch (no multiplicative boost — it saturates all signal to 255
    # and then THRESH_BINARY_INV would label only zero-photon gaps as tissue).
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Smooth + threshold: fluorescence cells are bright → THRESH_BINARY labels
    # pixels above threshold as foreground (correct for FLIM intensity images).
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)

    # Fill contours to get a solid mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_uint8 = np.zeros_like(img)
    cv2.drawContours(mask_uint8, contours, -1, 255, thickness=cv2.FILLED)

    if save_mask and path is not None:
        out_path = str(Path(path).with_suffix('')) + "_cell_mask.png"
        cv2.imwrite(out_path, mask_uint8)
        print(f"  Saved cell mask: {out_path}")

    return mask_uint8 > 0  # boolean mask


# ── Intensity-threshold tools ──────────────────────────────────────────────


def apply_intensity_threshold(intensity_image, threshold):
    """Return a boolean mask where True = pixel intensity >= threshold.

    Parameters
    ----------
    intensity_image : np.ndarray
        2-D array of total photon counts per pixel (Y, X).
    threshold : int or float
        Minimum total-photon count to keep a pixel.

    Returns
    -------
    mask : np.ndarray (bool)
        Same shape as *intensity_image*; True where intensity >= *threshold*.
    """
    return intensity_image >= threshold


def pick_intensity_threshold(intensity_image, initial=None):
    """Open an interactive matplotlib window to pick an intensity threshold.

    The window shows the intensity image with a slider bar. Dragging the
    slider updates a red overlay highlighting pixels that would be
    **excluded** (below threshold). The chosen value is returned when the
    user closes the window or presses *Enter* / clicks *Accept*.

    Parameters
    ----------
    intensity_image : np.ndarray
        2-D array of photon counts per pixel.
    initial : int or None
        Starting position of the slider. Defaults to ~5 % of the max
        intensity.

    Returns
    -------
    threshold : int
        The selected photon-count threshold.
    """
    import matplotlib
    matplotlib.use("TkAgg")          # need an interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    img = intensity_image.astype(float)
    vmax = img.max()
    if initial is None:
        initial = max(1, int(vmax * 0.05))

    # State container (mutable so the nested functions can write to it)
    state = {"threshold": int(initial)}

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.subplots_adjust(bottom=0.22)

    # Show intensity image
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_title(f"Intensity image  —  threshold = {state['threshold']} photons")
    ax.set_axis_off()

    # Red overlay for excluded pixels
    overlay_rgba = np.zeros((*img.shape, 4), dtype=float)
    mask_below = img < state["threshold"]
    overlay_rgba[mask_below] = [1, 0, 0, 0.35]
    overlay_im = ax.imshow(overlay_rgba, interpolation="nearest")

    # Pixel count annotation
    n_above = int((~mask_below).sum())
    n_total = img.size
    count_text = ax.text(
        0.01, 0.01,
        f"{n_above:,}/{n_total:,} pixels kept ({100 * n_above / n_total:.1f} %)",
        transform=ax.transAxes, fontsize=9,
        color="white", backgroundcolor=(0, 0, 0, 0.5),
        verticalalignment="bottom",
    )

    # Slider
    ax_slider = plt.axes([0.15, 0.08, 0.65, 0.03])
    slider = Slider(ax_slider, "Min photons", 0, max(int(vmax), 1),
                    valinit=initial, valstep=1, valfmt="%d")

    def _update(val):
        thr = int(val)
        state["threshold"] = thr
        mask_below = img < thr
        rgba = np.zeros((*img.shape, 4), dtype=float)
        rgba[mask_below] = [1, 0, 0, 0.35]
        overlay_im.set_data(rgba)
        n_above = int((~mask_below).sum())
        ax.set_title(f"Intensity image  —  threshold = {thr} photons")
        count_text.set_text(
            f"{n_above:,}/{n_total:,} pixels kept ({100 * n_above / n_total:.1f} %)"
        )
        fig.canvas.draw_idle()

    slider.on_changed(_update)

    # Accept button
    ax_btn = plt.axes([0.40, 0.02, 0.20, 0.04])
    btn = Button(ax_btn, "Accept", hovercolor="lightgreen")

    def _accept(event):
        plt.close(fig)

    btn.on_clicked(_accept)

    # Also close on Enter key
    def _on_key(event):
        if event.key in ("enter", "return"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)

    plt.show()   # blocks until window is closed

    chosen = state["threshold"]
    print(f"  Intensity threshold selected: {chosen} photons")
    return chosen