"""Top-level interactive launcher for phasor FLIM analysis.

Prompts the user for a *.ptu* file (or a previously saved *.npz* session),
an optional IRF calibration file, then opens the interactive phasor cursor
tool.  Results — phasor arrays **and** cursor positions / ellipse parameters
— can be saved to a single *.npz* file and re-opened later for continued
editing.

Usage
-----
From a script or terminal::

    from flimkit.phasor_launcher import launch_phasor
    launch_phasor()          # interactive prompts

Or via the CLI entry-point (``phasor_cli.py``)::

    python phasor_cli.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import inquirer


# ─────────────────────────────────────────────────────────────
# Helpers: yes/no, path prompts, file-dialog save
# ─────────────────────────────────────────────────────────────
def _yes_no(question: str) -> bool:
    """Ask a yes/no question via inquirer and return True for Yes."""
    ans = inquirer.prompt([inquirer.List(
        'yesno', message=question, choices=['Yes', 'No'])])
    return ans['yesno'] == 'Yes'


def _ask_path(message: str, *, optional: bool = False) -> str | None:
    """Ask the user for a file path via inquirer (with tab-completion)."""
    hint = " (leave blank to skip)" if optional else ""
    ans = inquirer.prompt([
        inquirer.Path('path',
                      message=f"{message}{hint}",
                      path_type=inquirer.Path.FILE,
                      exists=not optional)])
    val = (ans or {}).get('path', '').strip()
    if not val:
        return None
    return val


def _pick_save_file(title: str, default_name: str) -> str | None:
    """Open a native save-file dialog (tkinter) or fall back to input."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        root.update()
        path = filedialog.asksaveasfilename(
            title=title,
            defaultextension='.npz',
            initialfile=default_name,
            filetypes=[('NumPy archive', '*.npz'), ('All files', '*')])
        root.destroy()
        return path or None
    except Exception:
        path = input(f"Save path [{default_name}]: ").strip().strip('"')
        return path or default_name


# ─────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────
def save_session(path: str, *,
                 real_cal: np.ndarray,
                 imag_cal: np.ndarray,
                 mean: np.ndarray,
                 frequency: float,
                 cursors: list[dict],
                 params: dict,
                 ptu_file: str | None = None,
                 irf_file: str | None = None,
                 display_image: np.ndarray | None = None) -> None:
    """Persist phasor data **and** cursor state to a *.npz* file.

    Parameters
    ----------
    path : str
        Destination file path (will get *.npz* extension if absent).
    real_cal, imag_cal, mean : ndarray
        Calibrated phasor arrays.
    frequency : float
        Modulation frequency in MHz.
    cursors : list of dict
        Each entry has ``'center_g'``, ``'center_s'``, ``'color'``.
    params : dict
        Ellipse parameters (``radius``, ``radius_minor``, ``angle_mode``).
    ptu_file, irf_file : str or None
        Original source paths (stored as metadata for reference).
    display_image : ndarray or None
        Spatially-correct intensity image (nsync-based).
    """
    n = len(cursors)
    cursor_g = np.array([c['center_g'] for c in cursors], dtype=float) if n else np.array([], dtype=float)
    cursor_s = np.array([c['center_s'] for c in cursors], dtype=float) if n else np.array([], dtype=float)
    cursor_colors = np.array([c['color'] for c in cursors], dtype='U10') if n else np.array([], dtype='U10')

    save_kw = dict(
        real_cal=real_cal,
        imag_cal=imag_cal,
        mean=mean,
        frequency=np.float64(frequency),
        cursor_g=cursor_g,
        cursor_s=cursor_s,
        cursor_colors=cursor_colors,
        param_radius=np.float64(params.get('radius', 0.05)),
        param_radius_minor=np.float64(params.get('radius_minor', 0.03)),
        param_angle_mode=np.array(params.get('angle_mode', 'semicircle')),
        ptu_file=np.array(ptu_file or ''),
        irf_file=np.array(irf_file or ''),
    )
    if display_image is not None:
        save_kw['display_image'] = np.asarray(display_image)

    np.savez_compressed(path, **save_kw)
    print(f"✅  Session saved → {path}  ({n} cursor(s))")


def load_session(path: str) -> dict:
    """Load a previously saved session from a *.npz* file.

    Returns
    -------
    dict
        Keys: ``real_cal``, ``imag_cal``, ``mean``, ``frequency``,
        ``cursors`` (list[dict]), ``params`` (dict), ``ptu_file``,
        ``irf_file``.
    """
    d = np.load(path, allow_pickle=False)
    cursors = []
    g = d['cursor_g']
    s = d['cursor_s']
    colors = d['cursor_colors']
    for i in range(len(g)):
        cursors.append(dict(
            center_g=float(g[i]),
            center_s=float(s[i]),
            color=str(colors[i]),
        ))
    params = dict(
        radius=float(d['param_radius']),
        radius_minor=float(d['param_radius_minor']),
        angle_mode=str(d['param_angle_mode']),
    )
    return dict(
        real_cal=d['real_cal'],
        imag_cal=d['imag_cal'],
        mean=d['mean'],
        frequency=float(d['frequency']),
        cursors=cursors,
        params=params,
        ptu_file=str(d['ptu_file']) or None,
        irf_file=str(d['irf_file']) or None,
        display_image=d['display_image'] if 'display_image' in d else None,
    )


# ─────────────────────────────────────────────────────────────
# Pipeline: PTU → phasor → (optional) calibration
# ─────────────────────────────────────────────────────────────
def _process_ptu(ptu_path: str, irf_path: str | None = None) -> dict:
    """Load a PTU file, compute phasors, optionally calibrate with IRF.

    Returns dict with ``real_cal``, ``imag_cal``, ``mean``, ``frequency``,
    and ``display_image`` (nsync-based intensity for correct spatial overlay).
    """
    from phasorpy.phasor import phasor_from_signal
    from .PTU.tools import signal_from_PTUFile
    from .PTU.reader import PTUFile
    from .phasor.signal import get_phasor_irf, calibrate_signal_with_irf

    print(f"Loading PTU file: {ptu_path}")
    signal = signal_from_PTUFile(ptu_path, dtype=np.uint32, binning=4)
    frequency = float(signal.attrs['frequency'])

    print(f"Computing phasors (frequency = {frequency:.2f} MHz) …")
    mean, real, imag = phasor_from_signal(signal, axis='H')

    # Build a spatially-correct intensity image via raw_pixel_stack
    # (uses nsync timing → accurate pixel positions for the FOV overlay)
    ptu = PTUFile(str(ptu_path), verbose=False)
    display_image = ptu.raw_pixel_stack(binning=4).sum(axis=-1)  # (Y, X)

    if irf_path:
        print(f"Calibrating with IRF: {irf_path}")
        irf_time_ns, irf_counts = get_phasor_irf(irf_path)
        real_cal, imag_cal = calibrate_signal_with_irf(
            signal, real, imag, irf_time_ns, irf_counts, frequency)
    else:
        print("⚠  No IRF — using uncalibrated phasor coordinates.")
        real_cal, imag_cal = real, imag

    return dict(
        real_cal=np.asarray(real_cal),
        imag_cal=np.asarray(imag_cal),
        mean=np.asarray(mean),
        frequency=frequency,
        display_image=np.asarray(display_image, dtype=float),
    )


# ─────────────────────────────────────────────────────────────
# Main launcher
# ─────────────────────────────────────────────────────────────
def launch_phasor(ptu_path: str | None = None,
                  irf_path: str | None = None,
                  session_path: str | None = None,
                  *,
                  min_photons: float = 0.01,
                  max_cursors: int = 6,
                  figsize: tuple[float, float] = (8, 5)) -> dict:
    """Interactive phasor FLIM analysis with save / load support.

    If no arguments are supplied a file-dialog prompts for the input.

    Parameters
    ----------
    ptu_path : str, optional
        Path to a *.ptu* file.  Ignored when *session_path* is given.
    irf_path : str, optional
        Path to the IRF Excel calibration file.
    session_path : str, optional
        Path to a previously saved *.npz* session to resume.
    min_photons, max_cursors, figsize
        Forwarded to :func:`~flimkit.phasor.interactive.phasor_cursor_tool`.

    Returns
    -------
    state : dict
        The mutable state dict from ``phasor_cursor_tool``.
    """
    from .phasor.interactive import phasor_cursor_tool

    initial_cursors = None
    initial_params = None
    src_ptu = ptu_path
    src_irf = irf_path

    # ── Decide whether to load a session or process a PTU ────
    if session_path is None and ptu_path is None:
        # Interactive inquirer flow
        mode = inquirer.prompt([inquirer.List(
            'mode',
            message='What would you like to do?',
            choices=[
                'Analyse a new PTU file',
                'Resume a saved session (.npz)',
            ],
        )])['mode']

        if mode.startswith('Resume'):
            session_path = _ask_path('Path to saved .npz session')
            if session_path is None:
                print("No file specified — aborting.")
                return {}
        else:
            ptu_path = _ask_path('Path to PTU file')
            if ptu_path is None:
                print("No file specified — aborting.")
                return {}

    # ── Load saved session ───────────────────────────────────
    if session_path:
        print(f"Loading session: {session_path}")
        sess = load_session(session_path)
        data = dict(
            real_cal=sess['real_cal'],
            imag_cal=sess['imag_cal'],
            mean=sess['mean'],
            frequency=sess['frequency'],
            display_image=sess.get('display_image'),
        )
        initial_cursors = sess['cursors'] or None
        initial_params = sess['params']
        src_ptu = sess.get('ptu_file')
        src_irf = sess.get('irf_file')
        print(f"  frequency = {data['frequency']:.2f} MHz, "
              f"{len(sess['cursors'])} cursor(s) restored")
    else:
        # ── Prompt for optional IRF ──────────────────────────
        if irf_path is None:
            if _yes_no('Do you have an IRF calibration file? (recommended)'):
                irf_path = _ask_path('Path to IRF Excel file (.xlsx)')
        src_irf = irf_path
        data = _process_ptu(ptu_path, irf_path)

    # ── Build a save callback ────────────────────────────────
    _data = data          # capture for closure

    def _save_callback(state, params):
        stem = Path(src_ptu).stem if src_ptu else 'phasor_session'
        default_name = f"{stem}_session.npz"
        out = _pick_save_file("Save phasor session", default_name)
        if out:
            save_session(
                out,
                real_cal=_data['real_cal'],
                imag_cal=_data['imag_cal'],
                mean=_data['mean'],
                frequency=_data['frequency'],
                cursors=state['cursors'],
                params=params,
                ptu_file=src_ptu,
                irf_file=src_irf,
                display_image=_data.get('display_image'),
            )

    # ── Launch the interactive tool ──────────────────────────
    state = phasor_cursor_tool(
        data['real_cal'],
        data['imag_cal'],
        data['mean'],
        data['frequency'],        display_image=data.get('display_image'),        min_photons=min_photons,
        max_cursors=max_cursors,
        figsize=figsize,
        initial_cursors=initial_cursors,
        initial_params=initial_params,
        on_save=_save_callback,
    )
    return state


# ─────────────────────────────────────────────────────────────
# Inquirer-driven interactive session (called from phasor_cli.py
# or ``main.py`` menu)
# ─────────────────────────────────────────────────────────────
def phasor_inquire() -> dict:
    """Full guided prompt → launch_phasor().  No arguments required."""
    print("\n--- Interactive Phasor Analysis ---")
    return launch_phasor()   # all prompts happen inside


# ── Allow ``python -m flimkit.phasor_launcher`` ───────────────
if __name__ == '__main__':
    phasor_inquire()
