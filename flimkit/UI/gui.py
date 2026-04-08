#!/usr/bin/env python3


from __future__ import annotations

import re
import sys
import time
import inspect
import threading
import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from flimkit.UI.progress_window import ProgressWindow
from flimkit.UI.phasor_panel import PhasorViewPanel
from flimkit.UI import flim_display

# Modern theme support
try:
    import TKinterModernThemes as TKMT
    HAS_TKMT = True
except ImportError:
    HAS_TKMT = False

# Drag and drop support (optional – may conflict with theming)
try:
    from tkinterdnd2 import DND_FILES, DND_TEXT
    HAS_DND = True
except ImportError:
    HAS_DND = False

# GUI mode flag for tqdm (disables progress bars in GUI, keeps them in CLI)
GUI_MODE = False

# ─────────────────────────────────────────────────────────────────────────────
# Lazy config loader – deferred so flimkit.__init__ circular imports don't fire
# at module load time.  Call _C()['key'] anywhere you need a config value.
# ─────────────────────────────────────────────────────────────────────────────

_cfg: dict = {}


def _C() -> dict:
    if not _cfg:
        from flimkit.configs import (
            n_exp, Tau_min, Tau_max, D_mode, binning_factor,
            MIN_PHOTONS_PERPIX, Optimizer, lm_restarts, de_population,
            de_maxiter, n_workers, OUT_NAME, IRF_BINS, IRF_FIT_WIDTH,
            IRF_FWHM, channels, TAU_DISPLAY_MIN, TAU_DISPLAY_MAX,
            INTENSITY_DISPLAY_MIN, INTENSITY_DISPLAY_MAX,
            MACHINE_IRF_DIR, MACHINE_IRF_DEFAULT_PATH,
            MACHINE_IRF_ALIGN_ANCHOR, MACHINE_IRF_REDUCER,
        )
        _cfg.update(
            n_exp=n_exp, Tau_min=Tau_min, Tau_max=Tau_max, D_mode=D_mode,
            binning_factor=binning_factor, MIN_PHOTONS_PERPIX=MIN_PHOTONS_PERPIX,
            Optimizer=Optimizer, lm_restarts=lm_restarts,
            de_population=de_population, de_maxiter=de_maxiter,
            n_workers=n_workers, OUT_NAME=OUT_NAME,
            IRF_BINS=IRF_BINS, IRF_FIT_WIDTH=IRF_FIT_WIDTH, IRF_FWHM=IRF_FWHM,
            channels=channels,
            TAU_DISPLAY_MIN=TAU_DISPLAY_MIN, TAU_DISPLAY_MAX=TAU_DISPLAY_MAX,
            INTENSITY_DISPLAY_MIN=INTENSITY_DISPLAY_MIN,
            INTENSITY_DISPLAY_MAX=INTENSITY_DISPLAY_MAX,
            MACHINE_IRF_DIR=MACHINE_IRF_DIR,
            MACHINE_IRF_DEFAULT_PATH=MACHINE_IRF_DEFAULT_PATH,
            MACHINE_IRF_ALIGN_ANCHOR=MACHINE_IRF_ALIGN_ANCHOR,
            MACHINE_IRF_REDUCER=MACHINE_IRF_REDUCER,
        )
    return _cfg


# ─────────────────────────────────────────────────────────────────────────────
# Helper to parse fit summary from captured log (placeholder)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_summary(captured_log: str) -> list:
    """
    Parse the captured stdout/stderr for a summary table.
    This is a placeholder – replace with actual parsing if needed.
    Returns a list of (parameter, value, unit) rows.
    """
    rows = []
    # Example: find lines like "tau1 = 2.45 ns"
    for line in captured_log.splitlines():
        if "tau" in line.lower() and "=" in line:
            parts = line.split("=", 1)
            if len(parts) == 2:
                param = parts[0].strip()
                rest = parts[1].strip()
                val_unit = rest.split()
                if len(val_unit) >= 2:
                    rows.append((param, val_unit[0], val_unit[1]))
                else:
                    rows.append((param, rest, ""))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Stdout capture (handles progress bars with \r)
# ─────────────────────────────────────────────────────────────────────────────

class _Redirect:
    """Redirect stdout/stderr to ScrolledText; batches updates for performance (thread-safe)."""

    def __init__(self, widget: scrolledtext.ScrolledText, buf: list, root=None):
        self.widget = widget
        self.buf    = buf
        self.root   = root  # For thread-safe GUI updates
        self._batch = []  # Accumulate text before writing
        self._batch_size = 5000  # characters, or time-based flush
        self._last_flush = time.time()
        self._flush_interval = 0.5  # seconds

    def write(self, text: str):
        if not text:
            return
        self.buf.append(text)
        self._batch.append(text)
        
        # Flush if batch is large or timeout elapsed
        should_flush = False
        if len("".join(self._batch)) >= self._batch_size:
            should_flush = True
        elif time.time() - self._last_flush >= self._flush_interval:
            should_flush = True
        
        if should_flush:
            self._flush_batch()

    def _flush_batch(self):
        if not self._batch:
            return
        text = "".join(self._batch)
        self._batch.clear()
        
        # Use root.after() for thread-safe GUI updates if root is available
        if self.root:
            self.root.after(0, self._update_widget, text)
        else:
            # Fallback to direct update (not thread-safe but works in single-threaded context)
            self._update_widget(text)
    
    def _update_widget(self, text):
        """Update widget from main thread."""
        try:
            self.widget.configure(state="normal")
            self.widget.insert(tk.END, text)
            self.widget.see(tk.END)
            self.widget.configure(state="disabled")
            self.widget.update_idletasks()
        except Exception:
            pass  # Widget may have been destroyed
        self._last_flush = time.time()

    def flush(self):
        self._flush_batch()


class _FileRedirect:
    """Redirect stdout/stderr to a file for performance (no widget updates)."""

    def __init__(self, filepath: str, buf: list):
        self.filepath = filepath
        self.buf = buf
        self._file = None
        try:
            self._file = open(filepath, 'w', buffering=1)  # Line buffering
        except Exception:
            pass

    def write(self, text: str):
        if not text:
            return
        self.buf.append(text)
        if self._file:
            try:
                self._file.write(text)
            except Exception:
                pass

    def flush(self):
        if self._file:
            try:
                self._file.flush()
            except Exception:
                pass

    def close(self):
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None


# ─────────────────────────────────────────────────────────────────────────────
# Progress Window Manager – safely create progress windows from worker threads
# ─────────────────────────────────────────────────────────────────────────────

class ProgressWindowManager:
    """Manages nested progress windows for sub-operations, callable from worker threads."""
    
    def __init__(self, root):
        self.root = root
        self.windows = {}  # task_id -> ProgressWindow
        self.counter = 0
        self.lock = threading.Lock()
    
    def create_progress_window(self, task_name="Processing..."):
        """Create a progress window and return its ID. Thread-safe."""
        with self.lock:
            window_id = self.counter
            self.counter += 1
        
        # Create window on main thread
        window_ref = [None]
        event = threading.Event()
        
        def create_window():
            window_ref[0] = ProgressWindow(self.root, task_name=task_name)
            with self.lock:
                self.windows[window_id] = window_ref[0]
            event.set()
        
        self.root.after(0, create_window)
        event.wait(timeout=5)  # Wait up to 5 seconds for window to be created
        return window_id
    
    def update_progress(self, window_id, current, total):
        """Update progress for a window. Thread-safe."""
        def update():
            if window_id in self.windows:
                try:
                    self.windows[window_id].set_progress(current, maximum=total)
                except Exception:
                    pass  # Window may have been closed
        
        self.root.after(0, update)
    
    def set_status(self, window_id, msg):
        """Set status message for a window. Thread-safe."""
        def update():
            if window_id in self.windows:
                try:
                    self.windows[window_id].set_status(msg)
                except Exception:
                    pass
        
        self.root.after(0, update)
    
    def close_window(self, window_id):
        """Close a progress window. Thread-safe."""
        def close():
            if window_id in self.windows:
                try:
                    self.windows[window_id].close()
                except Exception:
                    pass
                finally:
                    with self.lock:
                        self.windows.pop(window_id, None)
        
        self.root.after(0, close)
    
    def close_all(self):
        """Close all progress windows."""
        with self.lock:
            ids = list(self.windows.keys())
        for wid in ids:
            self.close_window(wid)



class _FileTailer:
    """Stream log file content to a Text widget in real-time."""

    def __init__(self, filepath: str, widget: scrolledtext.ScrolledText, update_interval_ms: int = 200):
        self.filepath = filepath
        self.widget = widget
        self.update_interval_ms = update_interval_ms
        self._file = None
        self._last_pos = 0
        self._running = False

    def start(self, root):
        """Start tailing the file and updating the widget."""
        self._running = True
        self._poll_file(root)

    def _poll_file(self, root):
        """Poll the file for new content and update widget."""
        if not self._running:
            return
        
        try:
            if not Path(self.filepath).exists():
                root.after(self.update_interval_ms, lambda: self._poll_file(root))
                return
            
            # Read new content from file
            with open(self.filepath, 'r') as f:
                f.seek(self._last_pos)
                new_content = f.read()
                self._last_pos = f.tell()
            
            # Update widget if there's new content
            if new_content:
                self.widget.configure(state="normal")
                self.widget.insert(tk.END, new_content)
                self.widget.see(tk.END)
                self.widget.configure(state="disabled")
                self.widget.update_idletasks()
        except Exception:
            pass
        
        # Schedule next poll
        root.after(self.update_interval_ms, lambda: self._poll_file(root))

    def stop(self):
        """Stop tailing the file."""
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────

PAD = dict(padx=8, pady=4)


def _browse_file(var, title="Select file", filetypes=(("All", "*.*"),)):
    p = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if p:
        var.set(p)


def _browse_dir(var, title="Select directory"):
    p = filedialog.askdirectory(title=title)
    if p:
        var.set(p)


def _row(parent, label, var, row, browse_fn, width=45, state="normal"):
    ttk.Label(parent, text=label).grid(
        row=row, column=0, sticky="e", padx=6, pady=3)
    e = ttk.Entry(parent, textvariable=var, width=width, state=state)
    e.grid(row=row, column=1, sticky="ew", padx=4, pady=3)
    ttk.Button(parent, text="Browse...", command=browse_fn).grid(
        row=row, column=2, padx=4, pady=3)
    return e


def _section(parent, text: str) -> ttk.LabelFrame:
    return ttk.LabelFrame(parent, text=f"  {text}  ", padding=(10, 6))


def _tog(bvar: tk.BooleanVar, entry: ttk.Entry):
    entry.configure(state="normal" if bvar.get() else "disabled")


def _flt(sv: tk.StringVar) -> Optional[float]:
    v = sv.get().strip()
    return float(v) if v and v.lower() != "none" else None


def _thresh(bvar: tk.BooleanVar, sv: tk.StringVar):
    """Return threshold value, or None."""
    if not bvar.get():
        return None
    v = sv.get().strip()
    return int(v) if v else None


class IRFWidget:
    # Sentinel to detect that the path was auto-filled (not user-entered)
    _AUTO_FILL = object()

    CHOICES = [
        ("Leica analytical model (XLSX)  [recommended]", "irf_xlsx"),
        ("Machine IRF (.npy pre-built)",                 "machine_irf"),
        ("Scatter PTU (measured IRF)",                   "file"),
        ("Estimate from decay – raw",                    "raw"),
        ("Estimate from decay – parametric",             "parametric"),
        ("Gaussian (fallback)",                          "gaussian"),
    ]

    def __init__(self, parent, default="irf_xlsx", xlsx_var=None, machine_irf_default: str = ""):
        self.xlsx_var  = xlsx_var
        self._machine_irf_default = machine_irf_default
        self.sv_method = tk.StringVar(value=default)
        self.sv_path   = tk.StringVar()

        self.frame = _section(parent, "Instrument Response Function (IRF)")
        self.frame.columnconfigure(1, weight=1)

        for i, (lbl, val) in enumerate(self.CHOICES):
            ttk.Radiobutton(self.frame, text=lbl, variable=self.sv_method,
                            value=val, command=self._update).grid(
                row=i, column=0, columnspan=3, sticky="w", padx=4, pady=1)

        r = len(self.CHOICES)
        self._path_lbl = ttk.Label(self.frame, text="IRF / XLSX path")
        self._path_lbl.grid(row=r, column=0, sticky="e", padx=6, pady=3)
        self._path_e = ttk.Entry(self.frame, textvariable=self.sv_path, width=45)
        self._path_e.grid(row=r, column=1, sticky="ew", padx=4, pady=3)
        self._path_btn = ttk.Button(
            self.frame, text="Browse...",
            command=self._browse_irf_path)
        self._path_btn.grid(row=r, column=2, padx=4, pady=3)

        self._note = ttk.Label(
            self.frame,
            text="Uses the XLSX entered in Input Files above",
            foreground="grey")
        self._note.grid(row=r, column=0, columnspan=3, sticky="w", padx=8, pady=3)

        self._update()

    def _browse_irf_path(self):
        if self.sv_method.get() == "machine_irf":
            _browse_file(self.sv_path, "Select machine IRF",
                         [("NumPy array", "*.npy"), ("All", "*.*")])
        else:
            _browse_file(self.sv_path, "Select IRF file",
                         [("PTU / XLSX", "*.ptu *.xlsx"), ("All", "*.*")])

    def _show_browse(self):
        method = self.sv_method.get()
        self._path_lbl.config(
            text="Machine IRF (.npy) path" if method == "machine_irf" else "IRF / XLSX path")
        if method == "machine_irf" and not self.sv_path.get().endswith(".npy"):
            self.sv_path.set(self._machine_irf_default)
        self._path_lbl.grid()
        self._path_e.grid()
        self._path_btn.grid()
        self._note.grid_remove()

    def _show_note(self):
        self._path_lbl.grid_remove()
        self._path_e.grid_remove()
        self._path_btn.grid_remove()
        self._note.grid()

    def _hide_all(self):
        self._path_lbl.grid_remove()
        self._path_e.grid_remove()
        self._path_btn.grid_remove()
        self._note.grid_remove()

    def _update(self):
        method = self.sv_method.get()
        if method == "irf_xlsx":
            self._show_note() if self.xlsx_var is not None else self._show_browse()
        elif method in ("file", "machine_irf"):
            self._show_browse()
        else:
            self._hide_all()

    def grid(self, **kw):
        self.frame.grid(**kw)

    def get_args(self, xlsx_fallback: Optional[str] = None) -> dict:
        method = self.sv_method.get()
        path   = self.sv_path.get().strip() or None
        if method == "irf_xlsx":
            xlsx = (self.xlsx_var.get().strip() if self.xlsx_var else None) \
                   or xlsx_fallback or path
            return dict(irf=None, irf_xlsx=xlsx, estimate_irf="none", no_xlsx_irf=False, machine_irf=None)
        elif method == "machine_irf":
            return dict(irf=None, irf_xlsx=None, estimate_irf="machine_irf", no_xlsx_irf=True, machine_irf=path)
        elif method == "file":
            return dict(irf=path, irf_xlsx=None, estimate_irf="none", no_xlsx_irf=True, machine_irf=None)
        elif method in ("raw", "parametric"):
            return dict(irf=None, irf_xlsx=None, estimate_irf=method, no_xlsx_irf=True, machine_irf=None)
        else:
            return dict(irf=None, irf_xlsx=None, estimate_irf="none", no_xlsx_irf=True, machine_irf=None)


# ─────────────────────────────────────────────────────────────────────────────
# FOV Preview panel  –  Intensity image & decay curve (right-side viewer)
# ─────────────────────────────────────────────────────────────────────────────

class FOVPreviewPanel:
    """Real-time preview of FOV intensity image and decay curve."""

    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        # Create figure with GridSpec layout: intensity & FLIM side-by-side | decay below + colorbar
        from matplotlib.gridspec import GridSpec
        self._fig = Figure(figsize=(10, 8), dpi=100, facecolor="#ffffff")
        # 2 rows, 3 cols: col 0 & 1 for images, col 2 for colorbar
        gs = GridSpec(2, 3, figure=self._fig, height_ratios=[1, 0.6], width_ratios=[1, 1, 0.05], hspace=0.3, wspace=0.15)
        
        self._ax_img = self._fig.add_subplot(gs[0, 0])    # Intensity (top-left)
        self._ax_flim = self._fig.add_subplot(gs[0, 1])   # FLIM (top-right)
        self._ax_cbar = self._fig.add_subplot(gs[0, 2])   # Colorbar (top-right, narrow)
        self._ax_decay = self._fig.add_subplot(gs[1, :])  # Decay (bottom, full width)
        
        self._canvas_mpl = FigureCanvasTkAgg(self._fig, master=self.frame)
        self._canvas_mpl.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Status label
        self._status = tk.StringVar(value="No FOV loaded")
        ttk.Label(self.frame, textvariable=self._status, foreground="grey", font=("Courier", 8)).grid(
            row=1, column=0, sticky="w", padx=4, pady=(2, 4))

        # ── FLIM Color Scale Controls (Phase 2.2) ──
        ctrl_frame = ttk.LabelFrame(self.frame, text="FLIM Color Scale", padding=4)
        ctrl_frame.grid(row=2, column=0, sticky="ew", padx=4, pady=(0, 4))
        ctrl_frame.columnconfigure(1, weight=1)
        ctrl_frame.grid_remove()  # Hide initially for faster startup
        self._ctrl_frame = ctrl_frame
        
        # Row 0: Min/Max lifetime
        ttk.Label(ctrl_frame, text="τ range (ns):").grid(row=0, column=0, sticky="w")
        ttk.Label(ctrl_frame, text="Min:").grid(row=0, column=1, sticky="w", padx=(10, 2))
        self._sv_tau_min = tk.StringVar()
        ttk.Entry(ctrl_frame, textvariable=self._sv_tau_min, width=6).grid(row=0, column=2, sticky="w", padx=2)
        ttk.Label(ctrl_frame, text="Max:").grid(row=0, column=3, sticky="w", padx=(10, 2))
        self._sv_tau_max = tk.StringVar()
        ttk.Entry(ctrl_frame, textvariable=self._sv_tau_max, width=6).grid(row=0, column=4, sticky="w", padx=2)
        ttk.Button(ctrl_frame, text="Auto", width=6, command=self._auto_detect_scale).grid(row=0, column=5, sticky="w", padx=2)
        
        # Row 1: Gamma & Colormap
        ttk.Label(ctrl_frame, text="Γ:").grid(row=1, column=0, sticky="w")
        self._sv_gamma = tk.StringVar(value="1.0")
        ttk.Entry(ctrl_frame, textvariable=self._sv_gamma, width=6).grid(row=1, column=2, sticky="w", padx=2)
        
        ttk.Label(ctrl_frame, text="Colormap:").grid(row=1, column=3, sticky="w", padx=(10, 2))
        self._sv_cmap = tk.StringVar(value="viridis")
        self._cmap_combo = ttk.Combobox(ctrl_frame, textvariable=self._sv_cmap, 
                                 state="readonly", width=10)
        self._cmap_combo.grid(row=1, column=4, sticky="w", padx=2)
        
        # Defer populating colormap options
        self._cmap_combo['values'] = list(flim_display.COLORMAPS.keys())
        
        ttk.Button(ctrl_frame, text="Update", width=8, command=self._update_flim_display).grid(row=1, column=5, sticky="w", padx=2)

        self._ptu_path = None
        
        # ── FLIM image display state (Phase 1) ──
        self._lifetime_map = None  # Cached intensity-weighted lifetime map
        self._intensity_map = None  # Cached intensity (photon count) map
        self._flim_cbar = None  # Track colorbar for cleanup
        self._flim_color_scale = {  # Color scale parameters
            'vmin': None,   # Auto-detect
            'vmax': None,   # Auto-detect
            'gamma': 1.0,   # Linear
            'cmap': 'viridis',
        }
        self._n_exp = 1  # Number of exponential components in last fit

    def load_fov(self, ptu_path: Optional[str]):
        """Load and display intensity image + decay curve from PTU file."""
        if not ptu_path or not Path(ptu_path).exists():
            self._clear()
            self._status.set("Invalid PTU file")
            return

        try:
            self._ptu_path = ptu_path
            from flimkit.PTU.reader import PTUFile
            import numpy as np

            ptu = PTUFile(ptu_path, verbose=False)
            
            # Get intensity image
            stack = ptu.pixel_stack(channel=None, binning=1)
            intensity = stack.sum(axis=2)  # Sum over time bins
            
            # Get decay curve
            decay = ptu.summed_decay(channel=None)
            time_ns = ptu.time_ns

            # Plot intensity image
            self._ax_img.clear()
            # Clip at 99th percentile for better contrast
            intensity_clipped = np.clip(intensity, 0, np.percentile(intensity, 99))
            self._ax_img.imshow(intensity_clipped, cmap="inferno", origin="upper")
            self._ax_img.set_title("Intensity Image", fontsize=10, fontweight="bold")
            self._ax_img.set_xlabel("X (pixels)")
            self._ax_img.set_ylabel("Y (pixels)")
            self._ax_img.tick_params(labelsize=8)

            # Clear FLIM axes (no fit data yet)
            self._ax_flim.clear()
            self._ax_flim.text(0.5, 0.5, "Waiting for fit...", ha='center', va='center',
                              transform=self._ax_flim.transAxes, fontsize=9, color='#888')
            self._ax_flim.set_title("FLIM Lifetime", fontsize=10, fontweight="bold")

            # Plot decay curve
            self._ax_decay.clear()
            self._ax_decay.semilogy(time_ns, decay, color="steelblue", linewidth=1.5)
            self._ax_decay.set_title("Summed Decay", fontsize=10, fontweight="bold")
            self._ax_decay.set_xlabel("Time (ns)")
            self._ax_decay.set_ylabel("Photon Count")
            self._ax_decay.grid(True, alpha=0.3)
            self._ax_decay.tick_params(labelsize=8)

            self._canvas_mpl.draw_idle()

            n_photons = int(decay.sum())
            img_shape = intensity.shape
            self._status.set(f"✓ {Path(ptu_path).name} | {img_shape[0]}×{img_shape[1]}px | {n_photons} photons")

        except Exception as e:
            self._clear()
            self._status.set(f"Error loading FOV: {str(e)[:50]}")

    def display_fit_results(self, ptu_path: str, fit_result: dict):
        """Display fit results with IRF and fitted decay overlaid on summed decay."""
        try:
            from flimkit.PTU.reader import PTUFile
            import numpy as np

            # Get data from fit result
            global_summary = fit_result.get('global_summary', {})
            global_popt = fit_result.get('global_popt')
            irf_prompt = fit_result.get('irf_prompt')
            time_ns_from_result = fit_result.get('time_ns')
            decay_from_result = fit_result.get('decay')
            canvas = fit_result.get('canvas')
            
            # (debug prints removed)
            
            # Use result data if available, otherwise load from PTU
            if decay_from_result is not None and time_ns_from_result is not None:
                decay = decay_from_result
                time_ns = time_ns_from_result
                print(f"  - Using decay/time_ns from fit result")
            else:
                if ptu_path and Path(ptu_path).exists():
                    ptu = PTUFile(ptu_path, verbose=False)
                    decay = ptu.summed_decay(channel=None)
                    time_ns = ptu.time_ns
                    print(f"  - Loaded decay/time_ns from PTU file")
                else:
                    print(f"  - No valid PTU path for decay loading")
                    decay = None
                    time_ns = None
            
            # Get intensity image: prefer canvas (from tile fitting), fallback to PTU
            intensity = None
            if canvas is not None and 'intensity' in canvas:
                intensity = canvas['intensity']
                print(f"  - Using intensity from canvas (tile fitting)")
            elif ptu_path and Path(ptu_path).exists():
                ptu = PTUFile(ptu_path, verbose=False)
                stack = ptu.pixel_stack(channel=None, binning=1)
                intensity = stack.sum(axis=2)
                print(f"  - Loaded intensity from PTU file")
            
            if intensity is None:
                intensity = np.ones((512, 512), dtype=np.float32)  # Placeholder
                print(f"  - No intensity data available, using placeholder")
            
            # ── Compute FLIM lifetime map (Phase 1) ──
            from flimkit.UI.flim_display import compute_intensity_weighted_lifetime
            
            pixel_maps = fit_result.get('pixel_maps')  # For single-FOV fits
            if pixel_maps is None and canvas is not None:
                # For tile fits, extract pixel_maps from canvas
                pixel_maps = {k: v for k, v in canvas.items() 
                             if k not in ('intensity', 'coverage')}
            
            nexp = global_summary.get('n_exp', len(global_summary.get('taus_ns', [])))
            lifetime_map = None
            
            if pixel_maps and nexp > 0:
                try:
                    lifetime_map = compute_intensity_weighted_lifetime(
                        pixel_maps, intensity, n_exp=nexp
                    )
                    print(f"  - Computed intensity-weighted lifetime map ({lifetime_map.shape})")
                    print(f"    Lifetime map range: {np.nanmin(lifetime_map):.4f} – {np.nanmax(lifetime_map):.4f} ns")
                    print(f"    pixel_maps keys: {list(pixel_maps.keys())[:10]}")  # Show first 10 keys
                except Exception as e:
                    print(f"  - Warning: Could not compute lifetime map: {e}")
                    import traceback
                    traceback.print_exc()
                    lifetime_map = None
            
            # Cache for interactive updates (Phase 2+)
            self._lifetime_map = lifetime_map
            self._intensity_map = intensity
            self._n_exp = nexp
            
            # Extract fit data
            taus_fit = global_summary.get('taus_ns', [])
            nexp = len(taus_fit)
            print(f"  - nexp: {nexp}, taus_ns: {taus_fit}")
            print(f"  - global_summary keys: {list(global_summary.keys())}")
            
            # Debug: check for model
            model = global_summary.get('model')
            print(f"  - model in global_summary: {model is not None}")
            if model is not None:
                print(f"    model shape: {model.shape if hasattr(model, 'shape') else len(model)}")
                print(f"    model min/max: {model.min():.2e} / {model.max():.2e}")
            
            
            # Update intensity image
            self._ax_img.clear()
            # Clip at 99th percentile for better contrast
            intensity_clipped = np.clip(intensity, 0, np.percentile(intensity, 99))
            self._ax_img.imshow(intensity_clipped, cmap="inferno", origin="upper")
            self._ax_img.set_title("Intensity Image", fontsize=10, fontweight="bold")
            self._ax_img.set_xlabel("X (pixels)")
            self._ax_img.set_ylabel("Y (pixels)")
            self._ax_img.tick_params(labelsize=8)

            # Update FLIM lifetime image (Phase 2.1)
            self._ax_flim.clear()
            if self._lifetime_map is not None and np.any(~np.isnan(self._lifetime_map)):
                # Apply color scaling
                scaled = flim_display.apply_color_scale(
                    self._lifetime_map,
                    vmin=self._flim_color_scale['vmin'],
                    vmax=self._flim_color_scale['vmax'],
                    gamma=self._flim_color_scale['gamma'],
                )
                
                # Get colormap and set NaN to black
                cmap = flim_display.get_colormap(self._flim_color_scale['cmap'])
                cmap.set_bad(color='black')
                
                im = self._ax_flim.imshow(scaled, cmap=cmap, origin="upper", vmin=0, vmax=1)
                self._ax_flim.set_title("FLIM Lifetime (ns)", fontsize=10, fontweight="bold")
                self._ax_flim.set_xlabel("X (pixels)")
                self._ax_flim.set_ylabel("Y (pixels)")
                self._ax_flim.tick_params(labelsize=8)
                
                # Colorbar with actual data range from lifetime map
                valid_data = self._lifetime_map[~np.isnan(self._lifetime_map)]
                if valid_data.size > 0:
                    data_min = np.min(valid_data)
                    data_max = np.max(valid_data)
                    
                    # Clear colorbar axes
                    self._ax_cbar.clear()
                    
                    # Create colorbar using dedicated axes
                    cbar = self._fig.colorbar(im, cax=self._ax_cbar)
                    cbar.set_label(f"τ (ns)", fontsize=8)
                    self._flim_cbar = cbar
                    
                    # Manually set colorbar tick labels to lifetime values
                    n_ticks = 5
                    tick_positions = np.linspace(0, 1, n_ticks)
                    tick_values = data_min + tick_positions * (data_max - data_min)
                    cbar.set_ticks(tick_positions)
                    cbar.set_ticklabels([f"{v:.2f}" for v in tick_values], fontsize=7)
                else:
                    self._ax_cbar.clear()
            else:
                self._ax_flim.text(0.5, 0.6, "No FLIM data", ha='center', va='center',
                                  transform=self._ax_flim.transAxes, fontsize=9, color='#888')
                self._ax_flim.text(0.5, 0.35, "(enable per-pixel fitting)", ha='center', va='center',
                                  transform=self._ax_flim.transAxes, fontsize=8, color='#666', style='italic')
                self._ax_flim.set_title("FLIM Lifetime", fontsize=10, fontweight="bold")

            # Plot decay with fit and IRF
            self._ax_decay.clear()
            
            if decay is None or len(decay) == 0:
                print(f"  - No decay data available for plotting")
                self._ax_decay.text(0.5, 0.5, "No decay data", ha='center', va='center',
                                  transform=self._ax_decay.transAxes)
            else:
                # Plot measured decay
                print(f"  - Plotting measured decay: {len(decay)} points, {len(time_ns)} time points")
                self._ax_decay.semilogy(time_ns, decay, 'o-', color="steelblue", 
                                        linewidth=1.5, markersize=3, label="Measured", alpha=0.7)
                
                # Plot IRF if available
                if irf_prompt is not None and len(irf_prompt) > 0:
                    irf_max = irf_prompt.max()
                    if irf_max > 0:
                        # Scale IRF to ~20% of max decay for visibility
                        irf_scaled = (irf_prompt / irf_max) * decay.max() * 0.2
                        irf_time = time_ns[:len(irf_prompt)]
                        print(f"  - Plotting IRF: {len(irf_prompt)} points")
                        self._ax_decay.semilogy(irf_time, np.maximum(irf_scaled, 1e-2), 
                                              linewidth=2.0, color="orange", label="IRF", alpha=0.8)
                
                # Plot fitted decay if we have parameters or model
                model = global_summary.get('model')
                if model is not None and len(model) > 0:
                    print(f"  - Plotting fitted model: {len(model)} points")
                    self._ax_decay.semilogy(time_ns, model, linewidth=2.0, 
                                          color="red", label="Fitted", alpha=0.8)
                elif global_popt is not None and nexp > 0:
                    print(f"  - Have global_popt but no precomputed model (model={model})")
                else:
                    print(f"  - No global_popt ({global_popt is not None}) or nexp<=0 ({nexp}) for model")
            
            self._ax_decay.set_title(f"Summed Decay{f' ({nexp}-exp fit)' if nexp > 0 else ''}", 
                                    fontsize=10, fontweight="bold")
            self._ax_decay.set_xlabel("Time (ns)")
            self._ax_decay.set_ylabel("Photon Count")
            if decay is not None and len(decay) > 0:
                self._ax_decay.legend(fontsize=8, loc="upper right")
            self._ax_decay.grid(True, alpha=0.3)
            self._ax_decay.tick_params(labelsize=8)

            # Show control frame now that we have FLIM data
            self._ctrl_frame.grid()
            
            self._canvas_mpl.draw_idle()

            # Update status with fit summary
            status = f"✓ Fit complete"
            chi2_tail = global_summary.get('reduced_chi2_tail')
            if chi2_tail is not None:
                status += f" | χ²_r(tail)={chi2_tail:.3f}"
            if nexp > 0:
                taus = [global_summary.get(f'taus_ns', [])[i] if i < len(global_summary.get('taus_ns', [])) else None 
                        for i in range(nexp)]
                taus_str = ", ".join([f"{t:.3f}" for t in taus if t is not None])
                status += f" | τ=[{taus_str}] ns"
            self._status.set(status)
            print(f"  - Status: {status}")

        except Exception as e:
            import traceback
            print(f"[FOV Preview] Error displaying fit results:")
            traceback.print_exc()
            self._status.set(f"Error: {str(e)[:60]}")
            self._status.set(f"Error displaying fit: {str(e)[:50]}")

    def load_stitched_roi(self, output_dir: str):
        """Load and display stitched ROI intensity image."""
        if not output_dir:
            self._clear()
            self._status.set("No output directory")
            return
        
        try:
            from pathlib import Path
            import numpy as np
            import tifffile
            
            out_path = Path(output_dir)
            
            # Find the stitched intensity TIFF file
            intensity_files = sorted(out_path.glob("*_stitched_intensity.tif"))
            if not intensity_files:
                self._clear()
                self._status.set("No stitched image found")
                return
            
            intensity_file = intensity_files[0]
            intensity = tifffile.imread(str(intensity_file))
            
            # Clear axes and display stitched image
            self._ax_img.clear()
            # Clip at 99th percentile for better contrast
            intensity_clipped = np.clip(intensity, 0, np.percentile(intensity, 99))
            self._ax_img.imshow(intensity_clipped, cmap="inferno", origin="upper")
            self._ax_img.set_title("Stitched ROI", fontsize=10, fontweight="bold")
            self._ax_img.set_xlabel("X (pixels)")
            self._ax_img.set_ylabel("Y (pixels)")
            self._ax_img.tick_params(labelsize=8)
            
            # Clear FLIM axes
            self._ax_flim.clear()
            self._ax_flim.text(0.5, 0.5, "Waiting for fit...", ha='center', va='center',
                              transform=self._ax_flim.transAxes, fontsize=9, color='#888')
            self._ax_flim.set_title("FLIM Lifetime", fontsize=10, fontweight="bold")
            
            # Clear decay plot
            self._ax_decay.clear()
            self._ax_decay.text(0.5, 0.5, "Stitch complete", 
                               ha="center", va="center", transform=self._ax_decay.transAxes,
                               fontsize=10, color="steelblue")
            
            self._canvas_mpl.draw_idle()
            
            img_shape = intensity.shape
            self._status.set(f"✓ Stitched ROI | {img_shape[0]}×{img_shape[1]}px")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._clear()
            self._status.set(f"Error loading stitched: {str(e)[:50]}")

    def _clear(self):
        """Clear all axes."""
        self._ax_img.clear()
        self._ax_flim.clear()
        self._ax_decay.clear()
        self._ax_cbar.clear()
        self._flim_cbar = None
        self._ax_img.set_title("No FOV loaded")
        self._ax_flim.set_title("FLIM Lifetime")
        self._ax_decay.text(0.5, 0.5, "Load a PTU file →", 
                           ha="center", va="center", transform=self._ax_decay.transAxes,
                           fontsize=10, color="#888")
        self._ctrl_frame.grid_remove()  # Hide controls when clearing
        self._canvas_mpl.draw_idle()

    def _auto_detect_scale(self):
        import numpy as np
        
        if self._lifetime_map is None:
            return
        valid_data = self._lifetime_map[~np.isnan(self._lifetime_map)]
        if valid_data.size > 0:
            vmin = np.percentile(valid_data, 2)
            vmax = np.percentile(valid_data, 98)
            self._sv_tau_min.set(f"{vmin:.2f}")
            self._sv_tau_max.set(f"{vmax:.2f}")
            self._update_flim_display()

    def _update_flim_display(self):
        import numpy as np
        
        if self._lifetime_map is None or not np.any(~np.isnan(self._lifetime_map)):
            return
        
        try:
            # Parse user inputs
            try:
                vmin = float(self._sv_tau_min.get()) if self._sv_tau_min.get() else None
            except ValueError:
                vmin = None
            try:
                vmax = float(self._sv_tau_max.get()) if self._sv_tau_max.get() else None
            except ValueError:
                vmax = None
            try:
                gamma = float(self._sv_gamma.get())
                if gamma <= 0:
                    gamma = 1.0
            except ValueError:
                gamma = 1.0
            
            cmap_name = self._sv_cmap.get()
            
            # Update color scale cache
            self._flim_color_scale['vmin'] = vmin
            self._flim_color_scale['vmax'] = vmax
            self._flim_color_scale['gamma'] = gamma
            self._flim_color_scale['cmap'] = cmap_name
            
            # Recompute scaled image
            scaled = flim_display.apply_color_scale(
                self._lifetime_map, vmin=vmin, vmax=vmax, gamma=gamma
            )
            
            # Redraw FLIM axes
            self._ax_flim.clear()
            # Clear colorbar axes
            self._ax_cbar.clear()
            self._flim_cbar = None
            cmap = flim_display.get_colormap(cmap_name)
            cmap.set_bad(color='black')
            
            im = self._ax_flim.imshow(scaled, cmap=cmap, origin="upper", vmin=0, vmax=1)
            self._ax_flim.set_title("FLIM Lifetime (ns)", fontsize=10, fontweight="bold")
            self._ax_flim.set_xlabel("X (pixels)")
            self._ax_flim.set_ylabel("Y (pixels)")
            self._ax_flim.tick_params(labelsize=8)
            
            # Update colorbar
            valid_data = self._lifetime_map[~np.isnan(self._lifetime_map)]
            if valid_data.size > 0:
                data_min = vmin if vmin is not None else np.min(valid_data)
                data_max = vmax if vmax is not None else np.max(valid_data)
                
                # Clear and reuse dedicated colorbar axes
                self._ax_cbar.clear()
                cbar = self._fig.colorbar(im, cax=self._ax_cbar)
                cbar.set_label("τ (ns)", fontsize=8)
                self._flim_cbar = cbar
                
                n_ticks = 5
                tick_positions = np.linspace(0, 1, n_ticks)
                tick_values = data_min + tick_positions * (data_max - data_min)
                cbar.set_ticks(tick_positions)
                cbar.set_ticklabels([f"{v:.2f}" for v in tick_values], fontsize=7)
            else:
                self._ax_cbar.clear()
            
            self._canvas_mpl.draw_idle()
        except Exception as e:
            print(f"Error updating FLIM display: {e}")

    def grid(self, **kw):
        self.frame.grid(**kw)


# ─────────────────────────────────────────────────────────────────────────────
# Results panel  –  Progress  |  Fit Summary  |  Images
# ─────────────────────────────────────────────────────────────────────────────

class ResultsPanel:

    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        self._nb = ttk.Notebook(self.frame)
        self._nb.grid(row=0, column=0, sticky="nsew")

        self._build_progress()
        self._build_summary()

        self._status = tk.StringVar(value="Ready.")
        ttk.Label(self.frame, textvariable=self._status, foreground="grey").grid(
            row=1, column=0, sticky="w", padx=4, pady=(2, 4))

    def _build_progress(self):
        f = ttk.Frame(self._nb, padding=4)
        self._nb.add(f, text="  Progress  ")
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)
        self.log = scrolledtext.ScrolledText(
            f, state="disabled", wrap="word",
            font=("Courier", 9), background="#1e1e1e", foreground="#d4d4d4")
        self.log.grid(row=0, column=0, sticky="nsew")

        btn_bar = ttk.Frame(f)
        btn_bar.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(btn_bar, text="Save log…", command=self._save_log).pack(side="left",  padx=4)
        ttk.Button(btn_bar, text="Clear log", command=self._clear_log).pack(side="right", padx=4)

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", tk.END)
        self.log.configure(state="disabled")

    def _save_log(self):
        text = self.log.get("1.0", tk.END)
        if not text.strip():
            messagebox.showinfo("Nothing to save", "The log is empty.")
            return
        path = filedialog.asksaveasfilename(
            title="Save log as…",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All", "*.*")])
        if path:
            Path(path).write_text(text, encoding="utf-8")
            self._status.set(f"Log saved → {Path(path).name}")

    def _build_summary(self):
        f = ttk.Frame(self._nb, padding=4)
        self._nb.add(f, text="  Fit Summary  ")
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)

        cols = ("Parameter", "Value", "Unit")
        tv = ttk.Treeview(f, columns=cols, show="headings")
        tv.heading("Parameter", text="Parameter", anchor="w")
        tv.heading("Value",     text="Value",     anchor="e")
        tv.heading("Unit",      text="Unit",      anchor="w")
        tv.column("Parameter", width=300, anchor="w", stretch=True)
        tv.column("Value",     width=110, anchor="e", stretch=False)
        tv.column("Unit",      width=70,  anchor="w", stretch=False)

        sb = ttk.Scrollbar(f, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        tv.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        tv.tag_configure("odd",  background="#f5f7fa", foreground="#000000")
        tv.tag_configure("even", background="#ffffff", foreground="#000000")
        tv.tag_configure("warn", foreground="#c0550a", background="#fff8f0")
        self._tv = tv

    def populate_summary(self, rows: list):
        for item in self._tv.get_children():
            self._tv.delete(item)
        
        # Force layout to ensure widget has proper dimensions
        self._tv.update()
        
        for i, (param, val, unit) in enumerate(rows):
            tag = "warn" if param.startswith('⚠') else ("odd" if i % 2 else "even")
            self._tv.insert("", tk.END, values=(param, val, unit), tags=(tag,))
        
        # Force final redraw
        self._tv.update_idletasks()
        
        if rows:
            self._nb.select(1)
            self._nb.update_idletasks()

    def set_status(self, msg: str):
        self._status.set(msg)

    def grid(self, **kw):
        self.frame.grid(**kw)

    def load_images(self, folder: Optional[str]):
        self._imgs = []
        if folder and Path(folder).is_dir():
            self._folder = folder
            for pat in ("*.png", "*.tif", "*.tiff"):
                self._imgs += sorted(Path(folder).glob(pat))
        self._img_i = 0
        self._draw_img()
        if self._imgs:
            self._nb.select(2)

    def _draw_img(self):
        self._ax.cla()
        self._ax.set_facecolor("#2b2b2b")
        self._ax.axis("off")
        if not self._imgs:
            self._img_lbl.set("No images found")
            self._ax.text(0.5, 0.5, "No images found",
                          ha="center", va="center", color="grey", fontsize=11,
                          transform=self._ax.transAxes)
        else:
            path = self._imgs[self._img_i]
            self._img_lbl.set(f"{path.name}  ({self._img_i + 1}/{len(self._imgs)})")
            try:
                img = mpimg.imread(str(path))
                self._ax.imshow(img, aspect="equal")
            except Exception as e:
                self._ax.text(0.5, 0.5, f"Cannot load image:\n{e}",
                              ha="center", va="center", color="red",
                              fontsize=9, transform=self._ax.transAxes)
        self._canvas_mpl.draw_idle()

    def _img_prev(self):
        if self._imgs:
            self._img_i = (self._img_i - 1) % len(self._imgs)
            self._draw_img()

    def _img_next(self):
        if self._imgs:
            self._img_i = (self._img_i + 1) % len(self._imgs)
            self._draw_img()

    def _save_img(self):
        """Save the currently displayed image to a user-chosen file."""
        if not self._imgs:
            messagebox.showinfo("No image", "No image is currently displayed.")
            return
        src = self._imgs[self._img_i]
        path = filedialog.asksaveasfilename(
            title="Save current image as…",
            initialfile=src.name,
            defaultextension=src.suffix,
            filetypes=[
                ("PNG",  "*.png"),
                ("TIFF", "*.tif *.tiff"),
                ("All",  "*.*"),
            ])
        if path:
            import shutil
            shutil.copy2(str(src), path)
            self._status.set(f"Image saved → {Path(path).name}")

    def _save_all_imgs(self):
        """Copy all output images to a user-chosen directory."""
        if not self._imgs:
            messagebox.showinfo("No images", "No images are available to save.")
            return
        dest = filedialog.askdirectory(title="Save all images to…")
        if not dest:
            return
        import shutil
        dest_path = Path(dest)
        for img in self._imgs:
            shutil.copy2(str(img), str(dest_path / img.name))
        self._status.set(f"{len(self._imgs)} image(s) saved → {dest_path.name}/")

    def _open_folder(self):
        import subprocess, platform
        if not self._folder or not Path(self._folder).exists():
            messagebox.showinfo("No folder", "No output folder available yet.")
            return
        s = platform.system()
        if   s == "Darwin":  subprocess.Popen(["open",     self._folder])
        elif s == "Windows": subprocess.Popen(["explorer", self._folder])
        else:                subprocess.Popen(["xdg-open", self._folder])

    def set_status(self, msg: str):
        self._status.set(msg)

    def grid(self, **kw):
        self.frame.grid(**kw)


# ─────────────────────────────────────────────────────────────────────────────
# Base class for the GUI (shared layout)
# ─────────────────────────────────────────────────────────────────────────────

class _UIBuilder:
    """Common UI building methods (used by both themed and fallback versions)."""

    def _make_scroll_tab(self, title: str) -> ttk.Frame:
        """Create a notebook tab whose contents are vertically scrollable."""
        outer = ttk.Frame(self._nb)
        self._nb.add(outer, text=title)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0)
        vbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")

        inner = ttk.Frame(canvas, padding=10)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_evt=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(evt):
            canvas.itemconfigure(window_id, width=evt.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        return inner

    def _fit_window_to_screen(self):
        """Clamp window geometry to visible screen bounds."""
        self.root.update_idletasks()
        sw = max(1, int(self.root.winfo_screenwidth()))
        sh = max(1, int(self.root.winfo_screenheight()))
        max_w = max(1200, sw - 40)  # Larger to accommodate preview panel
        max_h = max(700, sh - 40)
        req_w = int(self.root.winfo_reqwidth())
        req_h = int(self.root.winfo_reqheight())
        cur_w = int(self.root.winfo_width())
        cur_h = int(self.root.winfo_height())
        target_w = min(max(cur_w, req_w, 1200), max_w)
        target_h = min(max(cur_h, req_h, 800), max_h)
        x = int(self.root.winfo_x())
        y = int(self.root.winfo_y())
        x = min(max(0, x), max(0, sw - target_w))
        y = min(max(0, y), max(0, sh - target_h))
        self.root.maxsize(sw, sh)
        self.root.geometry(f"{target_w}x{target_h}+{x}+{y}")

    def run_with_progress(self, task_fn, task_name="Working…", on_done=None, output_dir=None):
        """Run a function in a thread, showing a pop-out progress window with cancel."""
        win = ProgressWindow(self.root, task_name=task_name)
        cancel_event = win.cancelled

        def progress_callback(i, total):
            win.set_progress(i, maximum=total)
            if cancel_event.is_set():
                win.set_status("Cancelling…")

        def worker():
            orig_stdout, orig_stderr = sys.stdout, sys.stderr
            # Always redirect to UI's ScrolledText widget with thread-safe updates
            redir = _Redirect(self._res.log, self._buf, root=self.root)
            sys.stdout = redir
            sys.stderr = redir
            try:
                result = task_fn(progress_callback, cancel_event)
                self.root.after(0, lambda: win.close())
                if on_done:
                    self.root.after(0, lambda: on_done(result))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: win.set_status(f"Error: {exc}"))
                self.root.after(0, lambda: win.btn_cancel.config(text="Close", command=win.close))
            finally:
                if hasattr(redir, 'close'):
                    redir.close()
                else:
                    redir.flush()
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr

        threading.Thread(target=worker, daemon=True).start()

    def _init_ui(self):
        """Build the entire user interface with horizontal layout (left: tabs+results, right: FOV preview)."""
        self._buf: list = []

        # Main horizontal PanedWindow: left (tabs+results) | right (FOV preview)
        main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ─────────────────────────────────────────────────────────────────
        # LEFT PANEL: Vertical layout with tabs (top) and results (bottom)
        # ─────────────────────────────────────────────────────────────────
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=3)  # 3:2 weight ratio
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)

        # Vertical PanedWindow for left side
        left_paned = ttk.PanedWindow(left_frame, orient="vertical")
        left_paned.grid(row=0, column=0, sticky="nsew")
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)

        # Top: Notebook with fitting options
        nb_frame = ttk.Frame(left_paned)
        left_paned.add(nb_frame, weight=2)
        nb_frame.columnconfigure(0, weight=1)
        nb_frame.rowconfigure(0, weight=1)

        self._nb = ttk.Notebook(nb_frame)
        self._nb.grid(row=0, column=0, sticky="nsew")

        self._build_fov_tab()
        self._build_stitch_tab()
        self._build_batch_tab()
        self._build_machine_irf_tab()
        self._build_phasor_tab()

        # Bottom: Results panel (progress, summary, images)
        results_frame = ttk.Frame(left_paned)
        left_paned.add(results_frame, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self._res = ResultsPanel(results_frame)
        self._res.grid(row=0, column=0, sticky="nsew")

        # ─────────────────────────────────────────────────────────────────
        # RIGHT PANEL: FOV Preview
        # ─────────────────────────────────────────────────────────────────
        preview_frame = ttk.LabelFrame(main_paned, text="  FOV Preview  ", padding=4)
        main_paned.add(preview_frame, weight=2)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self._fov_preview = FOVPreviewPanel(preview_frame)
        self._fov_preview.grid(row=0, column=0, sticky="nsew")

        # Phasor panel shares the same right-panel cell; hidden until needed.
        self._phasor_panel = PhasorViewPanel(preview_frame, max_cursors=6)
        self._phasor_panel.frame.grid(row=0, column=0, sticky="nsew")
        self._phasor_panel.frame.grid_remove()

        # Swap right-panel content when the active tab changes.
        def _on_tab_changed(event):
            try:
                tab_text = self._nb.tab(self._nb.select(), "text").strip()
            except Exception:
                return
            if tab_text == "Phasor Analysis":
                self._fov_preview.frame.grid_remove()
                self._phasor_panel.frame.grid()
                preview_frame.configure(text="  Phasor Analysis  ")
            else:
                self._phasor_panel.frame.grid_remove()
                self._fov_preview.frame.grid()
                preview_frame.configure(text="  FOV Preview  ")

        self._nb.bind("<<NotebookTabChanged>>", _on_tab_changed)

        # Redirect stdout/stderr to the log widget
        redir = _Redirect(self._res.log, self._buf, root=self.root)
        sys.stdout = redir
        sys.stderr = redir

        # Ensure initial window fits within screen bounds
        self.root.after_idle(self._fit_window_to_screen)

        # Set close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Set window icon if available
        self._set_window_icon()

    def _set_window_icon(self):
        base_path = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(__file__).parent
        icon_paths = [
            base_path / "flimkit" / "icon.png",
            base_path / "icon.png",
            Path(__file__).parent / "flimkit" / "icon.png",
            Path(__file__).parent / "icon.png",
        ]
        for icon_path in icon_paths:
            if icon_path.exists():
                try:
                    from PIL import Image, ImageTk
                    icon_img = Image.open(str(icon_path))
                    icon_img.thumbnail((32, 32), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(icon_img)
                    self.root.iconphoto(False, photo)
                    self.root._icon_photo = photo
                    break
                except Exception as e:
                    print(f"Warning: Could not load icon from {icon_path}: {e}")
                    continue

    def _on_close(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.root.destroy()

    # -------------------------------------------------------------------------
    # TAB 1 – Single-FOV FLIM fit
    # -------------------------------------------------------------------------
    def _build_fov_tab(self):
        tab = self._make_scroll_tab("  Single FOV Fit  ")
        tab.columnconfigure(0, weight=1)

        ff = _section(tab, "Input Files")
        ff.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ff.columnconfigure(1, weight=1)
        self.sv_ptu  = tk.StringVar()
        self.sv_xlsx = tk.StringVar()
        
        # Track PTU file changes to auto-load preview
        self.sv_ptu.trace_add("write", self._on_fov_ptu_changed)
        
        _row(ff, "PTU file *", self.sv_ptu, 0,
             lambda: _browse_file(self.sv_ptu, "PTU file",
                                  [("PTU", "*.ptu"), ("All", "*.*")]))
        _row(ff, "XLSX file (optional)", self.sv_xlsx, 1,
             lambda: _browse_file(self.sv_xlsx, "XLSX file",
                                  [("Excel", "*.xlsx"), ("All", "*.*")]))

        self._irf_fov = IRFWidget(tab, default="irf_xlsx", xlsx_var=self.sv_xlsx,
                                   machine_irf_default=str(_C()["MACHINE_IRF_DEFAULT_PATH"]))
        self._irf_fov.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        fp = _section(tab, "Fitting Parameters")
        fp.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        ttk.Label(fp, text="Exponential components:").grid(
            row=0, column=0, sticky="w", **PAD)
        self.iv_nexp_fov = tk.IntVar(value=2)
        for n in (1, 2, 3):
            ttk.Radiobutton(fp, text=str(n), variable=self.iv_nexp_fov,
                            value=n).grid(row=0, column=n, sticky="w", padx=4)

        ttk.Label(fp, text="Fitting mode:").grid(row=1, column=0, sticky="w", **PAD)
        self.sv_mode_fov = tk.StringVar(value="summed")
        for c, (lbl, val) in enumerate(
                [("Summed only", "summed"), ("Per-pixel", "perPixel"), ("Both", "both")], 1):
            ttk.Radiobutton(fp, text=lbl, variable=self.sv_mode_fov,
                            value=val).grid(row=1, column=c, sticky="w", padx=4)

        ttk.Label(fp, text="Fit window (ns):").grid(row=2, column=0, sticky="w", **PAD)
        self.sv_tau_min_fov = tk.StringVar(value=str(_C()["Tau_min"]))
        self.sv_tau_max_fov = tk.StringVar(value=str(_C()["Tau_max"]))
        ttk.Entry(fp, textvariable=self.sv_tau_min_fov, width=7).grid(row=2, column=1, sticky="w", padx=4)
        ttk.Label(fp, text="to").grid(row=2, column=2)
        ttk.Entry(fp, textvariable=self.sv_tau_max_fov, width=7).grid(row=2, column=3, sticky="w", padx=4)
        ttk.Label(fp, text="ns  (fitting range)", foreground="grey").grid(row=2, column=4, sticky="w")

        ttk.Label(fp, text="Output prefix:").grid(row=3, column=0, sticky="w", **PAD)
        self.sv_out_fov = tk.StringVar(value="flim_out")
        ttk.Entry(fp, textvariable=self.sv_out_fov, width=35).grid(
            row=3, column=1, columnspan=3, sticky="ew", padx=4)

        fm = _section(tab, "Masking & Thresholding")
        fm.grid(row=3, column=0, sticky="ew", pady=(0, 6))

        self.bv_cell = tk.BooleanVar(value=False)
        ttk.Checkbutton(fm, text="Apply cell mask (Otsu on intensity image)",
                        variable=self.bv_cell).grid(
            row=0, column=0, columnspan=3, sticky="w", **PAD)

        self.bv_thr_fov = tk.BooleanVar(value=False)
        self.sv_thr_fov = tk.StringVar()
        ttk.Checkbutton(fm, text="Intensity threshold (min photons/px):",
                        variable=self.bv_thr_fov,
                        command=lambda: _tog(self.bv_thr_fov, self._thr_fov_e)).grid(
            row=1, column=0, sticky="w", **PAD)
        self._thr_fov_e = ttk.Entry(fm, textvariable=self.sv_thr_fov,
                                    width=8, state="disabled")
        self._thr_fov_e.grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(fm, text="(leave blank for no threshold)",
                  foreground="grey").grid(row=1, column=2, sticky="w")

        self._btn_fov = ttk.Button(tab, text="▶  Run Single-FOV Fit",
                                   command=self._run_fov)
        self._btn_fov.grid(row=4, column=0, pady=8, ipadx=20, ipady=4)

    # -------------------------------------------------------------------------
    # TAB 2 – Tile Stitch / Fit
    # -------------------------------------------------------------------------
    def _build_stitch_tab(self):
        tab = self._make_scroll_tab("  Tile Stitch / Fit  ")
        tab.columnconfigure(0, weight=1)

        ff = _section(tab, "Input Files")
        ff.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ff.columnconfigure(1, weight=1)
        self.sv_xlif    = tk.StringVar()
        self.sv_ptu_dir = tk.StringVar()
        self.sv_out_st  = tk.StringVar()

        _row(ff, "XLIF metadata *",      self.sv_xlif,    0,
             lambda: _browse_file(self.sv_xlif, "XLIF file",
                                  [("XLIF", "*.xlif"), ("All", "*.*")]))
        _row(ff, "PTU tile directory *", self.sv_ptu_dir, 1,
             lambda: _browse_dir(self.sv_ptu_dir, "PTU tile directory"))
        _row(ff, "Base output dir *",    self.sv_out_st,  2,
             lambda: _browse_dir(self.sv_out_st, "Output directory"))

        ttk.Label(ff, text="(A sub-folder named after the ROI will be created inside)",
                  foreground="grey").grid(row=3, column=1, columnspan=2,
                                         sticky="w", padx=4)
        self.bv_rotate = tk.BooleanVar(value=True)
        ttk.Checkbutton(ff, text="Rotate tiles 90° CW (recommended for Leica)",
                        variable=self.bv_rotate).grid(
            row=4, column=0, columnspan=3, sticky="w", padx=4, pady=(4, 0))

        # Pipeline mode
        fp = _section(tab, "Pipeline")
        fp.grid(row=1, column=0, sticky="ew", pady=(4, 2))
        self.sv_pipeline = tk.StringVar(value="stitch_only")
        for r, (val, lbl) in enumerate([
            ("stitch_only", "Stitch tiles only"),
            ("stitch_fit",  "Stitch then fit full ROI"),
            ("tile_fit",    "Per-tile fit  [recommended — fits each tile independently]"),
        ]):
            ttk.Radiobutton(fp, text=lbl, variable=self.sv_pipeline,
                            value=val, command=self._pipeline_changed).grid(
                row=r, column=0, sticky="w", padx=4, pady=1)

        self._fit_frame = ttk.Frame(tab)
        self._fit_frame.columnconfigure(0, weight=1)
        self._fit_frame.grid(row=2, column=0, sticky="ew")
        self._build_stitch_fit(self._fit_frame)
        self._fit_frame.grid_remove()

        self._btn_st = ttk.Button(tab, text="▶  Run Tile Stitch",
                                  command=self._run_stitch)
        self._btn_st.grid(row=3, column=0, pady=8, ipadx=20, ipady=4)

    def _build_stitch_fit(self, parent):
        self._irf_st = IRFWidget(parent, default="irf_xlsx",
                                  machine_irf_default=str(_C()["MACHINE_IRF_DEFAULT_PATH"]))
        self._irf_st.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self._irf_st.frame.columnconfigure(1, weight=1)

        fp = _section(parent, "Fitting Parameters")
        fp.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        ttk.Label(fp, text="Exponential components:").grid(
            row=0, column=0, sticky="w", **PAD)
        self.iv_nexp_st = tk.IntVar(value=2)
        for n in (1, 2, 3):
            ttk.Radiobutton(fp, text=str(n), variable=self.iv_nexp_st,
                            value=n).grid(row=0, column=n, sticky="w", padx=4)

        self.bv_perpix = tk.BooleanVar(value=False)
        ttk.Checkbutton(fp, text="Per-pixel fitting (produces lifetime maps; slower)",
                        variable=self.bv_perpix,
                        command=self._perpix_toggled).grid(
            row=1, column=0, columnspan=4, sticky="w", **PAD)

        self._pxf = ttk.Frame(fp)
        self._pxf.grid(row=2, column=0, columnspan=4, sticky="ew", padx=20)

        # Weighted map export options
        self.bv_save_tau_weighted = tk.BooleanVar(value=True)
        self.bv_save_int_weighted = tk.BooleanVar(value=True)
        self.bv_save_amp_weighted = tk.BooleanVar(value=False)
        self.bv_save_ind = tk.BooleanVar(value=False)

        ttk.Checkbutton(self._pxf, text="Export τ-weighted map",
                        variable=self.bv_save_tau_weighted).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(self._pxf, text="Export intensity-weighted map",
                        variable=self.bv_save_int_weighted).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(self._pxf, text="Export amplitude-weighted map",
                        variable=self.bv_save_amp_weighted).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(self._pxf, text="Save individual component maps (τ₁, τ₂, a₁, a₂)",
                        variable=self.bv_save_ind).grid(row=0, column=3, sticky="w")

        self.sv_tau_lo = tk.StringVar()
        self.sv_tau_hi = tk.StringVar()
        self.sv_int_lo = tk.StringVar()
        self.sv_int_hi = tk.StringVar()

        # Range controls for weighted maps
        ttk.Label(self._pxf, text="Lifetime display (ns):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(self._pxf, textvariable=self.sv_tau_lo, width=7).grid(row=1, column=1, padx=4)
        ttk.Label(self._pxf, text="to").grid(row=1, column=2)
        ttk.Entry(self._pxf, textvariable=self.sv_tau_hi, width=7).grid(row=1, column=3, padx=4)
        ttk.Label(self._pxf, text="(blank = auto)", foreground="grey").grid(row=1, column=4, padx=4)

        ttk.Label(self._pxf, text="Intensity display:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(self._pxf, textvariable=self.sv_int_lo, width=7).grid(row=2, column=1, padx=4)
        ttk.Label(self._pxf, text="to").grid(row=2, column=2)
        ttk.Entry(self._pxf, textvariable=self.sv_int_hi, width=7).grid(row=2, column=3, padx=4)
        ttk.Label(self._pxf, text="(blank = auto)", foreground="grey").grid(row=2, column=4, padx=4)

        ttk.Label(self._pxf, text="Fit window (ns):").grid(row=3, column=0, sticky="w", pady=2)
        self.sv_tau_fit_lo = tk.StringVar(value=str(_C()["Tau_min"]))
        self.sv_tau_fit_hi = tk.StringVar(value=str(_C()["Tau_max"]))
        ttk.Entry(self._pxf, textvariable=self.sv_tau_fit_lo, width=7).grid(row=3, column=1, padx=4)
        ttk.Label(self._pxf, text="to").grid(row=3, column=2)
        ttk.Entry(self._pxf, textvariable=self.sv_tau_fit_hi, width=7).grid(row=3, column=3, padx=4)
        ttk.Label(self._pxf, text="ns  (fitting range)", foreground="grey").grid(row=3, column=4, padx=4)

        self._pxf.grid_remove()

        fm = _section(parent, "Masking & Thresholding")
        fm.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        self.bv_thr_st = tk.BooleanVar(value=False)
        self.sv_thr_st = tk.StringVar()
        ttk.Checkbutton(fm, text="Intensity threshold (min photons/px):",
                        variable=self.bv_thr_st,
                        command=lambda: _tog(self.bv_thr_st, self._thr_st_e)).grid(
            row=0, column=0, sticky="w", **PAD)
        self._thr_st_e = ttk.Entry(fm, textvariable=self.sv_thr_st,
                                   width=8, state="disabled")
        self._thr_st_e.grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(fm, text="(leave blank for no threshold)",
                  foreground="grey").grid(row=0, column=2, sticky="w")

        # Registration
        freg = _section(parent, "Tile Registration")
        freg.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        self.bv_register = tk.BooleanVar(value=True)
        ttk.Checkbutton(freg, text="Phase-correlation registration (fixes stage Y/X drift)",
                        variable=self.bv_register).grid(
            row=0, column=0, columnspan=3, sticky="w", **PAD)
        ttk.Label(freg, text="Max shift (px):").grid(row=1, column=0, sticky="w", **PAD)
        self.sv_reg_max_shift = tk.StringVar(value="120")
        ttk.Entry(freg, textvariable=self.sv_reg_max_shift, width=6).grid(
            row=1, column=1, sticky="w", padx=4)
        ttk.Label(freg, text="(increase if drift > 120px)",
                  foreground="grey").grid(row=1, column=2, sticky="w")

        # Per-tile extras (shown only in tile_fit pipeline mode)
        self._tile_extras_frame = ttk.Frame(parent)
        self._tile_extras_frame.columnconfigure(0, weight=1)
        self._tile_extras_frame.grid(row=4, column=0, sticky="ew", pady=(0, 4))
        fte = _section(self._tile_extras_frame, "Per-Tile IRF Directory (optional)")
        fte.grid(row=0, column=0, sticky="ew")
        fte.columnconfigure(1, weight=1)
        self.sv_tile_irf_dir = tk.StringVar()
        _row(fte, "IRF XLSX dir", self.sv_tile_irf_dir, 0,
             lambda: _browse_dir(self.sv_tile_irf_dir, "Directory of per-tile IRF xlsx files"))
        ttk.Label(fte, text="One <tile_name>.xlsx per tile; leave blank to use IRF method above",
                  foreground="grey").grid(row=1, column=1, columnspan=2, sticky="w", padx=4)
        self._tile_extras_frame.grid_remove()

    def _pipeline_changed(self):
        mode = self.sv_pipeline.get()
        if mode == "stitch_only":
            self._fit_frame.grid_remove()
            self._btn_st.configure(text="▶  Run Tile Stitch")
        elif mode == "stitch_fit":
            self._fit_frame.grid()
            self._tile_extras_frame.grid_remove()
            self._btn_st.configure(text="▶  Run Stitch + Fit")
        else:  # tile_fit
            self._fit_frame.grid()
            self._tile_extras_frame.grid()
            self._btn_st.configure(text="▶  Run Per-Tile Fit")
        self.root.after_idle(self._fit_window_to_screen)

    def _perpix_toggled(self):
        if self.bv_perpix.get():
            self._pxf.grid()
        else:
            self._pxf.grid_remove()
        self.root.after_idle(self._fit_window_to_screen)

    # -------------------------------------------------------------------------
    # TAB 3 – Batch ROI Fit
    # -------------------------------------------------------------------------
    def _build_batch_tab(self):
        tab = self._make_scroll_tab("  Batch ROI Fit  ")
        tab.columnconfigure(0, weight=1)

        ff = _section(tab, "Input / Output")
        ff.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ff.columnconfigure(1, weight=1)
        self.sv_batch_xlif_dir = tk.StringVar()
        self.sv_batch_ptu_dir  = tk.StringVar()
        self.sv_batch_out_dir  = tk.StringVar()
        _row(ff, "XLIF folder *",     self.sv_batch_xlif_dir, 0,
             lambda: _browse_dir(self.sv_batch_xlif_dir, "Folder of XLIF files"))
        _row(ff, "PTU folder *",      self.sv_batch_ptu_dir,  1,
             lambda: _browse_dir(self.sv_batch_ptu_dir,  "PTU tile directory"))
        _row(ff, "Output base dir *", self.sv_batch_out_dir,  2,
             lambda: _browse_dir(self.sv_batch_out_dir,  "Base output directory"))
        ttk.Label(ff, text="One sub-folder per ROI created inside the output base dir.",
                  foreground="grey").grid(row=3, column=1, columnspan=2, sticky="w", padx=4)

        fi = _section(tab, "IRF")
        fi.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        fi.columnconfigure(1, weight=1)
        self.sv_batch_mirf = tk.StringVar(value=str(_C()["MACHINE_IRF_DEFAULT_PATH"]))
        _row(fi, "Machine IRF (.npy) *", self.sv_batch_mirf, 0,
             lambda: _browse_file(self.sv_batch_mirf, "Machine IRF",
                                  [("NumPy", "*.npy"), ("All", "*.*")]))

        fp = _section(tab, "Fitting Parameters")
        fp.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(fp, text="Exponential components:").grid(row=0, column=0, sticky="w", **PAD)
        self.iv_nexp_batch = tk.IntVar(value=2)
        for n in (1, 2, 3):
            ttk.Radiobutton(fp, text=str(n), variable=self.iv_nexp_batch,
                            value=n).grid(row=0, column=n, sticky="w", padx=4)
        ttk.Label(fp, text="Fit window (ns):").grid(row=1, column=0, sticky="w", **PAD)
        self.sv_batch_tau_min = tk.StringVar(value=str(_C()["Tau_min"]))
        self.sv_batch_tau_max = tk.StringVar(value=str(_C()["Tau_max"]))
        ttk.Entry(fp, textvariable=self.sv_batch_tau_min, width=7).grid(row=1, column=1, padx=4)
        ttk.Label(fp, text="to").grid(row=1, column=2)
        ttk.Entry(fp, textvariable=self.sv_batch_tau_max, width=7).grid(row=1, column=3, padx=4)
        ttk.Label(fp, text="ns", foreground="grey").grid(row=1, column=4, padx=4)
        ttk.Label(fp, text="Colour scale (ns):").grid(row=2, column=0, sticky="w", **PAD)
        self.sv_batch_tau_lo = tk.StringVar(
            value="" if _C()["TAU_DISPLAY_MIN"] is None else str(_C()["TAU_DISPLAY_MIN"]))
        self.sv_batch_tau_hi = tk.StringVar(
            value="" if _C()["TAU_DISPLAY_MAX"] is None else str(_C()["TAU_DISPLAY_MAX"]))
        ttk.Entry(fp, textvariable=self.sv_batch_tau_lo, width=7).grid(row=2, column=1, padx=4)
        ttk.Label(fp, text="to").grid(row=2, column=2)
        ttk.Entry(fp, textvariable=self.sv_batch_tau_hi, width=7).grid(row=2, column=3, padx=4)
        ttk.Label(fp, text="ns  (display only)", foreground="grey").grid(row=2, column=4, padx=4)

        freg = _section(tab, "Tile Registration")
        freg.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        self.bv_batch_register = tk.BooleanVar(value=True)
        ttk.Checkbutton(freg, text="Phase-correlation registration (fixes stage Y/X drift)",
                        variable=self.bv_batch_register).grid(
            row=0, column=0, columnspan=3, sticky="w", **PAD)
        ttk.Label(freg, text="Max shift (px):").grid(row=1, column=0, sticky="w", **PAD)
        self.sv_batch_reg_shift = tk.StringVar(value="120")
        ttk.Entry(freg, textvariable=self.sv_batch_reg_shift, width=6).grid(
            row=1, column=1, sticky="w", padx=4)
        ttk.Label(freg, text="(increase if drift > 120px)",
                  foreground="grey").grid(row=1, column=2, sticky="w")

        fm = _section(tab, "Masking")
        fm.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        self.bv_batch_thr = tk.BooleanVar(value=False)
        self.sv_batch_thr = tk.StringVar()
        ttk.Checkbutton(fm, text="Intensity threshold (min photons/px):",
                        variable=self.bv_batch_thr,
                        command=lambda: _tog(self.bv_batch_thr, self._batch_thr_e)).grid(
            row=0, column=0, sticky="w", **PAD)
        self._batch_thr_e = ttk.Entry(fm, textvariable=self.sv_batch_thr,
                                      width=8, state="disabled")
        self._batch_thr_e.grid(row=0, column=1, sticky="w", padx=4)

        fexp = _section(tab, "Image Export")
        fexp.grid(row=5, column=0, sticky="ew", pady=(0, 6))
        self.bv_batch_save_lifetime  = tk.BooleanVar(value=True)
        self.bv_batch_save_rgb       = tk.BooleanVar(value=True)
        self.bv_batch_save_intensity = tk.BooleanVar(value=True)
        self.bv_batch_save_npy       = tk.BooleanVar(value=True)
        self.bv_batch_save_ind       = tk.BooleanVar(value=False)
        ttk.Checkbutton(fexp, text="Lifetime image (uint16 TIFF)",
                        variable=self.bv_batch_save_lifetime).grid(row=0, column=0, sticky="w", **PAD)
        ttk.Checkbutton(fexp, text="Component RGB TIFF",
                        variable=self.bv_batch_save_rgb).grid(row=0, column=1, sticky="w", **PAD)
        ttk.Checkbutton(fexp, text="Intensity TIFF",
                        variable=self.bv_batch_save_intensity).grid(row=0, column=2, sticky="w", **PAD)
        ttk.Checkbutton(fexp, text="Raw maps (.npy)",
                        variable=self.bv_batch_save_npy).grid(row=1, column=0, sticky="w", **PAD)
        ttk.Checkbutton(fexp, text="Individual component maps (τ₁, a₁, τ₂…)",
                        variable=self.bv_batch_save_ind).grid(row=1, column=1, columnspan=2, sticky="w", **PAD)
        ttk.Label(fexp, text="Lifetime colour scale (ns):").grid(row=2, column=0, sticky="w", **PAD)
        ttk.Entry(fexp, textvariable=self.sv_batch_tau_lo, width=7).grid(row=2, column=1, sticky="w", padx=4)
        ttk.Label(fexp, text="to").grid(row=2, column=2)
        ttk.Entry(fexp, textvariable=self.sv_batch_tau_hi, width=7).grid(row=2, column=3, sticky="w", padx=4)
        ttk.Label(fexp, text="ns  (blank = auto)", foreground="grey").grid(row=2, column=4, sticky="w", padx=4)
        ttk.Label(fexp, text="Gamma (lifetime image):").grid(row=3, column=0, sticky="w", **PAD)
        self.sv_batch_gamma = tk.StringVar(value="0.4")
        ttk.Entry(fexp, textvariable=self.sv_batch_gamma, width=5).grid(row=3, column=1, sticky="w", padx=4)
        ttk.Label(fexp, text="(0.4 = boost dim tissue; 1.0 = linear)",
                  foreground="grey").grid(row=3, column=2, columnspan=3, sticky="w")
        ttk.Label(fexp, text="Intensity display max:").grid(row=4, column=0, sticky="w", **PAD)
        self.sv_batch_int_max = tk.StringVar()
        ttk.Entry(fexp, textvariable=self.sv_batch_int_max, width=8).grid(row=4, column=1, sticky="w", padx=4)
        ttk.Label(fexp, text="(blank = auto 99th percentile)",
                  foreground="grey").grid(row=4, column=2, columnspan=3, sticky="w")

        self._btn_batch = ttk.Button(tab, text="▶  Run Batch ROI Fit",
                                     command=self._run_batch)
        self._btn_batch.grid(row=6, column=0, pady=8, ipadx=20, ipady=4)

    def _run_batch(self):
        xlif_dir = self.sv_batch_xlif_dir.get().strip()
        ptu_dir  = self.sv_batch_ptu_dir.get().strip()
        out_dir  = self.sv_batch_out_dir.get().strip()
        for val, name in [(xlif_dir, "XLIF folder"),
                          (ptu_dir,  "PTU folder"),
                          (out_dir,  "Output directory")]:
            if not val or not Path(val).is_dir():
                messagebox.showerror("Missing input", f"Please select a valid {name}.")
                return
        xlif_files = sorted(Path(xlif_dir).glob("*.xlif"))
        if not xlif_files:
            messagebox.showerror("No XLIF files", f"No .xlif files found in:\n{xlif_dir}")
            return

        cfg       = _C()
        mirf      = self.sv_batch_mirf.get().strip() or str(cfg["MACHINE_IRF_DEFAULT_PATH"])
        n_exp     = self.iv_nexp_batch.get()
        tau_min   = float(self.sv_batch_tau_min.get() or cfg["Tau_min"])
        tau_max   = float(self.sv_batch_tau_max.get() or cfg["Tau_max"])
        tau_lo        = _flt(self.sv_batch_tau_lo) or cfg["TAU_DISPLAY_MIN"] or 0.0
        tau_hi        = _flt(self.sv_batch_tau_hi) or cfg["TAU_DISPLAY_MAX"] or 10.0
        save_lifetime = self.bv_batch_save_lifetime.get()
        save_rgb      = self.bv_batch_save_rgb.get()
        save_npy      = self.bv_batch_save_npy.get()
        save_ind      = self.bv_batch_save_ind.get()
        gamma         = float(self.sv_batch_gamma.get() or 0.4)
        int_max       = _flt(self.sv_batch_int_max) or None
        register  = self.bv_batch_register.get()
        reg_shift = int(self.sv_batch_reg_shift.get() or 120)
        thr       = _thresh(self.bv_batch_thr, self.sv_batch_thr)

        from flimkit.PTU.stitch    import fit_flim_tiles
        from flimkit.FLIM.assemble import (derive_global_tau, save_assembled_maps,
                                           assemble_tile_maps)
        from flimkit.utils.lifetime_image import make_lifetime_image, make_component_rgb_tiff
        import gc, csv as csv_mod

        def task(progress_callback, cancel_event):
            csv_path = Path(out_dir) / "batch_roi_fit_summary.csv"
            header_written = False
            n_total = len(xlif_files)

            for idx, xlif_path in enumerate(xlif_files):
                if cancel_event.is_set():
                    print("\nBatch cancelled.")
                    break
                progress_callback(idx, n_total)
                ptu_basename = xlif_path.stem
                roi_clean    = ptu_basename.replace(" ", "_")
                roi_out      = Path(out_dir) / roi_clean
                roi_out.mkdir(parents=True, exist_ok=True)
                print(f"\n{'='*50}\n  [{idx+1}/{n_total}] {ptu_basename}\n{'='*50}")

                try:
                    fit_args = argparse.Namespace(
                        nexp=n_exp, tau_min=tau_min, tau_max=tau_max,
                        optimizer="de", restarts=1,
                        de_population=cfg["de_population"],
                        de_maxiter=cfg["de_maxiter"],
                        workers=cfg["n_workers"],
                        binning=1,
                        min_photons=cfg["MIN_PHOTONS_PERPIX"],
                        intensity_threshold=thr,
                        register_tiles=register,
                        reg_max_shift_px=reg_shift,
                        machine_irf=mirf,
                        tau_display_min=tau_lo,
                        tau_display_max=tau_hi,
                        intensity_display_min=0.0,
                        intensity_display_max=None,
                        irf_xlsx_dir=None, irf_xlsx_map=None,
                        ptu_basename=ptu_basename,
                        xlif=str(xlif_path),
                        ptu_dir=ptu_dir,
                        output_dir=str(roi_out),
                        no_plots=True, cell_mask=False,
                        debug_xlsx=False, print_config=False,
                        irf=None, irf_xlsx=None,
                        estimate_irf="machine_irf",
                        no_xlsx_irf=True,
                    )
                    (tile_results, canvas_h, canvas_w, _, _, _, _, _, _) = fit_flim_tiles(
                        xlif_path=xlif_path, ptu_dir=Path(ptu_dir),
                        output_dir=roi_out, args=fit_args,
                        ptu_basename=ptu_basename, rotate_tiles=True, verbose=True,
                    )
                    if not tile_results:
                        row = {"roi": ptu_basename, "status": "No tiles fitted"}
                    else:
                        canvas = assemble_tile_maps(tile_results, canvas_h, canvas_w, n_exp)
                        del tile_results; gc.collect()
                        summary = derive_global_tau(canvas, n_exp=n_exp)
                        save_assembled_maps(
                            canvas=canvas, global_summary=summary,
                            output_dir=roi_out, roi_name=roi_clean, n_exp=n_exp,
                            tau_display_min=tau_lo, tau_display_max=tau_hi,
                            intensity_display_max=int_max,
                        )
                        if save_lifetime:
                            make_lifetime_image(
                                canvas=canvas, output_dir=roi_out, roi_name=roi_clean,
                                tau_min_ns=tau_lo, tau_max_ns=tau_hi,
                                smooth_sigma_px=0.0, gamma=gamma, verbose=False,
                            )
                        if save_rgb:
                            make_component_rgb_tiff(
                                canvas=canvas, output_dir=roi_out,
                                roi_name=roi_clean, n_exp=n_exp, verbose=False,
                            )
                        if not save_npy:
                            for f_ in roi_out.glob("*.npy"):
                                if not f_.name.endswith("_time_axis_ns.npy"):
                                    f_.unlink(missing_ok=True)
                        del canvas; gc.collect()
                        row = {"roi": ptu_basename, "status": "OK", **summary}
                        print(f"  OK: {ptu_basename}")
                except Exception as exc:
                    import traceback; traceback.print_exc()
                    row = {"roi": ptu_basename,
                           "status": f"ERROR: {type(exc).__name__}: {str(exc)[:80]}"}

                with open(csv_path, "a", newline="") as fh:
                    writer = csv_mod.DictWriter(fh, fieldnames=list(row.keys()))
                    if not header_written:
                        writer.writeheader(); header_written = True
                    writer.writerow(row)

            progress_callback(n_total, n_total)
            print(f"\nBatch complete. CSV: {csv_path}")

        def on_done(result):
            self._set_buttons("normal")
            self._res.set_status("✓  Batch complete.")
            self._res.load_images(out_dir)

        self._set_buttons("disabled")
        self.run_with_progress(
            task, task_name=f"Batch ROI Fit ({len(xlif_files)} ROIs)", on_done=on_done, output_dir=out_dir)

    # -------------------------------------------------------------------------
    # TAB 4 – Machine IRF Builder
    # -------------------------------------------------------------------------
    def _build_machine_irf_tab(self):
        tab = self._make_scroll_tab("  Machine IRF Builder  ")
        tab.columnconfigure(0, weight=1)

        cfg = _C()

        ff = _section(tab, "Source Data")
        ff.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ff.columnconfigure(1, weight=1)
        self.sv_mirf_src = tk.StringVar()
        _row(
            ff,
            "PTU/XLSX folder *",
            self.sv_mirf_src,
            0,
            lambda: _browse_dir(self.sv_mirf_src, "Folder with paired .ptu and .xlsx"),
        )
        ttk.Label(
            ff,
            text="Builder uses matching <name>.ptu + <name>.xlsx pairs.",
            foreground="grey",
        ).grid(row=1, column=1, columnspan=2, sticky="w", padx=4)

        fp = _section(tab, "Build Settings")
        fp.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        fp.columnconfigure(1, weight=1)

        self.sv_mirf_anchor = tk.StringVar(value=cfg["MACHINE_IRF_ALIGN_ANCHOR"])
        self.sv_mirf_reducer = tk.StringVar(value=cfg["MACHINE_IRF_REDUCER"])

        ttk.Label(fp, text="Align anchor:").grid(row=0, column=0, sticky="w", **PAD)
        ttk.Combobox(
            fp,
            textvariable=self.sv_mirf_anchor,
            values=["peak", "halfmax", "onset10", "slope"],
            state="readonly",
            width=12,
        ).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(fp, text="Reducer:").grid(row=1, column=0, sticky="w", **PAD)
        ttk.Combobox(
            fp,
            textvariable=self.sv_mirf_reducer,
            values=["median", "mean"],
            state="readonly",
            width=12,
        ).grid(row=1, column=1, sticky="w", padx=4)

        fo = _section(tab, "Output")
        fo.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        fo.columnconfigure(1, weight=1)
        self.sv_mirf_out_dir = tk.StringVar(value=str(cfg["MACHINE_IRF_DIR"]))
        self.sv_mirf_name = tk.StringVar(value="machine_irf_default")

        _row(
            fo,
            "Output directory *",
            self.sv_mirf_out_dir,
            0,
            lambda: _browse_dir(self.sv_mirf_out_dir, "Machine IRF output directory"),
        )
        ttk.Label(fo, text="Base filename:").grid(row=1, column=0, sticky="w", **PAD)
        ttk.Entry(fo, textvariable=self.sv_mirf_name, width=35).grid(
            row=1, column=1, columnspan=2, sticky="ew", padx=4
        )

        self._btn_mirf = ttk.Button(
            tab,
            text="▶  Build Machine IRF",
            command=self._run_build_machine_irf,
        )
        self._btn_mirf.grid(row=3, column=0, pady=8, ipadx=20, ipady=4)

    # -------------------------------------------------------------------------
    # TAB 5 – Phasor
    # -------------------------------------------------------------------------
    def _build_phasor_tab(self):
        # Plain (non-scrolling) outer frame so the figure can resize freely.
        outer = ttk.Frame(self._nb)
        self._nb.add(outer, text="  Phasor Analysis  ")
        outer.columnconfigure(0, weight=1)
        # ── Controls strip (fixed height, top) ───────────────────────────────
        ctrl = ttk.Frame(outer, padding=(6, 4))
        ctrl.grid(row=0, column=0, sticky="ew")
        ctrl.columnconfigure(0, weight=1)

        # Input mode
        mode_fr = _section(ctrl, "Input Mode")
        mode_fr.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        mode_fr.columnconfigure(1, weight=1)

        self.sv_ph_mode = tk.StringVar(value="new")
        ttk.Radiobutton(mode_fr, text="New PTU file",
                        variable=self.sv_ph_mode, value="new",
                        command=self._ph_mode_changed).grid(
            row=0, column=0, sticky="w", padx=4, pady=1)
        ttk.Radiobutton(mode_fr, text="Resume session (.npz)",
                        variable=self.sv_ph_mode, value="session",
                        command=self._ph_mode_changed).grid(
            row=0, column=1, sticky="w", padx=4, pady=1)

        # New-PTU sub-frame
        self._ph_new = ttk.Frame(ctrl)
        self._ph_new.columnconfigure(0, weight=1)
        self._ph_new.grid(row=1, column=0, sticky="ew")
        fn = _section(self._ph_new, "New Analysis")
        fn.grid(row=0, column=0, sticky="ew")
        fn.columnconfigure(1, weight=1)
        self.sv_ph_ptu  = tk.StringVar()
        self.sv_ph_irf  = tk.StringVar()
        self.sv_ph_mirf = tk.StringVar(
            value=str(_C()["MACHINE_IRF_DEFAULT_PATH"]))
        _row(fn, "PTU file *",             self.sv_ph_ptu,  0,
             lambda: _browse_file(self.sv_ph_ptu, "PTU file",
                                  [("PTU", "*.ptu"), ("All", "*.*")]))
        _row(fn, "IRF XLSX (optional)",    self.sv_ph_irf,  1,
             lambda: _browse_file(self.sv_ph_irf, "IRF XLSX",
                                  [("Excel", "*.xlsx"), ("All", "*.*")]))
        _row(fn, "Machine IRF (optional)", self.sv_ph_mirf, 2,
             lambda: _browse_file(self.sv_ph_mirf, "Machine IRF",
                                  [("NumPy", "*.npy"), ("All", "*.*")]))
        ttk.Label(fn, text="XLSX takes priority if both supplied",
                  foreground="grey").grid(
            row=3, column=1, columnspan=2, sticky="w", padx=4)

        # Session sub-frame
        self._ph_sess = ttk.Frame(ctrl)
        self._ph_sess.columnconfigure(0, weight=1)
        self._ph_sess.grid(row=2, column=0, sticky="ew")
        fs = _section(self._ph_sess, "Resume Session")
        fs.grid(row=0, column=0, sticky="ew")
        fs.columnconfigure(1, weight=1)
        self.sv_ph_session = tk.StringVar()
        _row(fs, "Session (.npz) *", self.sv_ph_session, 0,
             lambda: _browse_file(self.sv_ph_session, "Session file",
                                  [("NPZ", "*.npz"), ("All", "*.*")]))
        self._ph_sess.grid_remove()

        # Display options
        opt_fr = _section(ctrl, "Display Options")
        opt_fr.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(opt_fr, text="Min photons (fraction):").grid(
            row=0, column=0, sticky="w", **PAD)
        self.sv_ph_minph = tk.StringVar(value="0.01")
        ttk.Entry(opt_fr, textvariable=self.sv_ph_minph, width=8).grid(
            row=0, column=1, sticky="w", padx=4)
        ttk.Label(opt_fr, text="Max cursors:").grid(
            row=0, column=2, sticky="w", padx=8)
        self.sv_ph_maxc = tk.StringVar(value="6")
        ttk.Entry(opt_fr, textvariable=self.sv_ph_maxc, width=4).grid(
            row=0, column=3, sticky="w", padx=4)

        # Run button
        self._btn_ph = ttk.Button(ctrl, text="▶  Load & Analyse",
                                   command=self._run_phasor)
        self._btn_ph.grid(row=4, column=0, pady=(6, 2), ipadx=16, ipady=3,
                          sticky="w")

        # (PhasorViewPanel lives in the right FOV-preview panel — see _init_ui)

    def _ph_mode_changed(self):
        if self.sv_ph_mode.get() == "new":
            self._ph_new.grid()
            self._ph_sess.grid_remove()
        else:
            self._ph_new.grid_remove()
            self._ph_sess.grid()

    # -------------------------------------------------------------------------
    # FOV Preview auto-load
    # -------------------------------------------------------------------------
    def _on_fov_ptu_changed(self, var, index, mode):
        """Auto-load FOV preview when PTU file is selected."""
        ptu_path = self.sv_ptu.get().strip()
        if ptu_path and hasattr(self, '_fov_preview'):
            # Defer loading to give the UI a chance to update
            self.root.after(100, lambda: self._fov_preview.load_fov(ptu_path))

    # -------------------------------------------------------------------------
    # Run handlers
    # -------------------------------------------------------------------------
    def _run_fov(self):
        ptu = self.sv_ptu.get().strip()
        if not ptu or not Path(ptu).exists():
            messagebox.showerror("Missing input", "Please select a valid PTU file.")
            return

        from flimkit.interactive import _run_flim_fit

        cfg = _C()
        irf = self._irf_fov.get_args(xlsx_fallback=self.sv_xlsx.get().strip())

        a = argparse.Namespace()
        a.ptu           = ptu
        a.xlsx          = self.sv_xlsx.get().strip() or None
        a.debug_xlsx    = False
        a.print_config  = False
        a.irf           = irf["irf"]
        a.irf_xlsx      = irf["irf_xlsx"]
        a.estimate_irf  = irf["estimate_irf"]
        a.no_xlsx_irf   = irf["no_xlsx_irf"]
        a.machine_irf   = irf.get("machine_irf") or str(_C()["MACHINE_IRF_DEFAULT_PATH"])
        a.irf_bins      = cfg["IRF_BINS"]
        a.irf_fit_width = cfg["IRF_FIT_WIDTH"]
        a.irf_fwhm      = cfg["IRF_FWHM"]
        a.nexp          = self.iv_nexp_fov.get()
        a.tau_min       = float(self.sv_tau_min_fov.get() or cfg["Tau_min"])
        a.tau_max       = float(self.sv_tau_max_fov.get() or cfg["Tau_max"])
        a.mode          = self.sv_mode_fov.get()
        a.binning       = cfg["binning_factor"]
        a.min_photons   = cfg["MIN_PHOTONS_PERPIX"]
        a.optimizer     = cfg["Optimizer"]
        a.restarts      = cfg["lm_restarts"]
        a.de_population = cfg["de_population"]
        a.de_maxiter    = cfg["de_maxiter"]
        a.workers       = cfg["n_workers"]
        a.no_polish     = False
        a.channel       = cfg["channels"]
        _out_raw = self.sv_out_fov.get().strip() or cfg["OUT_NAME"]
        # If the prefix has no directory component, anchor it to the PTU's
        # parent so output files are never written to a read-only CWD
        # (which happens inside a frozen .app bundle).
        if Path(_out_raw).parent == Path("."):
            a.out = str(Path(ptu).parent / _out_raw)
        else:
            a.out = _out_raw
        a.no_plots      = False
        a.cell_mask     = self.bv_cell.get()
        a.intensity_threshold = _thresh(self.bv_thr_fov, self.sv_thr_fov)

        out_dir = str(Path(a.out).parent)
        self._launch(
            lambda progress_callback=None, cancel_event=None: _run_flim_fit(a, progress_callback, cancel_event),
            output_dir=out_dir,
            ptu_path=ptu,
            task_name="Single-FOV Fit"
        )

    def _run_stitch(self):
        xlif     = self.sv_xlif.get().strip()
        ptu_dir  = self.sv_ptu_dir.get().strip()
        out_base = self.sv_out_st.get().strip()

        for val, name in [(xlif, "XLIF file"),
                          (ptu_dir, "PTU directory"),
                          (out_base, "Output directory")]:
            if not val:
                messagebox.showerror("Missing input", f"Please specify the {name}.")
                return

        pipeline = self.sv_pipeline.get()
        from flimkit.PTU.stitch import stitch_flim_tiles

        roi_name   = Path(xlif).stem.replace(" ", "_")
        output_dir = str(Path(out_base) / roi_name)

        a = argparse.Namespace()
        a.xlif         = xlif
        a.ptu_dir      = ptu_dir
        a.output_dir   = output_dir
        a.ptu_basename = Path(xlif).stem
        a.rotate_tiles = self.bv_rotate.get()

        cfg = _C()
        irf = self._irf_st.get_args()
        a.irf           = irf["irf"]
        a.irf_xlsx      = irf["irf_xlsx"]
        a.no_xlsx_irf   = irf["no_xlsx_irf"]
        a.estimate_irf  = irf["estimate_irf"] if irf["estimate_irf"] != "none" else "gaussian"
        a.machine_irf   = irf.get("machine_irf") or str(cfg["MACHINE_IRF_DEFAULT_PATH"])
        a.nexp          = self.iv_nexp_st.get()
        a.tau_min       = float(self.sv_tau_fit_lo.get() or cfg["Tau_min"])
        a.tau_max       = float(self.sv_tau_fit_hi.get() or cfg["Tau_max"])
        a.register_tiles     = self.bv_register.get()
        a.reg_max_shift_px   = int(self.sv_reg_max_shift.get() or 120)
        a.binning       = cfg["binning_factor"]
        a.min_photons   = cfg["MIN_PHOTONS_PERPIX"]
        a.optimizer     = "de"
        a.restarts      = cfg["lm_restarts"]
        a.de_population = cfg["de_population"]
        a.de_maxiter    = cfg["de_maxiter"]
        a.workers       = cfg["n_workers"]
        a.no_polish     = False
        a.channel       = cfg["channels"]
        a.irf_fwhm      = cfg["IRF_FWHM"]
        a.irf_bins      = cfg["IRF_BINS"]
        a.irf_fit_width = cfg["IRF_FIT_WIDTH"]
        a.tau_display_min        = _flt(self.sv_tau_lo)
        a.tau_display_max        = _flt(self.sv_tau_hi)
        a.intensity_display_min  = _flt(self.sv_int_lo)
        a.intensity_display_max  = _flt(self.sv_int_hi)
        a.intensity_threshold    = _thresh(self.bv_thr_st, self.sv_thr_st)
        a.save_individual        = self.bv_save_ind.get()
        a.save_tau_weighted      = self.bv_save_tau_weighted.get()
        a.save_int_weighted      = self.bv_save_int_weighted.get()
        a.save_amp_weighted      = self.bv_save_amp_weighted.get()

        if pipeline == "tile_fit":
            # Per-tile: each tile gets its own fit; per-pixel forced; no per-tile plots
            a.mode         = "both"
            a.no_plots     = True
            a.cell_mask    = False
            a.debug_xlsx   = False
            a.print_config = False
            a.xlsx         = None
            a.out          = None
            a.irf_xlsx_dir = self.sv_tile_irf_dir.get().strip() or None
        else:
            a.mode    = "both" if self.bv_perpix.get() else "summed"
            a.no_plots = False
            a.irf_xlsx_dir = None

        def on_done(result):
            self._set_buttons("normal")
            self._res.set_status("✓  Complete.")
            self._res._nb.select(0)
            captured = "".join(self._buf)
            rows = _parse_summary(captured)

            
            # For tile_fit, result is a dict with fit data
            if pipeline == "tile_fit" and isinstance(result, dict):
                print(f"[on_done] Executing tile_fit branch")
                fit_result = result
                global_summary = fit_result.get('global_summary', {})
                global_popt = fit_result.get('global_popt')
                
                # Extract summary from fit result
                if global_summary:
                    extracted_rows = self._extract_summary_rows(global_summary, global_popt)
                    if extracted_rows:
                        rows = extracted_rows
                        print(f"[on_done] Extracted {len(extracted_rows)} summary rows from global_summary")
                
                # Display fit results (decay + model) in FOV preview
                try:
                    self._fov_preview.display_fit_results(None, fit_result)
                except Exception as e:
                    import traceback
                    print(f"[Warning] Could not display fit results:")
                    traceback.print_exc()
            else:
                # For stitch_only and stitch_fit, load the stitched ROI
                print(f"[on_done] Executing else branch (load_stitched_roi)")
                try:
                    self._fov_preview.load_stitched_roi(a.output_dir)
                except Exception as e:
                    print(f"Warning: Could not load stitched image: {e}")
            
            # Populate summary table
            if rows:
                print(f"[on_done] Populating summary with {len(rows)} rows")
                self._res.populate_summary(rows)
            else:
                print(f"[on_done] No rows to populate summary (rows={rows})")

        def task(progress_callback, cancel_event):
            if pipeline == "stitch_only":
                return stitch_flim_tiles(
                    xlif_path=a.xlif,
                    ptu_dir=a.ptu_dir,
                    output_dir=a.output_dir,
                    ptu_basename=a.ptu_basename,
                    rotate_tiles=a.rotate_tiles,
                    register_tiles=a.register_tiles,
                    reg_max_shift_px=a.reg_max_shift_px,
                    verbose=True,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                )
            elif pipeline == "stitch_fit":
                from flimkit.interactive import _run_stitch_and_fit
                return _run_stitch_and_fit(a, progress_callback=progress_callback,
                                           cancel_event=cancel_event)
            else:  # tile_fit
                from flimkit.interactive import _run_tile_fit
                return _run_tile_fit(a, progress_callback=progress_callback,
                                     cancel_event=cancel_event)

        self._set_buttons("disabled")
        task_name = {"stitch_only": "Stitching", "stitch_fit": "Stitch + Fit",
                     "tile_fit": "Per-Tile Fit"}[pipeline]
        self.run_with_progress(task, task_name=task_name, on_done=on_done, output_dir=a.output_dir)

    def _run_phasor(self):
        """Dispatch PTU loading to a worker thread; update the embedded panel on done."""
        try:
            min_ph  = float(self.sv_ph_minph.get() or 0.01)
            max_cur = int(self.sv_ph_maxc.get() or 6)
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Min photons and max cursors must be numeric.")
            return

        self._phasor_panel.max_cursors = max_cur

        if self.sv_ph_mode.get() == "session":
            sess = self.sv_ph_session.get().strip()
            if not sess or not Path(sess).exists():
                messagebox.showerror("Missing input",
                                     "Please select a valid .npz session file.")
                return

            def _worker():
                from flimkit.phasor_launcher import load_session
                return load_session(sess)

            def _done(result):
                if isinstance(result, Exception):
                    messagebox.showerror("Session load error", str(result))
                    return
                self._phasor_panel.load_session(result, min_photons=min_ph)
                self._res.set_status("✓  Phasor session loaded.")

            self._phasor_thread(_worker, _done, status="⏳  Loading session…")

        else:
            ptu = self.sv_ph_ptu.get().strip()
            if not ptu or not Path(ptu).exists():
                messagebox.showerror("Missing input",
                                     "Please select a valid PTU file.")
                return

            xlsx_irf = self.sv_ph_irf.get().strip() or None
            mach_irf = self.sv_ph_mirf.get().strip() or None
            irf_path = xlsx_irf or mach_irf

            def _worker():
                from flimkit.phasor_launcher import _process_ptu
                return _process_ptu(ptu, irf_path=irf_path)

            def _done(result):
                if isinstance(result, Exception):
                    messagebox.showerror("Phasor error", str(result))
                    return
                self._phasor_panel.set_data(
                    result['real_cal'],
                    result['imag_cal'],
                    result['mean'],
                    result['frequency'],
                    display_image=result.get('display_image'),
                    min_photons=min_ph,
                )
                self._res.set_status(
                    "✓  Phasor data loaded — click the phasor to place cursors.")

            self._phasor_thread(_worker, _done,
                                status="⏳  Loading PTU and computing phasors…")

    def _phasor_thread(self, worker_fn, done_cb, *, status="⏳  Working…"):
        """Run worker_fn in a daemon thread; call done_cb(result) on the main thread.
        Only disables the phasor run button while running — the rest of the UI
        stays responsive (unlike _launch which locks all buttons).
        """
        self._btn_ph.configure(state="disabled")
        self._res.set_status(status)

        def _run():
            try:
                result = worker_fn()
            except Exception as exc:
                import traceback
                traceback.print_exc()
                result = exc
            self.root.after(0, lambda: _finish(result))

        def _finish(result):
            self._btn_ph.configure(state="normal")
            done_cb(result)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _run_build_machine_irf(self):
        src_dir = self.sv_mirf_src.get().strip()
        out_dir = self.sv_mirf_out_dir.get().strip()
        out_name = self.sv_mirf_name.get().strip()
        anchor = self.sv_mirf_anchor.get().strip()
        reducer = self.sv_mirf_reducer.get().strip()

        if not src_dir or not Path(src_dir).exists():
            messagebox.showerror("Missing input", "Please select a valid PTU/XLSX source folder.")
            return
        if not out_dir:
            messagebox.showerror("Missing input", "Please select an output directory.")
            return
        if not out_name:
            messagebox.showerror("Missing input", "Please enter an output base filename.")
            return

        from flimkit.FLIM.irf_tools import build_machine_irf_from_folder

        def task_fn():
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            return build_machine_irf_from_folder(
                folder=src_dir,
                align_anchor=anchor,
                reducer=reducer,
                save=True,
                confirm_save=True,
                output_name=out_name,
                output_dir=out_dir,
                verbose=True,
            )

        self._launch(task_fn, output_dir=out_dir, task_name="Building Machine IRF")

    def _launch(self, fn, output_dir=None, ptu_path=None, task_name="Working…"):
        self._buf.clear()
        self._set_buttons("disabled")
        self._res.set_status("⏳  Running...")
        self._res._nb.select(0)

        # Create a progress window manager for sub-operations
        win_manager = ProgressWindowManager(self.root)
        
        # Create main progress window
        win = ProgressWindow(self.root, task_name=task_name)
        cancel_event = win.cancelled

        def progress_callback(i, total):
            win.set_progress(i, maximum=total)
            if cancel_event.is_set():
                win.set_status("Cancelling…")

        def _worker():
            orig_stdout, orig_stderr = sys.stdout, sys.stderr
            # Always redirect to UI's ScrolledText widget with thread-safe updates
            redir = _Redirect(self._res.log, self._buf, root=self.root)
            sys.stdout = redir
            sys.stderr = redir
            
            try:
                # Call function with progress_callback if it supports it
                sig = inspect.signature(fn)
                if 'progress_callback' in sig.parameters or 'cancel_event' in sig.parameters:
                    # Pass window manager for sub-operations
                    if 'progress_window_manager' in sig.parameters:
                        result = fn(progress_callback=progress_callback, cancel_event=cancel_event, 
                                   progress_window_manager=win_manager)
                    else:
                        result = fn(progress_callback=progress_callback, cancel_event=cancel_event)
                else:
                    # Function doesn't support callbacks, call normally
                    result = fn()
                
                self.root.after(0, lambda: win.close())
                self.root.after(0, lambda: win_manager.close_all())
                captured = "".join(self._buf)
                rows     = _parse_summary(captured)
                self.root.after(0, lambda: self._on_done(rows, output_dir, result, ptu_path))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self._res.set_status(f"✗  Error: {exc}"))
            finally:
                if hasattr(redir, 'close'):
                    redir.close()
                else:
                    redir.flush()
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr
                self.root.after(0, lambda: self._set_buttons("normal"))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_done(self, rows, output_dir, fit_result=None, ptu_path=None):
        self._res.set_status("✓  Finished.")
        
        # Priority 1: Extract summary from fit_result dict (most reliable)
        if fit_result is not None:
            global_summary = fit_result.get('global_summary')
            global_popt = fit_result.get('global_popt')
            
            # If fit_result has summary data, use it (overrides log parsing)
            if global_summary is not None:
                extracted_rows = self._extract_summary_rows(global_summary, global_popt)
                if extracted_rows:
                    rows = extracted_rows
        
        # Populate summary table with whatever rows we have
        if rows:
            self._res.populate_summary(rows)
        
        # Update FOV preview with fit results (works for both single-FOV and tile fitting)
        if fit_result is not None:
            try:
                self._fov_preview.display_fit_results(ptu_path, fit_result)
            except Exception as e:
                import traceback
                traceback.print_exc()

    def _extract_summary_rows(self, global_summary: dict, global_popt=None) -> list:
        """Extract fit summary rows from global_summary dict.

        Handles two schemas:
          • fit_summed schema  — keys: taus_ns, amps, fractions, tau_mean_amp_ns …
          • derive_global_tau schema — keys: tau_mean_amp_global_ns, tau1_mean_ns …
        """
        if not global_summary:
            return []

        rows = []

        # ── Schema A: fit_summed / fit_per_pixel (single-FOV fit) ─────────────
        taus  = global_summary.get('taus_ns', [])
        amps  = global_summary.get('amps',    [])
        fracs = global_summary.get('fractions', [])

        if taus is not None and len(taus) > 0:
            import numpy as np
            taus  = list(np.atleast_1d(taus))
            amps  = list(np.atleast_1d(amps))  if amps  is not None else []
            fracs = list(np.atleast_1d(fracs)) if fracs is not None else []
            for i in range(len(taus)):
                rows.append((f"τ{i+1}", f"{taus[i]:.4f}", "ns"))
                if i < len(amps):
                    rows.append((f"α{i+1}", f"{amps[i]:.3e}", ""))
                if i < len(fracs):
                    rows.append((f"f{i+1} (amp frac)", f"{fracs[i]:.4f}", ""))

        # Mean lifetimes — schema A names
        for key, label in [
            ('tau_mean_amp_ns', 'τ_mean (amp-weighted)'),
            ('tau_mean_int_ns', 'τ_mean (int-weighted)'),
        ]:
            v = global_summary.get(key)
            if v is not None:
                rows.append((label, f"{v:.4f}", "ns"))

        # ── Schema B: derive_global_tau (stitch / tile-fit pipeline) ──────────
        tau_global = global_summary.get('tau_mean_amp_global_ns')
        if tau_global is not None:
            rows.append(("τ_mean amp-wtd (global)", f"{tau_global:.4f}", "ns"))

        tau_std = global_summary.get('tau_std_amp_global_ns')
        if tau_std is not None:
            rows.append(("τ σ (pixel distrib.)", f"{tau_std:.4f}", "ns"))

        tau_med = global_summary.get('tau_median_amp_global_ns')
        if tau_med is not None:
            rows.append(("τ_median (amp-wtd)", f"{tau_med:.4f}", "ns"))

        n_px = global_summary.get('n_pixels_fitted')
        if n_px is not None:
            rows.append(("Pixels fitted", f"{n_px:,}", ""))

        # Per-component rows for tile schema  (tau1_mean_ns, a1_mean_frac …)
        k = 1
        while True:
            tau_k = global_summary.get(f'tau{k}_mean_ns')
            if tau_k is None:
                break
            rows.append((f"τ{k} mean", f"{tau_k:.4f}", "ns"))
            a_k = global_summary.get(f'a{k}_mean_frac')
            if a_k is not None:
                rows.append((f"f{k} mean (amp frac)", f"{a_k:.4f}", ""))
            k += 1

        # ── Shared fields (present in both schemas) ────────────────────────────
        bg_fit = global_summary.get('bg_fit')
        if bg_fit is not None:
            rows.append(("Background (fitted)", f"{bg_fit:.2f}", "cts/bin"))

        irf_shift = global_summary.get('irf_shift_bins')
        if irf_shift is not None:
            rows.append(("IRF shift", f"{irf_shift:.3f}", "bins"))

        irf_sigma = global_summary.get('irf_sigma_bins')
        if irf_sigma is not None:
            rows.append(("IRF σ (broadening)", f"{irf_sigma:.3f}", "bins"))

        irf_fwhm = global_summary.get('irf_fwhm_eff_ns')
        if irf_fwhm is not None:
            rows.append(("IRF FWHM (eff.)", f"{irf_fwhm:.4f}", "ns"))

        chi2_r_tail = global_summary.get('reduced_chi2_tail')
        if chi2_r_tail is not None:
            rows.append(("χ²_r(tail) Neyman", f"{chi2_r_tail:.4f}", ""))

        chi2_p_tail = global_summary.get('reduced_chi2_tail_pearson')
        if chi2_p_tail is not None:
            rows.append(("χ²_r(tail) Pearson", f"{chi2_p_tail:.4f}", ""))

        return rows

    def _set_buttons(self, state):
        for btn in (self._btn_fov, self._btn_st, self._btn_batch, self._btn_mirf, self._btn_ph):
            btn.configure(state=state)


# -----------------------------------------------------------------------------
# Themed version (uses TKinterModernThemes)
# -----------------------------------------------------------------------------
if HAS_TKMT:
    class FLIMKitGUIThemed(TKMT.ThemedTKinterFrame, _UIBuilder):
        def __init__(self, theme="sun-valley", mode="dark"):
            super().__init__("FLIMkit Analysis GUI", theme, mode,
                             usecommandlineargs=True, useconfigfile=True)
            self.root = self.master
            self.root.minsize(760, 700)
            self._init_ui()
            self.run()

# -----------------------------------------------------------------------------
# Fallback version (plain Tk, optional drag‑and‑drop)
# -----------------------------------------------------------------------------
class FLIMKitGUIFallback(_UIBuilder):
    def __init__(self, root):
        self.root = root
        self.root.title("FLIMkit Analysis GUI")
        self.root.minsize(760, 700)
        self._init_ui()
        self.root.mainloop()


# -----------------------------------------------------------------------------
# Entry point – choose themed or fallback
# -----------------------------------------------------------------------------
def launch_gui():
    global GUI_MODE
    GUI_MODE = True

    if HAS_TKMT:
        # Themed version: ThemedTKinterFrame creates its own root.
        app = FLIMKitGUIThemed(theme="sun-valley", mode="dark")
    else:
        # Fallback: create a plain Tk root (optionally with drag‑and‑drop)
        if HAS_DND:
            from tkinterdnd2 import Tk
            root = Tk()
        else:
            root = tk.Tk()
        # Apply a simple ttk theme for better appearance
        style = ttk.Style(root)
        for theme_name in ("clam", "alt", "default"):
            if theme_name in style.theme_names():
                style.theme_use(theme_name)
                break
        app = FLIMKitGUIFallback(root)


if __name__ == "__main__":
    launch_gui()