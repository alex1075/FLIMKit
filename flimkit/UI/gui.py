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
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from flimkit.UI.progress_window import ProgressWindow
from flimkit.UI.phasor_panel import PhasorViewPanel
from flimkit.UI import flim_display
from flimkit.UI.roi_tools import RoiManager, RoiAnalysisPanel

# Modern theme support
try:
    import TKinterModernThemes as TKMT
    HAS_TKMT = True
except ImportError:
    HAS_TKMT = False

try:
    from tkinterdnd2 import DND_FILES, DND_TEXT
    HAS_DND = True
except ImportError:
    HAS_DND = False

GUI_MODE = False


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



def _reconstruct_dict_from_session(session_data: dict, key: str) -> dict:
    """
    Inverse of the hoisting done in _save_roi_progress.
    Reconstructs a dict from JSON + hoisted numpy arrays stored separately.
    
    Args:
        session_data: The session/fit result dict containing "key_json" and "key_arr_*" entries
        key: Base key name (e.g. "global_summary" or "pixel_maps")
    
    Returns:
        Reconstructed dict with arrays reattached
    """
    import json
    result = {}
    json_str = session_data.get(f"{key}_json")
    if json_str:
        if isinstance(json_str, (bytes, np.ndarray)):
            json_str = json_str.item() if hasattr(json_str, 'item') else json_str.decode()
        try:
            result = json.loads(json_str)
        except:
            pass
    
    prefix = f"{key}_arr_"
    for skey, sval in session_data.items():
        if skey.startswith(prefix) and isinstance(sval, np.ndarray):
            result[skey[len(prefix):]] = sval
    
    return result


def _safe_array_from_json(value) -> np.ndarray:
    """
    Safely convert a value that may be a string representation of an array
    back to a real numpy array. Handles numpy scalar wrappers.
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bytes, np.ndarray)):
        if hasattr(value, 'item'):
            value = value.item()
        else:
            value = value.decode() if isinstance(value, bytes) else str(value)
    if isinstance(value, str):
        try:
            import re
            # Try to parse numpy array string format: [1.0 2.0 3.0] with optional formatting
            value = re.sub(r'\s+', ' ', value.strip())
            value = value.replace('e+', 'e+').replace('e-', 'e-')
            return np.fromstring(value.strip('[]'), sep=' ')
        except:
            pass
    return np.asarray(value)


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
    
    # Add drag-and-drop support if available
    if HAS_DND:
        try:
            def _drop_handler(event):
                data = event.data.strip()
                if data.startswith("{") and data.endswith("}"):
                    data = data[1:-1]
                var.set(data)
            
            e.drop_target_register(DND_FILES, DND_TEXT)
            e.dnd_bind("<<Drop>>", _drop_handler)
        except Exception:
            pass
    
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

        #  FLIM Color Scale Controls 
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
        
        #  FLIM image display state
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
        
        #  Region management
        self._roi_manager = RoiManager()  # Manages all drawn regions
        self._roi_patches = {}  # Map region_id -> matplotlib patch for rendering
        
        #  Drawing state
        self._drawing_mode = tk.StringVar(value="select")  # Linked to RoiAnalysisPanel mode
        self._is_drawing = False  # Currently drawing
        self._draw_coords = []  # Coordinates being collected
        self._temp_line = None  # Temporary line for visual feedback
        self._mouse_press_event = None  # Track last press event
        self._roi_analysis_panel = None  # Will be set by FLIMKitApp
        
        # Connect matplotlib event handlers to FLIM axes
        self._setup_drawing_events()

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

            # FIX 1: preserve drawn regions after PTU reload
            self._redraw_region_overlays()
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
            
            #  Compute FLIM lifetime map
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
            
            # Cache for interactive updates
            self._lifetime_map = lifetime_map
            self._intensity_map = intensity
            self._n_exp = nexp
            
            #  Store images in fit_result for export and reloading 
            # Add intensity if not already there
            if 'intensity' not in fit_result and intensity is not None:
                fit_result['intensity'] = intensity
            
            # Add lifetime map if computed
            if lifetime_map is not None:
                fit_result['lifetime'] = lifetime_map
            
            # Add pixel maps if available
            if pixel_maps:
                fit_result['pixel_maps'] = pixel_maps
            
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

            # Update FLIM lifetime image
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
            
            # Redraw region overlays on FLIM image
            self._redraw_region_overlays()

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
            
            # Save updated color scale to session (quick update)
            self._save_color_scale_update()
            
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
            
            # Redraw region overlays after color scale update
            self._redraw_region_overlays()
            
            self._canvas_mpl.draw_idle()
        except Exception as e:
            print(f"Error updating FLIM display: {e}")
    
    def _save_color_scale_update(self):
        """Save updated color scale to existing session file (quick update, no full recompute)."""
        try:
            # Only save if we have a PTU path and session file exists
            if not self._ptu_path:
                return
            
            from pathlib import Path
            import json
            import numpy as np
            
            ptu_path = Path(self._ptu_path)
            session_file = ptu_path.parent / f"{ptu_path.stem}.roi_session.npz"
            
            if not session_file.exists():
                return  # No session to update
            
            # Load existing session
            existing_data = np.load(session_file, allow_pickle=True)
            session_data = {key: existing_data[key].item() if existing_data[key].ndim == 0 else existing_data[key] 
                           for key in existing_data.files}
            
            # Update only color scale (don't touch fit data)
            session_data["fov_color_scale"] = json.dumps(self._flim_color_scale)
            
            # Save back to same file
            np.savez_compressed(session_file, **session_data)
            print(f"[Color Scale] ✓ Saved to {session_file.name}")
            
        except Exception as e:
            print(f"[Color Scale] Could not save update: {e}")

    def _save_regions_update(self):
        """Save updated regions to session file (create if doesn't exist)."""
        try:
            if not self._ptu_path:
                return
            
            from pathlib import Path
            import json
            import numpy as np
            from datetime import datetime
            
            ptu_path = Path(self._ptu_path)
            session_file = ptu_path.parent / f"{ptu_path.stem}.roi_session.npz"
            
            if session_file.exists():
                # Load existing session
                existing_data = np.load(session_file, allow_pickle=True)
                session_data = {key: existing_data[key].item() if existing_data[key].ndim == 0 else existing_data[key] 
                               for key in existing_data.files}
            else:
                # Create minimal session file with FOV preview data (regions drawn before fit)
                session_data = {
                    "timestamp": datetime.now().isoformat(),
                    "source": str(self._ptu_path),
                    "form_state_json": json.dumps({}, default=str),
                }
                
                # Save FOV preview data if available
                if self._lifetime_map is not None:
                    session_data["fov_lifetime_map"] = self._lifetime_map
                if self._intensity_map is not None:
                    session_data["fov_intensity_map"] = self._intensity_map
                session_data["fov_color_scale"] = json.dumps(self._flim_color_scale)
                session_data["fov_n_exp"] = self._n_exp
                if self._ptu_path:
                    session_data["fov_ptu_path"] = self._ptu_path
            
            # Update regions (always overwrite)
            session_data["fov_regions"] = self._roi_manager.to_json()
            
            # Save to file
            np.savez_compressed(session_file, **session_data)
            print(f"[ROI Manager] ✓ Saved {len(self._roi_manager.regions)} region(s) to {session_file.name}")
            
        except Exception as e:
            print(f"[ROI Manager] Could not save regions: {e}")
    
    def _load_regions_from_json(self, json_str: str):
        """Load regions from JSON string (called during session restore)."""
        try:
            self._roi_manager = RoiManager.from_json(json_str)
            print(f"[ROI Manager] ✓ Loaded {len(self._roi_manager.regions)} region(s)")
            self._redraw_region_overlays()
        except Exception as e:
            print(f"[ROI Manager] Could not load regions: {e}")
    
    def _redraw_region_overlays(self):
        """Redraw all region patches on the FLIM image axes."""
        import matplotlib.patches as mpatches
        from flimkit.UI.roi_tools import get_rectangle_patch, get_ellipse_patch, get_polygon_patch
        
        # Clear old patches
        for patch in self._roi_patches.values():
            if patch in self._ax_flim.patches:
                patch.remove()
        self._roi_patches = {}
        
        # Draw all regions
        for region in self._roi_manager.get_all_regions():
            region_id = region['id']
            tool_type = region['tool']
            coords = region['coords']
            color = self._roi_manager.get_color(region_id)
            linewidth = 2.5 if region_id == self._roi_manager.get_selected_id() else 1.5
            
            try:
                if tool_type == 'rect':
                    patch = get_rectangle_patch(coords, edgecolor=color, linewidth=linewidth)
                elif tool_type == 'ellipse':
                    patch = get_ellipse_patch(coords, edgecolor=color, linewidth=linewidth)
                elif tool_type in ('polygon', 'freehand'):
                    patch = get_polygon_patch(coords, edgecolor=color, linewidth=linewidth)
                else:
                    continue
                
                self._ax_flim.add_patch(patch)
                self._roi_patches[region_id] = patch
            except Exception as e:
                print(f"[ROI] Could not draw region {region_id}: {e}")
        
        self._canvas_mpl.draw_idle()
    
    def _setup_drawing_events(self):
        """Connect matplotlib event handlers to FLIM axes for drawin."""
        self._canvas_mpl.mpl_connect('button_press_event', self._on_draw_press)
        self._canvas_mpl.mpl_connect('motion_notify_event', self._on_draw_motion)
        self._canvas_mpl.mpl_connect('button_release_event', self._on_draw_release)
    
    def _on_draw_press(self, event):
        """Handle mouse press on FLIM image."""
        if not event.inaxes or event.inaxes != self._ax_flim:
            return
        
        mode = self._drawing_mode.get()
        if mode == "select":
            return  # No drawing in select mode
        
        self._is_drawing = True
        self._draw_coords = [[event.xdata, event.ydata]]
        self._mouse_press_event = event
        print(f"[Drawing] Started {mode} at ({event.xdata:.1f}, {event.ydata:.1f})")
    
    def _on_draw_motion(self, event):
        """Handle mouse motion during drawing."""
        if not self._is_drawing or not event.inaxes or event.inaxes != self._ax_flim:
            return
        
        mode = self._drawing_mode.get()
        
        # For rectangle/ellipse: show preview bbox
        if mode in ("rect", "ellipse") and len(self._draw_coords) > 0:
            if self._temp_line is not None:
                try:
                    self._temp_line.remove()
                except:
                    pass
                self._temp_line = None
            
            # Draw preview rectangle
            x0, y0 = self._draw_coords[0]
            x1, y1 = event.xdata, event.ydata
            
            from matplotlib.patches import Rectangle
            preview = Rectangle((min(x0, x1), min(y0, y1)), 
                               abs(x1 - x0), abs(y1 - y0),
                               edgecolor='cyan', facecolor='none', 
                               linewidth=1, linestyle='--', alpha=0.5)
            self._ax_flim.add_patch(preview)
            self._temp_line = preview
            self._canvas_mpl.draw_idle()
        
        # For polygon/freehand: collect intermediate points
        elif mode in ("polygon", "freehand"):
            self._draw_coords.append([event.xdata, event.ydata])
    
    def _on_draw_release(self, event):
        """Handle mouse release to complete drawing."""
        if not self._is_drawing or not event.inaxes or event.inaxes != self._ax_flim:
            return
        
        mode = self._drawing_mode.get()
        
        # Complete rectangle/ellipse with two points
        if mode in ("rect", "ellipse"):
            if len(self._draw_coords) > 0:
                self._draw_coords.append([event.xdata, event.ydata])
                self._finalize_drawing(mode)
        
        # For polygon: right-click or double-click to finish; single click adds point
        elif mode == "polygon":
            # Single click adds to polygon; need explicit finish (e.g., Escape key)
            # For now, any release adds a point
            if len(self._draw_coords) >= 3 and event.button == 3:  # Right-click to finish
                self._finalize_drawing(mode)
            else:
                # Add first point on press already; continue collecting
                pass
        
        elif mode == "freehand":
            # Release finishes the freehand shape
            if len(self._draw_coords) >= 3:
                self._finalize_drawing(mode)
        
        # Clear temporary drawing aid
        if self._temp_line is not None:
            try:
                self._temp_line.remove()
            except:
                pass
            self._temp_line = None
        
        self._is_drawing = False
    
    def _finalize_drawing(self, tool_type: str):
        """Complete drawing and add region to RoiManager."""
        if len(self._draw_coords) < 2:
            print(f"[Drawing] Cancelled {tool_type} (insufficient points)")
            self._draw_coords = []
            return
        
        try:
            # Add region to manager
            region_id = self._roi_manager.add_region(
                f"{tool_type}-{len(self._roi_manager.regions) + 1}",
                tool_type,
                self._draw_coords
            )
            self._redraw_region_overlays()
            self._save_regions_update()
            print(f"[Drawing] Added {tool_type} region {region_id}")
            
            # Notify RoiAnalysisPanel to refresh list
            if self._roi_analysis_panel:
                self._roi_analysis_panel._refresh_region_list()
        except Exception as e:
            print(f"[Drawing] Error finalizing: {e}")
        finally:
            self._draw_coords = []

    def grid(self, **kw):
        self.frame.grid(**kw)


class ResultsPanel:

    def __init__(self, parent, root=None):
        self.parent = parent
        self.root = root  # Reference to main window for dialogs
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        self._nb = ttk.Notebook(self.frame)
        self._nb.grid(row=0, column=0, sticky="nsew")

        self._build_progress()
        self._build_summary()
        
        # Store references for export and load functionality
        self._fit_result = None
        self._output_dir = None
        self._current_npz_path = None  # Track current fit NPZ file location
        self._export_callback = None
        self._load_callback = None
        self._save_npz_callback = None

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
        ttk.Button(btn_bar, text="Load Fitted Data…", command=self._load_fitted_data).pack(side="left", padx=4)
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

    def _on_export_clicked(self):
        """Handle export button click."""
        try:
            print(f"[Export Button] Clicked - callback={self._export_callback is not None}, fit_result={self._fit_result is not None}, output_dir={self._output_dir}")
            if self._export_callback and self._fit_result and self._output_dir:
                print(f"[Export Button] Calling callback...")
                self._export_callback(self._fit_result, self._output_dir)
            else:
                print(f"[Export Button] Missing: callback={self._export_callback} fit_result={self._fit_result is not None} output_dir={self._output_dir}")
        except Exception as e:
            print(f"[Export Button Error] {e}")
            import traceback
            traceback.print_exc()
    
    def set_fit_result(self, fit_result: dict, output_dir: str, npz_path: str = None):
        """Store fit result and enable export/save buttons."""
        self._fit_result = fit_result
        self._output_dir = output_dir
        if npz_path:
            self._current_npz_path = npz_path
        # Enable buttons if there are images to export
        has_images = any(isinstance(v, np.ndarray) for v in (fit_result or {}).values())
        self._export_btn.configure(state="normal" if has_images else "disabled")
        # Always enable save NPZ button when fit result is available
        self._save_npz_btn.configure(state="normal")
    
    def set_export_callback(self, callback):
        """Set the callback function for export button."""
        self._export_callback = callback
    
    def set_load_callback(self, callback):
        """Set the callback function for loading fitted data."""
        self._load_callback = callback
    
    def set_save_npz_callback(self, callback):
        """Set the callback function for saving NPZ."""
        self._save_npz_callback = callback
    
    def _on_save_npz_clicked(self):
        """Handle save NPZ button click."""
        try:
            if self._save_npz_callback and self._output_dir:
                self._save_npz_callback(self._output_dir)
        except Exception as e:
            print(f"[Save NPZ Error] {e}")
            import traceback
            traceback.print_exc()
    
    def _load_fitted_data(self):
        """Show file dialog to load previously fitted ROI data from NPZ."""
        npz_file = filedialog.askopenfilename(
            title="Load Fitted Data",
            filetypes=[("NumPy Archives", "*.npz"), ("All", "*.*")],
            defaultextension=".npz")
        if not npz_file:
            return
        
        if self._load_callback:
            self._load_callback(npz_file)

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
        
        # Add export and save buttons below treeview
        btn_bar = ttk.Frame(f)
        btn_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self._export_btn = ttk.Button(btn_bar, text="Export Images…", 
                                     command=self._on_export_clicked, state="disabled")
        self._export_btn.pack(side="left", padx=4)
        
        self._save_npz_btn = ttk.Button(btn_bar, text="Save NPZ", 
                                       command=self._on_save_npz_clicked, state="disabled")
        self._save_npz_btn.pack(side="left", padx=4)

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


class _UIBuilder:
    """Common UI building methods (used by both themed and fallback versions)."""

    def _make_scroll_frame(self, parent: ttk.Frame) -> tuple:
        """Create a vertically scrollable frame. Returns (outer_frame, inner_content_frame)."""
        outer = ttk.Frame(parent)
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

        def _on_mousewheel(evt):
            # Windows: evt.delta, Mac: evt.delta, Linux: evt.num
            if evt.num == 5 or evt.delta < 0:
                canvas.yview_scroll(3, "units")
            elif evt.num == 4 or evt.delta > 0:
                canvas.yview_scroll(-3, "units")

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        # Bind mousewheel events for all platforms
        canvas.bind("<MouseWheel>", _on_mousewheel)  # Windows & Mac
        canvas.bind("<Button-4>", _on_mousewheel)    # Linux scroll up
        canvas.bind("<Button-5>", _on_mousewheel)    # Linux scroll down
        inner.bind("<MouseWheel>", _on_mousewheel)
        inner.bind("<Button-4>", _on_mousewheel)
        inner.bind("<Button-5>", _on_mousewheel)

        # Store canvas and window_id references on the inner frame for later refresh
        inner._canvas = canvas
        inner._window_id = window_id
        outer._canvas = canvas
        outer._window_id = window_id

        return outer, inner

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
        """Build the entire user interface with left tab buttons and central content area."""
        self._buf: list = []
        self._current_session_file = None  # Track current session file for auto-save
        self._current_npz_path = None  # For backward compatibility
        self._last_loaded_ptu = None  # Guard against duplicate auto-loads

        # Main horizontal PanedWindow: left (tabs+content+results) | right (FOV preview)
        main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        
        # LEFT PANEL: Tab buttons (vertical) + Content area (vertical paned)
        
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=3)  # 3:2 weight ratio
        left_frame.columnconfigure(0, weight=0)  # Tab buttons (narrow)
        left_frame.columnconfigure(1, weight=1)  # Content area (expand)
        left_frame.rowconfigure(0, weight=1)

        # Left sidebar with vertical buttons
        btn_frame = ttk.Frame(left_frame, width=100)
        btn_frame.grid(row=0, column=0, sticky="ns", padx=(0, 4))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.grid_propagate(False)

        ttk.Label(btn_frame, text="Modes", font=("TkDefaultFont", 9, "bold")).pack(pady=(4, 8))

        self._form_buttons = {}
        self._form_frames = {}
        self._current_form = None

        def make_button_click(form_name):
            def on_click():
                self._switch_form(form_name)
            return on_click

        # Create buttons for each form
        form_list = [
            ("fov", "Single FOV Fit"),
            ("stitch", "Tile Stitch/Fit"),
            ("batch", "Batch Processing"),
            ("irf", "Machine IRF"),
            ("phasor", "Phasor Analysis"),
        ]

        for form_id, form_label in form_list:
            btn = ttk.Button(btn_frame, text=form_label, command=make_button_click(form_id))
            btn.pack(fill="x", padx=2, pady=2)
            self._form_buttons[form_id] = btn

        # Vertical PanedWindow: content (top) | results (bottom)
        content_paned = ttk.PanedWindow(left_frame, orient="vertical")
        content_paned.grid(row=0, column=1, sticky="nsew")
        left_frame.columnconfigure(1, weight=1)
        left_frame.rowconfigure(0, weight=1)

        # Create wrapper frames for each form (initially hidden)
        form_wrapper = ttk.Frame(content_paned)
        content_paned.add(form_wrapper, weight=2)
        form_wrapper.columnconfigure(0, weight=1)
        form_wrapper.rowconfigure(0, weight=1)

        #  Analysis Notebook: Fit Settings | ROI Analysis 
        # Only shown for FOV and Stitch modes
        self._analysis_tabs = ttk.Notebook(form_wrapper)
        self._analysis_tabs.grid(row=0, column=0, sticky="nsew")
        self._analysis_tabs.grid_remove()  # Hidden by default
        
        # Tab 0: Fit Settings (form content goes here)
        fit_settings_outer = ttk.Frame(self._analysis_tabs)
        self._analysis_tabs.add(fit_settings_outer, text="  Fit Settings  ")
        fit_settings_outer.columnconfigure(0, weight=1)
        fit_settings_outer.rowconfigure(0, weight=1)
        self._fit_settings_tab = fit_settings_outer
        
        # Tab 1: ROI Analysis
        roi_frame = ttk.Frame(self._analysis_tabs, padding=4)
        self._analysis_tabs.add(roi_frame, text="  ROI Analysis  ")
        roi_frame.columnconfigure(0, weight=1)
        roi_frame.rowconfigure(0, weight=1)
        self._roi_analysis_panel = RoiAnalysisPanel(roi_frame)
        self._roi_analysis_panel.grid(row=0, column=0, sticky="nsew")
        self._roi_analysis_frame = roi_frame

        self._form_content_frame = form_wrapper
        self._form_inner_frames = {}

        # Create scrollable frames for each form
        # For FOV/Stitch: forms go in "Fit Settings" tab and can be toggled
        # For others: forms in traditional layout
        for form_id, form_label in form_list:
            if form_id in ("fov", "stitch"):
                # FOV and Stitch forms go inside the Fit Settings tab
                outer, inner = self._make_scroll_frame(self._fit_settings_tab)
                outer.grid(row=0, column=0, sticky="nsew")
                outer.grid_remove()  # Hide initially; show on demand
                self._form_inner_frames[form_id] = (outer, inner)
            else:
                # Batch, IRF, Phasor forms use traditional layout
                outer, inner = self._make_scroll_frame(form_wrapper)
                outer.grid(row=0, column=0, sticky="nsew")
                outer.grid_remove()
                self._form_inner_frames[form_id] = (outer, inner)
            self._form_frames[form_id] = self._form_inner_frames[form_id]

        # Bottom: Results panel (progress, summary, images)
        results_frame = ttk.Frame(content_paned)
        content_paned.add(results_frame, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self._res = ResultsPanel(results_frame, root=self.root)
        self._res.grid(row=0, column=0, sticky="nsew")
        # Set callbacks for export, save, and load buttons
        self._res.set_export_callback(self._show_export_dialog)
        self._res.set_load_callback(self._load_fitted_data_from_file)
        self._res.set_save_npz_callback(self._save_npz_quick)

        # Build form content for each form
        self._build_fov_tab()
        self._build_stitch_tab()
        self._build_batch_tab()
        self._build_machine_irf_tab()
        self._build_phasor_tab()

        
        # RIGHT PANEL: FOV Preview
        
        preview_frame = ttk.LabelFrame(main_paned, text="  FOV Preview  ", padding=4)
        main_paned.add(preview_frame, weight=2)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self._fov_preview = FOVPreviewPanel(preview_frame)
        self._fov_preview.grid(row=0, column=0, sticky="nsew")
        
        # Connect ROI Analysis panel to FOV preview 
        self._roi_analysis_panel.fov_preview = self._fov_preview
        self._fov_preview._roi_analysis_panel = self._roi_analysis_panel

        # Phasor panel shares the same right-panel cell; hidden until needed.
        self._phasor_panel = PhasorViewPanel(preview_frame, max_cursors=6)
        self._phasor_panel.frame.grid(row=0, column=0, sticky="nsew")
        self._phasor_panel.frame.grid_remove()
        self._preview_frame_label = preview_frame

        # Show first form by default
        self._switch_form("fov")

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

    def _refresh_scrollable_frame(self, form_id: str):
        """Refresh the scrollable frame canvas after it's been shown. Fixes display issues when switching back to hidden forms."""
        if form_id not in self._form_inner_frames:
            return
        
        outer, inner = self._form_inner_frames[form_id]
        
        # Schedule refresh on next event loop iteration to allow grid() to complete
        def do_refresh():
            try:
                # Process all pending events first
                self.root.update_idletasks()
                
                # For notebook-contained forms (FOV/Stitch), also update the notebook
                if form_id in ("fov", "stitch"):
                    # Update the notebook to reflect size changes
                    self._fit_settings_tab.update_idletasks()
                    self._analysis_tabs.update_idletasks()
                
                # Now refresh the canvas
                if hasattr(outer, '_canvas') and hasattr(outer, '_window_id'):
                    canvas = outer._canvas
                    window_id = outer._window_id
                    
                    # Make sure outer frame is properly laid out
                    outer.update()
                    
                    # Get canvas dimensions
                    canvas_width = canvas.winfo_width()
                    canvas_height = canvas.winfo_height()
                    
                    # Ensure canvas window width matches canvas width
                    if canvas_width > 1:
                        canvas.itemconfigure(window_id, width=canvas_width)
                    
                    # Update inner frame to get its requested size
                    inner.update()
                    inner_height = inner.winfo_reqheight()
                    
                    # Configure window height to inner frame's requested height
                    # This ensures scrollbar works properly
                    if inner_height > 1:
                        canvas.itemconfigure(window_id, height=inner_height)
                    
                    # Recalculate scroll region based on all content
                    bbox = canvas.bbox("all")
                    if bbox:
                        canvas.configure(scrollregion=bbox)
                    else:
                        # Fallback: use reasonable defaults
                        h = max(canvas_height, inner_height, 300) if inner_height > 1 else 500
                        w = max(canvas_width, 300) if canvas_width > 1 else 400
                        canvas.configure(scrollregion=(0, 0, w, h))
                    
                    # Reset scroll position to top
                    canvas.yview_moveto(0)
                    
                    # Final update to ensure everything is rendered
                    canvas.update()
            except Exception as e:
                # Silently ignore errors if widgets have been destroyed
                pass
        
        # Schedule with after() to let grid() complete first (200ms)
        self.root.after(200, do_refresh)

    def _switch_form(self, form_id: str):
        """Switch to the specified form and update preview panel accordingly."""
        # Hide all buttons' active state
        for btn in self._form_buttons.values():
            btn.state(["!pressed"])

        # Show selected form
        if form_id in self._form_inner_frames:
            self._form_buttons[form_id].state(["pressed"])
            self._current_form = form_id

            # For FOV and Stitch: hide other analysis forms and show their notebook
            if form_id in ("fov", "stitch"):
                # Hide ALL other forms (batch, irf, phasor, and the other FOV/Stitch)
                for fid in ("batch", "irf", "phasor"):
                    if fid in self._form_inner_frames:
                        self._form_inner_frames[fid][0].grid_remove()
                
                # Hide the other FOV/Stitch form
                other_id = "stitch" if form_id == "fov" else "fov"
                other_frame = self._form_inner_frames[other_id][0]
                other_frame.grid_remove()
                
                # Now show the selected form
                selected_frame = self._form_inner_frames[form_id][0]
                selected_frame.grid(row=0, column=0, sticky="nsew")
                
                # Raise the frame to ensure it's on top
                selected_frame.lift()
                selected_frame.tkraise()
                
                # Now show the notebook
                self._analysis_tabs.grid(row=0, column=0, sticky="nsew")
                self._analysis_tabs.lift()
                self._analysis_tabs.tkraise()
                
                # Select Fit Settings tab
                self._analysis_tabs.select(0)
                
                # Force notebook update before refresh
                self._fit_settings_tab.update()
                
                # Refresh canvas to ensure content displays properly
                self._refresh_scrollable_frame(form_id)
            else:
                # For other forms: hide notebook first
                self._analysis_tabs.grid_remove()
                
                # Hide all other traditional forms AND FOV/Stitch forms
                for fid in ("batch", "irf", "phasor", "fov", "stitch"):
                    if fid != form_id and fid in self._form_inner_frames:
                        self._form_inner_frames[fid][0].grid_remove()
                
                # Show the selected form
                if form_id in self._form_inner_frames:
                    selected_frame = self._form_inner_frames[form_id][0]
                    selected_frame.grid(row=0, column=0, sticky="nsew")
                    selected_frame.lift()
                    selected_frame.tkraise()
                    # Refresh canvas to ensure content displays properly
                    self._refresh_scrollable_frame(form_id)

            # Update preview panel based on form
            if form_id == "phasor":
                self._fov_preview.frame.grid_remove()
                self._phasor_panel.frame.grid()
                self._preview_frame_label.configure(text="  Phasor Analysis  ")
            else:
                self._phasor_panel.frame.grid_remove()
                self._fov_preview.frame.grid()
                self._preview_frame_label.configure(text="  FOV Preview  ")

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

    def _capture_form_state(self) -> dict:
        """Capture all current form field values into a serializable dict."""
        try:
            state = {
                # Current active form mode
                "active_form": self.iv_form.get() if hasattr(self, 'iv_form') else 0,
                
                # Input files (common to all)
                "ptu_file": self.sv_ptu.get() if hasattr(self, 'sv_ptu') else "",
                "xlsx_file": self.sv_xlsx.get() if hasattr(self, 'sv_xlsx') else "",
                
                # IRF settings (common to all)
                "irf_method": self._irf_fov.sv_method.get() if hasattr(self, '_irf_fov') else "irf_xlsx",
                "irf_file": self._irf_fov.sv_path.get() if hasattr(self, '_irf_fov') else "",
                
                # FOV mode fitting parameters
                "nexp_fov": self.iv_nexp_fov.get() if hasattr(self, 'iv_nexp_fov') else 3,
                "tau_fit_lo": self.sv_tau_fit_lo.get() if hasattr(self, 'sv_tau_fit_lo') else "0.1",
                "tau_fit_hi": self.sv_tau_fit_hi.get() if hasattr(self, 'sv_tau_fit_hi') else "10.0",
                
                # Stitch mode parameters
                "nexp_st": self.iv_nexp_st.get() if hasattr(self, 'iv_nexp_st') else 3,
                
                # Batch mode parameters
                "nexp_batch": self.iv_nexp_batch.get() if hasattr(self, 'iv_nexp_batch') else 3,
                
                # Other common settings
                "register": self.bv_register.get() if hasattr(self, 'bv_register') else False,
                "channel": self.sv_channel_focus.get() if hasattr(self, 'sv_channel_focus') else "auto",
                "threshold": self.sv_int_threshold.get() if hasattr(self, 'sv_int_threshold') else "5",

                "out_fov":        self.sv_out_fov.get() if hasattr(self, 'sv_out_fov') else "",
                "mode_fov":       self.sv_mode_fov.get() if hasattr(self, 'sv_mode_fov') else "both",
                "tau_min_fov":    self.sv_tau_min_fov.get() if hasattr(self, 'sv_tau_min_fov') else "",
                "tau_max_fov":    self.sv_tau_max_fov.get() if hasattr(self, 'sv_tau_max_fov') else "",
                "thr_fov_en":     self.bv_thr_fov.get() if hasattr(self, 'bv_thr_fov') else False,
                "thr_fov_val":    self.sv_thr_fov.get() if hasattr(self, 'sv_thr_fov') else "5",
                "cell_mask":      self.bv_cell.get() if hasattr(self, 'bv_cell') else False,
            }
            print(f"[Session] Captured form state: active_form={state.get('active_form')}")
            return state
        except Exception as e:
            print(f"[Session] Could not capture form state: {e}")
            return {}

    def _restore_form_state(self, state: dict):
        """Restore all form field values from captured state dict."""
        try:
            # Restore input files (always available)
            if "ptu_file" in state and hasattr(self, 'sv_ptu'):
                self.sv_ptu.set(state["ptu_file"])
            if "xlsx_file" in state and hasattr(self, 'sv_xlsx'):
                self.sv_xlsx.set(state["xlsx_file"])
            
            # Restore IRF settings
            if "irf_method" in state and hasattr(self, '_irf_fov'):
                self._irf_fov.sv_method.set(state["irf_method"])
            if "irf_file" in state and hasattr(self, '_irf_fov'):
                self._irf_fov.sv_path.set(state["irf_file"])
            
            # Restore fitting parameters (for all modes)
            if "nexp_fov" in state and hasattr(self, 'iv_nexp_fov'):
                self.iv_nexp_fov.set(state["nexp_fov"])
            if "nexp_st" in state and hasattr(self, 'iv_nexp_st'):
                self.iv_nexp_st.set(state["nexp_st"])
            if "nexp_batch" in state and hasattr(self, 'iv_nexp_batch'):
                self.iv_nexp_batch.set(state["nexp_batch"])
            
            if "tau_fit_lo" in state and hasattr(self, 'sv_tau_fit_lo'):
                self.sv_tau_fit_lo.set(state["tau_fit_lo"])
            if "tau_fit_hi" in state and hasattr(self, 'sv_tau_fit_hi'):
                self.sv_tau_fit_hi.set(state["tau_fit_hi"])

            if "out_fov" in state and hasattr(self, 'sv_out_fov'): self.sv_out_fov.set(state["out_fov"])
            if "mode_fov" in state and hasattr(self, 'sv_mode_fov'): self.sv_mode_fov.set(state["mode_fov"])
            if "tau_min_fov" in state and hasattr(self, 'sv_tau_min_fov'): self.sv_tau_min_fov.set(state["tau_min_fov"])
            if "tau_max_fov" in state and hasattr(self, 'sv_tau_max_fov'): self.sv_tau_max_fov.set(state["tau_max_fov"])
            if "thr_fov_en" in state and hasattr(self, 'bv_thr_fov'): self.bv_thr_fov.set(state["thr_fov_en"])
            if "thr_fov_val" in state and hasattr(self, 'sv_thr_fov'): self.sv_thr_fov.set(state["thr_fov_val"])
            if "cell_mask" in state and hasattr(self, 'bv_cell'): self.bv_cell.set(state["cell_mask"])    
                        
            # Restore other settings
            if "register" in state and hasattr(self, 'bv_register'):
                self.bv_register.set(state["register"])
            if "channel" in state and hasattr(self, 'sv_channel_focus'):
                self.sv_channel_focus.set(state["channel"])
            if "threshold" in state and hasattr(self, 'sv_int_threshold'):
                self.sv_int_threshold.set(state["threshold"])
            
            # Restore active form mode (switch to the right tab)
            if "active_form" in state and hasattr(self, 'iv_form'):
                self.iv_form.set(state["active_form"])
                self._switch_form([None, "fov", "stitch", "batch", "irf", "phasor"][state["active_form"]])
            
            print(f"[Session] Restored form state")
        except Exception as e:
            print(f"[Session] Could not restore form state: {e}")
            import traceback
            traceback.print_exc()

    def _save_roi_progress(self, path: str, fit_result: dict, summary_rows: list):
        """Save comprehensive session: fit results, form state, and file paths - ALL IN ONE FILE."""
        try:
            from pathlib import Path
            import json
            from datetime import datetime
            import numpy as np
            
            base_path = Path(path)
            # For PTU files, save next to the PTU; for directories, save inside it
            if base_path.is_file():
                session_file = base_path.parent / f"{base_path.stem}.roi_session.npz"
            else:
                session_file = base_path / "roi_session.npz"
            
            # Capture current form state
            form_state = self._capture_form_state()
            
            # Build comprehensive session data - ONE FILE WITH EVERYTHING
            session_data = {
                "timestamp": datetime.now().isoformat(),
                "source": str(path),
                "form_state_json": json.dumps(form_state, default=str),  # All form settings
            }
            
            # Save all numpy arrays from fit_result
            print(f"[Session] Saving fit_result keys: {list(fit_result.keys())}")
            for key, val in fit_result.items():
                if isinstance(val, np.ndarray):
                    session_data[key] = val
                    print(f"  ✓ Saved array: {key} {val.shape}")
                elif isinstance(val, dict):
                    # Hoist any numpy arrays out of the dict before JSON-ing it,
                    # otherwise default=str silently corrupts them (e.g. global_summary['model']).
                    try:
                        hoisted = {}
                        json_safe = {}
                        for k2, v2 in val.items():
                            if isinstance(v2, np.ndarray):
                                arr_key = f"{key}_arr_{k2}"
                                hoisted[arr_key] = v2
                                print(f"  ✓ Hoisted array from {key}: {k2} {v2.shape}")
                            else:
                                json_safe[k2] = v2
                        session_data[f"{key}_json"] = json.dumps(json_safe, default=str)
                        session_data.update(hoisted)
                        print(f"  ✓ Saved dict: {key} ({len(hoisted)} arrays hoisted)")
                    except Exception as e:
                        print(f"  ✗ Could not save dict {key}: {e}")
                elif isinstance(val, (list, tuple)):
                    # Save lists/tuples as arrays if they're numeric
                    try:
                        arr = np.array(val)
                        if arr.dtype != object:
                            session_data[key] = arr
                            print(f"  ✓ Saved list: {key} {arr.shape}")
                    except:
                        pass
                elif val is not None and not callable(val):
                    # Save scalar metadata
                    try:
                        session_data[key] = val
                    except:
                        pass
            
            # Add summary table
            summary_params = []
            summary_values = []
            summary_units = []
            for param, value, unit in summary_rows:
                summary_params.append(param)
                summary_values.append(value)
                summary_units.append(unit)
            
            if summary_params:
                session_data["summary_params"] = np.array(summary_params, dtype=object)
                session_data["summary_values"] = np.array(summary_values, dtype=object)
                session_data["summary_units"] = np.array(summary_units, dtype=object)
            
            # Add FOV preview state
            if self._fov_preview._ptu_path:
                session_data["fov_ptu_path"] = self._fov_preview._ptu_path
            if self._fov_preview._lifetime_map is not None:
                session_data["fov_lifetime_map"] = self._fov_preview._lifetime_map
            if self._fov_preview._intensity_map is not None:
                session_data["fov_intensity_map"] = self._fov_preview._intensity_map
            session_data["fov_color_scale"] = json.dumps(self._fov_preview._flim_color_scale)
            session_data["fov_n_exp"] = self._fov_preview._n_exp
            
            # Save regions
            session_data["fov_regions"] = self._fov_preview._roi_manager.to_json()
            
            # Write single comprehensive session NPZ file
            np.savez_compressed(session_file, **session_data)
            print(f"✓ Session saved: {session_file}")
            print(f"  Saved {len(session_data)} items (fit results + form state + FOV preview)")
            
            # Also store the path for quick save
            self._current_session_file = str(session_file)
            
        except Exception as e:
            import traceback
            print(f"[Save Error] {e}")
            traceback.print_exc()

    def _load_roi_session(self, session_path: str) -> dict:
        """Load complete session from .roi_session.npz file."""
        try:
            import numpy as np
            import json
            
            data = np.load(session_path, allow_pickle=True)
            loaded = {}
            
            # Extract all arrays from NPZ
            for key in data.files:
                val = data[key]
                # Convert object arrays back to lists if needed
                if hasattr(val, 'dtype') and val.dtype == object:
                    val = val.tolist()
                loaded[key] = val
            
            print(f"✓ Loaded session from {session_path}")
            # Also store for later use
            self._current_session_file = session_path
            return loaded
        except Exception as e:
            print(f"[Load Session Error] {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _auto_load_session_for_ptu(self, ptu_path: str):
        """Check for and auto-load session file when PTU is selected."""
        try:
            from pathlib import Path
            import json
            import numpy as np
            
            ptu_path = Path(ptu_path)
            session_file = ptu_path.parent / f"{ptu_path.stem}.roi_session.npz"
            
            if session_file.exists():
                print(f"[Auto-Load] Found session for PTU: {session_file.name}")
                
                # Load the session
                session_data = self._load_roi_session(str(session_file))
                if not session_data:
                    return
                
                # Restore form state
                if "form_state_json" in session_data:
                    try:
                        form_state_str = session_data["form_state_json"]
                        # Handle numpy array scalar
                        if isinstance(form_state_str, np.ndarray):
                            form_state_str = form_state_str.item()
                        if isinstance(form_state_str, bytes):
                            form_state_str = form_state_str.decode('utf-8')
                        form_state = json.loads(form_state_str)
                        self._restore_form_state(form_state)
                    except Exception as e:
                        print(f"[Auto-Load] Could not restore form state: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Restore fit results to display
                try:
                    # Extract summary
                    params = session_data.get("summary_params", [])
                    values = session_data.get("summary_values", [])
                    units = session_data.get("summary_units", [])
                    
                    if isinstance(params, np.ndarray):
                        params = params.tolist()
                    if isinstance(values, np.ndarray):
                        values = values.tolist()
                    if isinstance(units, np.ndarray):
                        units = units.tolist()
                    
                    rows = []
                    for param, val, unit in zip(params, values, units):
                        if isinstance(param, bytes):
                            param = param.decode('utf-8')
                        if isinstance(val, bytes):
                            val = val.decode('utf-8')
                        if isinstance(unit, bytes):
                            unit = unit.decode('utf-8')
                        rows.append((str(param), str(val), str(unit)))
                    
                    # Display summary
                    if rows:
                        self._res.populate_summary(rows)
                    
                    # BUG 1 FIX: restore drawn regions unconditionally.
                    # fov_regions is saved even when no fit has been run yet,
                    # so we must not gate it on the map keys.
                    if "fov_regions" in session_data:
                        try:
                            import json as _json_roi
                            regions_json = session_data["fov_regions"]
                            if isinstance(regions_json, bytes):
                                regions_json = regions_json.decode("utf-8")
                            self._fov_preview._load_regions_from_json(regions_json)
                            
                            # Refresh the RoiAnalysisPanel tree list to show restored regions
                            if self._roi_analysis_panel is not None:
                                self._roi_analysis_panel._refresh_region_list()
                        except Exception as _e:
                            print(f"[Auto-Load] Could not restore regions: {_e}")

                    # Restore FOV preview
                    if "fov_intensity_map" in session_data and "fov_lifetime_map" in session_data:
                        intensity = session_data["fov_intensity_map"]
                        lifetime = session_data["fov_lifetime_map"]
                        
                        if isinstance(intensity, np.ndarray):
                            self._fov_preview._intensity_map = intensity
                        if isinstance(lifetime, np.ndarray):
                            self._fov_preview._lifetime_map = lifetime
                        
                        # Restore color scale
                        if "fov_color_scale" in session_data:
                            try:
                                import json
                                cs = session_data["fov_color_scale"]
                                if isinstance(cs, bytes):
                                    cs = cs.decode('utf-8')
                                self._fov_preview._flim_color_scale = json.loads(cs)
                            except:
                                pass

                        if "fov_n_exp" in session_data:
                            n_exp = session_data["fov_n_exp"]
                            if isinstance(n_exp, (np.integer, int)):
                                self._fov_preview._n_exp = int(n_exp)

                        # BUG 2 FIX: render the FLIM image now that maps, colour
                        # scale, n_exp, and regions are all restored.
                        # _update_flim_display calls _redraw_region_overlays() at
                        # its end, so patches appear on top of the rendered image.
                        try:
                            self._fov_preview._update_flim_display()
                        except Exception as _e:
                            print(f"[Auto-Load] Could not render FLIM display: {_e}")

                        try:
                            # Redraw the FOV preview decay with fit overlay
                            ax_decay = self._fov_preview._ax_decay
                            ax_decay.clear()
                            if "decay" in session_data and "time_ns" in session_data:
                                decay   = session_data["decay"]
                                time_ns = session_data["time_ns"]
                                if isinstance(decay, np.ndarray) and isinstance(time_ns, np.ndarray):
                                    ax_decay.semilogy(time_ns, decay, 'o-', color="steelblue",
                                                    linewidth=1.5, markersize=3, label="Measured", alpha=0.7)
                                    irf = session_data.get("irf_prompt")
                                    if irf is not None and isinstance(irf, np.ndarray) and irf.max() > 0:
                                        irf_scaled = (irf / irf.max()) * decay.max() * 0.2
                                        ax_decay.semilogy(time_ns[:len(irf)], np.maximum(irf_scaled, 1e-2),
                                                        color="orange", linewidth=2.0, label="IRF", alpha=0.8)
                                    # Reconstruct global_summary with hoisted arrays
                                    gs = _reconstruct_dict_from_session(session_data, "global_summary")
                                    model = gs.get('model')
                                    # Safely convert string model back to array if needed
                                    if model is not None and isinstance(model, str):
                                        model = _safe_array_from_json(model)
                                    if model is not None and len(model) > 0:
                                        ax_decay.semilogy(time_ns, model, color="red", linewidth=2.0,
                                                        label="Fitted", alpha=0.8)
                                    ax_decay.legend(fontsize=8, loc="upper right")
                            ax_decay.set_title("Summed Decay (reloaded)", fontsize=10, fontweight="bold")
                            ax_decay.set_xlabel("Time (ns)")
                            ax_decay.set_ylabel("Photon Count")
                            ax_decay.grid(True, alpha=0.3)
                            ax_decay.tick_params(labelsize=8)

                            # Update status
                            self._res._status.set("✓ Session restored — ready to export or re-fit")
                            self._fov_preview._ctrl_frame.grid()
                            self._res._export_btn.configure(state="normal")
                            self._fov_preview._canvas_mpl.draw_idle()
                            print("[Auto-Load] ✓ Session restored")
                        except Exception as e:
                            print(f"[Auto-Load] Could not redraw FOV: {e}")
                    
                except Exception as e:
                    print(f"[Auto-Load] Could not restore results: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[Auto-Load] No session found for {ptu_path.name}")
        except Exception as e:
            print(f"[Auto-Load Error] {e}")

    def load_roi_fit(self, npz_path: str) -> dict:
        """Load previously saved ROI fit data from NPZ file."""
        try:
            import numpy as np
            
            data = np.load(npz_path, allow_pickle=True)
            loaded = {}
            
            # Extract all arrays from NPZ
            for key in data.files:
                val = data[key]
                # Convert object arrays back to lists if needed
                if hasattr(val, 'dtype') and val.dtype == object:
                    val = val.tolist()
                loaded[key] = val
            
            print(f"✓ Loaded ROI fit from {npz_path}")
            return loaded
        except Exception as e:
            print(f"✗ Failed to load ROI fit: {e}")
            return {}

    def _save_npz_quick(self, output_dir: str):
        """Quick save session: prompt user for location and copy session file."""
        try:
            from pathlib import Path
            import shutil
            import numpy as np
            
            # Find the existing session file - prefer stored path, then look next to PTU
            session_source = None
            
            # First try the stored session path (from loading or just saved)
            if self._current_session_file:
                session_source = Path(self._current_session_file)
                if session_source.exists():
                    print(f"[Save Session] Using stored session path: {session_source}")
                else:
                    print(f"[Save Session] Stored path no longer exists: {session_source}")
                    session_source = None
            
            # Fallback: look for session file next to PTU file
            if not session_source:
                ptu_path = self._fov_preview._ptu_path if self._fov_preview else None
                
                # Convert ptu_path to string if it's an ndarray or bytes
                if ptu_path is not None:
                    if isinstance(ptu_path, np.ndarray):
                        ptu_path = ptu_path.item() if ptu_path.ndim == 0 else str(ptu_path[0])
                    if isinstance(ptu_path, bytes):
                        ptu_path = ptu_path.decode('utf-8')
                    ptu_path = str(ptu_path)
                
                if ptu_path:
                    base_path = Path(ptu_path)
                    if base_path.is_file():
                        session_source = base_path.parent / f"{base_path.stem}.roi_session.npz"
                        if session_source.exists():
                            print(f"[Save Session] Found session next to PTU: {session_source}")
                        else:
                            session_source = None
            
            if not session_source or not session_source.exists():
                messagebox.showwarning("No session data", "No saved session (.roi_session.npz) found.\n\nRun a fit first to create a session.")
                return
            
            # Ask user where to save
            output_path = Path(filedialog.askdirectory(
                title="Save Session File",
                initialdir=output_dir))
            
            if not output_path or output_path == Path():
                return  # User cancelled
            
            # Check if source and dest are the same file
            session_dest = output_path / session_source.name
            
            if session_source.samefile(session_dest) if session_dest.exists() else session_source == session_dest:
                messagebox.showinfo("Already Saved", 
                    f"Session already saved at:\n{session_source}")
                return
            
            # If destination exists, ask to override
            if session_dest.exists():
                response = messagebox.askyesno("File Exists", 
                    f"File already exists:\n{session_dest.name}\n\nOverride?")
                if not response:
                    return
            
            # Copy to chosen location
            session_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(session_source, session_dest)
            
            messagebox.showinfo("Success", f"Session saved:\n{session_dest.name}\nat {output_path}")
            print(f"✓ Session saved: {session_dest}")
            
        except Exception as e:
            print(f"[Save Session Error] {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to save session:\n{str(e)}")

    def _load_fitted_data_from_file(self, npz_path: str):
        """Load previously fitted data from NPZ and display in results panel."""
        try:
            import numpy as np
            import json
            from pathlib import Path
            
            fit_result = self.load_roi_fit(npz_path)
            if not fit_result:
                messagebox.showerror("Load Error", f"Failed to load fitted data from:\n{npz_path}")
                return
            
            # Extract summary data
            params = fit_result.get("summary_params", [])
            values = fit_result.get("summary_values", [])
            units = fit_result.get("summary_units", [])
            
            # Build rows for display
            rows = []
            if isinstance(params, np.ndarray):
                params = params.tolist()
            if isinstance(values, np.ndarray):
                values = values.tolist()
            if isinstance(units, np.ndarray):
                units = units.tolist()
            
            for param, val, unit in zip(params, values, units):
                if isinstance(param, bytes):
                    param = param.decode('utf-8')
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                if isinstance(unit, bytes):
                    unit = unit.decode('utf-8')
                rows.append((str(param), str(val), str(unit)))
            
            # Display in results panel
            try:
                if rows:
                    self._res.populate_summary(rows)
            except Exception as populate_err:
                import traceback
                traceback.print_exc()
            
            # Restore FOV preview state if available
            try:
                if "fov_ptu_path" in fit_result:
                    ptu_path = fit_result["fov_ptu_path"]
                    # Handle various types: bytes, ndarray, str
                    if isinstance(ptu_path, bytes):
                        ptu_path = ptu_path.decode('utf-8')
                    elif isinstance(ptu_path, np.ndarray):
                        # Convert 0-d array to scalar
                        ptu_path = ptu_path.item() if ptu_path.ndim == 0 else str(ptu_path[0])
                        if isinstance(ptu_path, bytes):
                            ptu_path = ptu_path.decode('utf-8')
                    self._fov_preview._ptu_path = str(ptu_path) if ptu_path else None
                
                if "fov_lifetime_map" in fit_result:
                    lifetime = fit_result["fov_lifetime_map"]
                    if isinstance(lifetime, np.ndarray):
                        self._fov_preview._lifetime_map = lifetime
                
                if "fov_intensity_map" in fit_result:
                    intensity = fit_result["fov_intensity_map"]
                    if isinstance(intensity, np.ndarray):
                        self._fov_preview._intensity_map = intensity
                
                if "fov_color_scale" in fit_result:
                    try:
                        cs = fit_result["fov_color_scale"]
                        if isinstance(cs, bytes):
                            cs = cs.decode('utf-8')
                        self._fov_preview._flim_color_scale = json.loads(cs)
                    except:
                        pass
            except Exception as fov_err:
                import traceback
                traceback.print_exc()
            
            import sys

            sys.stdout.flush()
            
            if "fov_regions" in fit_result:
                try:
                    regions_json = fit_result["fov_regions"]
                    
                    # Handle numpy array (0-d array containing string)
                    if isinstance(regions_json, np.ndarray):
                        regions_json = regions_json.item() if regions_json.ndim == 0 else regions_json[0]
                    
                    if isinstance(regions_json, bytes):
                        regions_json = regions_json.decode('utf-8')
                    
                    self._fov_preview._load_regions_from_json(regions_json)
                    
                    # Refresh the RoiAnalysisPanel tree list to show restored regions
                    if self._roi_analysis_panel is not None:
                        self._roi_analysis_panel._refresh_region_list()
                except Exception as e:
                    import traceback
                    traceback.print_exc()
            
            if "fov_n_exp" in fit_result:
                n_exp = fit_result["fov_n_exp"]
                if isinstance(n_exp, (np.integer, int)):
                    self._fov_preview._n_exp = int(n_exp)
            
            # Redraw FOV preview with restored data
            try:
                if self._fov_preview._lifetime_map is not None and self._fov_preview._intensity_map is not None:
                    from flimkit.UI import flim_display

                    intensity = self._fov_preview._intensity_map
                    lifetime  = self._fov_preview._lifetime_map

                    ax_img  = self._fov_preview._ax_img
                    ax_flim = self._fov_preview._ax_flim
                    ax_cbar = self._fov_preview._ax_cbar
                    fig     = self._fov_preview._fig

                    # — Intensity image —
                    ax_img.clear()
                    intensity_clipped = np.clip(intensity, 0, np.percentile(intensity, 99))
                    ax_img.imshow(intensity_clipped, cmap="inferno", origin="upper")
                    ax_img.set_title("Intensity Image", fontsize=10, fontweight="bold")
                    ax_img.set_xlabel("X (pixels)")
                    ax_img.set_ylabel("Y (pixels)")

                    # — FLIM image (FIX: use Colormap object so set_bad works) —
                    cs = self._fov_preview._flim_color_scale
                    scaled = flim_display.apply_color_scale(
                        lifetime,
                        vmin=cs.get('vmin'),
                        vmax=cs.get('vmax'),
                        gamma=cs.get('gamma', 1.0),
                    )
                    cmap_obj = flim_display.get_colormap(cs.get('cmap', 'viridis'))
                    cmap_obj.set_bad(color='black')                          # ← KEY FIX
                    ax_flim.clear()
                    ax_cbar.clear()
                    im = ax_flim.imshow(scaled, cmap=cmap_obj, origin="upper", vmin=0, vmax=1)
                    ax_flim.set_title("FLIM Lifetime (τ-weighted)", fontsize=10, fontweight="bold")
                    ax_flim.set_xlabel("X (pixels)")
                    ax_flim.set_ylabel("Y (pixels)")

                    # Colorbar
                    valid = lifetime[~np.isnan(lifetime)]
                    if valid.size > 0:
                        vmin_cb = cs.get('vmin') or float(np.nanmin(valid))
                        vmax_cb = cs.get('vmax') or float(np.nanmax(valid))
                        cbar = fig.colorbar(im, cax=ax_cbar)
                        cbar.set_label("τ (ns)", fontsize=8)
                        ticks = np.linspace(0, 1, 5)
                        cbar.set_ticks(ticks)
                        cbar.set_ticklabels([f"{vmin_cb + t*(vmax_cb - vmin_cb):.2f}" for t in ticks], fontsize=7)

                    # — Decay + model + IRF (FIX: draw into FOV preview, not non-existent _res._ax_decay) —
                    ax_decay = self._fov_preview._ax_decay
                    ax_decay.clear()
                    decay    = fit_result.get("decay")
                    time_ns  = fit_result.get("time_ns")
                    if decay is not None and time_ns is not None:
                        ax_decay.semilogy(time_ns, decay, 'o-', color="steelblue",
                                        linewidth=1.5, markersize=3, label="Measured", alpha=0.7)
                        irf = fit_result.get("irf_prompt")
                        if irf is not None and len(irf) > 0 and irf.max() > 0:
                            irf_scaled = (irf / irf.max()) * decay.max() * 0.2
                            ax_decay.semilogy(time_ns[:len(irf)], np.maximum(irf_scaled, 1e-2),
                                            color="orange", linewidth=2.0, label="IRF", alpha=0.8)
                        # Reconstruct global_summary with hoisted arrays
                        gs = _reconstruct_dict_from_session(fit_result, "global_summary")
                        model = gs.get('model')
                        # Safely convert string model back to array if needed
                        if model is not None and isinstance(model, str):
                            model = _safe_array_from_json(model)
                        if model is not None and len(model) > 0:
                            ax_decay.semilogy(time_ns, model, color="red", linewidth=2.0,
                                            label="Fitted", alpha=0.8)
                        ax_decay.legend(fontsize=8, loc="upper right")
                    ax_decay.set_title("Summed Decay", fontsize=10, fontweight="bold")
                    ax_decay.set_xlabel("Time (ns)")
                    ax_decay.set_ylabel("Photon Count")
                    ax_decay.grid(True, alpha=0.3)
                    ax_decay.tick_params(labelsize=8)

                    # Redraw region overlays
                    self._fov_preview._redraw_region_overlays()
                    
                    self._fov_preview._ctrl_frame.grid()
                    self._fov_preview._canvas_mpl.draw_idle()
                    print(f"[Load] Restored FOV preview from cached data")
            except Exception as e:
                import traceback
                print(f"[Load] Could not redraw FOV preview: {e}")
                traceback.print_exc()

            # — FIX: restore form state —
            if "form_state_json" in fit_result:
                try:
                    fs = fit_result["form_state_json"]
                    if isinstance(fs, np.ndarray): fs = fs.item()
                    if isinstance(fs, bytes): fs = fs.decode()
                    form_state = json.loads(fs)
                    # Pre-arm the guard so sv_ptu.set() inside _restore_form_state
                    # doesn't re-trigger load_fov and wipe the FLIM we just drew.
                    # FIX 2: always arm the guard so sv_ptu.set() inside
                    # _restore_form_state does not re-trigger load_fov.
                    ptu_in_session = form_state.get("ptu_file", "").strip()
                    # If session has no ptu path, keep the current sv_ptu value
                    # so the trace guard still returns early.
                    self._last_loaded_ptu = (
                        ptu_in_session
                        if ptu_in_session
                        else (self.sv_ptu.get().strip() if hasattr(self, "sv_ptu") else "")
                    )
                    self._restore_form_state(form_state)
                except Exception as e:
                    print(f"[Load] Could not restore form state: {e}")

            
            # Store fit result for export (use NPZ directory as output dir)
            output_dir = str(Path(npz_path).parent)
            self._res.set_fit_result(fit_result, output_dir, npz_path=npz_path)
            
            # Switch to Results panel
            self._switch_form("results")
            
            messagebox.showinfo("Success", f"Loaded fitted data from:\n{Path(npz_path).name}")
            
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load fitted data:\n{str(e)}")

    def _show_export_dialog(self, image_dict: dict, output_dir: str):
        """Show export options dialog for fit images and NPZ data."""
        try:
            print(f"[Export Dialog] Starting with {len(image_dict)} items")
            
            if not image_dict or not any(isinstance(v, np.ndarray) for v in image_dict.values()):
                messagebox.showinfo("No images", "No fit images available to export.")
                return
            
            # Find all numpy arrays that look like images (2D or 3D with reasonable size)
            available_images = {}
            for k, v in image_dict.items():
                if isinstance(v, np.ndarray):
                    # Must be 2D (grayscale) or 3D with 3 channels (RGB)
                    if (v.ndim == 2) or (v.ndim == 3 and v.shape[2] == 3):
                        available_images[k] = v
                        print(f"[Export] Found image: {k} shape={v.shape}")
            
            if not available_images:
                messagebox.showinfo("No images", "No valid FLIM result images found to export.")
                return
            
            print(f"[Export] {len(available_images)} images available: {list(available_images.keys())}")
            print(f"[Export Dialog] Creating Toplevel on root={self.root}")
            
            # Create dialog
            dlg = tk.Toplevel(self.root)
            dlg.title("Export Results")
            dlg.geometry("550x480")
            dlg.transient(self.root)
            dlg.grab_set()
            print(f"[Export Dialog] Dialog created successfully")
            
            # Options
            bv_scalebar = tk.BooleanVar(value=True)
            bv_annotations = tk.BooleanVar(value=True)
            image_vars = {}  # Initialize dictionary

            for key in available_images.keys():
                image_vars[key] = tk.BooleanVar(value=True)
            
            # Title
            ttk.Label(dlg, text="Export Results", font=("TkDefaultFont", 11, "bold")).pack(pady=10)
            
            # Image selection section (multi-column layout)
            img_frame = ttk.LabelFrame(dlg, text="Images to Export", padding=10)
            img_frame.pack(fill="both", expand=True, padx=20, pady=5)
            
            # Organize images into 3 columns
            sorted_images = sorted(available_images.keys())
            n_cols = 3
            n_rows = (len(sorted_images) + n_cols - 1) // n_cols
            
            # Configure columns to have equal weight
            for col in range(n_cols):
                img_frame.columnconfigure(col, weight=1)
            
            # Add checkboxes in grid layout
            for idx, key in enumerate(sorted_images):
                row = idx % n_rows
                col = idx // n_rows
                ttk.Checkbutton(img_frame, text=key.replace('_', ' ').title(), 
                               variable=image_vars[key]).grid(row=row, column=col, sticky="w", padx=5, pady=2)
            
            # Select all / None buttons (below the grid)
            sel_btn_frame = ttk.Frame(img_frame)
            sel_btn_frame.grid(row=n_rows, column=0, columnspan=n_cols, sticky="w", pady=(10, 0))
            def select_all():
                for v in image_vars.values():
                    v.set(True)
            def select_none():
                for v in image_vars.values():
                    v.set(False)
            ttk.Button(sel_btn_frame, text="All", command=select_all, width=8).pack(side="left", padx=2)
            ttk.Button(sel_btn_frame, text="None", command=select_none, width=8).pack(side="left", padx=2)
            
            # Image rendering options
            opt_frame = ttk.LabelFrame(dlg, text="Rendering Options", padding=10)
            opt_frame.pack(fill="x", padx=20, pady=5)
            ttk.Checkbutton(opt_frame, text="Include scale bar", variable=bv_scalebar).pack(anchor="w", pady=3)
            ttk.Checkbutton(opt_frame, text="Include ROI annotations", variable=bv_annotations).pack(anchor="w", pady=3)
            
            # Format selection
            fmt_frame = ttk.LabelFrame(dlg, text="Image Format", padding=10)
            fmt_frame.pack(fill="x", padx=20, pady=5)
            bv_format = tk.StringVar(value="png")
            ttk.Radiobutton(fmt_frame, text="PNG (smaller file size, web-friendly)", 
                           variable=bv_format, value="png").pack(anchor="w", pady=3)
            ttk.Radiobutton(fmt_frame, text="OME-TIFF (lossless, metadata-rich)", 
                           variable=bv_format, value="ometiff").pack(anchor="w", pady=3)
            
            # Export location
            loc_frame = ttk.LabelFrame(dlg, text="Save Location", padding=10)
            loc_frame.pack(fill="x", padx=20, pady=5)
            export_path = tk.StringVar(value=output_dir)
            ttk.Label(loc_frame, text="Path:").pack(side="left")
            ttk.Entry(loc_frame, textvariable=export_path, width=40).pack(side="left", padx=5, fill="x", expand=True)
            
            def browse_folder():
                from tkinter import filedialog
                folder = filedialog.askdirectory(initialdir=output_dir, title="Select export folder")
                if folder:
                    export_path.set(folder)
                    print(f"[Export] Save location changed to: {folder}")
            
            ttk.Button(loc_frame, text="Browse", command=browse_folder, width=8).pack(side="left", padx=2)
            
            def do_export():
                try:
                    # Filter images to only those selected
                    selected_images = {k: v for k, v in available_images.items() 
                                     if image_vars[k].get()}
                    if not selected_images:
                        messagebox.showwarning("No selection", "Please select at least one image to export.")
                        return
                    
                    export_dir = export_path.get()
                    if not export_dir.strip():
                        messagebox.showwarning("No path", "Please select an export directory.")
                        return
                    
                    fmt = bv_format.get()
                    print(f"[Export] Exporting {len(selected_images)} images in {fmt.upper()} format to {export_dir}")
                    self._export_images(selected_images, export_dir, 
                                       with_scalebar=bv_scalebar.get(),
                                       with_annotations=bv_annotations.get(),
                                       format=fmt)
                    dlg.destroy()
                    messagebox.showinfo("Success", f"Results exported to\n{export_dir}")
                except Exception as e:
                    print(f"[Export Error] {e}")
                    import traceback
                    traceback.print_exc()
                    messagebox.showerror("Export Error", f"Failed to export:\n{str(e)}")
            
            btn_frame = ttk.Frame(dlg)
            btn_frame.pack(pady=15, fill="x", padx=20)
            
            # Export button (main action)
            export_btn = ttk.Button(btn_frame, text="💾 Export", command=do_export)
            export_btn.pack(side="left", padx=5, fill="x", expand=True)
            
            # Cancel button
            cancel_btn = ttk.Button(btn_frame, text="Cancel", command=dlg.destroy)
            cancel_btn.pack(side="left", padx=5)
            
            print(f"[Export Dialog] Dialog fully created, awaiting user input")
            
        except Exception as e:
            print(f"[Export Dialog Error] {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Dialog Error", f"Failed to create export dialog:\n{str(e)}")

    def _export_images(self, image_dict: dict, output_dir: str, 
                      with_scalebar: bool = True, with_annotations: bool = True,
                      format: str = "png"):
        """Export intensity and lifetime images in PNG or OME-TIFF format."""
        try:
            from pathlib import Path
            import numpy as np
            import matplotlib.pyplot as plt
            
            fmt = format.lower()
            print(f"[Export Images] Exporting {len(image_dict)} images in {fmt.upper()} format to {output_dir}")
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            exported_count = 0
            
            if fmt == "ometiff":
                # Try to use tifffile for OME-TIFF export (lossless, preserves metadata)
                try:
                    import tifffile
                    
                    # Export intensity as OME-TIFF
                    if 'intensity' in image_dict and isinstance(image_dict['intensity'], np.ndarray):
                        try:
                            intensity = image_dict['intensity']
                            intensity_16bit = (intensity / intensity.max() * 65535).astype(np.uint16) if intensity.max() > 0 else intensity.astype(np.uint16)
                            
                            output_file = output_path / "intensity.ome.tiff"
                            tifffile.imwrite(output_file, intensity_16bit, photometric='minisblack',
                                           metadata={'description': 'FLIM Intensity Image'})
                            print(f"✓ Exported OME-TIFF intensity: {output_file.name} ({intensity.shape})")
                            exported_count += 1
                        except Exception as e:
                            print(f"[Export] Error exporting intensity TIFF: {e}")
                    
                    # Export lifetime as OME-TIFF
                    if 'lifetime' in image_dict and isinstance(image_dict['lifetime'], np.ndarray):
                        try:
                            lifetime = image_dict['lifetime']
                            # Replace NaN with 0
                            lifetime = np.nan_to_num(lifetime, nan=0.0)
                            lifetime_32bit = lifetime.astype(np.float32)
                            
                            output_file = output_path / "lifetime.ome.tiff"
                            tifffile.imwrite(output_file, lifetime_32bit, photometric='minisblack',
                                           metadata={'description': 'FLIM Lifetime Map (ns)'})
                            print(f"✓ Exported OME-TIFF lifetime: {output_file.name} ({lifetime.shape})")
                            exported_count += 1
                        except Exception as e:
                            print(f"[Export] Error exporting lifetime TIFF: {e}")
                            
                except ImportError:
                    print(f"[Export] tifffile not installed, falling back to PNG")
                    fmt = "png"
            
            if fmt == "png":
                # Export as PNG (high quality)
                # Export intensity image at maximum resolution
                if 'intensity' in image_dict and isinstance(image_dict['intensity'], np.ndarray):
                    try:
                        intensity = image_dict['intensity']
                        print(f"[Export] Intensity shape: {intensity.shape}")
                        
                        # Create figure with no margins for maximum resolution
                        h, w = intensity.shape
                        fig = plt.figure(figsize=(w/100, h/100), dpi=100, facecolor='black', edgecolor='black')  # 1:1 pixel ratio
                        ax = fig.add_axes([0, 0, 1, 1])  # Full figure area
                        ax.set_facecolor('black')
                        
                        intensity_clipped = np.clip(intensity, 0, np.percentile(intensity, 99))
                        im = ax.imshow(intensity_clipped, cmap='inferno', origin='upper', aspect='auto')
                        ax.axis('off')
                        
                        output_file = output_path / "intensity.png"
                        fig.savefig(output_file, dpi=100, bbox_inches='tight', pad_inches=0, facecolor='black')
                        plt.close(fig)
                        print(f"✓ Exported PNG intensity: {output_file.name} ({intensity.shape})")
                        exported_count += 1
                    except Exception as e:
                        print(f"[Export] Error exporting intensity PNG: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Export lifetime/FLIM map at maximum resolution
                if 'lifetime' in image_dict and isinstance(image_dict['lifetime'], np.ndarray):
                    try:
                        lifetime = image_dict['lifetime']
                        print(f"[Export] Lifetime shape: {lifetime.shape}")
                        
                        # Create figure with no margins for maximum resolution
                        h, w = lifetime.shape[:2]
                        fig = plt.figure(figsize=(w/100, h/100), dpi=100, facecolor='black', edgecolor='black')  # 1:1 pixel ratio
                        ax = fig.add_axes([0, 0, 1, 1])  # Full figure area
                        ax.set_facecolor('black')
                        
                        im = ax.imshow(lifetime, cmap='viridis', origin='upper', aspect='auto')
                        ax.axis('off')
                        
                        # Add ROI annotations if requested
                        if with_annotations and self._fov_preview._roi_manager.get_all_regions():
                            from flimkit.UI.roi_tools import get_rectangle_patch, get_ellipse_patch, get_polygon_patch
                            for region in self._fov_preview._roi_manager.get_all_regions():
                                region_id = region['id']
                                tool_type = region['tool']
                                coords = region['coords']
                                color = self._fov_preview._roi_manager.get_color(region_id)
                                try:
                                    if tool_type == 'rect':
                                        patch = get_rectangle_patch(coords, edgecolor=color, linewidth=2)
                                    elif tool_type == 'ellipse':
                                        patch = get_ellipse_patch(coords, edgecolor=color, linewidth=2)
                                    elif tool_type in ('polygon', 'freehand'):
                                        patch = get_polygon_patch(coords, edgecolor=color, linewidth=2)
                                    else:
                                        continue
                                    ax.add_patch(patch)
                                except Exception as e:
                                    print(f"[Export] Could not add ROI {region_id}: {e}")
                        
                        output_file = output_path / "lifetime.png"
                        fig.savefig(output_file, dpi=100, bbox_inches='tight', pad_inches=0, facecolor='black')
                        plt.close(fig)
                        print(f"✓ Exported PNG lifetime: {output_file.name} ({lifetime.shape})")
                        exported_count += 1
                    except Exception as e:
                        print(f"[Export] Error exporting lifetime PNG: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Export summed decay plot for reference (always PNG)
            try:
                if self._fov_preview is not None and hasattr(self._fov_preview, '_ax_decay'):
                    ax_decay = self._fov_preview._ax_decay
                    
                    # Create a high-quality decay plot
                    fig, ax = plt.subplots(figsize=(14, 8), dpi=150, facecolor='white', edgecolor='white')
                    ax.set_facecolor('white')
                    
                    # Clone all lines from the preview
                    for line in ax_decay.get_lines():
                        ax.plot(line.get_xdata(), line.get_ydata(), 
                               color=line.get_color(), linewidth=2.5,
                               marker=line.get_marker(), markersize=line.get_markersize(),
                               label=line.get_label(), alpha=line.get_alpha())
                    
                    ax.set_yscale('log')
                    ax.set_title('Summed Decay - Measured, IRF, and Fitted', fontsize=16, fontweight='bold', color='black')
                    ax.set_xlabel('Time (ns)', fontsize=13, color='black')
                    ax.set_ylabel('Photon Count', fontsize=13, color='black')
                    ax.tick_params(colors='black')
                    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
                    ax.grid(True, alpha=0.3, color='gray')
                    
                    output_file = output_path / "summed_decay.png"
                    fig.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
                    plt.close(fig)
                    print(f"✓ Exported decay plot: {output_file.name}")
                    exported_count += 1
            except Exception as e:
                print(f"[Export] Error exporting decay: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"✓ Export complete: {exported_count} high-resolution images to {output_path}")
            
            # Open the folder in finder
            try:
                import subprocess
                subprocess.Popen(['open', str(output_path)])
                print(f"✓ Opened export folder in Finder")
            except Exception as e:
                print(f"[Export] Could not open folder: {e}")
                
        except Exception as e:
            print(f"✗ Export images error: {e}")
            import traceback
            traceback.print_exc()

    def _export_npz_fit(self, fit_result: dict, output_dir: str, ptu_path: str = None):
        """Copy the existing NPZ fit file to export directory instead of recreating it."""
        try:
            from pathlib import Path
            import shutil
            
            # Find the existing NPZ file that was saved after fitting
            npz_source = None
            
            # If we have ptu_path, look for the NPZ next to it
            if ptu_path:
                base_path = Path(ptu_path)
                if base_path.is_file():
                    npz_source = base_path.parent / f"{base_path.stem}.roi_fit.npz"
            
            # If not found, try to infer from fit_result
            if not npz_source or not npz_source.exists():
                source = fit_result.get('source')
                if isinstance(source, bytes):
                    source = source.decode('utf-8')
                if source:
                    base_path = Path(source)
                    if base_path.is_file():
                        npz_source = base_path.parent / f"{base_path.stem}.roi_fit.npz"
                    else:
                        npz_source = base_path / "roi_fit.npz"
            
            # Copy the NPZ file to export directory
            if npz_source and npz_source.exists():
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                npz_dest = output_path / "fit_result.npz"
                shutil.copy2(npz_source, npz_dest)
                print(f"✓ NPZ fit data copied to export: {npz_dest.name}")
            else:
                print(f"[Export] Could not find existing NPZ file to copy")
        except Exception as e:
            print(f"✗ NPZ export error: {e}")
            import traceback
            traceback.print_exc()
            import traceback
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # TAB 1 – Single-FOV FLIM fit
    # -------------------------------------------------------------------------
    def _build_fov_tab(self):
        outer, tab = self._form_inner_frames["fov"]
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
        outer, tab = self._form_inner_frames["stitch"]
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
        outer, tab = self._form_inner_frames["batch"]
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
        outer, tab = self._form_inner_frames["irf"]
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
        # Get outer and inner frames from tuple
        outer, inner = self._form_inner_frames["phasor"]
        inner.columnconfigure(0, weight=1)
        #  Controls strip (fixed height, top) ─
        ctrl = ttk.Frame(inner, padding=(6, 4))
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
        
        # Skip if PTU hasn't actually changed (prevents loop from trace callback)
        if ptu_path == self._last_loaded_ptu:
            return
        
        self._last_loaded_ptu = ptu_path
        
        if ptu_path and hasattr(self, '_fov_preview'):
            # Defer loading to give the UI a chance to update
            self.root.after(100, lambda: self._fov_preview.load_fov(ptu_path))
            # Auto-load session if it exists (restore form + results)
            self.root.after(150, lambda: self._auto_load_session_for_ptu(ptu_path))

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
                    # Refresh ROI Analysis panel
                    if hasattr(self, '_roi_analysis_panel'):
                        self._roi_analysis_panel._refresh_region_list()
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
                # Refresh ROI Analysis panel 
                if hasattr(self, '_roi_analysis_panel'):
                    self._roi_analysis_panel._refresh_region_list()
            except Exception as e:
                import traceback
                traceback.print_exc()
        
        # NPZ is saved automatically after fit completes (silently)
        # but can also be explicitly saved to a different location via "Save NPZ" button
        npz_file_path = None
        if ptu_path or output_dir:
            try:
                self._save_roi_progress(ptu_path or output_dir, fit_result, rows or [])
                # Figure out where the session file was saved
                from pathlib import Path
                base_path = Path(ptu_path) if ptu_path else Path(output_dir)
                if base_path.is_file():
                    npz_file_path = str(base_path.parent / f"{base_path.stem}.roi_session.npz")
                else:
                    npz_file_path = str(base_path / "roi_session.npz")
            except Exception as e:
                print(f"[Info] Could not save ROI progress: {e}")
        
        # Store fit result with NPZ path so quick save knows where it is
        self._res.set_fit_result(fit_result or {}, output_dir, npz_path=npz_file_path)

    def _extract_summary_rows(self, global_summary: dict, global_popt=None) -> list:
        """Extract fit summary rows from global_summary dict.

        Handles two schemas:
          • fit_summed schema  — keys: taus_ns, amps, fractions, tau_mean_amp_ns …
          • derive_global_tau schema — keys: tau_mean_amp_global_ns, tau1_mean_ns …
        """
        if not global_summary:
            return []

        rows = []

        #  Schema A: fit_summed / fit_per_pixel (single-FOV fit) ─
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

        #  Schema B: derive_global_tau (stitch / tile-fit pipeline) 
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

        #  Shared fields (present in both schemas) 
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