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
import os
import sys
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("TkAgg")
# Dark-theme text defaults for all figures
matplotlib.rcParams.update({
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.titlecolor': 'white',
})
import matplotlib.image as mpimg
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from flimkit.UI.progress_window import ProgressWindow
from flimkit.UI.phasor_panel import PhasorViewPanel
from flimkit.UI import flim_display
from flimkit.UI.roi_tools import RoiManager, RoiAnalysisPanel
from flimkit.UI.project_panel import ProjectBrowserPanel

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

    def __init__(self, widget: scrolledtext.ScrolledText, buf: list, root=None, is_stderr=False):
        self.widget = widget
        self.buf    = buf
        self.root   = root  # For thread-safe GUI updates
        self._is_stderr = is_stderr
        self._batch = []  # Accumulate text before writing
        self._batch_size = 5000  # characters, or time-based flush
        self._last_flush = time.time()
        self._flush_interval = 0.5  # seconds

    def write(self, text: str):
        if not text:
            return
        self.buf.append(text)
        self._batch.append(text)

        # Forward stderr content to crash handler log
        if self._is_stderr:
            try:
                from flimkit.utils.crash_handler import log_event
                log_event(f"STDERR: {text.rstrip()}", level="warning")
            except Exception:
                pass
        
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
        ("Leica analytical model (XLSX)",                "irf_xlsx"),
        ("Machine IRF (.npy pre-built)",                 "machine_irf"),
        ("Machine IRF + full σ broadening",               "machine_irf_sigma_full"),
        ("Machine IRF + half σ broadening (σ≤0.5)",       "machine_irf_sigma_half"),
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
        if self.sv_method.get().startswith("machine_irf"):
            _browse_file(self.sv_path, "Select machine IRF",
                         [("NumPy array", "*.npy"), ("All", "*.*")])
        else:
            _browse_file(self.sv_path, "Select IRF file",
                         [("PTU / XLSX", "*.ptu *.xlsx"), ("All", "*.*")])

    def _show_browse(self):
        method = self.sv_method.get()
        self._path_lbl.config(
            text="Machine IRF (.npy) path" if method.startswith("machine_irf") else "IRF / XLSX path")
        if method.startswith("machine_irf") and not self.sv_path.get().endswith(".npy"):
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
        elif method in ("file",) or method.startswith("machine_irf"):
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
        elif method.startswith("machine_irf"):
            return dict(irf=None, irf_xlsx=None, estimate_irf=method, no_xlsx_irf=True, machine_irf=path)
        elif method == "file":
            return dict(irf=path, irf_xlsx=None, estimate_irf="none", no_xlsx_irf=True, machine_irf=None)
        elif method in ("raw", "parametric"):
            return dict(irf=None, irf_xlsx=None, estimate_irf=method, no_xlsx_irf=True, machine_irf=None)
        else:
            return dict(irf=None, irf_xlsx=None, estimate_irf="none", no_xlsx_irf=True, machine_irf=None)


# ── Expert Settings Dialog ──────────────────────────────────────────

# Default expert settings (mirrors configs.py)
_EXPERT_DEFAULTS = {
    "binning_factor": 1,
    "optimizer": "de",
    "lm_restarts": 8,
    "de_population": 30,
    "de_maxiter": 5000,
    "n_workers": -1,
    "cost_function": "poisson",
    "channels": "",
    "min_photons": 10,
}


class ExpertSettingsDialog(tk.Toplevel):
    """Modal dialog for advanced fitting parameters."""

    def __init__(self, parent, current: dict):
        super().__init__(parent)
        self.title("Expert Fit Settings")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result: Optional[dict] = None
        cfg = _C()

        # Merge defaults ← config ← current overrides
        vals = dict(_EXPERT_DEFAULTS)
        vals.update({
            "binning_factor": cfg["binning_factor"],
            "optimizer": cfg["Optimizer"],
            "lm_restarts": cfg["lm_restarts"],
            "de_population": cfg["de_population"],
            "de_maxiter": cfg["de_maxiter"],
            "n_workers": cfg["n_workers"],
            "min_photons": cfg["MIN_PHOTONS_PERPIX"],
        })
        vals.update(current)

        PAD = {"padx": 4, "pady": 3}
        row = 0
        f = ttk.Frame(self, padding=12)
        f.pack(fill="both", expand=True)

        # ── Optimizer ────────────────────────────────
        ttk.Label(f, text="Optimizer:").grid(row=row, column=0, sticky="w", **PAD)
        self._sv_optimizer = tk.StringVar(value=vals["optimizer"])
        opt_frame = ttk.Frame(f)
        opt_frame.grid(row=row, column=1, columnspan=3, sticky="w", **PAD)
        ttk.Radiobutton(opt_frame, text="Differential Evolution (DE)",
                        variable=self._sv_optimizer, value="de").pack(side="left", padx=(0, 8))
        ttk.Radiobutton(opt_frame, text="Levenberg-Marquardt (LM)",
                        variable=self._sv_optimizer, value="lm_multistart").pack(side="left")

        # ── DE settings ─────────────────────────────
        row += 1
        ttk.Label(f, text="DE population:").grid(row=row, column=0, sticky="w", **PAD)
        self._sv_de_pop = tk.StringVar(value=str(vals["de_population"]))
        ttk.Entry(f, textvariable=self._sv_de_pop, width=8).grid(row=row, column=1, sticky="w", **PAD)
        ttk.Label(f, text="DE max iterations:").grid(row=row, column=2, sticky="w", **PAD)
        self._sv_de_maxiter = tk.StringVar(value=str(vals["de_maxiter"]))
        ttk.Entry(f, textvariable=self._sv_de_maxiter, width=8).grid(row=row, column=3, sticky="w", **PAD)

        # ── LM settings ─────────────────────────────
        row += 1
        ttk.Label(f, text="LM random restarts:").grid(row=row, column=0, sticky="w", **PAD)
        self._sv_lm_restarts = tk.StringVar(value=str(vals["lm_restarts"]))
        ttk.Entry(f, textvariable=self._sv_lm_restarts, width=8).grid(row=row, column=1, sticky="w", **PAD)

        # ── Binning ──────────────────────────────────
        row += 1
        ttk.Label(f, text="Spatial binning (NxN):").grid(row=row, column=0, sticky="w", **PAD)
        self._sv_binning = tk.StringVar(value=str(vals["binning_factor"]))
        ttk.Entry(f, textvariable=self._sv_binning, width=8).grid(row=row, column=1, sticky="w", **PAD)
        ttk.Label(f, text="(1 = no binning)", foreground="grey").grid(row=row, column=2, columnspan=2, sticky="w", **PAD)

        # ── Workers ──────────────────────────────────
        row += 1
        ttk.Label(f, text="CPU workers:").grid(row=row, column=0, sticky="w", **PAD)
        self._sv_workers = tk.StringVar(value=str(vals["n_workers"]))
        ttk.Entry(f, textvariable=self._sv_workers, width=8).grid(row=row, column=1, sticky="w", **PAD)
        ttk.Label(f, text="(-1 = all cores)", foreground="grey").grid(row=row, column=2, columnspan=2, sticky="w", **PAD)

        # ── Min photons ──────────────────────────────
        row += 1
        ttk.Label(f, text="Min photons/pixel:").grid(row=row, column=0, sticky="w", **PAD)
        self._sv_min_ph = tk.StringVar(value=str(vals["min_photons"]))
        ttk.Entry(f, textvariable=self._sv_min_ph, width=8).grid(row=row, column=1, sticky="w", **PAD)

        # ── Cost function ────────────────────────────
        row += 1
        ttk.Label(f, text="Cost function:").grid(row=row, column=0, sticky="w", **PAD)
        self._sv_cost = tk.StringVar(value=vals["cost_function"])
        cf_frame = ttk.Frame(f)
        cf_frame.grid(row=row, column=1, columnspan=3, sticky="w", **PAD)
        ttk.Radiobutton(cf_frame, text="Poisson deviance",
                        variable=self._sv_cost, value="poisson").pack(side="left", padx=(0, 8))
        ttk.Radiobutton(cf_frame, text="Chi² (legacy)",
                        variable=self._sv_cost, value="chi2").pack(side="left")

        # ── Channels ─────────────────────────────────
        row += 1
        ttk.Label(f, text="Channel filter:").grid(row=row, column=0, sticky="w", **PAD)
        self._sv_channels = tk.StringVar(value=str(vals.get("channels", "") or ""))
        ttk.Entry(f, textvariable=self._sv_channels, width=12).grid(row=row, column=1, sticky="w", **PAD)
        ttk.Label(f, text="(blank = all channels)", foreground="grey").grid(row=row, column=2, columnspan=2, sticky="w", **PAD)

        # ── Buttons ──────────────────────────────────
        row += 1
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=row, column=0, columnspan=4, pady=(12, 0))
        ttk.Button(btn_frame, text="Confirm", command=self._confirm).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Reset Defaults", command=self._reset).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side="left", padx=4)

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        # Centre on parent
        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

    def _collect(self) -> dict:
        ch = self._sv_channels.get().strip()
        return {
            "optimizer": self._sv_optimizer.get(),
            "de_population": int(self._sv_de_pop.get() or 30),
            "de_maxiter": int(self._sv_de_maxiter.get() or 5000),
            "lm_restarts": int(self._sv_lm_restarts.get() or 8),
            "binning_factor": int(self._sv_binning.get() or 1),
            "n_workers": int(self._sv_workers.get() or -1),
            "min_photons": int(self._sv_min_ph.get() or 10),
            "cost_function": self._sv_cost.get(),
            "channels": int(ch) if ch.isdigit() else (None if ch == "" else ch),
        }

    def _confirm(self):
        try:
            self.result = self._collect()
        except ValueError as e:
            messagebox.showerror("Invalid value", str(e), parent=self)
            return
        self.destroy()

    def _reset(self):
        d = _EXPERT_DEFAULTS
        self._sv_optimizer.set(d["optimizer"])
        self._sv_de_pop.set(str(d["de_population"]))
        self._sv_de_maxiter.set(str(d["de_maxiter"]))
        self._sv_lm_restarts.set(str(d["lm_restarts"]))
        self._sv_binning.set(str(d["binning_factor"]))
        self._sv_workers.set(str(d["n_workers"]))
        self._sv_min_ph.set(str(d["min_photons"]))
        self._sv_cost.set(d["cost_function"])
        self._sv_channels.set("")


class FOVPreviewPanel:
    """Real-time preview of FOV intensity image and decay curve."""

    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        # Create figure with GridSpec layout
        from matplotlib.gridspec import GridSpec
        self._fig = Figure(figsize=(10, 8), dpi=100, facecolor="black")
        self._decay_visible = True  # Track decay panel visibility
        self._display_mode = "flim"  # "flim" or "intensity" — which image gets the main slot
        # Initial layout: 2 rows, 3 cols
        gs = GridSpec(2, 3, figure=self._fig, height_ratios=[1, 0.6], width_ratios=[1, 1, 0.05], hspace=0.3, wspace=0.15)
        
        self._ax_img = self._fig.add_subplot(gs[0, 0])    # Intensity (top-left)
        self._ax_flim = self._fig.add_subplot(gs[0, 1])   # FLIM (top-right)
        self._ax_cbar = self._fig.add_subplot(gs[0, 2])   # Colorbar (top-right, narrow)
        self._ax_decay = self._fig.add_subplot(gs[1, :])  # Decay (bottom, full width)
        for _ax in (self._ax_img, self._ax_flim):
            _ax.set_facecolor('black')
        self._ax_decay.set_facecolor('white')
        self._ax_decay.tick_params(colors='white')
        self._ax_decay.xaxis.label.set_color('white')
        self._ax_decay.yaxis.label.set_color('white')
        self._ax_decay.title.set_color('white')
        self._strip_image_axes(self._ax_img)
        self._strip_image_axes(self._ax_flim)
        
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

        # Row 2: Show/Hide decay toggle + display mode
        self._bv_show_decay = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_frame, text="Show Decay Plot",
                        variable=self._bv_show_decay,
                        command=self._toggle_decay).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # Display mode selector (FLIM vs Intensity) — relevant when decay hidden
        self._sv_display_mode = tk.StringVar(value="flim")
        dm_frame = ttk.Frame(ctrl_frame)
        dm_frame.grid(row=2, column=3, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(dm_frame, text="View:").pack(side="left", padx=(0, 4))
        ttk.Radiobutton(dm_frame, text="FLIM", variable=self._sv_display_mode,
                        value="flim", command=self._on_display_mode_changed).pack(side="left")
        ttk.Radiobutton(dm_frame, text="Intensity", variable=self._sv_display_mode,
                        value="intensity", command=self._on_display_mode_changed).pack(side="left")

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
        self._roi_drag = None  # {id, ox, oy} when dragging a ROI
        
        # Connect matplotlib event handlers to FLIM axes
        self._setup_drawing_events()
        self._cached_decay_lines = []  # Persist decay data across layout rebuilds
        self._cached_decay_title = ""
        self._cached_decay_yscale = "log"

        # Connect scroll-wheel zoom on image axes
        self._setup_zoom()

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
            self._ax_img.set_title("Intensity", fontsize=9, fontweight="bold")
            self._strip_image_axes(self._ax_img)

            # Clear FLIM axes (no fit data yet)
            self._ax_flim.clear()
            self._ax_flim.text(0.5, 0.5, "Waiting for fit...", ha='center', va='center',
                              transform=self._ax_flim.transAxes, fontsize=9, color='#888')
            self._ax_flim.set_title("FLIM Lifetime", fontsize=10, fontweight="bold")

            # Plot decay curve
            self._ax_decay.clear()
            self._ax_decay.set_facecolor('white')
            self._ax_decay.semilogy(time_ns, decay, color="steelblue", linewidth=1.5)
            self._ax_decay.set_title("Summed Decay", fontsize=10, fontweight="bold", color='white')
            self._ax_decay.set_xlabel("Time (ns)", color='white')
            self._ax_decay.set_ylabel("Photon Count", color='white')
            self._ax_decay.grid(True, alpha=0.3)
            self._ax_decay.tick_params(labelsize=8, colors='white')

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
            
            # Get intensity image: prefer canvas (from tile fitting), then fit_result, then PTU fallback
            intensity = None
            if canvas is not None and 'intensity' in canvas:
                intensity = canvas['intensity']
                print(f"  - Using intensity from canvas (tile fitting)")
            elif 'intensity' in fit_result:
                intensity = fit_result['intensity']
                if isinstance(intensity, np.ndarray):
                    print(f"  - Using intensity from fit_result")
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
            if nexp == 0:
                # derive_global_tau schema (tile_fit) stores tau1_mean_ns / tau2_mean_ns …
                # rather than a taus_ns list.  Count how many tau{k}_mean_ns keys exist.
                nexp = sum(1 for k in range(1, 4) if f'tau{k}_mean_ns' in global_summary)
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
            
            # Upsample lifetime map to full-res intensity shape if they differ
            # (happens when per-pixel fitting used binning > 1)
            if (lifetime_map is not None and intensity is not None
                    and lifetime_map.shape != intensity.shape[:2]):
                import cv2 as _cv2
                th, tw = intensity.shape[:2]
                lifetime_map = _cv2.resize(
                    lifetime_map.astype(np.float32), (tw, th),
                    interpolation=_cv2.INTER_NEAREST)
                print(f"  - Upsampled lifetime_map to {th}×{tw} to match intensity")

            # Upsample lifetime_map to full-res intensity shape when binning>1
            if (lifetime_map is not None and intensity is not None
                    and lifetime_map.shape != intensity.shape[:2]):
                try:
                    import cv2 as _cv2
                    th, tw = intensity.shape[:2]
                    lifetime_map = _cv2.resize(
                        lifetime_map.astype(np.float32), (tw, th),
                        interpolation=_cv2.INTER_NEAREST)
                    print(f"  - Upsampled lifetime_map to {th}×{tw} px")
                except Exception as _upe:
                    print(f"  - Could not upsample lifetime_map: {_upe}")

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
            
            # Extract fit data (taus_ns present in fit_summed schema only;
            # derive_global_tau / tile_fit uses tau1_mean_ns etc.).
            # nexp resolved above — do NOT overwrite it here.
            taus_fit = global_summary.get('taus_ns', [])
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
            self._ax_img.set_title("Intensity", fontsize=9, fontweight="bold")
            self._strip_image_axes(self._ax_img)

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
                self._ax_flim.set_title("FLIM Lifetime (ns)", fontsize=9, fontweight="bold")
                self._strip_image_axes(self._ax_flim)
                
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
            self._ax_decay.set_facecolor('white')
            
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
                                    fontsize=10, fontweight="bold", color='white')
            self._ax_decay.set_xlabel("Time (ns)", color='white')
            self._ax_decay.set_ylabel("Photon Count", color='white')
            if decay is not None and len(decay) > 0:
                self._ax_decay.legend(fontsize=8, loc="upper right", labelcolor='black')
            self._ax_decay.grid(True, alpha=0.3)
            self._ax_decay.tick_params(labelsize=8, colors='white')

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
        """Load and display stitched ROI intensity image and lifetime map."""
        if not output_dir:
            self._clear()
            self._status.set("No output directory")
            return
        
        try:
            from pathlib import Path
            import numpy as np
            import tifffile
            
            out_path = Path(output_dir)
            
            # Find and load the stitched intensity TIFF file.
            # Stitch-pipeline writes  *_stitched_intensity.tif
            # Tile-fit pipeline writes *_intensity.tif (via save_assembled_maps)
            intensity_files = sorted(out_path.glob("*_stitched_intensity.tif"))
            if not intensity_files:
                intensity_files = sorted(out_path.glob("*_intensity.tif"))
            if not intensity_files:
                self._clear()
                self._status.set("No stitched image found")
                return
            
            intensity = tifffile.imread(str(intensity_files[0]))
            
            # Clear axes and display stitched image
            self._ax_img.clear()
            intensity_clipped = np.clip(intensity, 0, np.percentile(intensity, 99))
            self._ax_img.imshow(intensity_clipped, cmap="inferno", origin="upper")
            self._ax_img.set_title("Stitched ROI", fontsize=9, fontweight="bold")
            self._strip_image_axes(self._ax_img)
            
            # Try to load lifetime map - check multiple sources
            lifetime_data = None
            lifetime_min, lifetime_max = None, None
            
            # Priority 1: Try full-range TIFF (best quality)
            lifetime_full = sorted(out_path.glob("*_tau_intensity_weighted_fullrange.tif"))
            if lifetime_full:
                try:
                    lifetime_data = tifffile.imread(str(lifetime_full[0])).astype(np.float32)
                    valid = np.isfinite(lifetime_data)
                    if valid.any():
                        lifetime_min = float(np.nanmin(lifetime_data[valid]))
                        lifetime_max = float(np.nanpercentile(lifetime_data[valid], 98))
                        print(f"  ✓ Loaded full-range lifetime: {lifetime_min:.2f}–{lifetime_max:.2f} ns")
                except Exception as e:
                    print(f"  - Could not load full-range lifetime: {e}")
            
            # Priority 2: Fall back to display-scaled TIFF
            if lifetime_data is None:
                lifetime_disp = sorted(out_path.glob("*_tau_intensity_weighted.tif"))
                if lifetime_disp:
                    try:
                        lifetime_data = tifffile.imread(str(lifetime_disp[0])).astype(np.float32)
                        # Convert uint16 back to ns (assumes 0-5 ns scale)
                        lifetime_data = lifetime_data / 65535.0 * 5.0
                        lifetime_min, lifetime_max = 0.0, 5.0
                        print(f"  ✓ Loaded display-scaled lifetime: 0–5 ns")
                    except Exception as e:
                        print(f"  - Could not load display-scaled lifetime: {e}")
            
            # Display lifetime map if available
            if lifetime_data is not None:
                self._ax_flim.clear()
                
                # Safe defaults
                if lifetime_min is None or lifetime_max is None or lifetime_max <= lifetime_min:
                    lifetime_min = 0.0
                    lifetime_max = 5.0
                    if lifetime_max <= lifetime_min:
                        lifetime_max = lifetime_min + 0.1
                
                # Normalize to 0-1 for imshow
                lifetime_norm = np.clip((lifetime_data - lifetime_min) / (lifetime_max - lifetime_min), 0, 1)
                
                im = self._ax_flim.imshow(lifetime_norm, cmap="viridis", origin="upper", vmin=0, vmax=1)
                self._ax_flim.set_title(f"FLIM Lifetime ({lifetime_min:.2f}–{lifetime_max:.2f} ns)", 
                                       fontsize=9, fontweight="bold")
                self._strip_image_axes(self._ax_flim)
                
                # Colorbar
                self._ax_cbar.clear()
                cbar = self._fig.colorbar(im, cax=self._ax_cbar, label="τ (ns)")
                # Format ticks
                _min, _max = lifetime_min, lifetime_max
                def _fmt_ns(x, pos):
                    return f"{_min + x * (_max - _min):.1f}"
                from matplotlib.ticker import FuncFormatter
                cbar.ax.yaxis.set_major_formatter(FuncFormatter(_fmt_ns))
                cbar.ax.tick_params(labelsize=7)
            else:
                self._ax_flim.clear()
                self._ax_flim.text(0.5, 0.5, "Lifetime map not available", ha='center', va='center',
                                  transform=self._ax_flim.transAxes, fontsize=9, color='#888')
                self._ax_flim.set_title("FLIM Lifetime", fontsize=10, fontweight="bold")
            
            # Decay plot
            self._ax_decay.clear()
            self._ax_decay.set_facecolor('white')
            self._ax_decay.text(0.5, 0.5, "Per-tile fit complete ✓", 
                               ha="center", va="center", transform=self._ax_decay.transAxes,
                               fontsize=10, color="forestgreen", fontweight="bold")
            
            self._canvas_mpl.draw_idle()
            img_shape = intensity.shape
            self._status.set(f"✓ Tile fit | {img_shape[0]}×{img_shape[1]}px")
            
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
        self._ax_decay.set_facecolor('white')
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
            self._ax_flim.set_title("FLIM Lifetime (ns)", fontsize=9, fontweight="bold")
            self._strip_image_axes(self._ax_flim)
            
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
        """Redraw all region patches on the visible image axes."""
        import matplotlib.patches as mpatches
        from flimkit.UI.roi_tools import get_rectangle_patch, get_ellipse_patch, get_polygon_patch
        
        # Determine which axes to draw on
        target_axes = [ax for ax in (self._ax_flim, self._ax_img) if ax.get_visible()]
        
        # Clear old patches — _roi_patches maps region_id -> list of patches
        for patches in self._roi_patches.values():
            for patch in (patches if isinstance(patches, list) else [patches]):
                try:
                    patch.remove()
                except (ValueError, NotImplementedError):
                    pass
        self._roi_patches = {}
        
        # Draw all regions on visible axes
        for region in self._roi_manager.get_all_regions():
            region_id = region['id']
            tool_type = region['tool']
            coords = region['coords']
            color = self._roi_manager.get_color(region_id)
            linewidth = 2.5 if region_id == self._roi_manager.get_selected_id() else 1.5
            
            patches_for_region = []
            for ax in target_axes:
                try:
                    if tool_type == 'rect':
                        patch = get_rectangle_patch(coords, edgecolor=color, linewidth=linewidth)
                    elif tool_type == 'ellipse':
                        patch = get_ellipse_patch(coords, edgecolor=color, linewidth=linewidth)
                    elif tool_type in ('polygon', 'freehand'):
                        patch = get_polygon_patch(coords, edgecolor=color, linewidth=linewidth)
                    else:
                        continue
                    
                    ax.add_patch(patch)
                    patches_for_region.append(patch)
                except Exception as e:
                    print(f"[ROI] Could not draw region {region_id}: {e}")
            if patches_for_region:
                self._roi_patches[region_id] = patches_for_region
        
        self._canvas_mpl.draw_idle()

    @staticmethod
    def _strip_image_axes(ax):
        """Remove ticks, tick labels, and axis labels from an image axes."""
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    def _setup_zoom(self):
        """Connect scroll-wheel zoom and middle-click pan on image axes."""
        self._zoom_cid = self._canvas_mpl.mpl_connect('scroll_event', self._on_scroll_zoom)
        self._pan_press_cid = self._canvas_mpl.mpl_connect('button_press_event', self._on_pan_press)
        self._pan_release_cid = self._canvas_mpl.mpl_connect('button_release_event', self._on_pan_release)
        self._pan_motion_cid = self._canvas_mpl.mpl_connect('motion_notify_event', self._on_pan_motion)
        self._pan_origin = None

    def _on_scroll_zoom(self, event):
        """Zoom in/out on image axes with scroll wheel."""
        ax = event.inaxes
        if ax is None or ax not in (self._ax_img, self._ax_flim):
            return
        if event.xdata is None or event.ydata is None:
            return

        base_scale = 1.3
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Zoom centred on cursor
        x_range = (xlim[1] - xlim[0]) * scale_factor
        y_range = (ylim[1] - ylim[0]) * scale_factor

        ax.set_xlim([event.xdata - x_range * (event.xdata - xlim[0]) / (xlim[1] - xlim[0]),
                     event.xdata + x_range * (xlim[1] - event.xdata) / (xlim[1] - xlim[0])])
        ax.set_ylim([event.ydata - y_range * (event.ydata - ylim[0]) / (ylim[1] - ylim[0]),
                     event.ydata + y_range * (ylim[1] - event.ydata) / (ylim[1] - ylim[0])])

        self._canvas_mpl.draw_idle()

    def _on_pan_press(self, event):
        """Left-click pans the image; right-click drags the selected ROI."""
        if event.button == 1 and self._drawing_mode.get() != "select":
            return  # Left-click reserved for drawing in non-select modes
        ax = event.inaxes
        if ax is None or ax not in self._active_image_axes():
            return
        if event.xdata is None:
            return
        # Right-click: drag the selected ROI (or hit-test under cursor)
        if event.button == 3:
            selected_id = self._roi_manager.get_selected_id()
            # If no ROI selected, try to pick one under the cursor
            if selected_id is None:
                selected_id = self._hit_test_roi(event.xdata, event.ydata, ax)
            if selected_id is not None:
                self._start_roi_drag(selected_id, event.xdata, event.ydata)
                return
        # Left-click (or right-click that missed an ROI): pan the image
        self._pan_origin = (event.xdata, event.ydata, ax)

    def _on_pan_release(self, event):
        """End panning or ROI dragging."""
        if self._roi_drag is not None:
            self._finish_roi_drag()
        self._pan_origin = None

    def _on_pan_motion(self, event):
        """Pan image axes or drag ROI."""
        # ROI dragging takes priority
        if self._roi_drag is not None:
            if event.xdata is not None and event.ydata is not None:
                self._update_roi_drag(event.xdata, event.ydata)
            return
        if self._pan_origin is None:
            return
        ox, oy, ax = self._pan_origin
        if event.inaxes != ax or event.xdata is None:
            return
        dx = ox - event.xdata
        dy = oy - event.ydata
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
        ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
        self._canvas_mpl.draw_idle()

    # ── ROI hit-testing and dragging ──────────────────────────────

    def _hit_test_roi(self, x, y, ax):
        """Return the region_id of the ROI patch under (x, y), or None."""
        for region_id, patches in self._roi_patches.items():
            for patch in (patches if isinstance(patches, list) else [patches]):
                if patch.axes is ax and patch.contains_point(ax.transData.transform((x, y))):
                    return region_id
        return None

    def _start_roi_drag(self, region_id, x, y):
        """Begin dragging a ROI."""
        self._roi_drag = {'id': region_id, 'ox': x, 'oy': y}
        self._roi_manager.select_region(region_id)
        self._redraw_region_overlays()
        if self._roi_analysis_panel:
            self._roi_analysis_panel._refresh_region_list()

    def _update_roi_drag(self, x, y):
        """Move the dragged ROI by the mouse delta."""
        drag = self._roi_drag
        dx = x - drag['ox']
        dy = y - drag['oy']
        region = self._roi_manager.get_region(drag['id'])
        if region is None:
            self._roi_drag = None
            return
        new_coords = [[c[0] + dx, c[1] + dy] for c in region['coords']]
        self._roi_manager.update_region(drag['id'], coords=new_coords)
        drag['ox'] = x
        drag['oy'] = y
        self._redraw_region_overlays()

    def _finish_roi_drag(self):
        """Complete ROI drag and persist changes."""
        self._roi_drag = None
        self._save_regions_update()
        if self._roi_analysis_panel:
            self._roi_analysis_panel._refresh_region_list()

    def _on_display_mode_changed(self):
        """Switch which image is shown in the main slot when decay is hidden."""
        new_mode = self._sv_display_mode.get()
        if new_mode == self._display_mode:
            return
        self._display_mode = new_mode
        self._rebuild_layout()

    def _toggle_decay(self):
        """Show or hide the decay plot, rearranging the layout accordingly."""
        show = self._bv_show_decay.get()
        if show == self._decay_visible:
            return
        self._decay_visible = show
        self._rebuild_layout()

    def _rebuild_layout(self):
        """Rebuild the GridSpec layout based on decay visibility.

        Decay visible (default):
            Row 0: [Intensity] [FLIM] [cbar]     height_ratio 1
            Row 1: [       Decay          ]       height_ratio 0.6

        Decay hidden:
            Row 0: [    FLIM   ] [cbar]           height_ratio 1
            Row 1: [ Intensity ]                  height_ratio 0.5
        """
        from matplotlib.gridspec import GridSpec

        # Preserve any current image data from the axes
        flim_title = self._ax_flim.get_title() if self._ax_flim.get_visible() else "FLIM Lifetime (ns)"
        img_title = self._ax_img.get_title() if self._ax_img.get_visible() else "Intensity"

        # Store decay line data so it can be redrawn (update persistent cache)
        current_lines = []
        for line in self._ax_decay.get_lines():
            current_lines.append({
                'x': line.get_xdata().copy(),
                'y': line.get_ydata().copy(),
                'color': line.get_color(),
                'lw': line.get_linewidth(),
                'label': line.get_label(),
                'alpha': line.get_alpha(),
                'marker': line.get_marker(),
                'ms': line.get_markersize(),
            })
        if current_lines:
            self._cached_decay_lines = current_lines
            self._cached_decay_title = self._ax_decay.get_title()
            self._cached_decay_yscale = self._ax_decay.get_yscale()
        decay_lines = self._cached_decay_lines
        decay_title = self._cached_decay_title
        decay_yscale = self._cached_decay_yscale

        # Remove old axes
        for ax in (self._ax_img, self._ax_flim, self._ax_cbar, self._ax_decay):
            ax.remove()

        # Build new gridspec
        if self._decay_visible:
            gs = GridSpec(2, 3, figure=self._fig,
                          height_ratios=[1, 0.6],
                          width_ratios=[1, 1, 0.05],
                          hspace=0.3, wspace=0.15)
            self._ax_img   = self._fig.add_subplot(gs[0, 0])
            self._ax_flim  = self._fig.add_subplot(gs[0, 1])
            self._ax_cbar  = self._fig.add_subplot(gs[0, 2])
            self._ax_decay = self._fig.add_subplot(gs[1, :])
        else:
            if self._display_mode == 'intensity':
                # Single large intensity image, no colorbar needed
                gs = GridSpec(1, 1, figure=self._fig)
                self._ax_img   = self._fig.add_subplot(gs[0, 0])
                # Hidden placeholders
                self._ax_flim  = self._fig.add_axes([0, 0, 0.01, 0.01])
                self._ax_flim.set_visible(False)
                self._ax_cbar  = self._fig.add_axes([0, 0, 0.01, 0.01])
                self._ax_cbar.set_visible(False)
            else:
                # Single large FLIM image + colorbar
                gs = GridSpec(1, 2, figure=self._fig,
                              width_ratios=[1, 0.05],
                              wspace=0.08)
                self._ax_flim  = self._fig.add_subplot(gs[0, 0])
                self._ax_cbar  = self._fig.add_subplot(gs[0, 1])
                # Hidden placeholder
                self._ax_img   = self._fig.add_axes([0, 0, 0.01, 0.01])
                self._ax_img.set_visible(False)
            self._ax_decay = self._fig.add_axes([0, 0, 0.01, 0.01])
            self._ax_decay.set_visible(False)

        for _ax in (self._ax_img, self._ax_flim):
            _ax.set_facecolor('black')

        # Re-populate images from cached map data
        if self._ax_img.get_visible() and self._intensity_map is not None:
            import numpy as np
            intensity_clipped = np.clip(self._intensity_map, 0,
                                        np.percentile(self._intensity_map, 99))
            self._ax_img.imshow(intensity_clipped, cmap="inferno", origin="upper")
            self._ax_img.set_title(img_title, fontsize=9, fontweight="bold")
            self._strip_image_axes(self._ax_img)
        elif self._ax_img.get_visible():
            self._ax_img.set_title(img_title, fontsize=9, fontweight="bold")
            self._strip_image_axes(self._ax_img)

        if self._ax_flim.get_visible():
            if self._lifetime_map is not None:
                import numpy as np
                scaled = flim_display.apply_color_scale(
                    self._lifetime_map,
                    vmin=self._flim_color_scale['vmin'],
                    vmax=self._flim_color_scale['vmax'],
                    gamma=self._flim_color_scale['gamma'],
                )
                cmap = flim_display.get_colormap(self._flim_color_scale['cmap'])
                cmap.set_bad(color='black')
                im = self._ax_flim.imshow(scaled, cmap=cmap, origin="upper",
                                           vmin=0, vmax=1)
                # Rebuild colorbar
                if self._ax_cbar.get_visible():
                    self._ax_cbar.clear()
                    self._flim_cbar = None
                    valid = self._lifetime_map[~np.isnan(self._lifetime_map)]
                    if valid.size > 0:
                        cs = self._flim_color_scale
                        d_min = cs['vmin'] if cs['vmin'] is not None else float(np.min(valid))
                        d_max = cs['vmax'] if cs['vmax'] is not None else float(np.max(valid))
                        cbar = self._fig.colorbar(im, cax=self._ax_cbar)
                        cbar.set_label("τ (ns)", fontsize=8)
                        self._flim_cbar = cbar
                        n_ticks = 5
                        tp = np.linspace(0, 1, n_ticks)
                        tv = d_min + tp * (d_max - d_min)
                        cbar.set_ticks(tp)
                        cbar.set_ticklabels([f"{v:.2f}" for v in tv], fontsize=7)
            self._ax_flim.set_title(flim_title, fontsize=9, fontweight="bold")
            self._strip_image_axes(self._ax_flim)

        # Re-populate decay if visible
        if self._decay_visible and decay_lines:
            for ld in decay_lines:
                self._ax_decay.plot(
                    ld['x'], ld['y'],
                    color=ld['color'], linewidth=ld['lw'],
                    label=ld['label'], alpha=ld['alpha'],
                    marker=ld['marker'], markersize=ld['ms'],
                )
            self._ax_decay.set_yscale(decay_yscale)
            self._ax_decay.set_title(decay_title, fontsize=10, fontweight="bold", color='white')
            self._ax_decay.set_xlabel("Time (ns)", color='white')
            self._ax_decay.set_ylabel("Photon Count", color='white')
            self._ax_decay.set_facecolor('white')
            self._ax_decay.tick_params(labelsize=8, colors='white')
            self._ax_decay.grid(True, alpha=0.3)

        # Re-draw ROI overlays on the new FLIM axes
        self._redraw_region_overlays()

        # Reconnect drawing events to new axes
        self._setup_drawing_events()

        self._canvas_mpl.draw_idle()

    def _setup_drawing_events(self):
        """Connect matplotlib event handlers to FLIM axes for drawing."""
        # Disconnect old handlers to prevent accumulation across layout rebuilds
        for cid in getattr(self, '_draw_cids', []):
            self._canvas_mpl.mpl_disconnect(cid)
        self._draw_cids = [
            self._canvas_mpl.mpl_connect('button_press_event', self._on_draw_press),
            self._canvas_mpl.mpl_connect('motion_notify_event', self._on_draw_motion),
            self._canvas_mpl.mpl_connect('button_release_event', self._on_draw_release),
        ]
    
    def _active_image_axes(self):
        """Return the set of image axes that should accept drawing events."""
        return {ax for ax in (self._ax_img, self._ax_flim) if ax.get_visible()}

    def _on_draw_press(self, event):
        """Handle mouse press on image axes."""
        if not event.inaxes or event.inaxes not in self._active_image_axes():
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
        if not self._is_drawing or not event.inaxes or event.inaxes not in self._active_image_axes():
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
                               linewidth=1, linestyle='', alpha=0.5)
            event.inaxes.add_patch(preview)
            self._temp_line = preview
            self._canvas_mpl.draw_idle()
        
        # For polygon/freehand: collect intermediate points
        elif mode in ("polygon", "freehand"):
            self._draw_coords.append([event.xdata, event.ydata])
    
    def _on_draw_release(self, event):
        """Handle mouse release to complete drawing."""
        if not self._is_drawing or not event.inaxes or event.inaxes not in self._active_image_axes():
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
        self._scan_name = None  # Current FOV/scan stem for export filenames
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
            initialfile=f"{self._scan_name}_log.txt" if self._scan_name else "",
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
    
    def set_fit_result(self, fit_result: dict, output_dir: str, npz_path: str = None, scan_name: str = None):
        """Store fit result and enable export/save buttons."""
        self._fit_result = fit_result
        self._output_dir = output_dir
        if npz_path:
            self._current_npz_path = npz_path
        if scan_name:
            self._scan_name = scan_name
        # Enable buttons if there are images to export
        has_images = any(isinstance(v, np.ndarray) for v in (fit_result or {}).values())
        self._export_btn.configure(state="normal" if has_images else "disabled")
    
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
    
    def _export_summed_csv(self):
        """Export summed fit data (summary table) to CSV."""
        try:
            import csv
            from pathlib import Path
            
            init_name = f"{self._scan_name}_summed_fit.csv" if self._scan_name else None
            csv_file = filedialog.asksaveasfilename(
                title="Export Summed Fit Data",
                initialfile=init_name,
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialdir=self._output_dir)
            
            if not csv_file:
                return
            
            # Get all rows from summary table
            rows = []
            for item in self._tv.get_children():
                values = self._tv.item(item)['values']
                rows.append(values)  # (Parameter, Value, Unit)
            
            if not rows:
                messagebox.showwarning("No Data", "No summary data to export.")
                return
            
            # Write to CSV
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Parameter", "Value", "Unit"])
                writer.writerows(rows)
            
            messagebox.showinfo("Export Success", f"Summed fit data exported to:\n{Path(csv_file).name}")
            self._status.set(f"Exported → {Path(csv_file).name}")
            print(f"[Export] Summed fit CSV: {csv_file}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export CSV:\n{e}")
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
        
        # Add export button below treeview
        btn_bar = ttk.Frame(f)
        btn_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self._export_btn = ttk.Button(btn_bar, text="Export Images…", 
                                     command=self._on_export_clicked, state="disabled")
        self._export_btn.pack(side="left", padx=4)

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
            # Expand inner frame to fill canvas width and minimum height
            inner_height = inner.winfo_reqheight()
            canvas_height = canvas.winfo_height()
            if canvas_height > 1:  # Canvas has been sized
                canvas.itemconfigure(window_id, height=max(inner_height, canvas_height))
            canvas.itemconfigure(window_id, width=canvas.winfo_width() if canvas.winfo_width() > 1 else None)

        def _on_canvas_configure(evt):
            if evt.width > 1:
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

    def _set_pane_positions(self):
        """Set initial sash positions for 15:45:40 layout (Project:Settings:FOV)."""
        try:
            # Update to ensure accurate measurements
            self.root.update_idletasks()
            
            # Get main paned window width (excluding padding)
            main_width = self._main_paned.winfo_width()
            if main_width < 100:
                return
            
            # Calculate positions for 15:45:40 ratio
            # Total: left_paned (60%) | preview (40%)
            left_width = int(main_width * 0.6)
            
            # Set main paned sash (between left_paned and preview)
            self._main_paned.sashpos(0, left_width)
            
            # Within left_paned: Project (15%) | Settings (45%)
            project_width = int(main_width * 0.15)
            
            # Set left paned sash (between Project and Settings)
            self._left_paned.sashpos(0, project_width)
            
        except Exception as e:
            print(f"[Layout] Could not set pane positions: {e}")

    def run_with_progress(self, task_fn, task_name="Working…", on_done=None, output_dir=None):
        """Run a function in a thread, showing a pop-out progress window with cancel."""
        from flimkit.utils.crash_handler import log_event
        log_event(f"Task started: {task_name}")
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
            redir_err = _Redirect(self._res.log, self._buf, root=self.root, is_stderr=True)
            sys.stdout = redir
            sys.stderr = redir_err
            try:
                result = task_fn(progress_callback, cancel_event)
                self.root.after(0, lambda: win.close())
                if on_done:
                    self.root.after(0, lambda: on_done(result))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                from flimkit.utils.crash_handler import log_exception
                log_exception(f"run_with_progress: {task_name}")
                self.root.after(0, lambda e=exc: win.set_status(f"Error: {e}"))
                self.root.after(0, lambda: win.btn_cancel.config(text="Close", command=win.close))
            finally:
                if hasattr(redir, 'close'):
                    redir.close()
                else:
                    redir.flush()
                if hasattr(redir_err, 'close'):
                    redir_err.close()
                else:
                    redir_err.flush()
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr

        threading.Thread(target=worker, daemon=True).start()

    def _build_menu_bar(self):
        """Build the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # ===== FILE MENU =====
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        
        file_menu.add_command(label="Restore NPZ...", command=self._menu_restore_npz)
        file_menu.add_command(label="Save NPZ", command=self._menu_save_npz)
        file_menu.add_command(label="Save NPZ As...", command=self._menu_save_npz_as)
        file_menu.add_separator()
        file_menu.add_command(label="Open Project Folder...", command=self._menu_open_project_folder)
        file_menu.add_separator()
        
        # Export submenu
        export_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Export", menu=export_menu)
        export_menu.add_command(label="Export Summed Fit CSV", command=self._menu_export_fit_csv)
        export_menu.add_command(label="Export ROI Table CSV", command=self._menu_export_roi_csv)
        export_menu.add_command(label="Export ROI as GeoJSON", command=self._menu_export_roi_geojson)
        export_menu.add_command(label="Export All ROIs as GeoJSON", command=self._menu_export_all_rois_geojson)
        
        file_menu.add_command(label="Import GeoJSON...", command=self._menu_import_geojson)
        
        file_menu.add_separator()
        
        # Recent Files submenu
        self._recent_files_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Files", menu=self._recent_files_menu)
        self._recent_files = self._load_recent_list()  # [{"path": ..., "type": "file"|"project"}]
        self._update_recent_files_menu()
        
        file_menu.add_separator()
        file_menu.add_command(label="Preferences...", command=self._menu_preferences)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # ===== EDIT MENU =====
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        
        edit_menu.add_command(label="Undo", command=self._menu_undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self._menu_redo, accelerator="Ctrl+Shift+Z")
        edit_menu.add_separator()
        edit_menu.add_command(label="Reset", command=self._menu_reset)
        
        # Bind keyboard shortcuts
        self.root.bind("<Control-z>", lambda e: self._menu_undo())
        self.root.bind("<Control-Shift-Z>", lambda e: self._menu_redo())
        
        # ===== TOOLS MENU =====
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        tools_menu.add_command(label="Machine IRF Builder", command=self._menu_irf_builder)
        tools_menu.add_command(label="Batch Processing", command=self._menu_batch_processing)
        
        # ===== HELP MENU =====
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        help_menu.add_command(label="About", command=self._menu_about)
        help_menu.add_command(label="Documentation", command=self._menu_documentation)
        help_menu.add_separator()
        help_menu.add_command(label="View Error Logs", command=self._menu_view_error_logs)
        help_menu.add_command(label="Export Error Logs", command=self._menu_export_error_logs)
    
    _RECENT_FILE = os.path.join(os.path.expanduser("~"), ".flimkit", "recent.json")
    _MAX_RECENT = 10

    def _load_recent_list(self):
        """Load recent items from ~/.flimkit/recent.json."""
        try:
            with open(self._RECENT_FILE, "r") as f:
                import json
                data = json.load(f)
            return [e for e in data if isinstance(e, dict) and "path" in e and "type" in e]
        except (FileNotFoundError, ValueError):
            return []

    def _save_recent_list(self):
        """Persist recent items to ~/.flimkit/recent.json."""
        import json
        os.makedirs(os.path.dirname(self._RECENT_FILE), exist_ok=True)
        with open(self._RECENT_FILE, "w") as f:
            json.dump(self._recent_files, f, indent=2)

    def _update_recent_files_menu(self):
        """Update the Recent Files submenu."""
        self._recent_files_menu.delete(0, tk.END)
        if self._recent_files:
            for entry in self._recent_files:
                path = entry["path"]
                kind = entry.get("type", "file")
                prefix = "[Project] " if kind == "project" else ""
                label = f"{prefix}{path}"
                self._recent_files_menu.add_command(
                    label=label,
                    command=lambda e=entry: self._load_recent_item(e)
                )
            self._recent_files_menu.add_separator()
            self._recent_files_menu.add_command(label="Clear Recent", command=self._clear_recent_files)
        else:
            self._recent_files_menu.add_command(label="(No recent items)", state="disabled")

    def _load_recent_item(self, entry):
        """Load a recent file or project."""
        path = entry["path"]
        kind = entry.get("type", "file")
        if kind == "project":
            if os.path.isdir(path):
                print(f"[Menu] Opening recent project: {path}")
                if hasattr(self, '_proj_browser') and self._proj_browser:
                    self._proj_browser.load_folder(path)
                self._add_to_recent(path, "project")
            else:
                print(f"[Menu] Project folder not found: {path}")
        else:
            if os.path.isfile(path):
                print(f"[Menu] Loading recent file: {path}")
                self.sv_ptu.set(path)
                self._add_to_recent(path, "file")
            else:
                print(f"[Menu] File not found: {path}")

    def _add_to_recent(self, filepath, kind="file"):
        """Add a file or project to recent items (keep last N)."""
        path_str = str(filepath)
        self._recent_files = [e for e in self._recent_files if e["path"] != path_str]
        self._recent_files.insert(0, {"path": path_str, "type": kind})
        if len(self._recent_files) > self._MAX_RECENT:
            self._recent_files = self._recent_files[:self._MAX_RECENT]
        self._update_recent_files_menu()
        self._save_recent_list()

    def _clear_recent_files(self):
        """Clear the recent files list."""
        self._recent_files = []
        self._update_recent_files_menu()
        self._save_recent_list()
    
    # ===== FILE MENU CALLBACKS =====
    def _current_scan_stem(self) -> str:
        """Return the stem of the currently loaded PTU file, or empty string."""
        if hasattr(self, 'sv_ptu'):
            p = self.sv_ptu.get().strip()
            if p:
                return Path(p).stem
        return ""

    def _menu_restore_npz(self):
        """Restore NPZ file."""
        if self._res and hasattr(self._res, '_load_fitted_data'):
            self._res._load_fitted_data()
    
    def _menu_save_npz(self):
        """Save NPZ file."""
        if self._res and hasattr(self._res, '_on_save_npz_clicked'):
            self._res._on_save_npz_clicked()
    
    def _menu_save_npz_as(self):
        """Save NPZ as new file."""
        from tkinter import filedialog
        scan = self._current_scan_stem()
        npz_file = filedialog.asksaveasfilename(
            title="Save NPZ As",
            initialfile=f"{scan}.roi_session.npz" if scan else "",
            defaultextension=".npz",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")])
        if npz_file:
            print(f"[Menu] Saving NPZ as: {npz_file}")
            # TODO: Implement save logic with chosen filename
    
    def _menu_open_project_folder(self):
        """Open project folder dialog and load project."""
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="Select Project Folder")
        if folder:
            print(f"[Menu] Opening project folder: {folder}")
            # Delegate to project browser to load the folder
            if hasattr(self, '_proj_browser') and self._proj_browser:
                self._proj_browser.load_folder(folder)
            self._add_to_recent(folder, "project")
    
    def _menu_export_fit_csv(self):
        """Export summed fit CSV."""
        print("[Menu] Export Fit CSV")
        # Delegate to existing method if available
        if hasattr(self, '_res') and self._res:
            self._res._export_summed_csv()
    
    def _menu_export_roi_csv(self):
        """Export ROI table CSV."""
        print("[Menu] Export ROI CSV")
        # Delegate to ROI panel
        if self._roi_analysis_panel:
            self._roi_analysis_panel._export_all_rois_csv()
    
    def _menu_export_roi_geojson(self):
        """Export single ROI as GeoJSON."""
        print("[Menu] Export ROI GeoJSON")
        # Delegate to ROI panel
        if self._roi_analysis_panel:
            self._roi_analysis_panel._export_selected_region()
    
    def _menu_export_all_rois_geojson(self):
        """Export all ROIs as GeoJSON."""
        print("[Menu] Export All ROIs GeoJSON")
        # Delegate to ROI panel
        if self._roi_analysis_panel:
            self._roi_analysis_panel._export_all_rois_geojson()
    
    def _menu_import_geojson(self):
        """Import ROIs from GeoJSON file."""
        print("[Menu] Import GeoJSON")
        # Delegate to ROI panel
        if self._roi_analysis_panel:
            self._roi_analysis_panel._import_rois_geojson()
    
    def _menu_preferences(self):
        """Open preferences dialog."""
        from flimkit.utils.config_manager import cfg
        prefs = cfg.get_section("preferences")

        pref_win = tk.Toplevel(self.root)
        pref_win.title("Preferences")
        pref_win.geometry("500x400")
        pref_win.resizable(False, False)
        
        # Main frame with padding
        main_frame = ttk.Frame(pref_win, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title label
        ttk.Label(main_frame, text="FLIMKit Preferences", font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Create a notebook for tabs
        note = ttk.Notebook(main_frame)
        note.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # === Display Tab ===
        disp_frame = ttk.Frame(note, padding=10)
        note.add(disp_frame, text="Display")
        
        ttk.Label(disp_frame, text="Colormap:", font=("TkDefaultFont", 10)).pack(anchor="w", pady=(5, 0))
        cmap_var = tk.StringVar(value=prefs.get("colormap", "viridis"))
        ttk.Combobox(disp_frame, textvariable=cmap_var, 
                     values=["viridis", "plasma", "gray", "jet"], state="readonly").pack(anchor="w", pady=(0, 10))
        
        ttk.Label(disp_frame, text="Font Size:", font=("TkDefaultFont", 10)).pack(anchor="w", pady=(5, 0))
        font_var = tk.IntVar(value=prefs.get("font_size", 9))
        ttk.Spinbox(disp_frame, from_=8, to=14, textvariable=font_var, width=10).pack(anchor="w", pady=(0, 10))
        
        # === Analysis Tab ===
        anal_frame = ttk.Frame(note, padding=10)
        note.add(anal_frame, text="Analysis")
        
        ttk.Label(anal_frame, text="Default Number of Exponents:", font=("TkDefaultFont", 10)).pack(anchor="w", pady=(5, 0))
        exp_var = tk.IntVar(value=prefs.get("default_nexp", 2))
        ttk.Spinbox(anal_frame, from_=1, to=5, textvariable=exp_var, width=10).pack(anchor="w", pady=(0, 10))
        
        ttk.Label(anal_frame, text="Export Format:", font=("TkDefaultFont", 10)).pack(anchor="w", pady=(5, 0))
        fmt_var = tk.StringVar(value=prefs.get("export_format", "CSV"))
        ttk.Combobox(anal_frame, textvariable=fmt_var, 
                     values=["CSV", "Excel", "NumPy"], state="readonly").pack(anchor="w", pady=(0, 10))
        
        # === Files Tab ===
        files_frame = ttk.Frame(note, padding=10)
        note.add(files_frame, text="Files")
        
        ttk.Label(files_frame, text="Output Directory:", font=("TkDefaultFont", 10)).pack(anchor="w", pady=(5, 0))
        saved_outdir = prefs.get("output_directory", "") or os.path.expanduser("~/FLIMKit/output")
        output_var = tk.StringVar(value=saved_outdir)
        ttk.Entry(files_frame, textvariable=output_var, width=40).pack(anchor="w", pady=(0, 5))
        ttk.Button(files_frame, text="Browse...", width=10,
                   command=lambda: output_var.set(filedialog.askdirectory())).pack(anchor="w", pady=(0, 10))
        
        ttk.Label(files_frame, text="Auto-save NPZ:", font=("TkDefaultFont", 10)).pack(anchor="w", pady=(5, 0))
        autosave_var = tk.BooleanVar(value=prefs.get("auto_save_npz", True))
        ttk.Checkbutton(files_frame, text="Enable auto-save", variable=autosave_var).pack(anchor="w", pady=(0, 10))
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 0))
        
        def save_prefs():
            cfg.update_section("preferences", {
                "colormap": cmap_var.get(),
                "font_size": font_var.get(),
                "default_nexp": exp_var.get(),
                "export_format": fmt_var.get(),
                "output_directory": output_var.get(),
                "auto_save_npz": autosave_var.get(),
            })
            print(f"[Preferences] Saved to {cfg._CONFIG_FILE if hasattr(cfg, '_CONFIG_FILE') else '~/.flimkit/config.yaml'}")
            pref_win.destroy()
        
        ttk.Button(btn_frame, text="Save", command=save_prefs).pack(side="right", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=pref_win.destroy).pack(side="right", padx=5)
    
    # ===== EDIT MENU CALLBACKS =====
    def _menu_undo(self):
        """Undo last action."""
        print("[Menu] Undo")
        # TODO: Implement when undo/redo system is built
    
    def _menu_redo(self):
        """Redo last undone action."""
        print("[Menu] Redo")
        # TODO: Implement when undo/redo system is built
    
    def _menu_reset(self):
        """Reset current analysis."""
        from tkinter import messagebox
        if messagebox.askyesno("Reset", "Clear all regions and results? This cannot be undone."):
            print("[Menu] Resetting analysis...")
            # Clear ROI regions
            if self._roi_analysis_panel and self._roi_analysis_panel.fov_preview:
                roi_manager = self._roi_analysis_panel.fov_preview._roi_manager
                region_ids = [r['id'] for r in roi_manager.get_all_regions()]
                for region_id in region_ids:
                    roi_manager.remove_region(region_id)
                self._roi_analysis_panel._refresh_region_list()
                self._roi_analysis_panel.fov_preview._redraw_region_overlays()
            # Clear results panel
            if self._res:
                self._res._tv.delete(*self._res._tv.get_children())
                pass  # Export and save buttons now in File menu
            messagebox.showinfo("Reset Complete", "Analysis cleared.")
            print("[Menu] Reset complete.")
    
    # ===== TOOLS MENU CALLBACKS =====
    def _menu_irf_builder(self):
        """Open Machine IRF Builder in separate window."""
        print("[Menu] Machine IRF Builder")
        # Switch to IRF form view
        if hasattr(self, '_switch_form'):
            self._switch_form("irf")
        # TODO: (Optional) Open as separate window in future
    
    def _menu_batch_processing(self):
        """Switch to batch processing view."""
        print("[Menu] Batch Processing")
        # Switch to batch form
        if hasattr(self, '_switch_form'):
            self._switch_form("batch")
    
    # ===== HELP MENU CALLBACKS =====
    def _menu_about(self):
        """Show about dialog."""
        from flimkit._version import __version__
        about_text = f"""FLIMKit Analysis GUI

Version: {__version__}

A comprehensive FLIM data analysis platform with:
• Single FOV & tile stitching
• ROI-based lifetime analysis
• Machine IRF calibration
• Batch processing
• GeoJSON & CSV export

Built with Python, Tkinter, NumPy, and SciPy.
        """
        messagebox.showinfo("About FLIMKit", about_text)
    
    def _menu_documentation(self):
        """Open documentation."""
        import webbrowser
        import os
        doc_file = os.path.join(os.path.dirname(__file__), "../../documentation.md")
        if os.path.exists(doc_file):
            webbrowser.open('file://' + os.path.realpath(doc_file))
        else:
            messagebox.showinfo("Documentation", "See README.md in the project root for documentation.")
    
    def _menu_view_error_logs(self):
        """View error logs."""
        from flimkit.utils.crash_handler import build_export_report, get_log_dir
        import glob

        log_dir = get_log_dir()
        log_files = glob.glob(os.path.join(log_dir, "*.log")) if os.path.exists(log_dir) else []
        if log_files:
            try:
                report = build_export_report(include_all_sessions=False)
                from tkinter.scrolledtext import ScrolledText
                win = tk.Toplevel(self.root)
                win.title("Error Logs — Current Session")
                win.geometry("700x500")
                text_widget = ScrolledText(win, wrap=tk.WORD)
                text_widget.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
                text_widget.insert(tk.END, report)
                text_widget.config(state=tk.DISABLED)
            except Exception as e:
                messagebox.showerror("Error", f"Could not read log file: {e}")
        else:
            messagebox.showinfo("No Logs", "No error logs found.")


    def _menu_export_error_logs(self):
        """Export error logs to file with system info."""
        from flimkit.utils.crash_handler import build_export_report, get_log_dir
        import glob

        log_dir = get_log_dir()
        log_files = glob.glob(os.path.join(log_dir, "*.log")) if os.path.exists(log_dir) else []

        if not log_files:
            messagebox.showwarning("No Logs", "No error logs found to export.")
            return

        export_file = filedialog.asksaveasfilename(
            title="Save Error Report",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if export_file:
            try:
                report = build_export_report(include_all_sessions=True)
                with open(export_file, 'w', encoding='utf-8') as out_f:
                    out_f.write(report)
                messagebox.showinfo("Export Success", f"Error report exported to:\n{export_file}")
                print(f"[Menu] Error report exported to: {export_file}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export logs: {e}")

    def _find_scroll_canvas(self, widget):
        """Walk widget ancestry to find the nearest scrollable canvas."""
        w = widget
        for _ in range(30):          # max depth
            try:
                if hasattr(w, '_canvas'):
                    return w._canvas
                w = w.master
                if w is None:
                    break
            except Exception:
                break
        return None

    def _setup_global_scroll(self):
        """Bind mousewheel globally so scrolling works over any child widget."""
        def _scroll(evt):
            # Find canvas responsible for the widget under the cursor
            try:
                widget = self.root.winfo_containing(evt.x_root, evt.y_root)
                if widget is None:
                    return
                canvas = self._find_scroll_canvas(widget)
                if canvas is None:
                    return
                # Only scroll if canvas has a scrollable region
                sr = canvas.cget('scrollregion')
                if not sr:
                    return
                delta = evt.delta if hasattr(evt, 'delta') else 0
                if evt.num == 5 or delta < 0:
                    canvas.yview_scroll(3, "units")
                elif evt.num == 4 or delta > 0:
                    canvas.yview_scroll(-3, "units")
            except Exception:
                pass

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind_all(seq, _scroll, add="+")

    def _setup_global_dnd(self):
        """Register DnD drop targets on all Entry widgets in the UI."""
        if not HAS_DND:
            return
        try:
            from tkinterdnd2 import DND_FILES, DND_TEXT
        except ImportError:
            return

        def _clean(data: str) -> str:
            """Handle brace-quoted macOS paths and multi-file drops."""
            data = data.strip()
            # Multi-file drop: take first path only
            if data.startswith("{"):
                # Extract content of first braced group
                end = data.find("}")
                if end != -1:
                    return data[1:end].strip()
            # Space-separated paths (take first)
            parts = data.split()
            if parts:
                return parts[0]
            return data

        def _register(widget):
            try:
                widget.drop_target_register(DND_FILES, DND_TEXT)
                # Get the textvariable name and look it up on the widget
                tv_name = widget.cget("textvariable")
                if tv_name:
                    tv = widget.nametowidget(str(tv_name)) if False else None
                    # Simpler: bind directly to widget.set via StringVar
                    def _drop(evt, w=widget):
                        path = _clean(evt.data)
                        try:
                            # Directly update the entry text
                            w.delete(0, "end")
                            w.insert(0, path)
                            # Also fire any associated traces
                            w.event_generate("<<Modified>>")
                        except Exception:
                            pass
                    widget.dnd_bind("<<Drop>>", _drop)
            except Exception:
                pass

        def _walk(widget):
            try:
                cls = widget.winfo_class()
                if cls in ("Entry", "TEntry"):
                    _register(widget)
                for child in widget.winfo_children():
                    _walk(child)
            except Exception:
                pass

        _walk(self.root)
        print("[DnD] Drop targets registered on all Entry widgets")

    def _init_ui(self):
        """Build the entire user interface with left tab buttons and central content area."""
        # Install Tk-level error handler for exceptions in event loop callbacks
        from flimkit.utils.crash_handler import install_tk_error_handler
        install_tk_error_handler(self.root)

        self._buf: list = []
        self._current_session_file = None  # Track current session file for auto-save
        self._current_npz_path = None  # For backward compatibility
        self._last_loaded_ptu = None  # Guard against duplicate auto-loads
        self._last_loaded_xlif = None  # Guard against duplicate auto-loads
        self._ptu_after_id = None   # Pending after() ID for FOV load
        self._xlif_after_id = None  # Pending after() ID for XLIF load
        self._form_buttons = {}  # Dictionary to store form mode buttons
        self._form_frames = {}  # Dictionary to store form frames
        self._form_inner_frames = {}  # Dictionary to store scrollable inner frames

        # Build menu bar
        self._build_menu_bar()

        # Mode Toolbar (below menu bar)
        self._mode_toolbar = ttk.Frame(self.root)
        self._mode_toolbar.grid(row=0, column=0, sticky="ew", padx=10, pady=(2, 0))
        self._mode_toolbar.columnconfigure(0, weight=1)

        ttk.Label(self._mode_toolbar, text="Mode:", font=("TkDefaultFont", 9, "bold")).pack(side="left", padx=(0, 10))

        self.current_mode = tk.StringVar(value="fov")

        btn_fov = ttk.Button(self._mode_toolbar, text="Single FOV Fit", width=16,
                             command=lambda: self._switch_form("fov"))
        btn_fov.pack(side="left", padx=2)
        self._form_buttons["fov"] = btn_fov

        btn_stitch = ttk.Button(self._mode_toolbar, text="Tile Stitch/Fit", width=16,
                                command=lambda: self._switch_form("stitch"))
        btn_stitch.pack(side="left", padx=2)
        self._form_buttons["stitch"] = btn_stitch

        btn_phasor = ttk.Button(self._mode_toolbar, text="Phasor Analysis", width=16,
                                command=lambda: self._switch_form("phasor"))
        btn_phasor.pack(side="left", padx=2)
        self._form_buttons["phasor"] = btn_phasor

        ttk.Separator(self._mode_toolbar, orient="vertical").pack(side="left", fill="y", padx=10, pady=2)

        self.mode_status = tk.StringVar(value="Current: Single FOV Fit")
        ttk.Label(self._mode_toolbar, textvariable=self.mode_status, foreground="grey").pack(side="left", padx=10)

        # Update status in _switch_form
        self._form_labels = {"fov": "Single FOV Fit", "stitch": "Tile Stitch/Fit", "phasor": "Phasor Analysis"}
        
        # Separator between toolbar and main content
        ttk.Separator(self.root, orient="horizontal").grid(row=1, column=0, sticky="ew", pady=(2, 0))

        # Main horizontal PanedWindow: left (tabs+content+results) | right (FOV preview)
        self._main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        self._main_paned.grid(row=2, column=0, sticky="nsew", padx=4, pady=(2, 4))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=0, minsize=0)  # toolbar row - fixed height
        self.root.rowconfigure(1, weight=0, minsize=0)  # separator row - fixed height
        self.root.rowconfigure(2, weight=1)  # main content - expands

        
        # LEFT PANEL: Project Browser + Content area (both resizable)
        
        self._left_paned = ttk.PanedWindow(self._main_paned, orient="horizontal")
        self._main_paned.add(self._left_paned, weight=3)  # 60% of total width (for Project 20% + Settings 40%)

        # Project Browser Panel (resizable)
        btn_frame = ttk.Frame(self._left_paned)
        self._left_paned.add(btn_frame, weight=1)  # 20% of total (1 out of 1+2)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.rowconfigure(0, weight=1)

        self._proj_browser = ProjectBrowserPanel(btn_frame, app=self, width=170)
        self._proj_browser.frame.grid(row=0, column=0, sticky="nsew")

        # Content pane in resizable section
        content_paned = ttk.PanedWindow(self._left_paned, orient="vertical")
        self._left_paned.add(content_paned, weight=2)  # 40% of total (2 out of 1+2)

        # Create wrapper frames for each form (initially hidden)
        form_wrapper = ttk.Frame(content_paned)
        content_paned.add(form_wrapper, weight=3)  # Increased from 2 to give more space to forms
        form_wrapper.columnconfigure(0, weight=1)
        form_wrapper.rowconfigure(0, weight=1)

        # Analysis Notebook: Fit Settings | ROI Analysis
        # Only shown for FOV mode
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
        
        # Stitch Notebook: Fit Settings | ROI Analysis
        # Only shown for Stitch mode
        self._stitch_tabs = ttk.Notebook(form_wrapper)
        self._stitch_tabs.grid(row=0, column=0, sticky="nsew")
        self._stitch_tabs.grid_remove()  # Hidden by default
        
        # Tab 0: Stitch Fit Settings
        stitch_settings_outer = ttk.Frame(self._stitch_tabs)
        self._stitch_tabs.add(stitch_settings_outer, text="  Fit Settings  ")
        stitch_settings_outer.columnconfigure(0, weight=1)
        stitch_settings_outer.rowconfigure(0, weight=1)
        self._stitch_settings_tab = stitch_settings_outer
        
        # Tab 1: Stitch ROI Analysis
        stitch_roi_frame = ttk.Frame(self._stitch_tabs, padding=4)
        self._stitch_tabs.add(stitch_roi_frame, text="  ROI Analysis  ")
        stitch_roi_frame.columnconfigure(0, weight=1)
        stitch_roi_frame.rowconfigure(0, weight=1)
        self._stitch_roi_analysis_frame = stitch_roi_frame

        self._form_content_frame = form_wrapper

        # Register batch and irf scroll frames (menu-only; no sidebar button)
        for _fid in ("batch", "irf"):
            _outer, _inner = self._make_scroll_frame(form_wrapper)
            _outer.grid(row=0, column=0, sticky="nsew")
            _outer.grid_remove()
            self._form_inner_frames[_fid] = (_outer, _inner)
            self._form_frames[_fid]       = (_outer, _inner)

        # Create scrollable frames for each form
        # FOV: inside notebook's Fit Settings tab
        # Stitch: inside stitch notebook's Fit Settings tab
        # Others: traditional layout
        form_list = [
            ("fov", "Single FOV Fit"),
            ("stitch", "Tile Stitch/Fit"),
            ("phasor", "Phasor Analysis"),
            ("batch", "Batch Processing"),
            ("irf", "Machine IRF Builder"),
        ]
        
        for form_id, form_label in form_list:
            if form_id == "fov":
                # FOV form goes inside the Fit Settings tab
                outer, inner = self._make_scroll_frame(self._fit_settings_tab)
                outer.grid(row=0, column=0, sticky="nsew")
                outer.grid_remove()  # Hide initially; show on demand
                self._form_inner_frames[form_id] = (outer, inner)
            elif form_id == "stitch":
                # Stitch form goes inside the Stitch Fit Settings tab
                outer, inner = self._make_scroll_frame(self._stitch_settings_tab)
                outer.grid(row=0, column=0, sticky="nsew")
                outer.grid_remove()  # Hide initially; show on demand
                self._form_inner_frames[form_id] = (outer, inner)
            else:
                # Phasor (and others) use traditional layout
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

        # Expert fit settings overrides (shared by FOV and stitch tabs)
        from flimkit.utils.config_manager import cfg as _cfg_mgr
        _saved_expert = _cfg_mgr.get_section("expert")
        _is_default = all(_saved_expert.get(k) == v for k, v in _EXPERT_DEFAULTS.items())
        self._expert_overrides: dict = {} if _is_default else _saved_expert

        # Build form content for each form
        self._build_fov_tab()
        self._build_stitch_tab()
        self._build_phasor_tab()
        self._build_batch_tab()
        self._build_machine_irf_tab()
        # Batch and Machine IRF are accessible via the Tools menu

        
        # RIGHT PANEL: FOV Preview
        
        preview_frame = ttk.LabelFrame(self._main_paned, text="  FOV Preview  ", padding=4)
        self._main_paned.add(preview_frame, weight=2)  # 40% of total
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self._fov_preview = FOVPreviewPanel(preview_frame)
        self._fov_preview.grid(row=0, column=0, sticky="nsew")
        
        # Connect ROI Analysis panel to FOV preview 
        self._roi_analysis_panel.fov_preview = self._fov_preview
        self._fov_preview._roi_analysis_panel = self._roi_analysis_panel

        # Phasor panel shares the same right-panel cell; hidden until needed.
        self._phasor_panel = PhasorViewPanel(preview_frame, max_cursors=6)
        self._phasor_panel.on_change = self._on_phasor_change
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
        
        # Set sash positions for 20:40:40 layout (Project:Settings:FOV)
        self.root.after_idle(self._set_pane_positions)

        # Global mousewheel scroll (works over any child widget)
        self._setup_global_scroll()

        # Global DnD registration (all Entry widgets)
        # Deferred so all widgets exist before we walk the tree.
        self.root.after(500, self._setup_global_dnd)

        # Set close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Set window icon if available
        self._set_window_icon()

    def _refresh_scrollable_frame(self, form_id: str):
        """Refresh the scrollable frame canvas after it's been shown."""
        if form_id not in self._form_inner_frames:
            return

        outer, inner = self._form_inner_frames[form_id]
        if not hasattr(outer, '_canvas'):
            return

        # Guard against re-entrant refreshes
        if hasattr(outer, '_refresh_scheduled') and outer._refresh_scheduled:
            return
        outer._refresh_scheduled = True

        canvas = outer._canvas
        window_id = outer._window_id

        def do_refresh():
            try:
                self.root.update_idletasks()
                canvas_width = canvas.winfo_width()
                canvas_height = canvas.winfo_height()
                if canvas_width > 1:
                    canvas.itemconfigure(window_id, width=canvas_width)
                inner.update()
                inner_height = inner.winfo_reqheight()
                if inner_height > 1:
                    canvas.itemconfigure(window_id, height=inner_height)
                bbox = canvas.bbox("all")
                if bbox:
                    canvas.configure(scrollregion=bbox)
                else:
                    h = max(canvas_height, inner_height, 300) if inner_height > 1 else 500
                    w = max(canvas_width, 300) if canvas_width > 1 else 400
                    canvas.configure(scrollregion=(0, 0, w, h))
                canvas.yview_moveto(0)
                canvas.update()
            except Exception:
                pass
            finally:
                outer._refresh_scheduled = False

        self.root.after_idle(lambda: self.root.after(80, do_refresh))
        
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
            # batch and irf are menu-only — no sidebar button to highlight
            
            self._current_form = form_id

            # FOV mode: use notebook with tabs
            if form_id == "fov":
                # Hide stitch notebook and phasor
                self._stitch_tabs.grid_remove()
                if "phasor" in self._form_inner_frames:
                    self._form_inner_frames["phasor"][0].grid_remove()

                # Restore the FOV ROI-analysis panel connection.
                # Switching to stitch mode overwrites _roi_analysis_panel on the
                # preview with _stitch_roi_panel; we must put it back so that FOV
                # drawing / analysis callbacks reach the visible panel.
                if hasattr(self, '_roi_analysis_panel'):
                    self._fov_preview._roi_analysis_panel = self._roi_analysis_panel
                
                # Now show FOV form
                fov_frame = self._form_inner_frames["fov"][0]
                fov_frame.grid(row=0, column=0, sticky="nsew")
                fov_frame.lift()
                fov_frame.tkraise()
                
                # Show notebook with Fit Settings + ROI Analysis tabs
                self._analysis_tabs.grid(row=0, column=0, sticky="nsew")
                self._analysis_tabs.lift()
                self._analysis_tabs.tkraise()
                
                # Select Fit Settings tab
                self._analysis_tabs.select(0)
                
                # Force layout update (update_idletasks avoids re-entrant event processing)
                self._fit_settings_tab.update_idletasks()
                
                # Refresh canvas to ensure content displays properly
                self._refresh_scrollable_frame(form_id)

            # Stitch mode: use notebook with tabs
            elif form_id == "stitch":
                # Hide FOV notebook and phasor
                self._analysis_tabs.grid_remove()
                if "phasor" in self._form_inner_frames:
                    self._form_inner_frames["phasor"][0].grid_remove()
                
                # Hide FOV form inside FOV notebook
                if "fov" in self._form_inner_frames:
                    self._form_inner_frames["fov"][0].grid_remove()
                
                # Show stitch form
                stitch_frame = self._form_inner_frames["stitch"][0]
                stitch_frame.grid(row=0, column=0, sticky="nsew")
                stitch_frame.lift()
                stitch_frame.tkraise()
                
                # Show stitch notebook with Fit Settings + ROI Analysis tabs
                self._stitch_tabs.grid(row=0, column=0, sticky="nsew")
                self._stitch_tabs.lift()
                self._stitch_tabs.tkraise()
                
                # Add ROI analysis panel to stitch ROI tab
                if not hasattr(self, '_stitch_roi_panel'):
                    # Create a separate ROI analysis panel for stitch mode
                    self._stitch_roi_panel = RoiAnalysisPanel(self._stitch_roi_analysis_frame)
                    self._stitch_roi_panel.grid(row=0, column=0, sticky="nsew")
                    # Connect to FOV preview
                    self._stitch_roi_panel.fov_preview = self._fov_preview
                    self._fov_preview._roi_analysis_panel = self._stitch_roi_panel
                
                # Select Fit Settings tab
                self._stitch_tabs.select(0)
                
                # Force layout update (update_idletasks avoids re-entrant event processing)
                self._stitch_settings_tab.update_idletasks()
                
                # Refresh canvas to ensure content displays properly
                self._refresh_scrollable_frame(form_id)

            # Other modes: traditional layout
            else:
                # Hide both notebooks
                self._analysis_tabs.grid_remove()
                self._stitch_tabs.grid_remove()
                
                # Hide all other traditional forms AND FOV/Stitch forms
                for fid in ("phasor", "fov", "stitch", "batch", "irf"):
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
                # Auto-populate phasor PTU + IRF from FOV fields if available
                if (hasattr(self, 'sv_ph_ptu') and hasattr(self, 'sv_ptu')
                        and not self.sv_ph_ptu.get().strip()):
                    fov_ptu = self.sv_ptu.get().strip()
                    if fov_ptu:
                        self.sv_ph_ptu.set(fov_ptu)
                    # Carry over IRF settings
                    if hasattr(self, '_irf_fov'):
                        method = self._irf_fov.sv_method.get()
                        if method == "irf_xlsx" and hasattr(self, 'sv_xlsx'):
                            xlsx = self.sv_xlsx.get().strip()
                            if xlsx and not self.sv_ph_irf.get().strip():
                                self.sv_ph_irf.set(xlsx)
                        elif method == "machine_irf":
                            mirf = self._irf_fov.sv_path.get().strip()
                            if mirf and not self.sv_ph_mirf.get().strip():
                                self.sv_ph_mirf.set(mirf)
            elif form_id in ("batch", "irf"):
                # No preview needed for batch/irf — hide both panels
                self._phasor_panel.frame.grid_remove()
                self._fov_preview.frame.grid_remove()
                label = "  Machine IRF Builder  " if form_id == "irf" else "  Batch Processing  "
                self._preview_frame_label.configure(text=label)
                # Show the IRF plot canvas if it exists (created after first build)
                if form_id == "irf" and hasattr(self, "_irf_plot_frame"):
                    self._irf_plot_frame.grid()
            else:
                self._phasor_panel.frame.grid_remove()
                if hasattr(self, "_irf_plot_frame"):
                    self._irf_plot_frame.grid_remove()
                self._fov_preview.frame.grid()

        if hasattr(self, 'mode_status'):
            self.mode_status.set(f"Current: {self._form_labels.get(form_id, form_id)}")
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
                "active_form": getattr(self, "_current_form", "fov"),
                
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

                # ── Stitch / tile-fit specific ──────────────────────────────
                "xlif_file":      self.sv_xlif.get()    if hasattr(self, 'sv_xlif')    else "",
                "ptu_dir":        self.sv_ptu_dir.get() if hasattr(self, 'sv_ptu_dir') else "",
                "out_st":         self.sv_out_st.get()  if hasattr(self, 'sv_out_st')  else "",
                "pipeline":       self.sv_pipeline.get() if hasattr(self, 'sv_pipeline') else "stitch_only",
                "bv_rotate":      self.bv_rotate.get()  if hasattr(self, 'bv_rotate')  else True,
                "bv_perpix":      self.bv_perpix.get()  if hasattr(self, 'bv_perpix')  else False,
                "tau_lo":         self.sv_tau_lo.get()  if hasattr(self, 'sv_tau_lo')  else "",
                "tau_hi":         self.sv_tau_hi.get()  if hasattr(self, 'sv_tau_hi')  else "",
                "int_lo":         self.sv_int_lo.get()  if hasattr(self, 'sv_int_lo')  else "",
                "int_hi":         self.sv_int_hi.get()  if hasattr(self, 'sv_int_hi')  else "",
                "thr_st_en":      self.bv_thr_st.get()  if hasattr(self, 'bv_thr_st')  else False,
                "thr_st_val":     self.sv_thr_st.get()  if hasattr(self, 'sv_thr_st')  else "",
                "bv_register":    self.bv_register.get() if hasattr(self, 'bv_register') else True,
                "reg_max_shift":  self.sv_reg_max_shift.get() if hasattr(self, 'sv_reg_max_shift') else "120",
                "irf_st_method":  self._irf_st.sv_method.get() if hasattr(self, '_irf_st') else "irf_xlsx",
                "irf_st_path":    self._irf_st.sv_path.get()   if hasattr(self, '_irf_st') else "",
                "tile_irf_dir":   self.sv_tile_irf_dir.get() if hasattr(self, 'sv_tile_irf_dir') else "",

                # Expert fit settings overrides
                "expert_overrides": self._expert_overrides if hasattr(self, '_expert_overrides') else {},
            }
            print(f"[Session] Captured form state: active_form={state.get('active_form')}")
            return state
        except Exception as e:
            print(f"[Session] Could not capture form state: {e}")
            return {}

    def _restore_form_state(self, state: dict):
        """Restore all form field values from captured state dict."""
        try:
            # Restore input files (skip PTU/XLIF—they're mode-specific and
            # should come fresh from project browser or user input, not old state)
            # if "ptu_file" in state and hasattr(self, 'sv_ptu'):
            #     self.sv_ptu.set(state["ptu_file"])
            # if "xlsx_file" in state and hasattr(self, 'sv_xlsx'):
            #     self.sv_xlsx.set(state["xlsx_file"])
            
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
            
            # ── Stitch-form fields ────────────────────────────────────────────
            # Skip xlif_file and ptu_dir—they're mode-specific and should come
            # fresh from project browser, not from old state
            # if "xlif_file" in state and hasattr(self, 'sv_xlif'):
            #     self.sv_xlif.set(state["xlif_file"])
            # if "ptu_dir" in state and hasattr(self, 'sv_ptu_dir'):
            #     self.sv_ptu_dir.set(state["ptu_dir"])
            if "out_st" in state and hasattr(self, 'sv_out_st'):
                self.sv_out_st.set(state["out_st"])
            if "bv_rotate" in state and hasattr(self, 'bv_rotate'):
                self.bv_rotate.set(state["bv_rotate"])
            if "tau_lo" in state and hasattr(self, 'sv_tau_lo'):
                self.sv_tau_lo.set(state["tau_lo"])
            if "tau_hi" in state and hasattr(self, 'sv_tau_hi'):
                self.sv_tau_hi.set(state["tau_hi"])
            if "int_lo" in state and hasattr(self, 'sv_int_lo'):
                self.sv_int_lo.set(state["int_lo"])
            if "int_hi" in state and hasattr(self, 'sv_int_hi'):
                self.sv_int_hi.set(state["int_hi"])
            if "thr_st_en" in state and hasattr(self, 'bv_thr_st'):
                self.bv_thr_st.set(state["thr_st_en"])
            if "thr_st_val" in state and hasattr(self, 'sv_thr_st'):
                self.sv_thr_st.set(state["thr_st_val"])
            if "bv_register" in state and hasattr(self, 'bv_register'):
                self.bv_register.set(state["bv_register"])
            if "reg_max_shift" in state and hasattr(self, 'sv_reg_max_shift'):
                self.sv_reg_max_shift.set(state["reg_max_shift"])
            if "tile_irf_dir" in state and hasattr(self, 'sv_tile_irf_dir'):
                self.sv_tile_irf_dir.set(state["tile_irf_dir"])
            # Stitch IRF widget
            if "irf_st_method" in state and hasattr(self, '_irf_st'):
                self._irf_st.sv_method.set(state["irf_st_method"])
                self._irf_st._update()   # refresh path/note visibility
            if "irf_st_path" in state and hasattr(self, '_irf_st'):
                self._irf_st.sv_path.set(state["irf_st_path"])

            # Refresh IRF widget visibility after method restore
            if hasattr(self, '_irf_fov'):
                self._irf_fov._update()

            # Pipeline + per-pixel – restore conditional frames
            if "pipeline" in state and hasattr(self, 'sv_pipeline'):
                self.sv_pipeline.set(state["pipeline"])
                self._pipeline_changed()   # show/hide _fit_frame + tile_extras
            if "bv_perpix" in state and hasattr(self, 'bv_perpix'):
                self.bv_perpix.set(state["bv_perpix"])
                self._perpix_toggled()     # show/hide _pxf

            # ── Stitch-form fields ───────────────────────────────────────
            # Skip xlif_file and ptu_dir—they're mode-specific
            # if "xlif_file"     in state and hasattr(self, 'sv_xlif'):          self.sv_xlif.set(state["xlif_file"])
            # if "ptu_dir"       in state and hasattr(self, 'sv_ptu_dir'):        self.sv_ptu_dir.set(state["ptu_dir"])
            if "out_st"        in state and hasattr(self, 'sv_out_st'):         self.sv_out_st.set(state["out_st"])
            if "bv_rotate"     in state and hasattr(self, 'bv_rotate'):         self.bv_rotate.set(state["bv_rotate"])
            if "tau_lo"        in state and hasattr(self, 'sv_tau_lo'):         self.sv_tau_lo.set(state["tau_lo"])
            if "tau_hi"        in state and hasattr(self, 'sv_tau_hi'):         self.sv_tau_hi.set(state["tau_hi"])
            if "int_lo"        in state and hasattr(self, 'sv_int_lo'):         self.sv_int_lo.set(state["int_lo"])
            if "int_hi"        in state and hasattr(self, 'sv_int_hi'):         self.sv_int_hi.set(state["int_hi"])
            if "thr_st_en"     in state and hasattr(self, 'bv_thr_st'):         self.bv_thr_st.set(state["thr_st_en"])
            if "thr_st_val"    in state and hasattr(self, 'sv_thr_st'):         self.sv_thr_st.set(state["thr_st_val"])
            if "bv_register"   in state and hasattr(self, 'bv_register'):       self.bv_register.set(state["bv_register"])
            if "reg_max_shift" in state and hasattr(self, 'sv_reg_max_shift'): self.sv_reg_max_shift.set(state["reg_max_shift"])
            if "tile_irf_dir"  in state and hasattr(self, 'sv_tile_irf_dir'): self.sv_tile_irf_dir.set(state["tile_irf_dir"])
            if "irf_st_method" in state and hasattr(self, '_irf_st'):
                self._irf_st.sv_method.set(state["irf_st_method"])
                self._irf_st._update()   # refresh path/note visibility
            if "irf_st_path"   in state and hasattr(self, '_irf_st'): self._irf_st.sv_path.set(state["irf_st_path"])
            if hasattr(self, '_irf_fov'): self._irf_fov._update()
            # Trigger pipeline/perpix commands so conditional frames appear
            if "pipeline"  in state and hasattr(self, 'sv_pipeline'):
                self.sv_pipeline.set(state["pipeline"]); self._pipeline_changed()
            if "bv_perpix" in state and hasattr(self, 'bv_perpix'):
                self.bv_perpix.set(state["bv_perpix"]); self._perpix_toggled()

            # Restore expert fit settings
            if "expert_overrides" in state and hasattr(self, '_expert_overrides'):
                ex = state["expert_overrides"]
                if isinstance(ex, dict):
                    self._expert_overrides = ex
                    self._update_expert_banners()

            # Restore active form mode
            if "active_form" in state:
                _form = state["active_form"]
                # legacy NPZ stored an int index; convert to form-id string
                if isinstance(_form, int):
                    _form = [None, "fov", "stitch", "batch", "irf", "phasor"][_form] or "fov"
                if _form in self._form_buttons:
                    self._switch_form(_form)
            
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
                                    ax_decay.legend(fontsize=8, loc="upper right", labelcolor='black')
                            ax_decay.set_title("Summed Decay (reloaded)", fontsize=10, fontweight="bold", color='white')
                            ax_decay.set_xlabel("Time (ns)", color='white')
                            ax_decay.set_ylabel("Photon Count", color='white')
                            ax_decay.grid(True, alpha=0.3)
                            ax_decay.tick_params(labelsize=8, colors='white')

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

    def _auto_load_session_for_stitch(self, output_dir: str):
        """Check for and auto-load session file for a stitched ROI."""
        try:
            from pathlib import Path
            import json
            import numpy as np

            session_file = Path(output_dir) / "roi_session.npz"
            if not session_file.exists():
                return

            print(f"[Auto-Load] Found tile session: {session_file.name}")
            session_data = np.load(session_file, allow_pickle=True)

            # Convert to dict for easier handling
            loaded = {key: session_data[key] for key in session_data.files}

            # Restore form state
            if "form_state_json" in loaded:
                try:
                    form_state_str = loaded["form_state_json"]
                    if isinstance(form_state_str, np.ndarray):
                        form_state_str = form_state_str.item()
                    if isinstance(form_state_str, bytes):
                        form_state_str = form_state_str.decode('utf-8')
                    form_state = json.loads(form_state_str)
                    self._restore_form_state(form_state)
                except Exception as e:
                    print(f"[Auto-Load] Could not restore form state: {e}")

            # Restore summary table
            if "summary_params" in loaded:
                params = loaded["summary_params"]
                values = loaded["summary_values"]
                units = loaded["summary_units"]

                if isinstance(params, np.ndarray):
                    params = params.tolist()
                if isinstance(values, np.ndarray):
                    values = values.tolist()
                if isinstance(units, np.ndarray):
                    units = units.tolist()

                rows = []
                for p, v, u in zip(params, values, units):
                    if isinstance(p, bytes): p = p.decode('utf-8')
                    if isinstance(v, bytes): v = v.decode('utf-8')
                    if isinstance(u, bytes): u = u.decode('utf-8')
                    rows.append((str(p), str(v), str(u)))

                if rows:
                    self._res.populate_summary(rows)

            # Restore FOV preview state (color scale, n_exp)
            if "fov_color_scale" in loaded:
                try:
                    cs = loaded["fov_color_scale"]
                    if isinstance(cs, bytes):
                        cs = cs.decode('utf-8')
                    self._fov_preview._flim_color_scale = json.loads(cs)
                except:
                    pass

            if "fov_n_exp" in loaded:
                n_exp = loaded["fov_n_exp"]
                if isinstance(n_exp, (np.integer, int)):
                    self._fov_preview._n_exp = int(n_exp)

            # Restore ROI regions (before redrawing FLIM image)
            if "fov_regions" in loaded:
                regions_json = loaded["fov_regions"]
                if isinstance(regions_json, np.ndarray):
                    regions_json = regions_json.item()
                if isinstance(regions_json, bytes):
                    regions_json = regions_json.decode('utf-8')
                if regions_json:  # Only load if not empty
                    self._fov_preview._load_regions_from_json(regions_json)
                    if self._roi_analysis_panel:
                        self._roi_analysis_panel._refresh_region_list()

            # Build a fit_result dict from the loaded data
            fit_result = {}
            for key, val in loaded.items():
                if key in ("summary_params", "summary_values", "summary_units",
                        "form_state_json", "fov_regions", "fov_color_scale", "fov_n_exp"):
                    continue
                fit_result[key] = val

            # Reconstruct global_summary with hoisted arrays (like in _load_fitted_data_from_file)
            if "global_summary_json" in fit_result:
                from flimkit.UI.gui import _reconstruct_dict_from_session
                fit_result["global_summary"] = _reconstruct_dict_from_session(fit_result, "global_summary")

            # Ensure intensity TIFF is loaded before displaying fit results.
            # Call load_stitched_roi to get the intensity image from TIFF files,
            # but only if we haven't loaded it yet.
            if self._fov_preview._intensity_map is None:
                self._fov_preview.load_stitched_roi(output_dir)

            # Display the decay plot and FLIM image
            # display_fit_results will compute and draw the FLIM lifetime map
            # from pixel_maps in the fit result, so it will overwrite any TIFF-based
            # lifetime that load_stitched_roi might have started.
            self._fov_preview.display_fit_results(None, fit_result)
            
            # Ensure canvas is fully rendered before proceeding
            self._fov_preview._canvas_mpl.draw_idle()

            # Store fit result for export/save buttons
            self._res.set_fit_result(fit_result, output_dir, npz_path=str(session_file),
                                     scan_name=self._current_scan_stem())

            # Update status
            self._res._status.set("✓ Session restored — ready to export or re-fit")
            self._fov_preview._ctrl_frame.grid()
            self._res._export_btn.configure(state="normal")

            print("[Auto-Load] Tile session fully restored")

        except Exception as e:
            print(f"[Auto-Load] Error loading tile session: {e}")
            import traceback
            traceback.print_exc()

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

    def _load_fitted_data_from_file(self, npz_path: str, suppress_popups: bool = False):
        """Load previously fitted data from NPZ and display in results panel.
        
        Args:
            npz_path: Path to the NPZ file to load
            suppress_popups: If True, suppress success/error messagebox popups (e.g., when loading from project tree)
        """
        try:
            import numpy as np
            import json
            from pathlib import Path
            
            fit_result = self.load_roi_fit(npz_path)
            if not fit_result:
                if not suppress_popups:
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
                        ax_decay.legend(fontsize=8, loc="upper right", labelcolor='black')
                    ax_decay.set_title("Summed Decay", fontsize=10, fontweight="bold", color='white')
                    ax_decay.set_xlabel("Time (ns)", color='white')
                    ax_decay.set_ylabel("Photon Count", color='white')
                    ax_decay.grid(True, alpha=0.3)
                    ax_decay.tick_params(labelsize=8, colors='white')

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
            self._res.set_fit_result(fit_result, output_dir, npz_path=npz_path,
                                     scan_name=self._current_scan_stem())
            
            # Stay on the current form (no "results" form exists)
            if not suppress_popups:
                messagebox.showinfo("Success", f"Loaded fitted data from:\n{Path(npz_path).name}")
            
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            if not suppress_popups:
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

            # Resolve pixel size (µm/px) for scale bar
            pixel_size_um = self._get_pixel_size_um()
            if with_scalebar and pixel_size_um is None:
                print("[Export] No pixel size available — scale bar will be omitted")
                with_scalebar = False
            
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

                        if with_scalebar:
                            self._draw_scale_bar(ax, w, h, pixel_size_um)
                        
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

                        if with_scalebar:
                            self._draw_scale_bar(ax, w, h, pixel_size_um)
                        
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
                    ax.legend(fontsize=12, loc='upper right', framealpha=0.9, labelcolor='black')
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

    def _get_pixel_size_um(self) -> "float | None":
        """Return the pixel size in µm for the currently loaded scan, or None."""
        try:
            from pathlib import Path

            # 1) Stitch mode — check stitch metadata JSON for pixel_size_um
            if getattr(self, '_current_form', None) == 'stitch':
                xlif = self.sv_xlif.get().strip() if hasattr(self, 'sv_xlif') else ''
                if xlif:
                    from flimkit.utils.xml_utils import get_pixel_size_from_xlif
                    pixel_size_m, _ = get_pixel_size_from_xlif(Path(xlif))
                    if pixel_size_m and pixel_size_m > 0:
                        return pixel_size_m * 1e6

            # 2) FOV mode — read ImgHdr_PixRes from the PTU header
            ptu_path = getattr(self._fov_preview, '_ptu_path', None)
            if ptu_path and Path(ptu_path).exists():
                from flimkit.PTU.reader import PTUFile
                ptu = PTUFile(str(ptu_path), verbose=False)
                pix_res = ptu.tags.get('ImgHdr_PixRes', 0)
                if pix_res and float(pix_res) > 0:
                    # ImgHdr_PixRes is in metres per pixel
                    return float(pix_res) * 1e6
        except Exception as e:
            print(f"[Export] Could not determine pixel size: {e}")
        return None

    @staticmethod
    def _draw_scale_bar(ax, img_w_px: int, img_h_px: int, pixel_size_um: float):
        """Draw a scale bar in the bottom-right corner of *ax*.

        Chooses a "nice" bar length (1, 2, 5, 10, 20, 50, 100, 200, 500 µm)
        that occupies roughly 15-25 % of the image width.
        """
        from matplotlib.patches import FancyBboxPatch

        fov_um = img_w_px * pixel_size_um
        target = fov_um * 0.20  # aim for ~20 % of width

        # Pick the largest "nice" value ≤ target
        nice = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        bar_um = nice[0]
        for n in nice:
            if n <= target:
                bar_um = n
            else:
                break

        bar_px = bar_um / pixel_size_um
        bar_h  = max(3, img_h_px * 0.015)  # bar thickness ≈ 1.5 % of height

        margin_x = img_w_px * 0.03
        margin_y = img_h_px * 0.03
        x0 = img_w_px - margin_x - bar_px
        y0 = img_h_px - margin_y - bar_h

        # Semi-transparent background behind bar + label
        label = f"{bar_um} µm"
        fontsize = max(7, min(14, img_h_px * 0.035))
        pad_x = bar_px * 0.08
        pad_y = fontsize * 1.8  # room for text above bar
        bg = FancyBboxPatch(
            (x0 - pad_x, y0 - pad_y),
            bar_px + 2 * pad_x,
            bar_h + pad_y + margin_y * 0.5,
            boxstyle="round,pad=4",
            facecolor="black", edgecolor="none", alpha=0.55,
            zorder=9,
        )
        ax.add_patch(bg)

        # White bar
        from matplotlib.patches import Rectangle
        bar = Rectangle((x0, y0), bar_px, bar_h,
                         facecolor="white", edgecolor="none", zorder=10)
        ax.add_patch(bar)

        # Label centred above bar
        ax.text(x0 + bar_px / 2, y0 - fontsize * 0.35,
                label, color="white", fontsize=fontsize,
                ha="center", va="bottom", zorder=10)

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

    
    # TAB 1 – Single-FOV FLIM fit
    
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

        # Exponential components (row 0)
        ttk.Label(fp, text="Exponential components:").grid(
            row=0, column=0, sticky="w", **PAD)
        self.iv_nexp_fov = tk.IntVar(value=2)
        for n in (1, 2, 3):
            ttk.Radiobutton(fp, text=str(n), variable=self.iv_nexp_fov,
                            value=n).grid(row=0, column=n, sticky="w", padx=1)

         #Fitting mode row (independent, no column-width interference) 
        mode_row = ttk.Frame(fp)
        mode_row.grid(row=1, column=0, columnspan=5, sticky="w", pady=(2, 0))

        ttk.Label(mode_row, text="Fitting mode:").pack(side="left", padx=(0, 10))
        self.sv_mode_fov = tk.StringVar(value="both")

        # Pack radio buttons tightly together
        radio_frame = ttk.Frame(mode_row)
        radio_frame.pack(side="left")
        ttk.Radiobutton(radio_frame, text="Full", variable=self.sv_mode_fov,
                        value="both").pack(side="left", padx=2)
        ttk.Radiobutton(radio_frame, text="Fast", variable=self.sv_mode_fov,
                        value="summed").pack(side="left", padx=2)
        # ttk.Radiobutton(radio_frame, text="Per‑pixel", variable=self.sv_mode_fov,
        #                 value="perPixel").pack(side="left", padx=2)

        # Optional explanatory text for "Summed"
        ttk.Label(mode_row, text="(fast = no FLIM image)",
                  foreground="grey").pack(side="left", padx=(10, 0))

        # Fit window (now row 2)
        ttk.Label(fp, text="Fit window (ns):").grid(row=2, column=0, sticky="w", **PAD)
        self.sv_tau_min_fov = tk.StringVar(value=str(_C()["Tau_min"]))
        self.sv_tau_max_fov = tk.StringVar(value=str(_C()["Tau_max"]))
        ttk.Entry(fp, textvariable=self.sv_tau_min_fov, width=7).grid(row=2, column=1, sticky="w", padx=4)
        ttk.Label(fp, text="to").grid(row=2, column=2)
        ttk.Entry(fp, textvariable=self.sv_tau_max_fov, width=7).grid(row=2, column=3, sticky="w", padx=4)
        ttk.Label(fp, text="ns  (fitting range)", foreground="grey").grid(row=2, column=4, sticky="w")

        # Output prefix (now row 3)
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

        # Expert settings banner (hidden until expert settings are confirmed)
        self._expert_banner_fov = ttk.Label(
            tab, text="⚙  Custom expert settings active",
            foreground="#e8a838", font=("TkDefaultFont", 9, "bold"))
        self._expert_banner_fov.grid(row=4, column=0, sticky="w", padx=8)
        self._expert_banner_fov.grid_remove()

        # Bottom row: Expert Settings + Run button
        btn_row_fov = ttk.Frame(tab)
        btn_row_fov.grid(row=5, column=0, pady=8)
        ttk.Button(btn_row_fov, text="⚙  Expert Settings",
                   command=self._open_expert_settings).pack(side="left", padx=4)
        self._btn_fov = ttk.Button(btn_row_fov, text="▶  Run Single-FOV Fit",
                                   command=self._run_fov)
        self._btn_fov.pack(side="left", padx=4, ipadx=20, ipady=4)

    
    # TAB 2 – Tile Stitch / Fit
    
    def _build_stitch_tab(self):
        outer, tab = self._form_inner_frames["stitch"]
        tab.columnconfigure(0, weight=1)

        ff = _section(tab, "Input Files")
        ff.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ff.columnconfigure(1, weight=1)
        self.sv_xlif    = tk.StringVar()
        self.sv_xlif.trace_add("write", self._on_xlif_changed)
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

        # Expert settings banner (hidden until expert settings are confirmed)
        self._expert_banner_st = ttk.Label(
            tab, text="⚙  Custom expert settings active",
            foreground="#e8a838", font=("TkDefaultFont", 9, "bold"))
        self._expert_banner_st.grid(row=3, column=0, sticky="w", padx=8)
        self._expert_banner_st.grid_remove()

        # Bottom row: Expert Settings + Run button
        btn_row_st = ttk.Frame(tab)
        btn_row_st.grid(row=4, column=0, pady=8)
        self._btn_expert_st = ttk.Button(btn_row_st, text="⚙  Expert Settings",
                   command=self._open_expert_settings)
        self._btn_expert_st.pack(side="left", padx=4)
        self._btn_st = ttk.Button(btn_row_st, text="▶  Run Tile Stitch",
                                  command=self._run_stitch)
        self._btn_st.pack(side="left", padx=4, ipadx=20, ipady=4)

    def _build_stitch_fit(self, parent):
        self._irf_st = IRFWidget(parent, default="machine_irf",
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
        ttk.Checkbutton(fp, text="Per-pixel fitting [REQUIRED FOR ROI ANALYSIS]",
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
                        variable=self.bv_save_tau_weighted).grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Checkbutton(self._pxf, text="Export intensity-weighted map",
                        variable=self.bv_save_int_weighted).grid(row=0, column=1, sticky="w", padx=(0, 8))
        ttk.Checkbutton(self._pxf, text="Export amplitude-weighted map",
                        variable=self.bv_save_amp_weighted).grid(row=1, column=0, sticky="w", padx=(0, 8))
        ttk.Checkbutton(self._pxf, text="Save individual component maps",
                        variable=self.bv_save_ind).grid(row=1, column=1, sticky="w")

        self.sv_tau_lo = tk.StringVar()
        self.sv_tau_hi = tk.StringVar()
        self.sv_int_lo = tk.StringVar()
        self.sv_int_hi = tk.StringVar()

        # Range controls for weighted maps
        ttk.Label(self._pxf, text="Lifetime display (ns):").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(self._pxf, textvariable=self.sv_tau_lo, width=7).grid(row=2, column=1, padx=4)
        ttk.Label(self._pxf, text="to").grid(row=2, column=2)
        ttk.Entry(self._pxf, textvariable=self.sv_tau_hi, width=7).grid(row=2, column=3, padx=4)
        ttk.Label(self._pxf, text="(blank = auto)", foreground="grey").grid(row=2, column=4, padx=4)

        ttk.Label(self._pxf, text="Intensity display:").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(self._pxf, textvariable=self.sv_int_lo, width=7).grid(row=3, column=1, padx=4)
        ttk.Label(self._pxf, text="to").grid(row=3, column=2)
        ttk.Entry(self._pxf, textvariable=self.sv_int_hi, width=7).grid(row=3, column=3, padx=4)
        ttk.Label(self._pxf, text="(blank = auto)", foreground="grey").grid(row=3, column=4, padx=4)

        self._pxf.grid_remove()

        # Fit window — applies to all fitting modes, not just per-pixel
        self.sv_tau_fit_lo = tk.StringVar(value=str(_C()["Tau_min"]))
        self.sv_tau_fit_hi = tk.StringVar(value=str(_C()["Tau_max"]))
        ttk.Label(fp, text="Fit window (ns):").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(fp, textvariable=self.sv_tau_fit_lo, width=7).grid(row=3, column=1, padx=4)
        ttk.Label(fp, text="to").grid(row=3, column=2)
        ttk.Entry(fp, textvariable=self.sv_tau_fit_hi, width=7).grid(row=3, column=3, padx=4)
        ttk.Label(fp, text="ns  (fitting range)", foreground="grey").grid(row=3, column=4, padx=4)

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

    def _apply_expert_overrides(self, a):
        """Apply expert settings overrides to an argparse.Namespace."""
        ex = self._expert_overrides
        if not ex:
            return
        if "optimizer" in ex:
            a.optimizer = ex["optimizer"]
        if "de_population" in ex:
            a.de_population = ex["de_population"]
        if "de_maxiter" in ex:
            a.de_maxiter = ex["de_maxiter"]
        if "lm_restarts" in ex:
            a.restarts = ex["lm_restarts"]
        if "binning_factor" in ex:
            a.binning = ex["binning_factor"]
        if "n_workers" in ex:
            a.workers = ex["n_workers"]
        if "min_photons" in ex:
            a.min_photons = ex["min_photons"]
        if "cost_function" in ex:
            a.cost_function = ex["cost_function"]
        if "channels" in ex:
            a.channel = ex["channels"]

    def _open_expert_settings(self):
        """Open the expert settings dialog and update banners accordingly."""
        from flimkit.utils.config_manager import cfg
        # Merge persisted config with in-memory overrides
        saved = cfg.get_section("expert")
        merged = dict(saved)
        merged.update(self._expert_overrides)
        dlg = ExpertSettingsDialog(self.root, merged)
        self.root.wait_window(dlg)
        if dlg.result is not None:
            # Check if all values match defaults → treat as "no overrides"
            is_default = all(
                dlg.result.get(k) == v for k, v in _EXPERT_DEFAULTS.items()
            )
            if is_default:
                self._expert_overrides = {}
            else:
                self._expert_overrides = dlg.result
            # Persist to config.yaml
            cfg.update_section("expert", dlg.result)
            # Also save to project.json if a project is open
            if hasattr(self, '_proj_browser') and self._proj_browser and self._proj_browser._project:
                self._proj_browser._project.config["expert"] = dlg.result
                self._proj_browser._project.save()
                cfg.load_project_overrides(self._proj_browser._project.config)
            self._update_expert_banners()

    def _update_expert_banners(self):
        """Show or hide the expert settings banners on FOV and stitch tabs."""
        active = bool(self._expert_overrides)
        for banner in (self._expert_banner_fov, self._expert_banner_st):
            if active:
                banner.grid()
            else:
                banner.grid_remove()

    def _pipeline_changed(self):
        mode = self.sv_pipeline.get()
        if mode == "stitch_only":
            self._fit_frame.grid_remove()
            self._btn_st.configure(text="▶  Run Tile Stitch")
            self._btn_expert_st.pack_forget()
            self._expert_banner_st.grid_remove()
        elif mode == "stitch_fit":
            self._fit_frame.grid()
            self._tile_extras_frame.grid_remove()
            self._btn_st.configure(text="▶  Run Stitch + Fit")
            self._btn_expert_st.pack(side="left", padx=4, before=self._btn_st)
            self._update_expert_banners()
        else:  # tile_fit
            self._fit_frame.grid()
            self._tile_extras_frame.grid()
            self._btn_st.configure(text="▶  Run Per-Tile Fit")
            self._btn_expert_st.pack(side="left", padx=4, before=self._btn_st)
            self._update_expert_banners()
        self._update_form_scrollbar("stitch")
        self.root.after_idle(self._fit_window_to_screen)

    def _perpix_toggled(self):
        if self.bv_perpix.get():
            self._pxf.grid()
        else:
            self._pxf.grid_remove()
        # Update scrollbar for the stitch form when content changes
        self._update_form_scrollbar("stitch")
        self.root.after_idle(self._fit_window_to_screen)
    
    def _update_form_scrollbar(self, form_id: str):
        """Force scrollregion and canvas-window height after content changes."""
        if form_id not in self._form_inner_frames:
            return
        try:
            outer, inner = self._form_inner_frames[form_id]
            if not hasattr(outer, '_canvas'):
                return
            canvas    = outer._canvas
            window_id = outer._window_id

            def _refresh():
                # inner.update() (not update_idletasks) forces Tkinter to finish
                # all geometry passes so winfo_reqheight() is accurate for any
                # newly shown/hidden child widgets (e.g. _pxf, _fit_frame).
                try:
                    inner.update()
                except Exception:
                    pass
                bbox = canvas.bbox("all")
                if bbox:
                    canvas.configure(scrollregion=bbox)
                new_h    = inner.winfo_reqheight()
                canvas_h = canvas.winfo_height()
                target_h = max(new_h, canvas_h if canvas_h > 1 else 0)
                if target_h > 0:
                    canvas.itemconfigure(window_id, height=target_h)
                cw = canvas.winfo_width()
                if cw > 1:
                    canvas.itemconfigure(window_id, width=cw)

            # Two-pass: after_idle lets grid() commit, then 80 ms absorbs any
            # deferred relayouts inside notebook tabs.
            self.root.after_idle(lambda: self.root.after(80, _refresh))
        except Exception as e:
            print(f"[Scrollbar] {form_id}: {e}")


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

    
    # TAB 4 – Machine IRF Builder
    
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

    
    # TAB 5 – Phasor
    
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

    
    # FOV Preview auto-load
    
    def _on_fov_ptu_changed(self, var, index, mode):
        ptu_path = self.sv_ptu.get().strip()
        if not ptu_path:
            return

        # Skip if already loading or if the path hasn't changed
        if ptu_path == self._last_loaded_ptu:
            return
        if getattr(self, '_loading_ptu', False):
            return

        self._last_loaded_ptu = ptu_path
        self._loading_ptu = True
        self._add_to_recent(ptu_path, "file")

        def load():
            try:
                self._fov_preview.load_fov(ptu_path)
                self._auto_load_session_for_ptu(ptu_path)
            finally:
                self._loading_ptu = False

        if self._ptu_after_id is not None:
            self.root.after_cancel(self._ptu_after_id)
        self._ptu_after_id = self.root.after(100, load)

    def _cancel_pending_scan_loads(self):
        """Cancel any pending after() callbacks from trace-driven scan loads."""
        if self._ptu_after_id is not None:
            self.root.after_cancel(self._ptu_after_id)
            self._ptu_after_id = None
        if self._xlif_after_id is not None:
            self.root.after_cancel(self._xlif_after_id)
            self._xlif_after_id = None
        self._loading_ptu = False
        self._loading_xlif = False

    def _on_xlif_changed(self, var, index, mode):
        """Auto-load stitched preview and session when XLIF file is selected."""
        xlif_path = self.sv_xlif.get().strip()
        if not xlif_path:
            return

        # Prevent duplicate loads
        if xlif_path == self._last_loaded_xlif:
            return
        if getattr(self, '_loading_xlif', False):
            return

        self._last_loaded_xlif = xlif_path
        self._loading_xlif = True

        # Find the corresponding ScanRecord in the project
        stem = Path(xlif_path).stem
        output_dir = None
        session_file = None

        if hasattr(self, '_proj_browser') and self._proj_browser._project:
            rec = self._proj_browser._project.scans.get(stem)
            if rec and rec.out_st:
                # rec.out_st can be either:
                #   - Base output dir (e.g. /project/output) → append stem to get ROI dir
                #   - ROI-specific dir (e.g. /project/output/A_Control) → use as-is
                # Check if the path already ends with the stem to handle both cases
                out_path = Path(rec.out_st)
                roi_name = stem.replace(" ", "_")
                if out_path.name == roi_name:
                    # Already the ROI-specific dir
                    output_dir = str(out_path)
                else:
                    # Just the base dir, append stem
                    output_dir = str(out_path / roi_name)
                session_file = Path(output_dir) / "roi_session.npz"
                if not session_file.exists():
                    # Session might be missing, but output_dir is still correct for preview
                    session_file = None

        # Fallback to UI fields if project record not available
        if output_dir is None:
            ptu_dir = self.sv_ptu_dir.get().strip()
            out_base = self.sv_out_st.get().strip()
            if ptu_dir and out_base:
                roi_name = stem.replace(" ", "_")
                output_dir = str(Path(out_base) / roi_name)

        if output_dir is None:
            self._loading_xlif = False
            return  # Not enough info

        # Cancel any previous pending XLIF load
        if self._xlif_after_id is not None:
            self.root.after_cancel(self._xlif_after_id)

        # Load stitched preview and session in a single deferred callback
        _sf = session_file  # capture for closure
        def _do_xlif_load():
            try:
                self._fov_preview.load_stitched_roi(output_dir)
                if _sf and _sf.exists():
                    self._auto_load_session_for_stitch(output_dir)
            finally:
                self._loading_xlif = False

        self._xlif_after_id = self.root.after(100, _do_xlif_load)
    
    # Run handlers
    
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

        # Apply expert overrides (if any)
        self._apply_expert_overrides(a)

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

        # Apply expert overrides (if any)
        self._apply_expert_overrides(a)

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
                
                # Display the assembled lifetime image from tile fit output directory.
                # load_stitched_roi reads the TIFFs saved by save_assembled_maps /
                # make_lifetime_image.  display_fit_results then overlays the pooled
                # decay, IRF and fitted model on _ax_decay (which load_stitched_roi
                # never touches).
                try:
                    self._fov_preview.load_stitched_roi(a.output_dir)
                    print(f"[on_done] Loaded stitched ROI from {a.output_dir}")
                except Exception as e:
                    print(f"[Warning] Could not load stitched image: {e}")

                try:
                    # ptu_path=None is fine: display_fit_results falls back to
                    # canvas['intensity'] for the image and uses decay/time_ns from
                    # the fit_result dict for the decay panel.
                    self._fov_preview.display_fit_results(None, fit_result)
                    print(f"[on_done] Displayed fit results (decay + FLIM) from fit_result")
                except Exception as e:
                    import traceback
                    print(f"[Warning] Could not display fit results: {e}")
                    traceback.print_exc()
                
                # Save tile fit results to NPZ via _save_roi_progress.
                # That method (used by the FOV pipeline) iterates fit_result.items(),
                # saves all numpy arrays directly, and hoists arrays out of nested
                # dicts such as 'canvas'.  This correctly captures decay, irf_prompt,
                # time_ns and every per-pixel canvas array (tau1, tau_mean_amp, etc.).
                try:
                    self._save_roi_progress(str(a.output_dir), fit_result, rows or [])
                    npz_path = Path(a.output_dir) / "roi_session.npz"
                    self._res.set_fit_result(
                        fit_result, str(a.output_dir),
                        npz_path=str(npz_path) if npz_path.exists() else None,
                        scan_name=self._current_scan_stem(),
                    )
                    print(f"[on_done] Saved tile fit session → {npz_path}")
                    # Notify project browser: remember output dir, refresh indicators.
                    if hasattr(self, '_proj_browser'):
                        xlif_stem = Path(a.xlif).stem if hasattr(a, 'xlif') else None
                        if xlif_stem:
                            # a.output_dir is the roi-specific dir  (e.g. /project/output/R_2).
                            # ScanRecord.session_path appends roi_clean to out_st, so we must
                            # store the PARENT (base output dir) here, not the roi-specific dir.
                            # Storing the roi-specific dir causes double-nesting and makes
                            # has_session always False, keeping the indicator stuck at ○.
                            self._proj_browser.on_fit_done(
                                xlif_stem,
                                out_st   = str(Path(a.output_dir).parent),
                                ptu_dir  = getattr(a, 'ptu_dir',    None),
                            )
                except Exception as e:
                    import traceback
                    print(f"[Warning] Could not save NPZ: {e}")
                    traceback.print_exc()
                
                # Refresh ROI Analysis panel
                try:
                    if hasattr(self, '_stitch_roi_panel'):
                        self._stitch_roi_panel._refresh_region_list()
                except:
                    pass
            else:
                # For stitch_only and stitch_fit, load the stitched ROI
                print(f"[on_done] Executing stitch branch (load_stitched_roi)")
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

            self._phasor_thread(_worker, _done, status="  Loading session…")

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
                # Auto-save phasor session next to the PTU file
                self._auto_save_phasor(ptu)
                self._res.set_status(
                    "✓  Phasor data loaded — click the phasor to place cursors.")

            self._phasor_thread(_worker, _done,
                                status="  Loading PTU and computing phasors…")

    def _phasor_thread(self, worker_fn, done_cb, *, status="  Working…"):
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

    def _auto_save_phasor(self, ptu_path: str):
        """Auto-save phasor session next to the PTU file as {stem}_phasor.npz."""
        try:
            sd = self._phasor_panel.get_session_dict()
            if sd.get('real_cal') is None:
                return
            p = Path(ptu_path)
            save_path = str(p.parent / f"{p.stem}_phasor.npz")
            from flimkit.phasor_launcher import save_session
            save_session(
                save_path,
                real_cal=sd['real_cal'],
                imag_cal=sd['imag_cal'],
                mean=sd['mean'],
                frequency=sd['frequency'],
                cursors=sd['cursors'],
                params=sd['params'],
                ptu_file=ptu_path,
                display_image=sd.get('display_image'),
            )
            print(f"[Phasor] Auto-saved session → {save_path}")
            # Notify project browser
            if hasattr(self, '_proj_browser'):
                self._proj_browser.on_phasor_done(p.stem)
        except Exception as e:
            print(f"[Phasor] Auto-save failed: {e}")

    def _on_phasor_change(self, panel):
        """Called when the user modifies cursors / params on the phasor panel."""
        ptu = self.sv_ph_ptu.get().strip() if hasattr(self, 'sv_ph_ptu') else ""
        if ptu and Path(ptu).exists():
            self._auto_save_phasor(ptu)

    def _restore_phasor_session(self, npz_path: str):
        """Load a phasor .npz session into the phasor panel."""
        try:
            min_ph = float(self.sv_ph_minph.get() or 0.01)
        except ValueError:
            min_ph = 0.01
        def _worker():
            from flimkit.phasor_launcher import load_session
            return load_session(npz_path)
        def _done(result):
            if isinstance(result, Exception):
                print(f"[Phasor] Could not restore session: {result}")
                return
            self._phasor_panel.load_session(result, min_photons=min_ph)
            self._res.set_status("✓  Phasor session restored.")
        self._phasor_thread(_worker, _done, status="  Restoring phasor session…")

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

        def on_done_irf(result):
            self._set_buttons("normal")
            self._res.set_status("✓  Machine IRF built.")
            if result is None or not isinstance(result, dict):
                return
            irf  = result.get("irf")
            meta = result.get("metadata", {})
            if irf is None:
                return
            tcspc_ns = float(meta.get("tcspc_res_ns_mean", 0.05))
            import numpy as np
            time_ns = np.arange(len(irf)) * tcspc_ns
            # Build or reuse IRF plot canvas in the right panel
            preview_parent = self._preview_frame_label
            if not hasattr(self, "_irf_plot_frame"):
                import tkinter as _tk
                from tkinter import ttk as _ttk
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                self._irf_plot_frame = _ttk.Frame(preview_parent)
                self._irf_plot_frame.grid(row=0, column=0, sticky="nsew")
                self._irf_fig = Figure(figsize=(6, 4), dpi=100, facecolor="#1e1e1e")
                self._irf_ax  = self._irf_fig.add_subplot(111)
                self._irf_canvas_mpl = FigureCanvasTkAgg(self._irf_fig, master=self._irf_plot_frame)
                self._irf_canvas_mpl.get_tk_widget().pack(fill="both", expand=True)
            # Draw the IRF
            ax = self._irf_ax
            ax.clear()
            ax.set_facecolor("#1e1e1e")
            self._irf_fig.patch.set_facecolor("#1e1e1e")
            n_pairs = meta.get("n_pairs", "?")
            anchor  = meta.get("align_anchor", "")
            reducer = meta.get("reducer", "")
            ax.plot(time_ns, irf, color="#00d4ff", linewidth=2, label=f"Machine IRF ({reducer})")
            ax.set_xlabel("Time (ns)", color="white")
            ax.set_ylabel("Amplitude (normalised)", color="white")
            ax.set_title(f"Machine IRF  |  {n_pairs} pairs  |  anchor={anchor}",
                         color="white", fontsize=10)
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#555")
            # Mark FWHM
            import numpy as _np
            half_max = irf.max() / 2
            above = _np.where(irf >= half_max)[0]
            if len(above) > 1:
                fwhm_ns = (above[-1] - above[0]) * tcspc_ns
                ax.axhline(half_max, color="#ff9900", linewidth=1,
                           linestyle="", alpha=0.7, label=f"FWHM={fwhm_ns*1000:.0f} ps")
                ax.axvspan(above[0]*tcspc_ns, above[-1]*tcspc_ns,
                           alpha=0.12, color="#ff9900")
            ax.legend(fontsize=8, facecolor="#2a2a2a", edgecolor="#555", labelcolor="white")
            self._irf_canvas_mpl.draw_idle()
            # Make it visible
            self._irf_plot_frame.grid()
            self._fov_preview.frame.grid_remove()
            self._preview_frame_label.configure(text="  Machine IRF Builder  ")

        self._launch(task_fn, output_dir=out_dir, task_name="Building Machine IRF",
                     _on_done_override=on_done_irf)

    def _launch(self, fn, output_dir=None, ptu_path=None, task_name="Working…",
                _on_done_override=None):
        self._buf.clear()
        self._set_buttons("disabled")
        self._res.set_status("  Running...")
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
            redir_err = _Redirect(self._res.log, self._buf, root=self.root, is_stderr=True)
            sys.stdout = redir
            sys.stderr = redir_err
            
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
                if _on_done_override is not None:
                    self.root.after(0, lambda r=result: _on_done_override(r))
                else:
                    self.root.after(0, lambda: self._on_done(rows, output_dir, result, ptu_path))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                from flimkit.utils.crash_handler import log_exception
                log_exception(f"_launch: {task_name}")
                self.root.after(0, lambda e=exc: self._res.set_status(f"✗  Error: {e}"))
            finally:
                if hasattr(redir, 'close'):
                    redir.close()
                else:
                    redir.flush()
                if hasattr(redir_err, 'close'):
                    redir_err.close()
                else:
                    redir_err.flush()
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
        self._res.set_fit_result(fit_result or {}, output_dir, npz_path=npz_file_path,
                                 scan_name=Path(ptu_path).stem if ptu_path else self._current_scan_stem())

        # Notify project browser so the session indicator (● / ○) refreshes
        # and the output prefix is remembered for next launch.
        if hasattr(self, '_proj_browser'):
            stem = Path(ptu_path).stem if ptu_path else None
            prefix = self.sv_out_fov.get().strip() if hasattr(self, 'sv_out_fov') else None
            if stem:
                self._proj_browser.on_fit_done(stem, output_prefix=prefix)

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
        for btn in (self._btn_fov, self._btn_st, self._btn_ph):
            btn.configure(state=state)



# Themed version (uses TKinterModernThemes)

if HAS_TKMT:
    class FLIMKitGUIThemed(TKMT.ThemedTKinterFrame, _UIBuilder):
        def __init__(self, theme="sun-valley", mode="dark"):
            super().__init__("FLIMkit Analysis GUI", theme, mode,
                             usecommandlineargs=True, useconfigfile=True)
            self.root = self.master
            self.root.minsize(760, 700)
            self._init_ui()
            # Skip TKMT's makeResizable — it sets weight=1 on every row,
            # overriding our weight=0 on the toolbar/separator rows.
            self.run(cleanresize=False)


# Fallback version (plain Tk, optional drag‑and‑drop)

class FLIMKitGUIFallback(_UIBuilder):
    def __init__(self, root):
        self.root = root
        self.root.title("FLIMkit Analysis GUI")
        self.root.minsize(760, 700)
        self._init_ui()
        self.root.mainloop()



# Entry point – choose themed or fallback

def launch_gui():
    global GUI_MODE
    GUI_MODE = True

    # Install crash handler and session logging
    from flimkit.utils.crash_handler import init_crash_handler
    init_crash_handler()

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