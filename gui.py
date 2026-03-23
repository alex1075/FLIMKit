#! /usr/bin/env python3
#!/usr/bin/env python3
"""
gui.py – Tkinter GUI front-end for FLIMkit
==========================================
Lives at the PROJECT ROOT (alongside the flimkit/ package folder).

Launch
------
  ./gui.py
  python gui.py
  python -m flimkit.gui   (also works if copied inside flimkit/)
"""

from __future__ import annotations

import re
import sys
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
from flimkit.utils.progress_window import ProgressWindow

# Modern theme support
try:
    import TKinterModernThemes as TKMT
    HAS_TKMT = True
except ImportError:
    HAS_TKMT = False

# Drag and drop support
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
# Stdout capture
# ─────────────────────────────────────────────────────────────────────────────

class _Redirect:
    """Redirect stdout/stderr to ScrolledText widget.
    
    Progress bars print each update as a new line for simplicity.
    """

    # ANSI escape code pattern to strip from output
    _ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')

    def __init__(self, widget: scrolledtext.ScrolledText, buf: list):
        self.widget = widget
        self.buf = buf

    def write(self, text: str):
        if not text:
            return
        self.buf.append(text)
        self.widget.configure(state="normal")
        
        # Strip ANSI escape codes
        text = self._ANSI_ESCAPE.sub('', text)
        # Treat carriage returns as newlines (progress bars print as new lines)
        text = text.replace('\r', '\n')
        
        # Simply append all text
        if text:
            self.widget.insert(tk.END, text)
        
        self.widget.see(tk.END)
        self.widget.configure(state="disabled")
        self.widget.update_idletasks()

    def flush(self):
        pass


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


def _on_mousewheel(event, canvas):
    """Handle mousewheel scrolling on Canvas."""
    if event.num == 5 or event.delta < 0:
        # Scroll down
        canvas.yview_scroll(3, "units")
    elif event.num == 4 or event.delta > 0:
        # Scroll up
        canvas.yview_scroll(-3, "units")


def _bind_mousewheel(widget, canvas):
    """Bind mousewheel events to canvas for scrolling."""
    # macOS uses <MouseWheel>
    widget.bind("<MouseWheel>", lambda e: _on_mousewheel(e, canvas))
    # Linux uses <Button-4> and <Button-5>
    widget.bind("<Button-4>", lambda e: _on_mousewheel(e, canvas))
    widget.bind("<Button-5>", lambda e: _on_mousewheel(e, canvas))
    
    # Recursively bind to all child widgets
    try:
        for child in widget.winfo_children():
            _bind_mousewheel(child, canvas)
    except (tk.TclError, AttributeError):
        pass


def _row(parent, label, var, row, browse_fn, width=45, state="normal"):
    ttk.Label(parent, text=label).grid(
        row=row, column=0, sticky="e", padx=6, pady=3)
    e = ttk.Entry(parent, textvariable=var, width=width, state=state)
    e.grid(row=row, column=1, sticky="ew", padx=4, pady=3)
    ttk.Button(parent, text="Browse...", command=browse_fn).grid(
        row=row, column=2, padx=4, pady=3)
    _enable_dnd_for_entry(e, var)
    return e


def _enable_dnd_for_entry(entry_widget: ttk.Entry, string_var: tk.StringVar):
    """Enable drag-and-drop for an entry widget bound to a StringVar."""
    if not HAS_DND:
        return
    
    def drop(event):
        """Handle file/folder drop events."""
        try:
            data = event.data.strip()
            if not data:
                return
            
            paths = []
            
            # Check if data contains brace-wrapped paths (typical for multiple selections)
            if '{' in data:
                # Extract paths from braces like: {/path1} {/path2}
                import re
                matches = re.findall(r'\{([^}]+)\}', data)
                paths = matches
            else:
                # Single path (possibly with spaces, no braces)
                # Remove any quotes
                path = data.strip('"\'')
                if path:
                    paths = [path]
            
            # Take the first path
            if paths:
                path = paths[0].strip()
                if path:
                    string_var.set(path)
        except Exception as e:
            # Silently fail if something goes wrong
            pass
    
    try:
        entry_widget.drop_target_register(DND_FILES, DND_TEXT)
        entry_widget.dnd_bind('<<Drop>>', drop)
    except Exception:
        # If DND fails silently, still allow the GUI to work
        pass


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
        _enable_dnd_for_entry(self._path_e, self.sv_path)
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
        self._build_images()

        self._status = tk.StringVar(value="Ready.")
        ttk.Label(self.frame, textvariable=self._status, foreground="grey").grid(
            row=1, column=0, sticky="w", padx=4, pady=(2, 4))

    # ── Progress ─────────────────────────────────────────────────────────────

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

    # ── Fit Summary ───────────────────────────────────────────────────────────

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

        tv.tag_configure("odd",  background="#f5f7fa")
        tv.tag_configure("even", background="#ffffff")
        tv.tag_configure("warn", foreground="#c0550a", background="#fff8f0")
        self._tv = tv

    def populate_summary(self, rows: list):
        for item in self._tv.get_children():
            self._tv.delete(item)
        for i, (param, val, unit) in enumerate(rows):
            tag = "warn" if param.startswith('⚠') else ("odd" if i % 2 else "even")
            self._tv.insert("", tk.END, values=(param, val, unit), tags=(tag,))
        if rows:
            self._nb.select(1)

    # ── Images (matplotlib) ───────────────────────────────────────────────────

    def _build_images(self):
        f = ttk.Frame(self._nb, padding=4)
        self._nb.add(f, text="  Images  ")
        f.columnconfigure(0, weight=1)
        f.rowconfigure(1, weight=1)

        ctrl = ttk.Frame(f)
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Button(ctrl, text="← Prev",       command=self._img_prev).pack(side="left",  padx=4)
        ttk.Button(ctrl, text="Next →",       command=self._img_next).pack(side="left",  padx=4)
        ttk.Button(ctrl, text="Save image…",  command=self._save_img).pack(side="left",  padx=4)
        ttk.Button(ctrl, text="Save all…",    command=self._save_all_imgs).pack(side="left",  padx=4)
        ttk.Button(ctrl, text="Open folder",  command=self._open_folder).pack(side="right", padx=4)
        self._img_lbl = tk.StringVar(value="No images yet")
        ttk.Label(ctrl, textvariable=self._img_lbl).pack(side="left", padx=8)

        self._fig = Figure(figsize=(6, 4), dpi=100, facecolor="#2b2b2b")
        self._ax  = self._fig.add_subplot(111)
        self._ax.set_facecolor("#2b2b2b")
        self._ax.axis("off")

        self._canvas_mpl = FigureCanvasTkAgg(self._fig, master=f)
        self._canvas_mpl.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        self._imgs: list         = []
        self._img_i: int         = 0
        self._folder: Optional[str] = None

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
        self._fig.tight_layout(pad=0.3)
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
# Main GUI
# ─────────────────────────────────────────────────────────────────────────────

class FLIMKitGUI(TKMT.ThemedTKinterFrame):
    
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
            # Rebind mousewheel to newly added children
            _bind_mousewheel(inner, canvas)

        def _on_canvas_configure(evt):
            canvas.itemconfigure(window_id, width=evt.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        
        # Bind mousewheel events to canvas and outer frame for scrolling
        _bind_mousewheel(outer, canvas)
        _bind_mousewheel(canvas, canvas)

        return inner

    def _fit_window_to_screen(self):
        """Clamp window geometry to visible screen bounds.

        Some Linux window managers can place/rescale Tk windows off-screen when
        the requested size jumps after showing additional controls.
        """
        self.root.update_idletasks()

        sw = max(1, int(self.root.winfo_screenwidth()))
        sh = max(1, int(self.root.winfo_screenheight()))

        # Leave a small safety margin so decorations stay visible.
        max_w = max(520, sw - 40)
        max_h = max(420, sh - 40)

        req_w = int(self.root.winfo_reqwidth())
        req_h = int(self.root.winfo_reqheight())
        cur_w = int(self.root.winfo_width())
        cur_h = int(self.root.winfo_height())

        target_w = min(max(cur_w, req_w, 760), max_w)
        target_h = min(max(cur_h, req_h, 700), max_h)

        x = int(self.root.winfo_x())
        y = int(self.root.winfo_y())
        x = min(max(0, x), max(0, sw - target_w))
        y = min(max(0, y), max(0, sh - target_h))

        self.root.maxsize(sw, sh)
        self.root.geometry(f"{target_w}x{target_h}+{x}+{y}")

    def run_with_progress(self, task_fn, task_name="Working…", on_done=None):
        """Run a function in a thread, showing a pop-out progress window with cancel."""
        win = ProgressWindow(self.root, task_name=task_name)
        cancel_event = win.cancelled

        def progress_callback(i, total):
            win.set_progress(i, maximum=total)
            if cancel_event.is_set():
                win.set_status("Cancelling…")

        def worker():
            # Redirect stdout/stderr to GUI buffer for this thread
            orig_stdout, orig_stderr = sys.stdout, sys.stderr
            redir = _Redirect(self._res.log, self._buf)
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
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr

        threading.Thread(target=worker, daemon=True).start()

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("FLIMkit Analysis GUI")
        root.resizable(True, True)
        root.minsize(760, 700)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self._buf: list = []

        # Use ttk.PanedWindow to allow draggable resizing of top panel vs results panel
        paned = ttk.PanedWindow(root, orient="vertical")
        paned.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Top panel (notebook with fitting options)
        nb_frame = ttk.Frame(paned)
        paned.add(nb_frame, weight=2)
        nb_frame.columnconfigure(0, weight=1)
        nb_frame.rowconfigure(0, weight=1)

        self._nb = ttk.Notebook(nb_frame)
        self._nb.grid(row=0, column=0, sticky="nsew")

        self._build_fov_tab()
        self._build_stitch_tab()
        self._build_batch_tab()
        self._build_machine_irf_tab()
        self._build_phasor_tab()

        # Results panel (progress, summary, images)
        results_frame = ttk.Frame(paned)
        paned.add(results_frame, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self._res = ResultsPanel(results_frame)
        self._res.grid(row=0, column=0, sticky="nsew")

        redir = _Redirect(self._res.log, self._buf)
        sys.stdout = redir
        sys.stderr = redir

        # Ensure initial window fits within current screen bounds.
        self.root.after_idle(self._fit_window_to_screen)

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 – Single-FOV FLIM fit
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_fov_tab(self):
        tab = self._make_scroll_tab("  Single FOV Fit  ")
        tab.columnconfigure(0, weight=1)

        ff = _section(tab, "Input Files")
        ff.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ff.columnconfigure(1, weight=1)
        self.sv_ptu  = tk.StringVar()
        self.sv_xlsx = tk.StringVar()
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
        self._out_fov_e = ttk.Entry(fp, textvariable=self.sv_out_fov, width=35)
        self._out_fov_e.grid(row=3, column=1, columnspan=3, sticky="ew", padx=4)
        _enable_dnd_for_entry(self._out_fov_e, self.sv_out_fov)

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

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 – Tile Stitch  ±  Fit
    # ═══════════════════════════════════════════════════════════════════════════

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
        self._tile_extras_frame.grid(row=3, column=0, sticky="ew", pady=(0, 4))
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


    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3 – Batch ROI Fit
    # ═══════════════════════════════════════════════════════════════════════════

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
                    tile_results, canvas_h, canvas_w, _ = fit_flim_tiles(
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
            task, task_name=f"Batch ROI Fit ({len(xlif_files)} ROIs)", on_done=on_done)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4 – Machine IRF Builder
    # ═══════════════════════════════════════════════════════════════════════════

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
        self._mirf_name_e = ttk.Entry(fo, textvariable=self.sv_mirf_name, width=35)
        self._mirf_name_e.grid(row=1, column=1, columnspan=2, sticky="ew", padx=4)

        self._btn_mirf = ttk.Button(
            tab,
            text="▶  Build Machine IRF",
            command=self._run_build_machine_irf,
        )
        self._btn_mirf.grid(row=3, column=0, pady=8, ipadx=20, ipady=4)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 5 – Phasor
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_phasor_tab(self):
        tab = self._make_scroll_tab("  Phasor Analysis  ")
        tab.columnconfigure(0, weight=1)

        fm = _section(tab, "Input Mode")
        fm.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        fm.columnconfigure(1, weight=1)

        self.sv_ph_mode = tk.StringVar(value="new")
        ttk.Radiobutton(fm, text="Analyse a new PTU file",
                        variable=self.sv_ph_mode, value="new",
                        command=self._ph_mode_changed).grid(
            row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Radiobutton(fm, text="Resume a saved session (.npz)",
                        variable=self.sv_ph_mode, value="session",
                        command=self._ph_mode_changed).grid(
            row=1, column=0, sticky="w", padx=4, pady=2)

        self._ph_new = ttk.Frame(tab)
        self._ph_new.columnconfigure(0, weight=1)
        self._ph_new.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        fn = _section(self._ph_new, "New Analysis")
        fn.grid(row=0, column=0, sticky="ew")
        fn.columnconfigure(1, weight=1)
        self.sv_ph_ptu = tk.StringVar()
        self.sv_ph_irf = tk.StringVar()
        _row(fn, "PTU file *",          self.sv_ph_ptu, 0,
             lambda: _browse_file(self.sv_ph_ptu, "PTU file",
                                  [("PTU", "*.ptu"), ("All", "*.*")]))
        _row(fn, "IRF XLSX (optional)", self.sv_ph_irf, 1,
             lambda: _browse_file(self.sv_ph_irf, "IRF XLSX",
                                  [("Excel", "*.xlsx"), ("All", "*.*")]))

        self._ph_sess = ttk.Frame(tab)
        self._ph_sess.columnconfigure(0, weight=1)
        self._ph_sess.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        fs = _section(self._ph_sess, "Resume Session")
        fs.grid(row=0, column=0, sticky="ew")
        fs.columnconfigure(1, weight=1)
        self.sv_ph_session = tk.StringVar()
        _row(fs, "Session file (.npz) *", self.sv_ph_session, 0,
             lambda: _browse_file(self.sv_ph_session, "Session file",
                                  [("NPZ", "*.npz"), ("All", "*.*")]))
        self._ph_sess.grid_remove()

        fo = _section(tab, "Display Options")
        fo.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(fo, text="Min photons (fraction):").grid(
            row=0, column=0, sticky="w", **PAD)
        self.sv_ph_minph = tk.StringVar(value="0.01")
        ttk.Entry(fo, textvariable=self.sv_ph_minph, width=8).grid(
            row=0, column=1, sticky="w", padx=4)
        ttk.Label(fo, text="Max cursors:").grid(row=0, column=2, sticky="w", padx=8)
        self.sv_ph_maxc = tk.StringVar(value="6")
        ttk.Entry(fo, textvariable=self.sv_ph_maxc, width=4).grid(
            row=0, column=3, sticky="w", padx=4)
        ttk.Label(fo,
                  text="The phasor tool opens in its own interactive matplotlib window.",
                  foreground="grey").grid(
            row=1, column=0, columnspan=4, sticky="w", padx=4, pady=(4, 0))

        self._btn_ph = ttk.Button(tab, text="▶  Launch Phasor Tool",
                                  command=self._run_phasor)
        self._btn_ph.grid(row=4, column=0, pady=8, ipadx=20, ipady=4)

    def _ph_mode_changed(self):
        if self.sv_ph_mode.get() == "new":
            self._ph_new.grid()
            self._ph_sess.grid_remove()
        else:
            self._ph_new.grid_remove()
            self._ph_sess.grid()
        self.root.after_idle(self._fit_window_to_screen)

    # ═══════════════════════════════════════════════════════════════════════════
    # Run handlers – flimkit imports happen here, safely after package init
    # ═══════════════════════════════════════════════════════════════════════════

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
        a.out           = self.sv_out_fov.get().strip() or cfg["OUT_NAME"]
        a.no_plots      = False
        a.cell_mask     = self.bv_cell.get()
        a.intensity_threshold = _thresh(self.bv_thr_fov, self.sv_thr_fov)

        out_dir = str(Path(a.out).parent) \
                  if Path(a.out).parent != Path(".") \
                  else str(Path(ptu).parent)
        self._launch(lambda: _run_flim_fit(a), out_dir)

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
            if rows:
                self._res.populate_summary(rows)

        def task(progress_callback, cancel_event):
            if pipeline == "stitch_only":
                return stitch_flim_tiles(
                    xlif_path=a.xlif,
                    ptu_dir=a.ptu_dir,
                    output_dir=a.output_dir,
                    ptu_basename=a.ptu_basename,
                    rotate_tiles=a.rotate_tiles,
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
        self.run_with_progress(task, task_name=task_name, on_done=on_done)

    def _run_phasor(self):
        try:
            min_ph  = float(self.sv_ph_minph.get() or 0.01)
            max_cur = int(self.sv_ph_maxc.get()    or 6)
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Min photons and max cursors must be numeric.")
            return

        from flimkit.phasor_launcher import launch_phasor

        if self.sv_ph_mode.get() == "new":
            ptu = self.sv_ph_ptu.get().strip()
            if not ptu or not Path(ptu).exists():
                messagebox.showerror("Missing input", "Please select a valid PTU file.")
                return
            irf = self.sv_ph_irf.get().strip() or None
            fn  = lambda: launch_phasor(ptu_path=ptu, irf_path=irf,
                                        min_photons=min_ph, max_cursors=max_cur)
        else:
            sess = self.sv_ph_session.get().strip()
            if not sess or not Path(sess).exists():
                messagebox.showerror("Missing input",
                                     "Please select a valid .npz session file.")
                return
            fn = lambda: launch_phasor(session_path=sess,
                                       min_photons=min_ph, max_cursors=max_cur)

        # ── The phasor tool opens its own matplotlib figure via plt.show(),
        #    which requires the main Tk thread.  We call it directly here
        #    (inside a Tk callback) so TkAgg drives it as a normal Toplevel.
        #    The main GUI window stays open; its buttons are re-enabled once
        #    the phasor window is closed.
        self._set_buttons("disabled")
        self._res.set_status("⏳  Phasor tool running…  (close the phasor window to return)")
        self._res._nb.select(0)
        try:
            fn()
            self._res.set_status("✓  Phasor tool closed.")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._res.set_status(f"✗  Phasor error: {exc}")
            messagebox.showerror("Phasor error", str(exc))
        finally:
            self._set_buttons("normal")

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

        self._launch(task_fn, output_dir=out_dir)

    # ═══════════════════════════════════════════════════════════════════════════
    # Thread runner
    # ═══════════════════════════════════════════════════════════════════════════

    def _launch(self, fn, output_dir=None):
        self._buf.clear()
        self._set_buttons("disabled")
        self._res.set_status("⏳  Running...")
        self._res._nb.select(0)

        def _worker():
            try:
                fn()
                captured = "".join(self._buf)
                rows     = _parse_summary(captured)
                self.root.after(0, lambda: self._on_done(rows, output_dir))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self._res.set_status(f"✗  Error: {exc}"))
            finally:
                self.root.after(0, lambda: self._set_buttons("normal"))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_done(self, rows, output_dir):
        self._res.set_status("✓  Finished.")
        if rows:
            self._res.populate_summary(rows)
        if output_dir:
            self._res.load_images(output_dir)

    def _set_buttons(self, state):
        for btn in (self._btn_fov, self._btn_st, self._btn_batch, self._btn_mirf, self._btn_ph):
            btn.configure(state=state)

    def _on_close(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.root.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def launch_gui():
    global GUI_MODE
    GUI_MODE = True  # Disable tqdm progress bars in GUI mode
    
    # Create root window with tkinterdnd2 support if available
    if HAS_DND:
        from tkinterdnd2 import Tk
        root = Tk()
    else:
        root = tk.Tk()
    
    # Set window icon if available
    icon_path = Path(__file__).parent / "flimkit" / "icon.png"
    if icon_path.exists():
        try:
            from PIL import Image, ImageTk
            icon_img = Image.open(str(icon_path))
            # Resize icon to reasonable size for window manager (32x32)
            icon_img.thumbnail((32, 32), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(icon_img)
            root.iconphoto(False, photo)
            # Keep a reference to prevent garbage collection
            root._icon_photo = photo
        except Exception as e:
            print(f"Warning: Could not load icon from {icon_path}: {e}")
    
    # Apply Sun-Valley dark theme if available
    if HAS_TKMT:
        try:
            root = TKMT.ThemedTKinterFrame(root, "sun-valley", mode="dark")
        except Exception:
            # Fallback if theming fails
            style = ttk.Style(root.root if hasattr(root, 'root') else root)
            for theme in ("clam", "alt", "default"):
                if theme in style.theme_names():
                    style.theme_use(theme)
                    break
    else:
        style = ttk.Style(root)
        for theme in ("clam", "alt", "default"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
    
    # Get the root window if using TKMT
    window_root = root.root if hasattr(root, 'root') else root
    
    FLIMKitGUI(window_root)
    window_root.mainloop()


if __name__ == "__main__":
    launch_gui()