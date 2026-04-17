from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # Avoid circular import — only used for type hints.
    from flimkit.UI.gui import FLIMKitApp
    from flimkit.project import ProjectFile, ScanRecord


#  visual constants 

_ICON_FIT_ONLY    = "●"   # fit session only
_ICON_PHASOR_ONLY = "◐"   # phasor session only
_ICON_BOTH        = "◉"   # both fit + phasor
_ICON_NOSESSION   = "○"   # no session yet
_ICON_FOV  = "F"
_ICON_XLIF = "T"       # T for Tiled


class ProjectBrowserPanel:
    """
    Left-sidebar scan browser.

    Shows one row per scan:
        ● R_2   (tiled, session exists)
        ○ FOV1  (single FOV, no session)

    Clicking a row:
      - switches the active mode panel (Single FOV / Tile Stitch)
      - populates every relevant StringVar in FLIMKitApp
      - triggers auto-load of the session .npz if one exists
    """

    def __init__(self, parent: tk.Widget, app: "FLIMKitApp", width: int = 170):
        self._app   = app
        self._width = width

        self._project: Optional["ProjectFile"] = None
        self._stems:   list[str] = []   # parallel to Listbox rows

        #  outer frame (fixed width, no propagation) 
        self.frame = ttk.Frame(parent, width=width)
        self.frame.grid_propagate(False)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)   # listbox expands

        #  header row 
        hdr = ttk.Frame(self.frame)
        hdr.grid(row=0, column=0, sticky="ew", padx=4, pady=(6, 2))
        ttk.Label(hdr, text="Project",
                  font=("TkDefaultFont", 9, "bold")).pack(side="left")

        #  listbox + scrollbar 
        lb_outer = ttk.Frame(self.frame, relief="sunken", borderwidth=1)
        lb_outer.grid(row=1, column=0, sticky="nsew", padx=4, pady=2)
        lb_outer.columnconfigure(0, weight=1)
        lb_outer.rowconfigure(0, weight=1)

        sb = ttk.Scrollbar(lb_outer, orient="vertical")
        sb.grid(row=0, column=1, sticky="ns")

        self._lb = tk.Listbox(
            lb_outer,
            yscrollcommand = sb.set,
            selectmode     = "single",
            activestyle    = "none",
            font           = ("Courier", 9),
            relief         = "flat",
            borderwidth    = 0,
            highlightthickness = 0,
            exportselection = False,
        )
        self._lb.grid(row=0, column=0, sticky="nsew")
        sb.config(command=self._lb.yview)
        self._lb.bind("<<ListboxSelect>>", self._on_select)

        #  status line 
        self._sv_status = tk.StringVar(value="No project open")
        ttk.Label(
            self.frame,
            textvariable = self._sv_status,
            foreground   = "grey",
            font         = ("TkDefaultFont", 7),
            wraplength   = width - 12,
            anchor       = "w",
        ).grid(row=2, column=0, sticky="ew", padx=6, pady=(2, 6))

        # Setup drag and drop for folder
        self._setup_dnd()

    #  public interface 

    def grid(self, **kw):
        self.frame.grid(**kw)

    def on_fit_done(
        self,
        stem: str,
        out_st: Optional[str] = None,
        output_prefix: Optional[str] = None,
        ptu_dir: Optional[str] = None,
    ):
        """
        Call this after a fit completes so the project file is updated and
        the session indicator refreshes.

        Parameters
        ----------
        stem          : scan stem (no extension)
        out_st        : base output dir used by the stitch/tile pipeline
        output_prefix : output prefix used by the FOV pipeline
        ptu_dir       : PTU tile dir (pass if it changed during the run)
        """
        if self._project is None:
            return
        self._project.update_after_fit(
            stem,
            out_st        = out_st,
            output_prefix = output_prefix,
            ptu_dir       = ptu_dir,
        )
        self._project.save()
        self._refresh()

    def on_phasor_done(self, stem: str):
        """Call after a phasor session is auto-saved to refresh indicators."""
        if self._project is None:
            return
        self._project.update_after_phasor(stem)
        self._project.save()
        self._refresh()

    #  private helpers 

    def _setup_dnd(self):
        """Setup drag and drop for the project panel."""
        # tkinterdnd2 has compatibility issues on many systems
        # Users can use File > Open Project Folder... instead
        pass

    def load_folder(self, folder: str):
        """Load project from a given folder path."""
        if not folder:
            return
        from flimkit.project import ProjectFile
        self._project = ProjectFile.load_or_create(Path(folder))
        self._project.save()   # write project.json immediately
        # Apply per-project config overrides
        if self._project.config:
            from flimkit.utils.config_manager import cfg
            cfg.load_project_overrides(self._project.config)
        self._refresh()

    def _open_project(self):
        folder = filedialog.askdirectory(title="Open project folder")
        if not folder:
            return
        self.load_folder(folder)
        if hasattr(self._app, '_add_to_recent'):
            self._app._add_to_recent(folder, "project")

    def _refresh(self):
        """Rebuild the listbox from the current project."""
        if self._project is None:
            return

        self._lb.delete(0, tk.END)
        self._stems.clear()

        for stem, rec in self._project.sorted_scans():
            has_fit = rec.has_session
            has_ph  = rec.has_phasor_session
            if has_fit and has_ph:
                session_dot = _ICON_BOTH
            elif has_fit:
                session_dot = _ICON_FIT_ONLY
            elif has_ph:
                session_dot = _ICON_PHASOR_ONLY
            else:
                session_dot = _ICON_NOSESSION
            type_tag    = _ICON_XLIF if rec.scan_type == "xlif" else _ICON_FOV
            # Format: "◉ F my_fov_name"  or  "○ T R 2"
            label = f"{session_dot} {type_tag} {stem}"
            self._lb.insert(tk.END, label)
            self._stems.append(stem)

        n = len(self._stems)
        n_sess = sum(1 for r in self._project.scans.values()
                     if r.has_session or r.has_phasor_session)
        folder_name = self._project.project_dir.name
        self._sv_status.set(
            f"{folder_name}  |  {n} scan{'s' if n != 1 else ''}  |  {n_sess} saved"
        )

    def _on_select(self, _event=None):
        """Handle a Listbox click: populate the app form and load any session."""
        if getattr(self, '_selecting', False):
            return
        sel = self._lb.curselection()
        if not sel or self._project is None:
            return
        stem = self._stems[sel[0]]
        rec  = self._project.scans.get(stem)
        if rec is None:
            return
        self._selecting = True
        try:
            self._load_scan(rec)
        finally:
            self._selecting = False

    def _load_scan(self, rec: "ScanRecord"):
        app = self._app

        # Cancel any pending after() loads from previous scan selection
        if hasattr(app, '_cancel_pending_scan_loads'):
            app._cancel_pending_scan_loads()

        # Clear ROIs from the previous scan so they don't bleed into the new one
        if hasattr(app, '_fov_preview'):
            app._fov_preview._roi_manager.clear_all()
            app._fov_preview._roi_patches.clear()
            app._fov_preview._redraw_region_overlays()
        for panel in (getattr(app, '_roi_analysis_panel', None),
                      getattr(app, '_stitch_roi_panel', None)):
            if panel is not None:
                panel._refresh_region_list()

        if rec.scan_type == "fov":
            #  Single FOV ──────────────────────────────────
            # Determine which mode the user is in
            current_form = getattr(app, '_current_form', 'fov')
            want_phasor = (current_form == "phasor")

            if want_phasor:
                app._switch_form("phasor")
            else:
                app._switch_form("fov")
            
            # Set guard flag BEFORE sv_ptu.set() to prevent auto-load trace from firing
            # (we'll load the session explicitly below instead)
            if hasattr(app, '_last_loaded_ptu'):
                app._last_loaded_ptu = rec.source_path
            
            # Set PTU path (triggers preview, but auto-load trace will return early)
            app.sv_ptu.set(rec.source_path)

            # Always load the FOV intensity preview from the project panel
            if hasattr(app, '_fov_preview'):
                app._fov_preview.load_fov(rec.source_path)
            # Auto-populate XLSX if a paired file exists
            if rec.xlsx_path and hasattr(app, "sv_xlsx"):
                app.sv_xlsx.set(rec.xlsx_path)
            # If no XLSX, default to Machine IRF; otherwise default to Leica analytical
            if hasattr(app, "_irf_fov"):
                if rec.xlsx_path:
                    app._irf_fov.sv_method.set("irf_xlsx")
                else:
                    app._irf_fov.sv_method.set("machine_irf")
            # Auto-populate output prefix with PTU base name
            if hasattr(app, "sv_out_fov"):
                app.sv_out_fov.set(rec.stem)
            
            if want_phasor:
                # Populate phasor PTU field and auto-restore phasor session
                if hasattr(app, 'sv_ph_ptu'):
                    app.sv_ph_ptu.set(rec.source_path)
                if rec.phasor_session_path and hasattr(app, '_restore_phasor_session'):
                    app._restore_phasor_session(str(rec.phasor_session_path))
            else:
                # Restore fit session if available
                if rec.session_path and hasattr(app, "_load_fitted_data_from_file"):
                    app._load_fitted_data_from_file(str(rec.session_path), suppress_popups=True)

        else:
            #  XLIF / Tile scan 
            app._switch_form("stitch")

            # Set guard flag BEFORE sv_xlif.set() to prevent auto-load trace from firing
            # (we'll load the session explicitly below instead)
            if hasattr(app, '_last_loaded_xlif'):
                app._last_loaded_xlif = rec.source_path

            app.sv_xlif.set(rec.source_path)

            if rec.ptu_dir and hasattr(app, "sv_ptu_dir"):
                app.sv_ptu_dir.set(rec.ptu_dir)

            out_st = self._project.default_out_st(rec.stem) if self._project else ""
            if hasattr(app, "sv_out_st"):
                app.sv_out_st.set(out_st)

            # If a session exists, restore it using the File > Restore NPZ pathway (suppress popups for project tree)
            if rec.session_path and hasattr(app, "_load_fitted_data_from_file"):
                app._load_fitted_data_from_file(str(rec.session_path), suppress_popups=True)
