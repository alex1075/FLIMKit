from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

# phasorpy v0.9 — all signatures verified
from phasorpy.cursor import mask_from_elliptic_cursor, pseudo_color
from phasorpy.lifetime import phasor_to_apparent_lifetime, phasor_semicircle_intersect
from phasorpy.component import phasor_component_fraction
from phasorpy.plot import PhasorPlot


_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']


def _hex_to_rgb01(h: str) -> tuple:
    h = h.lstrip('#')
    return (int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)


def _tau_phi_scalar(g: float, s: float, freq_mhz: float) -> float:
    """Return τ_φ (ns) for a single (G, S) point.

    phasorpy.phasor_to_apparent_lifetime returns a scalar (not array) when
    the inputs are Python floats — avoids the float(1-element-array) TypeError.
    """
    tau_phi, _ = phasor_to_apparent_lifetime(float(g), float(s), freq_mhz)
    return float(tau_phi)


class PhasorViewPanel:
    """Embedded image-top / phasor-bottom interactive panel."""

    def __init__(self, parent, max_cursors: int = 6):
        self.max_cursors = max_cursors
        self.on_change = None   # optional callback(panel) after cursor/param changes


        self._real:  Optional[np.ndarray] = None
        self._imag:  Optional[np.ndarray] = None
        self._mean:  Optional[np.ndarray] = None
        self._disp:  Optional[np.ndarray] = None   # display/intensity image
        self._freq:  float = 80.0                  # MHz
        self._valid: Optional[np.ndarray] = None   # boolean (min-photons)


        self._cursors: list = []                    # {center_g, center_s, color}
        self._cursor_artists: list = []


        self._radius = tk.DoubleVar(value=0.05)
        self._ratio  = tk.DoubleVar(value=0.60)    # radius_minor = ratio × radius

        #  outer frame 
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)        # row 1 = figure, expands

        self._build_controls()
        self._build_figure()

        self._status_var = tk.StringVar(
            value="Load a PTU file to begin phasor analysis.")
        ttk.Label(self.frame, textvariable=self._status_var,
                  foreground="grey", font=("Courier", 8)).grid(
            row=2, column=0, sticky="w", padx=4, pady=(0, 2))


    def _build_controls(self):
        ctrl = ttk.Frame(self.frame)
        ctrl.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))

        ttk.Button(ctrl, text="✕  Clear all",
                   command=self._on_clear).pack(side="left", padx=(0, 4))
        ttk.Button(ctrl, text="↩  Undo",
                   command=self._on_undo).pack(side="left", padx=(0, 8))
        ttk.Button(ctrl, text="💾  Save session",
                   command=self._on_save).pack(side="left", padx=(0, 20))

        ttk.Label(ctrl, text="Radius:").pack(side="left")
        ttk.Scale(ctrl, variable=self._radius, from_=0.01, to=0.15,
                  orient="horizontal", length=90,
                  command=lambda _: self._on_param_change()).pack(
            side="left", padx=(2, 4))
        self._radius_lbl = ttk.Label(ctrl, text="0.050", width=5)
        self._radius_lbl.pack(side="left", padx=(0, 14))

        ttk.Label(ctrl, text="Minor/major:").pack(side="left")
        ttk.Scale(ctrl, variable=self._ratio, from_=0.10, to=1.00,
                  orient="horizontal", length=70,
                  command=lambda _: self._on_param_change()).pack(
            side="left", padx=(2, 4))
        self._ratio_lbl = ttk.Label(ctrl, text="0.60", width=4)
        self._ratio_lbl.pack(side="left")

        self._radius.trace_add("write", lambda *_: self._update_param_labels())
        self._ratio.trace_add("write",  lambda *_: self._update_param_labels())

    def _update_param_labels(self):
        self._radius_lbl.configure(text=f"{self._radius.get():.3f}")
        self._ratio_lbl.configure(text=f"{self._ratio.get():.2f}")

    def _build_figure(self):
        fig_frame = ttk.Frame(self.frame)
        fig_frame.grid(row=1, column=0, sticky="nsew")
        fig_frame.columnconfigure(0, weight=1)
        fig_frame.rowconfigure(0, weight=1)

        self._fig = Figure(figsize=(5, 7), dpi=100, facecolor="#f8f8f8")
        gs = self._fig.add_gridspec(
            2, 1, height_ratios=[1, 1.8],
            hspace=0.38, left=0.10, right=0.95, top=0.95, bottom=0.07)
        self._ax_img = self._fig.add_subplot(gs[0])
        self._ax_ph  = self._fig.add_subplot(gs[1])

        self._draw_placeholder()

        self._canvas = FigureCanvasTkAgg(self._fig, master=fig_frame)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        tb_frame = ttk.Frame(fig_frame)
        tb_frame.grid(row=1, column=0, sticky="ew")
        self._toolbar = NavigationToolbar2Tk(self._canvas, tb_frame,
                                              pack_toolbar=True)
        self._toolbar.update()

        self._cid = self._canvas.mpl_connect("button_press_event",
                                               self._on_click)

    def _draw_placeholder(self):
        for ax in (self._ax_img, self._ax_ph):
            ax.set_facecolor("#e8e8e8")
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
        self._ax_img.set_title(
            "FOV image  (pseudo-colour overlay once cursors are placed)",
            fontsize=9, color="#999999")
        self._ax_ph.set_title(
            "Phasor plot  —  load a file to begin",
            fontsize=9, color="#999999")

    def set_data(self,
                 real_cal: np.ndarray,
                 imag_cal: np.ndarray,
                 mean: np.ndarray,
                 frequency: float,
                 display_image: Optional[np.ndarray] = None,
                 min_photons: float = 0.01):
        """Load phasor arrays and render the phasor histogram.
        Must be called from the Tk main thread.
        """
        self._real  = np.asarray(real_cal,      dtype=float).squeeze()
        self._imag  = np.asarray(imag_cal,      dtype=float).squeeze()
        self._mean  = np.asarray(mean,          dtype=float).squeeze()
        self._disp  = (np.asarray(display_image, dtype=float).squeeze()
                       if display_image is not None else self._mean.copy())
        self._freq  = float(frequency)
        self._valid = (self._mean >= min_photons) & ~np.isnan(self._real)

        self._cursors.clear()
        self._cursor_artists.clear()

        self._redraw_phasor()
        self._redraw_image(masks=None)
        self._canvas.draw_idle()

        n_valid = int(self._valid.sum())
        self._status_var.set(
            f"✓ {n_valid} valid pixels  |  {frequency:.1f} MHz  "
            f"|  click phasor to place cursor")

    def load_session(self, session: dict, min_photons: float = 0.01):
        """Restore a previously saved session dict (from phasor_launcher)."""
        self.set_data(
            session['real_cal'], session['imag_cal'],
            session['mean'], session['frequency'],
            display_image=session.get('display_image'),
            min_photons=min_photons,
        )
        for c in session.get('cursors', []):
            self._cursors.append(dict(
                center_g=float(c['center_g']),
                center_s=float(c['center_s']),
                color=c['color']))
        p = session.get('params', {})
        if 'radius' in p:
            self._radius.set(float(p['radius']))
        if 'radius_minor' in p and p.get('radius', 0) > 0:
            self._ratio.set(float(p['radius_minor']) / float(p['radius']))
        if self._cursors:
            self._redraw_cursors()
            self._analyse()

    def get_session_dict(self) -> dict:
        """Return a dict compatible with phasor_launcher.save_session."""
        r  = self._radius.get()
        rm = r * self._ratio.get()
        return dict(
            real_cal=self._real,
            imag_cal=self._imag,
            mean=self._mean,
            frequency=self._freq,
            display_image=self._disp,
            cursors=[dict(center_g=c['center_g'],
                          center_s=c['center_s'],
                          color=c['color'])
                     for c in self._cursors],
            params=dict(radius=r, radius_minor=rm, angle_mode='semicircle'),
        )

    def _redraw_phasor(self):
        """Re-draw the phasor histogram background (no cursors)."""
        self._ax_ph.cla()
        if self._real is None:
            return
        g = self._real[self._valid]
        s = self._imag[self._valid]
        # PhasorPlot sets up the semicircle, grid and axis limits in our axes
        pp = PhasorPlot(ax=self._ax_ph, frequency=self._freq)
        pp.hist2d(g, s, cmap='inferno', bins=256)
        self._ax_ph.set_title(
            f"Phasor  ({self._freq:.1f} MHz)  —  click to place cursor",
            fontsize=9)

    def _redraw_cursors(self):
        """Remove old cursor artists and re-draw all current cursors."""
        for art in self._cursor_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._cursor_artists.clear()

        if not self._cursors:
            return

        r     = self._radius.get()
        r_min = r * self._ratio.get()
        ax    = self._ax_ph

        for i, cur in enumerate(self._cursors):
            cg, cs, col = cur['center_g'], cur['center_s'], cur['color']
            ang = float(np.degrees(np.arctan2(cs, cg - 0.5) + np.pi / 2.0))

            ell = Ellipse((cg, cs), 2 * r, 2 * r_min,
                          angle=ang, facecolor=col, alpha=0.18,
                          edgecolor=col, linewidth=2,
                          linestyle='--', zorder=8)
            ax.add_patch(ell)
            self._cursor_artists.append(ell)

            (dot,) = ax.plot(cg, cs, 'o', color=col, ms=7, zorder=10)
            self._cursor_artists.append(dot)

            txt = ax.text(cg + 0.015, cs + 0.015, f'C{i+1}',
                          color=col, fontsize=9, fontweight='bold', zorder=11)
            self._cursor_artists.append(txt)

        # Semicircle-intersection tie-line for C1 + C2
        if len(self._cursors) >= 2:
            g0 = self._cursors[0]['center_g']
            s0 = self._cursors[0]['center_s']
            g1 = self._cursors[1]['center_g']
            s1 = self._cursors[1]['center_s']
            gi0, si0, gi1, si1 = phasor_semicircle_intersect(g0, s0, g1, s1)
            if not np.isnan(float(gi0)):
                (ln,)  = ax.plot([float(gi0), float(gi1)],
                                 [float(si0), float(si1)],
                                 'w-', lw=1.5, alpha=0.7, zorder=7)
                (pt1,) = ax.plot(float(gi0), float(si0),
                                 'g*', ms=11, zorder=12)
                (pt2,) = ax.plot(float(gi1), float(si1),
                                 'm*', ms=11, zorder=12)
                self._cursor_artists.extend([ln, pt1, pt2])

    def _redraw_image(self, masks: Optional[np.ndarray]):
        """Update the intensity/overlay image panel."""
        self._ax_img.cla()
        if self._disp is None:
            return

        if masks is not None and len(masks) > 0 and self._cursors:
            colors_rgb = np.array([_hex_to_rgb01(c['color'])
                                   for c in self._cursors])
            mask_list  = [masks[i] for i in range(len(self._cursors))]
            pc = pseudo_color(*mask_list,
                              intensity=self._disp,
                              colors=colors_rgb)
            self._ax_img.imshow(pc, origin='upper', interpolation='nearest')
            counts = ', '.join(
                f'C{i+1}:{int(masks[i].sum())}px'
                for i in range(len(self._cursors)))
            title = f"FOV overlay  ({counts})"
        else:
            self._ax_img.imshow(self._disp, cmap='inferno',
                                origin='upper', interpolation='nearest')
            title = "FOV image  (place cursors on phasor to colourize)"

        self._ax_img.set_title(title, fontsize=9)
        self._ax_img.set_xlabel("X (px)", fontsize=8)
        self._ax_img.set_ylabel("Y (px)", fontsize=8)
        self._ax_img.tick_params(labelsize=7)

    def _analyse(self):
        """Compute elliptic masks, update image overlay, print per-cursor stats."""
        if self._real is None or not self._cursors:
            self._redraw_image(None)
            self._canvas.draw_idle()
            return

        r     = self._radius.get()
        r_min = r * self._ratio.get()
        n     = len(self._cursors)

        cg = np.array([c['center_g'] for c in self._cursors])
        cs = np.array([c['center_s'] for c in self._cursors])

        masks = mask_from_elliptic_cursor(
            self._real, self._imag, cg, cs,
            radius=r, radius_minor=r_min, angle='semicircle')

        # Normalise: single cursor returns (Y, X), multiple returns (n, Y, X)
        if masks.ndim == self._real.ndim:
            masks = masks[np.newaxis]

        masks = masks & self._valid[np.newaxis]

        self._redraw_image(masks)

        #Per-cursor τ_φ stats 
        print(f"\n{'─' * 50}")
        for ci in range(n):
            m    = masks[ci]
            n_px = int(m.sum())
            if n_px == 0:
                print(f"  C{ci+1}: no pixels selected")
                continue
            tau_phi, _ = phasor_to_apparent_lifetime(
                self._real[m], self._imag[m], self._freq)
            med = float(np.nanmedian(tau_phi))
            lo  = float(np.nanpercentile(tau_phi, 5))
            hi  = float(np.nanpercentile(tau_phi, 95))
            print(f"  C{ci+1} ({self._cursors[ci]['color']}):  "
                  f"{n_px} px  |  τ_φ = {lo:.2f}–{hi:.2f} ns  "
                  f"(median {med:.2f} ns)")

        # Two-component decomposition C1 ↔ C2
        if n >= 2:
            g0 = self._cursors[0]['center_g']
            s0 = self._cursors[0]['center_s']
            g1 = self._cursors[1]['center_g']
            s1 = self._cursors[1]['center_s']
            gi0, si0, gi1, si1 = phasor_semicircle_intersect(g0, s0, g1, s1)

            if not np.isnan(float(gi0)):
                # Pass Python floats → phasor_to_apparent_lifetime returns scalars
                tau1 = _tau_phi_scalar(gi0, si0, self._freq)
                tau2 = _tau_phi_scalar(gi1, si1, self._freq)

                combined = np.any(masks, axis=0)
                frac = phasor_component_fraction(
                    self._real[combined],
                    self._imag[combined],
                    np.array([float(gi0), float(gi1)]),
                    np.array([float(si0), float(si1)]),
                )
                print(f"\n  ↳ 2-component (C1↔C2):")
                print(f"     τ₁ = {tau1:.3f} ns  "
                      f"τ₂ = {tau2:.3f} ns  "
                      f"mean frac(C1) = {float(np.mean(frac)):.3f}")

        print(f"{'─' * 50}")
        self._canvas.draw_idle()

    def _on_click(self, event):
        if self._real is None:
            return
        if event.inaxes is not self._ax_ph:
            return
        if event.button != 1:
            return
        if self._toolbar.mode != '':        # zoom/pan tool active — don't place cursor
            return
        if event.xdata is None or event.ydata is None:
            return
        if len(self._cursors) >= self.max_cursors:
            self._status_var.set(
                f"Max {self.max_cursors} cursors — clear or undo first.")
            return

        idx = len(self._cursors) % len(_COLORS)
        self._cursors.append(dict(
            center_g=event.xdata,
            center_s=event.ydata,
            color=_COLORS[idx]))

        self._redraw_cursors()
        self._analyse()
        self._notify_change()
        self._status_var.set(
            f"{len(self._cursors)} cursor(s)  |  "
            f"G={event.xdata:.4f}  S={event.ydata:.4f}")

    def _on_clear(self):
        if self._real is None:
            return
        self._cursors.clear()
        self._redraw_phasor()          # also clears cursor artists
        self._redraw_image(masks=None)
        self._canvas.draw_idle()
        self._notify_change()
        self._status_var.set("Cleared.  Click phasor to place cursors.")

    def _on_undo(self):
        if self._real is None or not self._cursors:
            return
        self._cursors.pop()
        self._redraw_phasor()
        self._redraw_cursors()
        self._analyse()
        self._notify_change()
        self._status_var.set(
            f"{len(self._cursors)} cursor(s)  |  last cursor removed.")

    def _notify_change(self):
        """Fire the optional on_change callback after cursor/param edits."""
        if self.on_change is not None:
            try:
                self.on_change(self)
            except Exception as e:
                print(f"[PhasorViewPanel] on_change callback error: {e}")

    def _on_param_change(self):
        """Radius / ratio slider changed — recompute immediately if cursors exist."""
        if self._cursors and self._real is not None:
            self._redraw_phasor()
            self._redraw_cursors()
            self._analyse()
            self._notify_change()

    def _on_save(self):
        if self._real is None:
            messagebox.showwarning("No data", "No phasor data loaded yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save phasor session",
            defaultextension=".npz",
            initialfile="phasor_session.npz",
            filetypes=[("NumPy archive", "*.npz"), ("All files", "*")])
        if not path:
            return
        try:
            from flimkit.phasor_launcher import save_session
            sd = self.get_session_dict()
            save_session(
                path,
                real_cal=sd['real_cal'],
                imag_cal=sd['imag_cal'],
                mean=sd['mean'],
                frequency=sd['frequency'],
                cursors=sd['cursors'],
                params=sd['params'],
                display_image=sd.get('display_image'),
            )
            self._status_var.set(f"✓ Session saved → {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
