"""Interactive elliptical-cursor tool for phasor plots.

Place multiple elliptical cursors on a calibrated phasor plot to select
regions of interest, compute apparent lifetimes, and perform two-component
decomposition using phasorpy built-ins.

Works in **both** Jupyter notebooks (ipywidgets) and standalone scripts
(matplotlib.widgets).  The environment is detected automatically.

Usage
-----
Notebook::

    from pyflim.phasor.interactive import phasor_cursor_tool
    state = phasor_cursor_tool(real_cal, imag_cal, mean, frequency)

Script::

    from pyflim.phasor.interactive import phasor_cursor_tool
    state = phasor_cursor_tool(real_cal, imag_cal, mean, frequency)
    # blocks until the window is closed
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from phasorpy.plot import PhasorPlot
from phasorpy.cursor import mask_from_elliptic_cursor, pseudo_color
from phasorpy.lifetime import (
    phasor_to_apparent_lifetime,
    phasor_semicircle_intersect,
)
from phasorpy.component import phasor_component_fraction

# Default palette for up to 6 cursors
_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']


def _in_notebook() -> bool:
    """Return True when running inside a Jupyter/IPython notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False


def _ellipse_angle_rad(cg: float, cs: float, mode: str) -> float:
    """Return the rotation angle (radians) that phasorpy uses for *mode*."""
    if mode == 'semicircle':
        return np.arctan2(cs, cg - 0.5) + np.pi / 2.0
    return np.arctan2(cs, cg)


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def phasor_cursor_tool(
    real_cal: np.ndarray,
    imag_cal: np.ndarray,
    mean: np.ndarray,
    frequency: float,
    *,
    min_photons: float = 0.01,
    max_cursors: int = 6,
    figsize: tuple[float, float] = (8, 5),
    initial_cursors: list[dict] | None = None,
    initial_params: dict | None = None,
    on_save: callable | None = None,
) -> dict:
    """Launch an interactive phasor cursor selection widget.

    Click on the phasor plot to place elliptical cursors.  The first two
    cursors additionally define a two-component decomposition line through
    the universal semicircle.

    Works in Jupyter notebooks (uses ipywidgets for sliders / buttons) and
    in plain Python scripts (uses matplotlib.widgets; call ``plt.show()``
    afterwards or let this function block).

    Parameters
    ----------
    real_cal : ndarray
        Calibrated phasor real (G) component, shape ``(…, Y, X)``.
    imag_cal : ndarray
        Calibrated phasor imaginary (S) component, same shape.
    mean : ndarray
        Mean intensity image, same spatial shape.
    frequency : float
        Laser-repetition / modulation frequency in **MHz**.
    min_photons : float, optional
        Minimum mean-intensity to consider a pixel valid (default 0.01).
    max_cursors : int, optional
        Maximum number of cursors (default 6).
    figsize : tuple, optional
        Size of the phasor figure.
    initial_cursors : list of dict, optional
        Pre-loaded cursors, each ``{'center_g', 'center_s', 'color'}``.
        Used to restore a previously saved session.
    initial_params : dict, optional
        Pre-loaded ellipse parameters ``{'radius', 'radius_minor',
        'angle_mode'}``.  Merged with defaults.
    on_save : callable, optional
        Callback ``on_save(state, params)`` invoked when the user clicks
        the *Save* button.  If *None*, no save button is shown.

    Returns
    -------
    state : dict
        Mutable dictionary that is updated live.  Keys include:

        * ``'cursors'`` – list of ``{'center_g', 'center_s', 'color'}``
        * ``'masks'`` – ndarray of boolean masks ``(n_cursors, Y, X)``
        * ``'fig'`` / ``'ax'`` – matplotlib figure and axes
    """
    notebook = _in_notebook()

    # ── Prepare data ─────────────────────────────────────────
    rc = np.asarray(real_cal).squeeze().astype(float)
    ic = np.asarray(imag_cal).squeeze().astype(float)
    mn = np.asarray(mean).squeeze().astype(float)
    valid = (mn >= min_photons) & ~np.isnan(rc)
    g_all = rc[valid]
    s_all = ic[valid]

    # ── Mutable state returned to caller ─────────────────────
    state: dict = {
        'cursors': [],
        'masks': None,
        'fig': None,
        'ax': None,
    }
    cursor_artists: list = []

    # ── Parameter store (values read by helpers) ─────────────
    params = dict(radius=0.05, radius_minor=0.03, angle_mode='semicircle')
    if initial_params:
        params.update({k: v for k, v in initial_params.items()
                       if k in params})

    # ── Output helper (notebook vs stdout) ───────────────────
    if notebook:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        results_out = widgets.Output()

        def _begin_output():
            results_out.__enter__()
            clear_output(wait=True)

        def _end_output():
            results_out.__exit__(None, None, None)
    else:
        def _begin_output():
            pass

        def _end_output():
            pass

    # ── Drawing helpers ──────────────────────────────────────
    def _redraw():
        ax = state['ax']
        for art in cursor_artists:
            try:
                art.remove()
            except Exception:
                pass
        cursor_artists.clear()

        radius = params['radius']
        r_minor = params['radius_minor']
        mode = params['angle_mode']
        cursors = state['cursors']

        for i, cur in enumerate(cursors):
            cg, cs, col = cur['center_g'], cur['center_s'], cur['color']
            ang_deg = np.degrees(_ellipse_angle_rad(cg, cs, mode))

            ell = Ellipse(
                (cg, cs), 2 * radius, 2 * r_minor,
                angle=ang_deg, facecolor=col, alpha=0.20,
                edgecolor=col, linewidth=2, linestyle='--', zorder=8)
            ax.add_patch(ell)
            cursor_artists.append(ell)

            (dot,) = ax.plot(cg, cs, 'o', color=col, ms=8, zorder=10)
            cursor_artists.append(dot)
            txt = ax.text(cg + 0.015, cs + 0.015, f'C{i+1}',
                          color=col, fontsize=10, fontweight='bold',
                          zorder=11)
            cursor_artists.append(txt)

        # Semicircle-intersection line for cursors 1 & 2
        if len(cursors) >= 2:
            g0, s0 = cursors[0]['center_g'], cursors[0]['center_s']
            g1, s1 = cursors[1]['center_g'], cursors[1]['center_s']
            gi0, si0, gi1, si1 = phasor_semicircle_intersect(
                g0, s0, g1, s1)
            if not np.isnan(gi0):
                (ln,) = ax.plot(
                    [float(gi0), float(gi1)],
                    [float(si0), float(si1)],
                    'r-', lw=1.5, alpha=0.6, zorder=7)
                cursor_artists.append(ln)
                (st1,) = ax.plot(float(gi0), float(si0),
                                 'g*', ms=12, zorder=12)
                (st2,) = ax.plot(float(gi1), float(si1),
                                 'm*', ms=12, zorder=12)
                cursor_artists.extend([st1, st2])

        ax.figure.canvas.draw_idle()

    # ── Analysis ─────────────────────────────────────────────
    def _analyse():
        cursors = state['cursors']
        if not cursors:
            state['masks'] = None
            _begin_output()
            print("Click on the phasor plot to place elliptical cursors.")
            _end_output()
            return

        radius = params['radius']
        r_minor = params['radius_minor']
        mode = params['angle_mode']
        n_cursors = len(cursors)

        centers_g = np.array([c['center_g'] for c in cursors])
        centers_s = np.array([c['center_s'] for c in cursors])

        masks = mask_from_elliptic_cursor(
            rc, ic, centers_g, centers_s,
            radius=radius, radius_minor=r_minor, angle=mode,
        )
        if masks.ndim == rc.ndim:
            masks = masks[np.newaxis]
        masks = masks & valid[np.newaxis]
        state['masks'] = masks

        _begin_output()

        has_decomp = n_cursors >= 2
        n_cols = 1 + n_cursors + (1 if has_decomp else 0)
        fig2, axes2 = plt.subplots(
            1, n_cols, figsize=(4.5 * n_cols, 4.5))
        if n_cols == 1:
            axes2 = [axes2]

        # Panel 0 – pseudo-colour overlay
        ax_pc = axes2[0]
        mask_list = [masks[i] for i in range(n_cursors)]
        colors_rgb = np.array([
            [int(c['color'][1:3], 16) / 255,
             int(c['color'][3:5], 16) / 255,
             int(c['color'][5:7], 16) / 255]
            for c in cursors
        ])
        pc_img = pseudo_color(
            *mask_list, intensity=mn, colors=colors_rgb)
        ax_pc.imshow(pc_img, origin='upper')
        ax_pc.set_title('Pseudo-color overlay')
        for i, c in enumerate(cursors):
            ax_pc.text(
                5, 15 + i * 15,
                f'C{i+1}: {int(masks[i].sum())} px',
                color=c['color'], fontsize=9, fontweight='bold')

        # Per-cursor τ_φ maps
        for ci in range(n_cursors):
            ax_cur = axes2[1 + ci]
            mask_i = masks[ci]
            n_px = int(mask_i.sum())
            if n_px == 0:
                ax_cur.set_title(f'C{ci+1}: no pixels')
                continue

            g_sel = rc[mask_i]
            s_sel = ic[mask_i]
            tau_phi, _ = phasor_to_apparent_lifetime(
                g_sel, s_sel, frequency)

            fmap = np.full_like(rc, np.nan)
            fmap[mask_i] = tau_phi
            vlo = np.nanpercentile(tau_phi, 2)
            vhi = np.nanpercentile(tau_phi, 98)
            im = ax_cur.imshow(
                fmap, cmap='turbo', vmin=vlo, vmax=vhi,
                origin='upper')
            plt.colorbar(im, ax=ax_cur, label='τ_φ (ns)')

            med = np.nanmedian(tau_phi)
            ax_cur.set_title(
                f'C{ci+1}: τ_φ  (n={n_px}, med={med:.2f} ns)')
            print(
                f"C{ci+1}: {n_px} px  "
                f"τ_φ = {np.nanmin(tau_phi):.2f}"
                f"–{np.nanmax(tau_phi):.2f} ns "
                f"(median {med:.2f})")

        # Two-component decomposition (C1 ↔ C2)
        if has_decomp:
            ax_dec = axes2[-1]
            g0, s0 = centers_g[0], centers_s[0]
            g1, s1 = centers_g[1], centers_s[1]
            gi0, si0, gi1, si1 = phasor_semicircle_intersect(
                g0, s0, g1, s1)

            if not (np.isnan(gi0) or np.isnan(gi1)):
                tau1 = float(phasor_to_apparent_lifetime(
                    gi0, si0, frequency)[0])
                tau2 = float(phasor_to_apparent_lifetime(
                    gi1, si1, frequency)[0])

                combined = np.any(masks, axis=0)
                g_comb = rc[combined]
                s_comb = ic[combined]

                frac = phasor_component_fraction(
                    g_comb, s_comb,
                    [float(gi0), float(gi1)],
                    [float(si0), float(si1)],
                )

                theta = np.linspace(0, np.pi, 200)
                ax_dec.plot(
                    0.5 + 0.5 * np.cos(theta),
                    0.5 * np.sin(theta),
                    'k-', lw=1, alpha=0.6)
                sc = ax_dec.scatter(
                    g_comb, s_comb, c=frac, s=2,
                    cmap='RdYlBu_r', vmin=0, vmax=1,
                    alpha=0.5, zorder=5)
                plt.colorbar(sc, ax=ax_dec, label='fraction C1')
                ax_dec.plot(
                    [float(gi0), float(gi1)],
                    [float(si0), float(si1)],
                    'r-', lw=2, zorder=10)
                ax_dec.plot(
                    float(gi0), float(si0), 'g*', ms=14,
                    zorder=15, label=f'τ₁={tau1:.2f} ns')
                ax_dec.plot(
                    float(gi1), float(si1), 'm*', ms=14,
                    zorder=15, label=f'τ₂={tau2:.2f} ns')
                ax_dec.set_xlabel('G')
                ax_dec.set_ylabel('S')
                ax_dec.set_xlim(0, 1.05)
                ax_dec.set_ylim(-0.02, 0.55)
                ax_dec.set_aspect('equal')
                ax_dec.legend(fontsize=9, loc='upper left')
                ax_dec.set_title('Two-component (C1↔C2)')

                print(
                    f"\n═══ Two-component decomposition (C1–C2) ═══")
                print(
                    f"  τ₁ = {tau1:.3f} ns  "
                    f"(G={float(gi0):.4f}, S={float(si0):.4f})")
                print(
                    f"  τ₂ = {tau2:.3f} ns  "
                    f"(G={float(gi1):.4f}, S={float(si1):.4f})")
                print(
                    f"  Mean fraction(τ₁) = "
                    f"{float(np.mean(frac)):.3f}")
            else:
                ax_dec.text(
                    0.5, 0.5,
                    'C1↔C2 line does not\nintersect semicircle',
                    ha='center', va='center',
                    transform=ax_dec.transAxes)
                ax_dec.set_title('Two-component (C1↔C2)')

        fig2.tight_layout()
        plt.show()
        _end_output()

    # ── Event handlers ───────────────────────────────────────
    def _on_click(event):
        if event.inaxes != state['ax'] or event.button != 1:
            return
        if event.xdata is None:
            return
        cursors = state['cursors']
        if len(cursors) >= max_cursors:
            return
        idx = len(cursors) % len(_COLORS)
        cursors.append(dict(
            center_g=event.xdata,
            center_s=event.ydata,
            color=_COLORS[idx]))
        _redraw()
        _analyse()

    def _on_clear(_ignored=None):
        state['cursors'].clear()
        state['masks'] = None
        _redraw()
        _begin_output()
        print("Cleared. Click on the phasor plot to place cursors.")
        _end_output()

    def _on_undo(_ignored=None):
        cursors = state['cursors']
        if cursors:
            cursors.pop()
            _redraw()
            if cursors:
                _analyse()
            else:
                state['masks'] = None
                _begin_output()
                print("All cursors removed. Click to place new ones.")
                _end_output()

    def _on_param_change(_ignored=None):
        _redraw()
        if state['cursors']:
            _analyse()

    # ══════════════════════════════════════════════════════════
    # Build the phasor plot
    # ══════════════════════════════════════════════════════════
    if notebook:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        # Leave room at the bottom for matplotlib sliders / buttons
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] + 1.8))
        fig.subplots_adjust(bottom=0.30)

    try:
        fig.canvas.toolbar_visible = True
        fig.canvas.header_visible = False
    except Exception:
        pass

    pp = PhasorPlot(
        frequency=frequency, ax=ax,
        title=f'Click to place elliptical cursors (up to {max_cursors})')
    pp.hist2d(g_all, s_all)

    state['fig'] = fig
    state['ax'] = ax

    fig.canvas.mpl_connect('button_press_event', _on_click)

    # ── Restore initial cursors if provided ──────────────────
    if initial_cursors:
        for cur in initial_cursors:
            state['cursors'].append(dict(
                center_g=cur['center_g'],
                center_s=cur['center_s'],
                color=cur.get('color', _COLORS[
                    len(state['cursors']) % len(_COLORS)])))
        _redraw()
        _analyse()

    # ── Save helper ──────────────────────────────────────────
    def _on_save(_ignored=None):
        if on_save is not None:
            on_save(state, params)

    # ── Export figure as image ───────────────────────────────
    def _on_export(_ignored=None):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            root.update()
            path = filedialog.asksaveasfilename(
                title='Export phasor figure',
                defaultextension='.png',
                initialfile='phasor_plot.png',
                filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'),
                           ('SVG', '*.svg'), ('All files', '*')])
            root.destroy()
        except Exception:
            path = input("Save image path [phasor_plot.png]: ").strip() or 'phasor_plot.png'
        if path:
            state['fig'].savefig(path, dpi=300, bbox_inches='tight')
            print(f"✅  Figure exported → {path}")

    # ── Peak detection on the live phasor ────────────────────
    _peak_artists: list = []          # track markers for toggle/clear

    def _on_peaks(_ignored=None):
        from .peaks import find_phasor_peaks, print_peaks, _annotate_peaks
        # Remove previous peak annotations if any
        for art in _peak_artists:
            try:
                art.remove()
            except Exception:
                pass
        _peak_artists.clear()

        _begin_output()
        peaks = find_phasor_peaks(rc, ic, mn, frequency,
                                  min_photons=min_photons)
        state['peaks'] = peaks
        print_peaks(peaks)
        arts = _annotate_peaks(
            state['ax'], peaks['n_peaks'],
            peaks['peak_g'], peaks['peak_s'], peaks['tau_phase'])
        _peak_artists.extend(arts)
        state['ax'].figure.canvas.draw_idle()
        _end_output()

    # ── Wire up controls ─────────────────────────────────────
    if notebook:
        import ipywidgets as widgets  # noqa: F811
        from IPython.display import display  # noqa: F811

        w_radius = widgets.FloatSlider(
            value=params['radius'], min=0.01, max=0.25, step=0.005,
            description='Major radius:', continuous_update=False,
            style={'description_width': 'initial'})
        w_radius_minor = widgets.FloatSlider(
            value=params['radius_minor'], min=0.005, max=0.15, step=0.005,
            description='Minor radius:', continuous_update=False,
            style={'description_width': 'initial'})
        w_angle_mode = widgets.Dropdown(
            options=['semicircle', 'phase'],
            value=params['angle_mode'], description='Alignment:',
            style={'description_width': 'initial'})
        btn_clear = widgets.Button(description='Clear all',
                                   button_style='warning')
        btn_undo = widgets.Button(description='Undo last', button_style='')
        btn_save = widgets.Button(description='💾 Save',
                                  button_style='success') if on_save else None
        btn_export = widgets.Button(description='📷 Export',
                                    button_style='info')
        btn_peaks = widgets.Button(description='🔍 Peaks',
                                   button_style='')

        def _ipyw_radius(change):
            params['radius'] = change['new']
            _on_param_change()

        def _ipyw_radius_minor(change):
            params['radius_minor'] = change['new']
            _on_param_change()

        def _ipyw_angle(change):
            params['angle_mode'] = change['new']
            _on_param_change()

        w_radius.observe(_ipyw_radius, names='value')
        w_radius_minor.observe(_ipyw_radius_minor, names='value')
        w_angle_mode.observe(_ipyw_angle, names='value')
        btn_clear.on_click(_on_clear)
        btn_undo.on_click(_on_undo)
        btn_export.on_click(_on_export)
        btn_peaks.on_click(_on_peaks)
        if btn_save:
            btn_save.on_click(_on_save)

        btn_row = [btn_undo, btn_clear, btn_peaks, btn_export]
        if btn_save:
            btn_row.append(btn_save)
        controls = widgets.HBox([
            widgets.VBox([w_radius, w_radius_minor]),
            widgets.VBox([w_angle_mode,
                          widgets.HBox(btn_row)]),
        ])
        display(controls)
        display(results_out)

    else:
        # ── matplotlib.widgets for standalone scripts ────────
        from matplotlib.widgets import Slider, Button, RadioButtons

        ax_rad = fig.add_axes([0.15, 0.17, 0.55, 0.03])
        ax_rmin = fig.add_axes([0.15, 0.12, 0.55, 0.03])
        sl_radius = Slider(ax_rad, 'Major R', 0.01, 0.25,
                           valinit=params['radius'], valstep=0.005)
        sl_radius_minor = Slider(ax_rmin, 'Minor R', 0.005, 0.15,
                                 valinit=params['radius_minor'],
                                 valstep=0.005)

        ax_angle = fig.add_axes([0.15, 0.02, 0.15, 0.08])
        rb_angle = RadioButtons(ax_angle, ['semicircle', 'phase'],
                                active=0)

        ax_undo  = fig.add_axes([0.35, 0.02, 0.10, 0.05])
        ax_clr   = fig.add_axes([0.46, 0.02, 0.10, 0.05])
        ax_peaks = fig.add_axes([0.57, 0.02, 0.10, 0.05])
        ax_exp   = fig.add_axes([0.68, 0.02, 0.10, 0.05])
        mpl_btn_undo   = Button(ax_undo,  'Undo')
        mpl_btn_clear  = Button(ax_clr,   'Clear')
        mpl_btn_peaks  = Button(ax_peaks, 'Peaks')
        mpl_btn_export = Button(ax_exp,   'Export')
        if on_save:
            ax_save = fig.add_axes([0.79, 0.02, 0.10, 0.05])
            mpl_btn_save = Button(ax_save, 'Save')
        else:
            mpl_btn_save = None

        def _mpl_radius(val):
            params['radius'] = val
            _on_param_change()

        def _mpl_rminor(val):
            params['radius_minor'] = val
            _on_param_change()

        def _mpl_angle(label):
            params['angle_mode'] = label
            _on_param_change()

        sl_radius.on_changed(_mpl_radius)
        sl_radius_minor.on_changed(_mpl_rminor)
        rb_angle.on_clicked(_mpl_angle)
        mpl_btn_undo.on_clicked(lambda _: _on_undo())
        mpl_btn_clear.on_clicked(lambda _: _on_clear())
        mpl_btn_peaks.on_clicked(lambda _: _on_peaks())
        mpl_btn_export.on_clicked(lambda _: _on_export())
        if mpl_btn_save:
            mpl_btn_save.on_clicked(lambda _: _on_save())

        # Keep references alive so callbacks aren't GC'd
        state['_mpl_widgets'] = (sl_radius, sl_radius_minor, rb_angle,
                                 mpl_btn_undo, mpl_btn_clear,
                                 mpl_btn_peaks, mpl_btn_export,
                                 mpl_btn_save)

    print(f"Plotting {valid.sum()} valid pixels")
    print("ℹ  Click on the phasor to place elliptical cursors.  "
          "First two define the decomposition line.")

    if not notebook:
        plt.show()

    return state
