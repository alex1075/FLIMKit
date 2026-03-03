import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pyflim.configs import FLIM_CMAP


def plot_summed(decay, summary, ptu, xlsx, n_exp, strategy, out_prefix,
               irf_prompt=None):
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                          "axes.spines.top": False, "axes.spines.right": False})
    s    = summary
    t_ns = np.arange(ptu.n_bins) * ptu.tcspc_res * 1e9
    fs, fe = s["fit_window_bins"]

    fig = plt.figure(figsize=(12, 7))
    gs  = gridspec.GridSpec(2, 3, height_ratios=[3, 1], hspace=0.08, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[1, :2], sharex=ax1)
    ax3 = fig.add_subplot(gs[0,  2])
    ax4 = fig.add_subplot(gs[1,  2])
    ax4.axis("off")

    ax1.semilogy(t_ns, np.clip(decay, 1, None), ".", color="#aaa",
                 ms=2, rasterized=True, label="PTU data")

    # IRF — scale to ~10% of decay peak for visibility on log axis.
    # Mask bins below 0.1% of IRF peak so zeros don't pollute the log axis.
    if irf_prompt is not None:
        scale      = decay.max() * 0.1 / irf_prompt.max()
        irf_scaled = irf_prompt * scale
        irf_mask   = irf_scaled > irf_scaled.max() * 1e-3   # only plot non-negligible
        t_irf_plot = t_ns[irf_mask]
        v_irf_plot = irf_scaled[irf_mask]
        ax1.semilogy(t_irf_plot, v_irf_plot,
                     color="#f4a261", lw=1.5, ls="-.", alpha=0.85,
                     label=f"IRF (×{scale:.1e})")

    if xlsx is not None and xlsx.get('fit_t') is not None:
        ax1.semilogy(xlsx["fit_t"], np.clip(xlsx["fit_c"], 1, None),
                     "b-", lw=1.1, alpha=0.55, label="LAS X fit")
    ax1.semilogy(t_ns, np.clip(s["model"], 1, None), "r-", lw=2,
                 label=f"{n_exp}-exp reconv.")
    ax1.set_xlim(0, min(t_ns[-1], 22))
    ax1.set_ylabel("Counts")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_title(f"Summed Decay — {n_exp}-exp | IRF: {strategy}", fontweight="bold")
    ax1.axvspan(s["fit_window_ns"][0], s["fit_window_ns"][1],
                alpha=0.06, color="green")
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.fill_between(t_ns[fs:fe], np.clip(s["residuals"][fs:fe], -5, 5),
                     alpha=0.5, color="#457b9d")
    ax2.set_ylim(-5, 5)
    ax2.set_xlim(0, min(t_ns[-1], 22))
    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("W. Residuals")

    rv = np.clip(s["residuals"][fs:fe], -5, 5)
    ax3.hist(rv, bins=60, color="#2a9d8f", edgecolor="none", alpha=0.85)
    ax3.axvline(0, color="k", lw=0.8)
    ax3.set_xlabel("Weighted residual")
    ax3.set_ylabel("Frequency")
    ax3.set_title(f"Residuals  μ={rv.mean():.3f}  σ={rv.std():.3f}")

    lines = [f"χ²_r = {s['reduced_chi2']:.4f}",
             f"p    = {s['p_val']:.4f}",
             f"bg   = {s['bg_fit']:.1f} cts/bin",
             f"τ_mean(int) = {s['tau_mean_int_ns']:.4f} ns",
             f"τ_mean(amp) = {s['tau_mean_amp_ns']:.4f} ns",
             f"IRF FWHM(eff) = {s['irf_fwhm_eff_ns']:.4f} ns", ""]
    for i, (tau, frac) in enumerate(zip(s["taus_ns"], s["fractions"])):
        lines.append(f"τ{i+1}={tau:.4f} ns  f{i+1}={frac:.4f}")
    ax4.text(0.05, 0.97, "\n".join(lines), transform=ax4.transAxes,
             va="top", fontsize=9, family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", alpha=0.9))

    plt.suptitle("FLIM Reconvolution Fit — Leica FALCON / PicoHarp",
                 fontsize=12, fontweight="bold")
    out = f"{out_prefix}_summed_{n_exp}exp.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_pixel_maps(maps, n_exp, out_prefix, binning=1):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("#111")

    def _show(ax, data, title, cmap="viridis", vmin=None, vmax=None, unit="ns"):
        ax.set_facecolor("#111")
        valid = data[np.isfinite(data) & (data > 0)]
        if len(valid) == 0:
            ax.set_visible(False); return
        vlo = vmin if vmin is not None else np.percentile(valid, 2)
        vhi = vmax if vmax is not None else np.percentile(valid, 98)
        im  = ax.imshow(data, cmap=cmap, vmin=vlo, vmax=vhi, interpolation="nearest")
        cb  = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(unit, color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
        ax.set_title(title, color="white", fontweight="bold")
        ax.set_axis_off()

    _show(axes[0, 0], maps["intensity"],    "Intensity",         "hot",     unit="photons")
    _show(axes[0, 1], maps["tau_mean_int"], "τ_mean (int.-wt.)", FLIM_CMAP)
    _show(axes[0, 2], maps["tau_mean_amp"], "τ_mean (amp.-wt.)", FLIM_CMAP)
    for i in range(min(n_exp, 3)):
        _show(axes[1, i], maps.get(f"frac_{i+1}"), f"f{i+1}",
              "viridis", vmin=0, vmax=1, unit="fraction")

    plt.suptitle(f"FLIM Pixel Maps — {n_exp}-exp (τ fixed, α free)  "
                 f"binning={binning}×{binning}",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = f"{out_prefix}_pixelmaps_{n_exp}exp.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#111")
    plt.close()
    print(f"  Saved: {out}")


def plot_lifetime_histogram(maps, n_exp, out_prefix):
    tau = maps["tau_mean_int"]
    wt  = maps["intensity"]
    ok  = np.isfinite(tau) & (wt > 0)
    if ok.sum() < 2:
        return
    mu_w = np.average(tau[ok], weights=wt[ok])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(tau[ok], bins=100, weights=wt[ok], color="#2a9d8f", alpha=0.85)
    ax.axvline(mu_w, color="red", ls="--", lw=1.5,
               label=f"Weighted mean = {mu_w:.3f} ns")
    ax.set_xlabel("τ_mean (intensity-weighted) [ns]")
    ax.set_ylabel("Photon-weighted frequency")
    ax.set_title(f"Lifetime Distribution — {n_exp}-exp", fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = f"{out_prefix}_lifetime_hist_{n_exp}exp.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
