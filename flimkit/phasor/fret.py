from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class FRETChannelData:
    """Calibrated phasor arrays for one detection channel.

    Parameters
    ----------
    real_cal : ndarray
        Calibrated per-pixel phasor G (real) component, shape ``(Y, X)``.
    imag_cal : ndarray
        Calibrated per-pixel phasor S (imaginary) component, same shape.
    mean : ndarray
        Per-pixel mean photon intensity, same spatial shape.
    frequency : float
        Laser modulation / repetition frequency in MHz.
    min_photons : float
        Pixels with ``mean < min_photons`` are excluded from fitting.
    """

    real_cal:     np.ndarray
    imag_cal:     np.ndarray
    mean:         np.ndarray
    frequency:    float
    min_photons:  float = 0.01

    def __post_init__(self):
        self.real_cal = np.asarray(self.real_cal, dtype=float)
        self.imag_cal = np.asarray(self.imag_cal, dtype=float)
        self.mean     = np.asarray(self.mean,     dtype=float)
        if self.real_cal.shape != self.imag_cal.shape:
            raise ValueError(
                f"real_cal and imag_cal must have the same shape; "
                f"got {self.real_cal.shape} and {self.imag_cal.shape}."
            )
        if self.mean.shape != self.real_cal.shape:
            raise ValueError(
                f"mean must have the same shape as real_cal; "
                f"got {self.mean.shape} vs {self.real_cal.shape}."
            )
        if self.frequency <= 0:
            raise ValueError(
                f"frequency must be positive (MHz); got {self.frequency}."
            )

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask of pixels that meet the photon threshold and are finite."""
        return (self.mean >= self.min_photons) & np.isfinite(self.real_cal)

    @property
    def valid_g(self) -> np.ndarray:
        """G (real) values of valid pixels, flattened."""
        return self.real_cal[self.valid_mask]

    @property
    def valid_s(self) -> np.ndarray:
        """S (imaginary) values of valid pixels, flattened."""
        return self.imag_cal[self.valid_mask]


@dataclass
class FRETModelParameters:
    """Parameters describing the FRET system for the phasorpy forward model.

    Parameter names and defaults match
    ``phasorpy.lifetime.phasor_from_fret_donor`` and
    ``phasor_from_fret_acceptor`` directly.  All lifetimes are in **ns** and
    frequency in **MHz** (``unit_conversion = 1e-3``).

    Parameters
    ----------
    donor_lifetime : float
        Unquenched donor lifetime in ns.
    fret_efficiency : float
        FRET efficiency in [0, 1].
    donor_fretting : float
        Fraction of donor molecules participating in FRET, in [0, 1].
    donor_background : float
        Weight of background in the donor channel relative to unquenched donor.
    background_real, background_imag : float
        Phasor coordinates of the background signal.
    acceptor_lifetime : float or None
        Acceptor lifetime in ns.  Required for joint donor+acceptor fits.
    donor_bleedthrough : float
        Weight of donor signal in the acceptor channel relative to fully
        sensitized acceptor (see phasorpy docs).
    acceptor_bleedthrough : float
        Weight of directly excited acceptor relative to sensitized acceptor.
    acceptor_background : float
        Weight of background in the acceptor channel.
    """

    donor_lifetime:        float
    fret_efficiency:       float = 0.0
    donor_fretting:        float = 1.0
    donor_background:      float = 0.0
    background_real:       float = 0.0
    background_imag:       float = 0.0
    acceptor_lifetime:     Optional[float] = None
    donor_bleedthrough:    float = 0.0
    acceptor_bleedthrough: float = 0.0
    acceptor_background:   float = 0.0

    def __post_init__(self):
        if self.donor_lifetime <= 0:
            raise ValueError(
                f"donor_lifetime must be positive (ns); got {self.donor_lifetime}."
            )
        for attr, lo, hi in [
            ('fret_efficiency', 0.0, 1.0),
            ('donor_fretting',  0.0, 1.0),
        ]:
            v = getattr(self, attr)
            if not (lo <= v <= hi):
                raise ValueError(f"{attr} must be in [{lo}, {hi}]; got {v}.")


@dataclass
class FRETBounds:
    """Optimization bounds for each free FRET model parameter.

    Each field is a ``(lower, upper)`` tuple.  Set both values equal to fix
    that parameter at a constant during fitting.
    """

    fret_efficiency:       tuple[float, float] = (0.0,  1.0)
    donor_fretting:        tuple[float, float] = (0.0,  1.0)
    donor_background:      tuple[float, float] = (0.0, 10.0)
    donor_bleedthrough:    tuple[float, float] = (0.0, 10.0)
    acceptor_bleedthrough: tuple[float, float] = (0.0, 10.0)
    acceptor_background:   tuple[float, float] = (0.0, 10.0)

    def _as_scipy(self, attrs: tuple[str, ...]) -> dict:
        lo = [getattr(self, a)[0] for a in attrs]
        hi = [getattr(self, a)[1] for a in attrs]
        return dict(lb=lo, ub=hi)

    def donor_only_scipy(self) -> dict:
        """SciPy ``bounds`` dict for the three donor-only free parameters."""
        return self._as_scipy(
            ('fret_efficiency', 'donor_fretting', 'donor_background')
        )

    def joint_scipy(self) -> dict:
        """SciPy ``bounds`` dict for the six joint free parameters."""
        return self._as_scipy((
            'fret_efficiency', 'donor_fretting',
            'donor_background', 'donor_bleedthrough',
            'acceptor_bleedthrough', 'acceptor_background',
        ))


@dataclass
class FRETResult:
    """Result of a FRET phasor fitting operation.

    Attributes
    ----------
    fret_efficiency : float
        Fitted FRET efficiency.
    donor_fretting : float
        Fitted fraction of donors participating in FRET.
    donor_background : float
        Fitted background weight in the donor channel.
    donor_real_model, donor_imag_model : float
        Phasor coordinates of the fitted donor model.
    residual : float
        Final weighted sum-of-squared residuals.
    donor_bleedthrough, acceptor_bleedthrough, acceptor_background : float
        Fitted nuisance terms (joint fits only; 0.0 for donor-only fits).
    acceptor_real_model, acceptor_imag_model : float or None
        Phasor coordinates of the fitted acceptor model (joint fits only).
    converged : bool
        Whether the optimizer converged.
    message : str
        Optimizer status message.
    """

    fret_efficiency:       float
    donor_fretting:        float
    donor_background:      float
    donor_real_model:      float
    donor_imag_model:      float
    residual:              float
    donor_bleedthrough:    float = 0.0
    acceptor_bleedthrough: float = 0.0
    acceptor_background:   float = 0.0
    acceptor_real_model:   Optional[float] = None
    acceptor_imag_model:   Optional[float] = None
    converged:             bool = True
    message:               str  = ""

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        print("\u2550\u2550\u2550 FRET fit result \u2550\u2550\u2550")
        print(f"  FRET efficiency    : {self.fret_efficiency:.4f}")
        print(f"  Donor fretting     : {self.donor_fretting:.4f}")
        print(f"  Donor background   : {self.donor_background:.4f}")
        print(f"  Donor model  (G,S) : "
              f"({self.donor_real_model:.4f}, {self.donor_imag_model:.4f})")
        if self.acceptor_real_model is not None:
            print(f"  Acceptor model (G,S): "
                  f"({self.acceptor_real_model:.4f}, {self.acceptor_imag_model:.4f})")
            print(f"  Donor bleedthrough  : {self.donor_bleedthrough:.4f}")
            print(f"  Acceptor bleedthrough: {self.acceptor_bleedthrough:.4f}")
            print(f"  Acceptor background : {self.acceptor_background:.4f}")
        print(f"  Residual           : {self.residual:.6g}")
        status = "\u2713 converged" if self.converged else "\u2717 did not converge"
        print(f"  Optimizer          : {status}  \u2014 {self.message}")

def _require_phasorpy_fret_api() -> None:
    """Raise ``ImportError`` with actionable guidance if the phasorpy FRET API
    is absent or the signatures differ from what FLIMKit expects.
    """
    try:
        from phasorpy.lifetime import (  # noqa: F401
            phasor_from_fret_donor,
            phasor_from_fret_acceptor,
        )
    except ImportError as exc:
        raise ImportError(
            "FRET analysis requires phasorpy >= 0.9 with "
            "phasor_from_fret_donor and phasor_from_fret_acceptor.  "
            "Install with:  pip install 'phasorpy>=0.9'"
        ) from exc

    import inspect
    from phasorpy.lifetime import phasor_from_fret_donor, phasor_from_fret_acceptor
    for fn, required in [
        (phasor_from_fret_donor,    {'frequency', 'donor_lifetime',
                                     'fret_efficiency', 'donor_fretting'}),
        (phasor_from_fret_acceptor, {'frequency', 'donor_lifetime',
                                     'acceptor_lifetime', 'fret_efficiency'}),
    ]:
        missing = required - set(inspect.signature(fn).parameters)
        if missing:
            raise ImportError(
                f"phasorpy.{fn.__name__} is missing expected parameters: "
                f"{missing}.  Please update phasorpy."
            )


def _single_lifetime_phasor(frequency: float, lifetime: float) -> tuple[float, float]:
    """Return the theoretical ``(G, S)`` phasor for a single-exponential lifetime.

    Parameters
    ----------
    frequency : float
        Modulation frequency in MHz.
    lifetime : float
        Fluorescence lifetime in ns.

    Returns
    -------
    (real, imag) : tuple of float
    """
    from phasorpy.lifetime import phasor_from_lifetime
    real, imag = phasor_from_lifetime(frequency, lifetime)
    return float(np.squeeze(real)), float(np.squeeze(imag))


def _fret_donor_phasor(
    frequency: float,
    donor_lifetime: float,
    *,
    fret_efficiency: float = 0.0,
    donor_fretting: float = 1.0,
    donor_background: float = 0.0,
    background_real: float = 0.0,
    background_imag: float = 0.0,
) -> tuple[float, float]:
    """Return the model donor-channel ``(G, S)`` phasor via phasorpy.

    Thin wrapper around ``phasorpy.lifetime.phasor_from_fret_donor`` with
    explicit MHz/ns unit conversion (``unit_conversion=1e-3``).

    Returns
    -------
    (real, imag) : tuple of float
    """
    from phasorpy.lifetime import phasor_from_fret_donor
    real, imag = phasor_from_fret_donor(
        frequency, donor_lifetime,
        fret_efficiency=fret_efficiency,
        donor_fretting=donor_fretting,
        donor_background=donor_background,
        background_real=background_real,
        background_imag=background_imag,
        unit_conversion=1e-3,
    )
    return float(np.squeeze(real)), float(np.squeeze(imag))


def _fret_acceptor_phasor(
    frequency: float,
    donor_lifetime: float,
    acceptor_lifetime: float,
    *,
    fret_efficiency: float = 0.0,
    donor_fretting: float = 1.0,
    donor_bleedthrough: float = 0.0,
    acceptor_bleedthrough: float = 0.0,
    acceptor_background: float = 0.0,
    background_real: float = 0.0,
    background_imag: float = 0.0,
) -> tuple[float, float]:
    """Return the model acceptor-channel ``(G, S)`` phasor via phasorpy.

    Thin wrapper around ``phasorpy.lifetime.phasor_from_fret_acceptor`` with
    explicit MHz/ns unit conversion (``unit_conversion=1e-3``).

    Returns
    -------
    (real, imag) : tuple of float
    """
    from phasorpy.lifetime import phasor_from_fret_acceptor
    real, imag = phasor_from_fret_acceptor(
        frequency, donor_lifetime, acceptor_lifetime,
        fret_efficiency=fret_efficiency,
        donor_fretting=donor_fretting,
        donor_bleedthrough=donor_bleedthrough,
        acceptor_bleedthrough=acceptor_bleedthrough,
        acceptor_background=acceptor_background,
        background_real=background_real,
        background_imag=background_imag,
        unit_conversion=1e-3,
    )
    return float(np.squeeze(real)), float(np.squeeze(imag))


def predict_fret_trajectory(
    frequency: float,
    donor_lifetime: float,
    *,
    acceptor_lifetime: Optional[float] = None,
    donor_fretting: float = 1.0,
    donor_background: float = 0.0,
    background_real: float = 0.0,
    background_imag: float = 0.0,
    donor_bleedthrough: float = 0.0,
    acceptor_bleedthrough: float = 0.0,
    acceptor_background: float = 0.0,
    n_points: int = 100,
) -> dict:
    """Generate model donor and acceptor phasor trajectories across FRET efficiency.

    Calls ``phasorpy.lifetime.phasor_from_fret_donor`` (and optionally
    ``phasor_from_fret_acceptor``) over a sweep of efficiency values from 0 to 1.
    Useful for overlaying the expected trajectory on a phasor histogram before
    fitting.

    Parameters
    ----------
    frequency : float
        Modulation frequency in MHz.
    donor_lifetime : float
        Unquenched donor lifetime in ns.
    acceptor_lifetime : float or None
        Acceptor lifetime in ns.  Supply to also compute the acceptor trajectory.
    donor_fretting : float
        Fraction of donors participating in FRET.
    donor_background, background_real, background_imag : float
        Donor channel background terms.
    donor_bleedthrough, acceptor_bleedthrough, acceptor_background : float
        Acceptor channel nuisance terms (only used when *acceptor_lifetime* is given).
    n_points : int
        Number of efficiency points spanning [0, 1].

    Returns
    -------
    dict with keys:

    * ``efficiency``  \u2013 1-D array of efficiency values (0 \u2192 1)
    * ``donor_g``, ``donor_s``       \u2013 donor model trajectory arrays
    * ``acceptor_g``, ``acceptor_s`` \u2013 acceptor model trajectory arrays
      (or ``None`` when *acceptor_lifetime* is not given)
    """
    _require_phasorpy_fret_api()
    efficiencies = np.linspace(0.0, 1.0, n_points)

    from phasorpy.lifetime import phasor_from_fret_donor
    donor_g, donor_s = phasor_from_fret_donor(
        frequency, donor_lifetime,
        fret_efficiency=efficiencies,
        donor_fretting=donor_fretting,
        donor_background=donor_background,
        background_real=background_real,
        background_imag=background_imag,
        unit_conversion=1e-3,
    )

    acceptor_g = acceptor_s = None
    if acceptor_lifetime is not None:
        from phasorpy.lifetime import phasor_from_fret_acceptor
        acceptor_g, acceptor_s = phasor_from_fret_acceptor(
            frequency, donor_lifetime, acceptor_lifetime,
            fret_efficiency=efficiencies,
            donor_fretting=donor_fretting,
            donor_bleedthrough=donor_bleedthrough,
            acceptor_bleedthrough=acceptor_bleedthrough,
            acceptor_background=acceptor_background,
            background_real=background_real,
            background_imag=background_imag,
            unit_conversion=1e-3,
        )

    return dict(
        efficiency=efficiencies,
        donor_g=np.asarray(donor_g),
        donor_s=np.asarray(donor_s),
        acceptor_g=np.asarray(acceptor_g) if acceptor_g is not None else None,
        acceptor_s=np.asarray(acceptor_s) if acceptor_s is not None else None,
    )


# Donor-Only Solver
def fit_donor_fret(
    donor: FRETChannelData,
    params: FRETModelParameters,
    bounds: Optional[FRETBounds] = None,
    *,
    weight_by_photons: bool = True,
) -> FRETResult:
    """Fit donor-channel FRET parameters to the photon-weighted phasor centroid.

    Minimises the Euclidean distance between the phasorpy donor FRET model
    and the weighted mean phasor of all pixels that pass the photon threshold
    in *donor*.  Free parameters are ``fret_efficiency``, ``donor_fretting``,
    and ``donor_background``; all other model values are held fixed from *params*.

    Parameters
    ----------
    donor : FRETChannelData
        Calibrated donor-channel phasor data.
    params : FRETModelParameters
        Starting values and fixed parameters.  ``donor_lifetime``,
        ``background_real``, and ``background_imag`` are held constant.
    bounds : FRETBounds or None
        Optimization bounds.  Defaults to ``FRETBounds()`` when ``None``.
    weight_by_photons : bool
        If ``True``, weight the centroid by per-pixel photon counts.

    Returns
    -------
    FRETResult
    """
    _require_phasorpy_fret_api()
    from phasorpy.lifetime import phasor_from_fret_donor
    from scipy.optimize import least_squares

    if bounds is None:
        bounds = FRETBounds()

    mask = donor.valid_mask
    g_vals = donor.real_cal[mask]
    s_vals = donor.imag_cal[mask]
    if weight_by_photons:
        w = donor.mean[mask]
        w = w / w.sum()
    else:
        w = np.ones(g_vals.size) / g_vals.size

    g_obs = float(np.dot(w, g_vals))
    s_obs = float(np.dot(w, s_vals))

    freq = donor.frequency
    tau_d = params.donor_lifetime
    bg_real = params.background_real
    bg_imag = params.background_imag

    def _residuals(x: np.ndarray) -> np.ndarray:
        E, f, bg = x
        g_m, s_m = phasor_from_fret_donor(
            freq, tau_d,
            fret_efficiency=E,
            donor_fretting=f,
            donor_background=bg,
            background_real=bg_real,
            background_imag=bg_imag,
            unit_conversion=1e-3,
        )
        return np.array([float(g_m) - g_obs, float(s_m) - s_obs])

    x0 = [params.fret_efficiency, params.donor_fretting, params.donor_background]
    scipy_bounds = bounds.donor_only_scipy()
    result = least_squares(
        _residuals, x0,
        bounds=(scipy_bounds['lb'], scipy_bounds['ub']),
        method='trf',
    )

    E_fit, f_fit, bg_fit = result.x
    g_model, s_model = phasor_from_fret_donor(
        freq, tau_d,
        fret_efficiency=E_fit,
        donor_fretting=f_fit,
        donor_background=bg_fit,
        background_real=bg_real,
        background_imag=bg_imag,
        unit_conversion=1e-3,
    )

    return FRETResult(
        fret_efficiency=float(E_fit),
        donor_fretting=float(f_fit),
        donor_background=float(bg_fit),
        donor_real_model=float(g_model),
        donor_imag_model=float(s_model),
        residual=float(result.cost),
        converged=bool(result.success),
        message=result.message,
    )


#Joint Donor+Acceptor Solver


def fit_joint_fret(
    donor: FRETChannelData,
    acceptor: FRETChannelData,
    params: FRETModelParameters,
    bounds: Optional[FRETBounds] = None,
    *,
    weight_by_photons: bool = True,
) -> FRETResult:
    """Fit FRET parameters jointly from donor and acceptor phasor centroids.

    Minimises the combined Euclidean distance between phasorpy forward-model
    predictions and the photon-weighted mean phasors of both channels.  The
    four residuals (donor G, donor S, acceptor G, acceptor S) are equally
    weighted.  Free parameters are ``fret_efficiency``, ``donor_fretting``,
    ``donor_background``, ``donor_bleedthrough``, ``acceptor_bleedthrough``,
    and ``acceptor_background``; ``donor_lifetime``, ``acceptor_lifetime``,
    ``background_real``, and ``background_imag`` are held fixed from *params*.

    Parameters
    ----------
    donor : FRETChannelData
        Calibrated donor-channel phasor data.
    acceptor : FRETChannelData
        Calibrated acceptor-channel phasor data.  Must have the same spatial
        shape as *donor*.
    params : FRETModelParameters
        Starting values and fixed parameters.  ``acceptor_lifetime`` must be
        set (not ``None``).
    bounds : FRETBounds or None
        Optimization bounds.  Defaults to ``FRETBounds()`` when ``None``.
    weight_by_photons : bool
        If ``True``, weight each channel's centroid by per-pixel photon counts.

    Returns
    -------
    FRETResult

    Raises
    ------
    ValueError
        If ``params.acceptor_lifetime`` is ``None`` or the donor and acceptor
        arrays have different shapes.
    """
    if params.acceptor_lifetime is None:
        raise ValueError(
            "fit_joint_fret requires params.acceptor_lifetime to be set."
        )
    if donor.real_cal.shape != acceptor.real_cal.shape:
        raise ValueError(
            f"donor and acceptor arrays must have the same shape; "
            f"got {donor.real_cal.shape} vs {acceptor.real_cal.shape}."
        )
    if donor.frequency != acceptor.frequency:
        raise ValueError(
            f"donor and acceptor must share the same frequency; "
            f"got {donor.frequency} vs {acceptor.frequency} MHz."
        )

    _require_phasorpy_fret_api()
    from phasorpy.lifetime import phasor_from_fret_donor, phasor_from_fret_acceptor
    from scipy.optimize import least_squares

    if bounds is None:
        bounds = FRETBounds()

    def _centroid(ch: FRETChannelData) -> tuple[float, float]:
        mask = ch.valid_mask
        g_vals = ch.real_cal[mask]
        s_vals = ch.imag_cal[mask]
        if weight_by_photons:
            w = ch.mean[mask]
            w = w / w.sum()
        else:
            w = np.ones(g_vals.size) / g_vals.size
        return float(np.dot(w, g_vals)), float(np.dot(w, s_vals))

    dg_obs, ds_obs = _centroid(donor)
    ag_obs, as_obs = _centroid(acceptor)

    freq   = donor.frequency
    tau_d  = params.donor_lifetime
    tau_a  = params.acceptor_lifetime
    bg_real = params.background_real
    bg_imag = params.background_imag

    def _residuals(x: np.ndarray) -> np.ndarray:
        E, f, d_bg, d_bt, a_bt, a_bg = x
        dg_m, ds_m = phasor_from_fret_donor(
            freq, tau_d,
            fret_efficiency=E,
            donor_fretting=f,
            donor_background=d_bg,
            background_real=bg_real,
            background_imag=bg_imag,
            unit_conversion=1e-3,
        )
        ag_m, as_m = phasor_from_fret_acceptor(
            freq, tau_d, tau_a,
            fret_efficiency=E,
            donor_fretting=f,
            donor_bleedthrough=d_bt,
            acceptor_bleedthrough=a_bt,
            acceptor_background=a_bg,
            background_real=bg_real,
            background_imag=bg_imag,
            unit_conversion=1e-3,
        )
        return np.array([
            float(dg_m) - dg_obs,
            float(ds_m) - ds_obs,
            float(ag_m) - ag_obs,
            float(as_m) - as_obs,
        ])

    x0 = [
        params.fret_efficiency,
        params.donor_fretting,
        params.donor_background,
        params.donor_bleedthrough,
        params.acceptor_bleedthrough,
        params.acceptor_background,
    ]
    scipy_bounds = bounds.joint_scipy()
    result = least_squares(
        _residuals, x0,
        bounds=(scipy_bounds['lb'], scipy_bounds['ub']),
        method='trf',
    )

    E_fit, f_fit, d_bg_fit, d_bt_fit, a_bt_fit, a_bg_fit = result.x

    dg_model, ds_model = phasor_from_fret_donor(
        freq, tau_d,
        fret_efficiency=E_fit,
        donor_fretting=f_fit,
        donor_background=d_bg_fit,
        background_real=bg_real,
        background_imag=bg_imag,
        unit_conversion=1e-3,
    )
    ag_model, as_model = phasor_from_fret_acceptor(
        freq, tau_d, tau_a,
        fret_efficiency=E_fit,
        donor_fretting=f_fit,
        donor_bleedthrough=d_bt_fit,
        acceptor_bleedthrough=a_bt_fit,
        acceptor_background=a_bg_fit,
        background_real=bg_real,
        background_imag=bg_imag,
        unit_conversion=1e-3,
    )

    return FRETResult(
        fret_efficiency=float(E_fit),
        donor_fretting=float(f_fit),
        donor_background=float(d_bg_fit),
        donor_real_model=float(dg_model),
        donor_imag_model=float(ds_model),
        residual=float(result.cost),
        donor_bleedthrough=float(d_bt_fit),
        acceptor_bleedthrough=float(a_bt_fit),
        acceptor_background=float(a_bg_fit),
        acceptor_real_model=float(ag_model),
        acceptor_imag_model=float(as_model),
        converged=bool(result.success),
        message=result.message,
    )


#Pixelwise Maps


def map_fret_efficiency(
    donor: FRETChannelData,
    params: FRETModelParameters,
    bounds: Optional[FRETBounds] = None,
    *,
    acceptor: Optional[FRETChannelData] = None,
    weight_by_photons: bool = True,
) -> dict:
    """Compute per-pixel FRET efficiency by fitting each pixel independently.

    Applies :func:`fit_donor_fret` (or :func:`fit_joint_fret` when *acceptor*
    is supplied) to every pixel that passes the photon threshold.  Pixels
    below the threshold are filled with ``nan``.

    Parameters
    ----------
    donor : FRETChannelData
        Calibrated donor-channel phasor data.
    params : FRETModelParameters
        Starting values and fixed parameters shared across all pixels.
    bounds : FRETBounds or None
        Optimization bounds.  Defaults to ``FRETBounds()`` when ``None``.
    acceptor : FRETChannelData or None
        Calibrated acceptor-channel phasor data.  When supplied, joint fitting
        is used; ``params.acceptor_lifetime`` must be set.
    weight_by_photons : bool
        Passed through to the per-pixel solver.

    Returns
    -------
    dict with keys:

    * ``efficiency``   -- 2-D array, shape ``(Y, X)``, ``nan`` where masked
    * ``fretting``     -- 2-D array of fitted ``donor_fretting``
    * ``residual``     -- 2-D array of per-pixel fit cost
    * ``converged``    -- 2-D bool array
    """
    if bounds is None:
        bounds = FRETBounds()

    Y, X = donor.real_cal.shape
    efficiency = np.full((Y, X), np.nan)
    fretting   = np.full((Y, X), np.nan)
    residual   = np.full((Y, X), np.nan)
    converged  = np.zeros((Y, X), dtype=bool)

    mask = donor.valid_mask
    if acceptor is not None:
        mask = mask & acceptor.valid_mask

    ys, xs = np.where(mask)

    for y, x in zip(ys, xs):
        px_donor = FRETChannelData(
            real_cal=donor.real_cal[y:y+1, x:x+1],
            imag_cal=donor.imag_cal[y:y+1, x:x+1],
            mean=donor.mean[y:y+1, x:x+1],
            frequency=donor.frequency,
            min_photons=donor.min_photons,
        )
        if acceptor is not None:
            px_acceptor = FRETChannelData(
                real_cal=acceptor.real_cal[y:y+1, x:x+1],
                imag_cal=acceptor.imag_cal[y:y+1, x:x+1],
                mean=acceptor.mean[y:y+1, x:x+1],
                frequency=acceptor.frequency,
                min_photons=acceptor.min_photons,
            )
            r = fit_joint_fret(
                px_donor, px_acceptor, params, bounds,
                weight_by_photons=weight_by_photons,
            )
        else:
            r = fit_donor_fret(
                px_donor, params, bounds,
                weight_by_photons=weight_by_photons,
            )
        efficiency[y, x] = r.fret_efficiency
        fretting[y, x]   = r.donor_fretting
        residual[y, x]   = r.residual
        converged[y, x]  = r.converged

    return dict(
        efficiency=efficiency,
        fretting=fretting,
        residual=residual,
        converged=converged,
    )


#Visualization


def plot_fret_trajectory(
    frequency: float,
    donor_lifetime: float,
    *,
    acceptor_lifetime: Optional[float] = None,
    donor_fretting: float = 1.0,
    n_points: int = 100,
    ax=None,
    donor_kw: Optional[dict] = None,
    acceptor_kw: Optional[dict] = None,
) -> tuple:
    """Overlay FRET model trajectory curves on a phasor axes.

    Parameters
    ----------
    frequency : float
        Modulation frequency in MHz.
    donor_lifetime : float
        Unquenched donor lifetime in ns.
    acceptor_lifetime : float or None
        Acceptor lifetime in ns.  Supply to also plot the acceptor trajectory.
    donor_fretting : float
        Fraction of donors participating in FRET.
    n_points : int
        Number of efficiency points spanning [0, 1].
    ax : matplotlib Axes or None
        Axes to draw on.  Creates a new figure/axes when ``None``.
    donor_kw : dict or None
        Keyword arguments forwarded to the donor ``ax.plot`` call.
        Defaults to ``{'color': 'steelblue', 'label': 'donor trajectory'}``.
    acceptor_kw : dict or None
        Keyword arguments forwarded to the acceptor ``ax.plot`` call.
        Defaults to ``{'color': 'tomato', 'label': 'acceptor trajectory'}``.

    Returns
    -------
    (ax, lines) : tuple
        *ax* is the Axes drawn on; *lines* is a list of the ``Line2D`` objects
        added (one for donor, optionally one for acceptor).
    """
    import matplotlib.pyplot as plt

    traj = predict_fret_trajectory(
        frequency, donor_lifetime,
        acceptor_lifetime=acceptor_lifetime,
        donor_fretting=donor_fretting,
        n_points=n_points,
    )

    if ax is None:
        _, ax = plt.subplots()

    _donor_kw = {'color': 'steelblue', 'label': 'donor trajectory'}
    if donor_kw:
        _donor_kw.update(donor_kw)

    lines = [ax.plot(traj['donor_g'], traj['donor_s'], **_donor_kw)[0]]

    if traj['acceptor_g'] is not None:
        _acceptor_kw = {'color': 'tomato', 'label': 'acceptor trajectory'}
        if acceptor_kw:
            _acceptor_kw.update(acceptor_kw)
        lines.append(
            ax.plot(traj['acceptor_g'], traj['acceptor_s'], **_acceptor_kw)[0]
        )

    return ax, lines


def plot_fret_fit(
    donor: FRETChannelData,
    result: FRETResult,
    frequency: float,
    donor_lifetime: float,
    *,
    acceptor: Optional[FRETChannelData] = None,
    acceptor_lifetime: Optional[float] = None,
    n_trajectory: int = 100,
    ax=None,
    scatter_kw: Optional[dict] = None,
    trajectory_kw: Optional[dict] = None,
) -> tuple:
    """Plot measured phasors, the fitted model point, and the FRET trajectory.

    Parameters
    ----------
    donor : FRETChannelData
        Calibrated donor phasor data (valid pixels are scatter-plotted).
    result : FRETResult
        Fitted result whose model point is highlighted.
    frequency : float
        Modulation frequency in MHz.
    donor_lifetime : float
        Unquenched donor lifetime in ns.
    acceptor : FRETChannelData or None
        Acceptor channel data to scatter-plot alongside the donor.
    acceptor_lifetime : float or None
        Required when *acceptor* is supplied to draw the acceptor trajectory.
    n_trajectory : int
        Number of efficiency points for the trajectory curve.
    ax : matplotlib Axes or None
        Axes to draw on.  Creates a new figure/axes when ``None``.
    scatter_kw : dict or None
        Keyword arguments forwarded to the donor scatter ``ax.plot`` call.
    trajectory_kw : dict or None
        Keyword arguments forwarded to :func:`plot_fret_trajectory`.

    Returns
    -------
    (ax, artists) : tuple
        *ax* is the Axes; *artists* is a dict with keys ``'donor_scatter'``,
        ``'donor_model'``, and optionally ``'acceptor_scatter'``,
        ``'acceptor_model'``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    _scatter_kw = {'marker': '.', 'alpha': 0.3, 'linestyle': 'none', 'color': 'steelblue'}
    if scatter_kw:
        _scatter_kw.update(scatter_kw)

    artists: dict = {}

    artists['donor_scatter'] = ax.plot(
        donor.valid_g, donor.valid_s, **_scatter_kw
    )[0]

    if acceptor is not None:
        _acc_scatter_kw = dict(_scatter_kw)
        _acc_scatter_kw['color'] = 'tomato'
        _acc_scatter_kw.pop('label', None)
        artists['acceptor_scatter'] = ax.plot(
            acceptor.valid_g, acceptor.valid_s, **_acc_scatter_kw
        )[0]

    _traj_kw = trajectory_kw or {}
    plot_fret_trajectory(
        frequency, donor_lifetime,
        acceptor_lifetime=acceptor_lifetime,
        donor_fretting=result.donor_fretting,
        n_points=n_trajectory,
        ax=ax,
        **_traj_kw,
    )

    artists['donor_model'] = ax.plot(
        result.donor_real_model, result.donor_imag_model,
        marker='*', markersize=12, color='steelblue',
        linestyle='none', label='donor fit',
        zorder=5,
    )[0]

    if result.acceptor_real_model is not None:
        artists['acceptor_model'] = ax.plot(
            result.acceptor_real_model, result.acceptor_imag_model,
            marker='*', markersize=12, color='tomato',
            linestyle='none', label='acceptor fit',
            zorder=5,
        )[0]

    return ax, artists
