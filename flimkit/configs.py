import sys
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

FLIM_CMAP = LinearSegmentedColormap.from_list(
    "flim", ["#000080","#0000ff","#00ffff","#00ff00","#ffff00","#ff0000"]
)

MIN_PHOTONS_PERPIX = 10

# Intensity threshold for masking low-signal pixels before fitting.
# Set to None (disabled) or an integer photon count.
# Use --intensity-threshold on the CLI, or 'interactive' to pick visually.
INTENSITY_THRESHOLD = None


# General fitting settings:
Tau_min = 0.145 # ns - set to 0.145 ns to avoid fitting to the IRF peak (which is typically around 0.1-0.12 ns for a system with 97 ps bins). Adjust as needed for other systems.
Tau_max = 45.0 # ns - set to 45 ns to allow fitting of long lifetimes. Can always manually override when running the code.

# Set default fitting mode. Options are "summed", "perPixel", and "both". Override with --mode when running the code.
D_mode = 'both'

# Default number of exponentials to fit. Options are 1, 2, or 3. Override with --nexp when running the code. 4+ exponentials are not supported as the fitting becomes unstable and unreliable.
n_exp = 3

# Default binning factor for per-pixel fitting. Set to 1 for no binning. Override with --binning when running the code.
binning_factor = 1
# Default optimizer for per-pixel fitting. Options are "lm_multistart" and "de".
Optimizer = "de"

# Levenberg-Marquardt settins:
lm_restarts = 8

# Default settings for DE optimizer:
de_population = 30
de_maxiter = 5000
n_workers = -1 # Use all available CPU cores for DE optimization. Override with --workers when running the code.

# IRF settings:
IRF_FWHM = None # Set to None to use the default of 1 bin width from the PTU file (e.g. 0.097 ns for 97 ps bins). Override with --irf-fwhm when running the code.
IRF_FIT_WIDTH = 1.5 # ns - width of the region around time zero to use for fitting the IRF. Adjust as needed for other systems.
IRF_BINS = 21 # Number of bins to use for the IRF when fitting with the "summed" mode. Adjust as needed for other systems and bin widths. Should be an odd number to have a bin centered on time zero.
Estimate_IRF = "none" # Options are "raw", "parametric", and "none". Set to "raw" to use the raw IRF from the data, "parametric" to fit a parametric function to the IRF, or "none" to not estimate the IRF (e.g. if you have a separate IRF file or are using a system with a very narrow IRF that doesn't need to be accounted for). Override with --estimate-irf when running the code.

# Machine IRF defaults (spreadsheet-free workflow)

# True when launched as a PyInstaller-compiled executable.
_is_frozen = getattr(sys, "frozen", False)

# Bundled read-only resources — always accessible via __file__ (even when frozen,
# PyInstaller extracts them to a temp dir that __file__ still points into).
_BUNDLED_MACHINE_IRF_DIR = Path(__file__).resolve().parent / "machine_irf"

# Writable location for user-generated IRF files.
# When frozen the package dir is inside a read-only bundle, so saves go to
# ~/.flimkit/machine_irf/ instead.  When running from source the behaviour is
# unchanged (saves alongside the package).
USER_MACHINE_IRF_DIR = Path.home() / ".flimkit" / "machine_irf"
MACHINE_IRF_DIR = USER_MACHINE_IRF_DIR if _is_frozen else _BUNDLED_MACHINE_IRF_DIR

# Default IRF path used when no explicit path is given.
# When frozen: prefer a user-saved copy (if it exists), else fall back to the
# bundled factory default so a fresh install still works out of the box.
_user_default = USER_MACHINE_IRF_DIR / "machine_irf_default.npy"
MACHINE_IRF_DEFAULT_PATH = (
    (_user_default if _user_default.exists() else _BUNDLED_MACHINE_IRF_DIR / "machine_irf_default.npy")
    if _is_frozen
    else _BUNDLED_MACHINE_IRF_DIR / "machine_irf_default.npy"
)
MACHINE_IRF_ALIGN_ANCHOR = "peak"   # learned Leica-style placement anchor
MACHINE_IRF_REDUCER = "median"      # aggregation across paired IRFs
MACHINE_IRF_FIT_STRATEGY = "fixed"  # notebook-chosen default: fixed machine IRF
MACHINE_IRF_FIT_BG = True
MACHINE_IRF_FIT_SIGMA = False
MACHINE_IRF_FIT_TAIL = False
MACHINE_IRF_DE_POPULATION = 30
MACHINE_IRF_DE_MAXITER = 5000

# Cost function for summed fit. Options are "poisson" (recommended) and "chi2" (legacy).
# "poisson" uses Poisson deviance (C-statistic) on raw counts — statistically correct.
# "chi2" normalises by peak and uses Neyman weights — underweights the tail.
Cost_function = "poisson"

# Display range for exported tau images (Leica LAS X-style clipping).
# Out-of-range pixels are clipped to the nearest boundary, matching LAS X behaviour.
# Set to None to keep the full fitted range (no clipping).
TAU_DISPLAY_MIN = None   # ns – minimum lifetime for weighted-tau images
TAU_DISPLAY_MAX = None   # ns – maximum lifetime for weighted-tau images

# Display range for exported intensity images (same clipping behaviour).
# Set to None to keep the full range.
INTENSITY_DISPLAY_MIN = None  # photon counts – minimum intensity
INTENSITY_DISPLAY_MAX = None  # photon counts – maximum intensity

# Other specific settings:
channels = None # Set to None to fit all channels in the PTU file. Override with --channel when running the code.
OUT_NAME = "flim_out" # Default output directory name. Override with --out when running the code.


config_message = f"""Default settings:
Intensity threshold: {INTENSITY_THRESHOLD} photons (None = disabled)
Tau_min: {Tau_min} ns
Tau_max: {Tau_max} ns
Tau display min: {TAU_DISPLAY_MIN} ns (None = no clip)
Tau display max: {TAU_DISPLAY_MAX} ns (None = no clip)
Intensity display min: {INTENSITY_DISPLAY_MIN} (None = no clip)
Intensity display max: {INTENSITY_DISPLAY_MAX} (None = no clip)
Fitting mode: {D_mode}  
Number of exponentials: {n_exp}
Cost function: {Cost_function}
Optimizer: {Optimizer}
Levenberg-Marquardt restarts: {lm_restarts}
Differential Evolution population size: {de_population}
Differential Evolution max iterations: {de_maxiter}
Number of workers for DE optimization: {n_workers}
IRF FWHM: {IRF_FWHM} ns
IRF fit width: {IRF_FIT_WIDTH} ns
IRF bins: {IRF_BINS}
IRF estimation method: {Estimate_IRF}
Machine IRF default path: {MACHINE_IRF_DEFAULT_PATH}
Machine IRF align anchor: {MACHINE_IRF_ALIGN_ANCHOR}
Machine IRF reducer: {MACHINE_IRF_REDUCER}
Machine IRF fit strategy: {MACHINE_IRF_FIT_STRATEGY}
Machine IRF fit bg/sigma/tail: {MACHINE_IRF_FIT_BG}/{MACHINE_IRF_FIT_SIGMA}/{MACHINE_IRF_FIT_TAIL}
Machine IRF DE population/maxiter: {MACHINE_IRF_DE_POPULATION}/{MACHINE_IRF_DE_MAXITER}
Channels to fit: {channels}
Output directory: {OUT_NAME}
Change any of these defaults by passing the corresponding argument when running the code. 
Run `python -m flim_fitter --help` for more details on available arguments and their usage. 
Change defaults in configs.py or override with command line arguments as needed for different datasets and systems.
"""

if __name__ == "__main__":
    print(config_message)