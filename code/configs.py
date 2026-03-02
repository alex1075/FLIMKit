from matplotlib.colors import LinearSegmentedColormap

FLIM_CMAP = LinearSegmentedColormap.from_list(
    "flim", ["#000080","#0000ff","#00ffff","#00ff00","#ffff00","#ff0000"]
)

MIN_PHOTONS_PERPIX = 10


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
de_population = 15
de_maxiter = 1000   
n_workers = -1 # Use all available CPU cores for DE optimization. Override with --workers when running the code.

# IRF settings:
IRF_FWHM = None # Set to None to use the default of 1 bin width from the PTU file (e.g. 0.097 ns for 97 ps bins). Override with --irf-fwhm when running the code.
IRF_FIT_WIDTH = 1.5 # ns - width of the region around time zero to use for fitting the IRF. Adjust as needed for other systems.
IRF_BINS = 21 # Number of bins to use for the IRF when fitting with the "summed" mode. Adjust as needed for other systems and bin widths. Should be an odd number to have a bin centered on time zero.
Estimate_IRF = "none" # Options are "raw", "parametric", and "none". Set to "raw" to use the raw IRF from the data, "parametric" to fit a parametric function to the IRF, or "none" to not estimate the IRF (e.g. if you have a separate IRF file or are using a system with a very narrow IRF that doesn't need to be accounted for). Override with --estimate-irf when running the code.

# Other specific settings:
channels = None # Set to None to fit all channels in the PTU file. Override with --channel when running the code.
OUT_NAME = "flim_out" # Default output directory name. Override with --out when running the code.


config_message = f"""Default settings:
Tau_min: {Tau_min} ns
Tau_max: {Tau_max} ns
Fitting mode: {D_mode}  
Number of exponentials: {n_exp}
Optimizer: {Optimizer}
Levenberg-Marquardt restarts: {lm_restarts}
Differential Evolution population size: {de_population}
Differential Evolution max iterations: {de_maxiter}
Number of workers for DE optimization: {n_workers}
IRF FWHM: {IRF_FWHM} ns
IRF fit width: {IRF_FIT_WIDTH} ns
IRF bins: {IRF_BINS}
IRF estimation method: {Estimate_IRF}
Channels to fit: {channels}
Output directory: {OUT_NAME}
Change any of these defaults by passing the corresponding argument when running the code. 
Run `python -m flim_fitter --help` for more details on available arguments and their usage. 
Change defaults in configs.py or override with command line arguments as needed for different datasets and systems.
"""

if __name__ == "__main__":
    print(config_message)