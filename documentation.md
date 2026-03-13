# FLIMKit Documentation

> **Version 0.3.1** — A Python toolkit for Fluorescence Lifetime Imaging Microscopy (FLIM)

> **Warning:** This project is in active development. Please cross-validate results with other software before drawing conclusions.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements & Installation](#requirements--installation)
3. [Quick Start](#quick-start)
4. [Workflows](#workflows)
   - [Guided Terminal UI](#guided-terminal-ui-mainpy)
   - [Machine IRF Setup (Required)](#machine-irf-setup-required)
   - [FLIM Reconvolution Fitting (CLI)](#flim-reconvolution-fitting-cli)
   - [Phasor Analysis (CLI)](#phasor-analysis-cli)
   - [Python API](#python-api)
5. [Configuration Reference](#configuration-reference)
6. [Module Reference](#module-reference)
   - [flimkit.PTU — PTU File I/O](#flimkitptu--ptu-file-io)
   - [flimkit.FLIM — Reconvolution Fitting](#flimkitflim--reconvolution-fitting)
   - [flimkit.phasor — Phasor Analysis](#flimkitphasor--phasor-analysis)
   - [flimkit.image — Image Utilities](#flimkitimage--image-utilities)
   - [flimkit.LIF — LIF Format Support](#flimkitlif--lif-format-support)
   - [flimkit.utils — Shared Utilities](#flimkitutils--shared-utilities)
7. [Project Structure](#project-structure)
8. [Testing](#testing)
9. [Outputs & File Formats](#outputs--file-formats)
10. [Troubleshooting](#troubleshooting)
11. [Roadmap](#roadmap)
12. [Contact](#contact)

---

## Overview

**FLIMKit** is a Python toolkit for analysing FLIM data acquired on a Leica SP8 (or compatible PTU-based systems). It provides two complementary analysis workflows:

| Workflow | Description |
|---|---|
| **Reconvolution fitting** | Mono/bi/tri-exponential lifetime fitting with IRF deconvolution, per-pixel and summed modes, and multi-tile ROI stitching. |
| **Phasor analysis** | Calibrated phasor plots with interactive elliptical cursors, automatic peak detection, two-component decomposition, and session save/load. |

Both workflows can be launched through:
- A **guided terminal UI** (`main.py`)
- **Standalone CLI scripts** (`fit_cli.py`, `phasor_cli.py`)
- **Programmatic Python API** (import `flimkit`)

---

## Requirements & Installation

### System Requirements

- **Python ≥ 3.11**
- macOS, Linux, or Windows

### Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array computation |
| `scipy` | Optimisers (Levenberg–Marquardt, Differential Evolution), signal processing |
| `matplotlib` | Plotting (decay curves, lifetime maps, phasor plots) |
| `xarray` | Labelled N-D arrays for FLIM signals |
| `phasorpy` (0.9) | Phasor computation, calibration, cursor masking, lifetime conversion |
| `ptufile` | Low-level PTU file reading |
| `inquirer` | Interactive terminal prompts |
| `ipywidgets` + `ipympl` | Jupyter notebook interactive support |
| `opencv-python` | Cell masking and image processing |
| `pandas` | Excel/XLSX IRF file parsing |
| `tifffile` | TIFF image I/O |
| `tqdm` | Progress bars |
| `colorama` | Coloured terminal output |

### Installation

```bash
git clone https://github.com/alex1075/FLIMKit.git
cd FLIMKit
pip install -r requirements.txt
```

### Validate Installation

```bash
python validate_installation.py
```

This runs 8 checks covering dependencies, module imports, XLIF parsing, stitching, fitting, and the phasor pipeline. All checks should pass.

---

## Quick Start

### 1. Interactive Mode (Recommended for New Users)

```bash
python main.py
```

This presents a menu:
1. **FLIM FIT a single FOV** — fit one PTU file
2. **Phasor analysis** — calibrated phasor plots
3. **Reconstruct a FOV and FLIM FIT** — stitch tiles + fit
4. **Just stitch multiple tiles together** — tile stitching only
5. **About** — version info and roadmap

### 2. Direct CLI

```bash
# Fit a PTU file with 2-exponential model
python fit_cli.py --ptu data.ptu --irf-xlsx irf.xlsx --nexp 2

# Phasor analysis
python phasor_cli.py --ptu data.ptu --irf irf.xlsx
```

Before routine interactive fitting, create a machine IRF once for your system/session (see Machine IRF Setup below).

### 3. Python API

```python
from flimkit.phasor_launcher import launch_phasor
state = launch_phasor('data.ptu', irf_path='irf.xlsx')
```

---

## Workflows

### Guided Terminal UI (`main.py`)

The main entry point launches an interactive menu using `inquirer`. Each workflow prompts for all necessary inputs (file paths, parameters, IRF method, etc.) with sensible defaults.

```bash
python main.py
```

**Menu options:**

| Option | Description |
|---|---|
| FLIM FIT a single FOV | Loads a single PTU file, builds an IRF, runs summed and/or per-pixel reconvolution fitting |
| Phasor analysis | Opens the interactive phasor cursor tool |
| Reconstruct a FOV and FLIM FIT | Stitches multi-tile PTU data from XLIF metadata, then runs fitting on the mosaic. The ROI name is extracted from the XLIF filename (e.g., `R 2.xlif` → `R_2`) and used to name the output subdirectory and all exported files. |
| Just stitch multiple tiles together | Produces intensity images and FLIM histogram cubes without fitting. Outputs are placed in a subdirectory named after the ROI. |

---

### Machine IRF Setup (Required)

For reconvolution fitting in interactive and GUI workflows, create a machine IRF first and reuse it across files from the same setup.

#### GUI method (recommended)

1. Launch the GUI:

```bash
python gui.py
```

2. Open the `Machine IRF Builder` tab.
3. Select a folder containing matched `<name>.ptu` and `<name>.xlsx` pairs.
4. Choose anchor/reducer and build the IRF.
5. Save as `machine_irf_default` into `flimkit/machine_irf/`.

#### Python API method

```python
from flimkit.FLIM.irf_tools import build_machine_irf_from_folder

result = build_machine_irf_from_folder(
    folder="/path/to/pairs",
    align_anchor="peak",
    reducer="median",
    save=True,
    confirm_save=True,
    output_name="machine_irf_default",
)
```

The default runtime path is `flimkit/machine_irf/machine_irf_default.npy`.

#### Minimum Pair Count by Target Quality

From the notebook subsampling run:
- The simple 10%-of-N=20 plateau rule reported minimum practical N = 4.
- That rule used only weighted lifetime MAE and is noisy/non-monotonic across random splits.
- Chi-squared stability improves with larger N, with variance dropping clearly at N=18-20.

Practical recommendation:
- Peak-placement rule learning only: 4-6 pairs can work.
- Stable machine IRF shape plus placement across conditions: use at least 10-12 pairs.
- Robust production behavior across objectives/samples: target 15-20 pairs.

Operational default: avoid fewer than about 10 pairs unless your data are highly homogeneous.

---

### FLIM Reconvolution Fitting (CLI)

```bash
python fit_cli.py [OPTIONS]
```

#### Required Arguments

| Argument | Description |
|---|---|
| `--ptu PATH` | Path to the PTU file to analyse |

#### IRF Arguments

| Argument | Description |
|---|---|
| `--irf PATH` | Scatter PTU file for a directly measured IRF (highest priority) |
| `--irf-xlsx PATH` | LAS X Excel export for analytical IRF fitting (recommended) |
| `--xlsx PATH` | LAS X export XLSX for overlay comparison and optional IRF source |
| `--no-xlsx-irf` | Load XLSX for comparison only; do not use its IRF |
| `--estimate-irf {raw,parametric,none}` | Estimate IRF from the decay rising edge (default: `none`) |
| `--irf-fwhm FLOAT` | IRF FWHM in ns (default: 1 bin width from PTU) |
| `--irf-bins INT` | Number of bins for the IRF (default: 21) |
| `--irf-fit-width FLOAT` | Width of the region around time zero for IRF fitting, in ns (default: 1.5) |

#### Fitting Arguments

| Argument | Description |
|---|---|
| `--nexp {1,2,3}` | Number of exponential components (default: 3) |
| `--tau-min FLOAT` | Minimum lifetime bound in ns (default: 0.145) |
| `--tau-max FLOAT` | Maximum lifetime bound in ns (default: 45.0) |
| `--mode {summed,perPixel,both}` | Fitting mode (default: `both`) |
| `--binning INT` | Spatial binning factor for per-pixel fitting (default: 1) |
| `--min-photons INT` | Minimum photons per pixel for per-pixel fitting (default: 10) |
| `--optimizer {lm_multistart,de}` | Optimiser for summed fitting (default: `de`) |
| `--restarts INT` | Number of restarts for LM optimiser (default: 8) |
| `--de-population INT` | DE population size (default: 30) |
| `--de-maxiter INT` | DE maximum iterations (default: 5000) |
| `--workers INT` | Number of CPU cores for DE (-1 = all) |
| `--no-polish` | Skip polishing step after DE optimisation |
| `--cost-function {poisson,chi2}` | Cost function for summed fit (default: `poisson`) |
| `--intensity-threshold INT\|interactive` | Min photons per pixel; pass `interactive` for visual slider |
| `--tau-display-min FLOAT` | Min lifetime (ns) for exported tau images; clips to this value (LAS X style) |
| `--tau-display-max FLOAT` | Max lifetime (ns) for exported tau images; clips to this value (LAS X style) |
| `--intensity-display-min FLOAT` | Min intensity for exported intensity images; clips to this value |
| `--intensity-display-max FLOAT` | Max intensity for exported intensity images; clips to this value |

#### Output Arguments

| Argument | Description |
|---|---|
| `--out NAME` | Output directory name (default: `flim_out`) |
| `--channel INT` | Detection channel (default: auto-detect) |
| `--no-plots` | Suppress plot generation |
| `--print-config` | Print default configuration and exit |
| `--debug-xlsx` | Print raw XLSX contents for debugging |

#### IRF Priority Order

When multiple IRF sources are available, the following priority applies:

1. `--irf` (scatter PTU) — fully measured, no tail or sigma needed
2. `--irf-xlsx` (analytical fit to LAS X export) — Gaussian + exponential tail model
3. `--xlsx` IRF columns (unless `--no-xlsx-irf`) — sparse rising edge, tail and sigma as free parameters
4. `--estimate-irf raw` — non-parametric IRF from raw decay rising edge
5. `--estimate-irf parametric` — Gaussian + exponential tail fit to rising edge
6. Gaussian IRF from FWHM (fallback)

> **Tip:** For best results, export an IRF from the LAS X FLIM tail-fit graph (right-click → export) and pass it with `--irf-xlsx`. The IRF is FOV- and ROI-specific, so you should export a separate IRF for each acquisition.

---

### Phasor Analysis (CLI)

```bash
python phasor_cli.py [OPTIONS]
```

| Argument | Description |
|---|---|
| `--ptu PATH` | Path to a `.ptu` file (skips first prompt) |
| `--irf PATH` | Path to IRF calibration Excel file |
| `--session PATH` | Resume a previously saved `.npz` session |

When called with no arguments, the CLI enters a fully guided `inquirer` flow.

#### Interactive Phasor Features

- **Click-to-place** elliptical cursors with adjustable size and orientation
- **Per-cursor apparent lifetime** (τ_φ) maps
- **Two-component decomposition** via semicircle intersection (first two cursors)
- **Peaks button** — automatic peak detection on the phasor histogram
- **Export button** — save the figure (with all ROIs drawn) as PNG/PDF/SVG
- **Save button** — persist arrays + cursor state to `.npz` for later editing

---

### Python API

#### Phasor Analysis

```python
from flimkit.phasor_launcher import launch_phasor, save_session, load_session

# Interactive prompts
state = launch_phasor()

# Pass paths directly
state = launch_phasor('data.ptu', irf_path='irf.xlsx')

# Resume a saved session
state = launch_phasor(session_path='session.npz')

# Save/load sessions programmatically
save_session('my_session.npz',
             real_cal=state['real_cal'],
             imag_cal=state['imag_cal'],
             mean=state['mean'],
             frequency=state['frequency'],
             cursors=state['cursors'],
             params=state['params'])

sess = load_session('my_session.npz')
```

#### PTU File Reading

```python
from flimkit.PTU.reader import PTUFile

ptu = PTUFile('data.ptu', verbose=True)

# Get summed decay histogram
decay = ptu.summed_decay(channel=None)  # auto-detect channel

# Get per-pixel stack (Y, X, H)
stack = ptu.pixel_stack(channel=None, binning=1)

# Access metadata
print(ptu.n_bins)      # number of TCSPC bins
print(ptu.tcspc_res)   # bin width in seconds
print(ptu.time_ns)     # time axis in nanoseconds
```

#### Signal Extraction (xarray)

```python
from flimkit.PTU.tools import signal_from_PTUFile
import numpy as np

signal = signal_from_PTUFile('data.ptu', dtype=np.uint32, binning=4)
# Returns an xarray DataArray with labelled dimensions
# signal.attrs['frequency'] contains the modulation frequency in MHz
```

#### Phasor Computation

```python
from flimkit.phasor.signal import (
    return_phasor_from_PTUFile,
    get_phasor_irf,
    calibrate_signal_with_irf
)

# Direct phasor from PTU
mean, real, imag = return_phasor_from_PTUFile('data.ptu')

# With IRF calibration
irf_time_ns, irf_counts = get_phasor_irf('irf.xlsx')
real_cal, imag_cal = calibrate_signal_with_irf(
    signal, real, imag, irf_time_ns, irf_counts, frequency
)
```

#### Phasor Peak Detection

```python
from flimkit.phasor.peaks import find_phasor_peaks

peaks = find_phasor_peaks(
    real_cal, imag_cal, mean, frequency,
    min_photons=0.01,
    n_bins=256,
    sigma=3.0,
    neighbourhood=15,
    threshold_frac=0.10,
)
```

#### Intensity Images & Cell Masking

```python
from flimkit.image.tools import (
    make_intensity_image,
    make_cell_mask,
    apply_intensity_threshold,
    pick_intensity_threshold
)

# Create intensity image from PTU
intensity = make_intensity_image('data.ptu', rotate_90_cw=True, save_image=True)

# Generate cell mask (Otsu thresholding + morphological cleanup)
mask = make_cell_mask(intensity, save_mask=True, path='output/')

# Apply photon-count threshold
int_mask = apply_intensity_threshold(intensity, threshold=50)

# Interactive threshold selection (slider)
threshold = pick_intensity_threshold(intensity)
```

#### Tile Stitching

```python
from flimkit.PTU.stitch import stitch_flim_tiles, load_flim_for_fitting
from pathlib import Path

# Stitch tiles
result = stitch_flim_tiles(
    xlif_path=Path('metadata/R 2.xlif'),
    ptu_dir=Path('PTU_tiles/'),
    output_dir=Path('stitched/R_2/'),
    ptu_basename='R 2',
    rotate_tiles=True,
)

# Load stitched data for fitting
stack, tcspc_res, n_bins = load_flim_for_fitting(
    Path('stitched/R_2/'),
    load_to_memory=True
)
decay = stack.sum(axis=(0, 1))  # summed decay from mosaic
```

#### XLIF Metadata Parsing

```python
from flimkit.utils.xml_utils import (
    parse_xlif_tile_positions,
    get_pixel_size_from_xlif,
    compute_tile_pixel_positions
)

tiles = parse_xlif_tile_positions('R 2.xlif', ptu_basename='R 2')
pixel_size_m, n_pixels = get_pixel_size_from_xlif('R 2.xlif')
tiles, width, height = compute_tile_pixel_positions(
    tiles, pixel_size_m=pixel_size_m, tile_size=n_pixels
)
```

---

## Configuration Reference

Default fitting parameters are defined in `flimkit/configs.py`. All can be overridden via CLI arguments.

### Fitting Defaults

| Parameter | Default | Description |
|---|---|---|
| `INTENSITY_THRESHOLD` | `None` | Minimum photon count per pixel (disabled by default) |
| `Tau_min` | 0.145 ns | Lower lifetime bound (avoids fitting the IRF peak) |
| `Tau_max` | 45.0 ns | Upper lifetime bound |
| `D_mode` | `'both'` | Fitting mode: `'summed'`, `'perPixel'`, or `'both'` |
| `n_exp` | 3 | Number of exponential components (1, 2, or 3) |
| `binning_factor` | 1 | Spatial binning for per-pixel fitting |
| `Optimizer` | `'de'` | Optimiser: `'de'` (Differential Evolution) or `'lm_multistart'` |
| `Cost_function` | `'poisson'` | Cost function: `'poisson'` (recommended) or `'chi2'` (legacy) |

### Optimiser Settings

| Parameter | Default | Description |
|---|---|---|
| `lm_restarts` | 8 | Levenberg–Marquardt multi-start restarts |
| `de_population` | 30 | DE population size |
| `de_maxiter` | 5000 | DE maximum iterations |
| `n_workers` | -1 | CPU cores for DE (-1 = all available) |

### Machine IRF Settings

| Parameter | Default | Description |
|---|---|---|
| `MACHINE_IRF_DIR` | `flimkit/machine_irf` | Directory containing saved machine IRF artifacts |
| `MACHINE_IRF_DEFAULT_PATH` | `flimkit/machine_irf/machine_irf_default.npy` | Default machine IRF file used at runtime |
| `MACHINE_IRF_ALIGN_ANCHOR` | `'peak'` | Landmark used during machine IRF construction |
| `MACHINE_IRF_REDUCER` | `'median'` | Aggregation mode used during machine IRF construction |
| `MACHINE_IRF_FIT_STRATEGY` | `'fixed'` | Runtime strategy selected for machine IRF fitting |
| `MACHINE_IRF_FIT_BG` | `True` | Fit background offset when using machine IRF |
| `MACHINE_IRF_FIT_SIGMA` | `False` | Fit additional Gaussian broadening when using machine IRF |
| `MACHINE_IRF_FIT_TAIL` | `False` | Fit exponential tail when using machine IRF |
| `MACHINE_IRF_DE_POPULATION` | 30 | DE population for machine IRF strategy |
| `MACHINE_IRF_DE_MAXITER` | 5000 | DE max iterations for machine IRF strategy |

### IRF Settings

| Parameter | Default | Description |
|---|---|---|
| `IRF_FWHM` | `None` | IRF FWHM in ns (`None` = 1 bin width from PTU file) |
| `IRF_FIT_WIDTH` | 1.5 ns | Width of region around time zero for IRF fitting |
| `IRF_BINS` | 21 | Number of bins for the IRF (should be odd) |
| `Estimate_IRF` | `'none'` | IRF estimation: `'raw'`, `'parametric'`, or `'none'` |

### Other Settings

| Parameter | Default | Description |
|---|---|---|
| `MIN_PHOTONS_PERPIX` | 10 | Minimum photons for per-pixel fitting |
| `channels` | `None` | Channels to fit (`None` = all) |
| `OUT_NAME` | `'flim_out'` | Default output directory name |

### Display Range Settings

These control how exported tau and intensity images are scaled. Out-of-range pixel values are **clipped to the nearest boundary**, matching the behaviour of Leica LAS X (rather than being zeroed).

| Parameter | Default | Description |
|---|---|---|
| `TAU_DISPLAY_MIN` | `None` | Minimum lifetime (ns) for weighted-tau images (`None` = no clip) |
| `TAU_DISPLAY_MAX` | `None` | Maximum lifetime (ns) for weighted-tau images (`None` = no clip) |
| `INTENSITY_DISPLAY_MIN` | `None` | Minimum photon count for intensity images (`None` = no clip) |
| `INTENSITY_DISPLAY_MAX` | `None` | Maximum photon count for intensity images (`None` = no clip) |

> **Tip — Leica LAS X behaviour:** When a display range is set, pixels with lifetime or
> intensity outside the range are clamped to the boundary value, not discarded. This
> preserves spatial information while confining the colour scale to a biologically
> relevant window (e.g. 0–5 ns for most fluorophores).

### Cost Functions

| Function | Description |
|---|---|
| `poisson` | Poisson deviance (C-statistic) on raw counts. Statistically correct, especially for low-count bins. Recommended. |
| `chi2` | Neyman chi-squared (legacy). Normalises by peak and uses Neyman weights. Underweights the tail region. |

### FLIM Colourmap

A custom sequential colourmap (`FLIM_CMAP`) is defined for lifetime images:
navy → blue → cyan → green → yellow → red

---

## Module Reference

### `flimkit.PTU` — PTU File I/O

#### `reader.py`
- **`PTUFile`** — Main class for reading PicoQuant PTU files. Parses T3 mode records, extracts TCSPC metadata (bin width, number of bins, image dimensions), and provides methods for summed decays and per-pixel histograms.
  - `PTUFile(path, verbose=False)` — Open and parse a PTU file
  - `.summed_decay(channel=None)` — Sum all photons into a single decay histogram
  - `.pixel_stack(channel=None, binning=1)` — Build a (Y, X, H) histogram stack
  - `.raw_pixel_stack(channel=None)` — Overflow-corrected pixel stack
  - `.n_bins`, `.tcspc_res`, `.time_ns` — TCSPC metadata
- **`PTUArray5D`** — Extended reader for multi-dimensional PTU data

#### `decode.py`
- Low-level T3 record decoding (PicoHarp, HydraHarp formats)
- `get_flim_histogram_from_ptufile()` — Extract FLIM histograms directly
- `create_time_axis()` — Build time axis from PTU metadata

#### `tools.py`
- **`signal_from_PTUFile(path, dtype, binning)`** — Load a PTU file and return an `xarray.DataArray` with labelled dimensions (`Y`, `X`, `H`) and metadata (`frequency`)
- **`filter_photons_with_mask(ptu_path, mask, ...)`** — Filter photons using a spatial mask, keeping only photons where `mask[y, x] == 0`

#### `stitch.py`
- **`stitch_flim_tiles(xlif_path, ptu_dir, output_dir, ...)`** — Stitch multi-tile PTU data into a single mosaic using XLIF metadata. Produces intensity TIFF, FLIM histogram cube (NPY), time axis, weight map, and JSON metadata.
- **`load_flim_for_fitting(output_dir, load_to_memory)`** — Load previously stitched data for fitting

---

### `flimkit.FLIM` — Reconvolution Fitting

#### `models.py`
- **`reconvolution_model(params, ...)`** — Compute a model decay curve by convolving exponential components with the IRF
- **`_DECost`** / **`_DECostLogTau`** — Chi-squared cost function classes for Differential Evolution (linear and log-tau parameterisation)
- **`_DECostPoisson`** / **`_DECostPoissonLogTau`** — Poisson deviance (C-statistic) cost function classes. Works on raw photon counts for statistically correct fits.

#### `fitters.py`
- **`fit_summed(decay, tcspc_res, n_bins, irf_prompt, ...)`** — Fit a summed FLIM decay via reconvolution. Supports both Poisson deviance and Neyman chi-squared cost functions, Differential Evolution and Levenberg–Marquardt optimisers.
  - Returns `(best_params, summary_dict)` where `summary_dict` contains the fitted model, residuals, fit window, χ² values, and component lifetimes/amplitudes.
- **`fit_per_pixel(stack, tcspc_res, n_bins, irf_prompt, ...)`** — Per-pixel lifetime fitting using non-negative least squares (NNLS) with the global fit as template. Much faster than full per-pixel optimisation.

#### `fit_tools.py`
- **`find_irf_peak_bin(decay)`** — Locate the IRF peak as the steepest rise in the summed decay (not the fluorescence peak, which is shifted right)
- **`estimate_bg(decay, peak_bin)`** — Estimate background level from the pre-peak region
- **`find_fit_end(decay, peak_bin, tau_max, tcspc_res, n_bins)`** — Determine the end of the fitting window
- **`_build_bounds()`** / **`_pack_p0()`** — Construct parameter bounds and initial guesses

#### `irf_tools.py`
- **`gaussian_irf_from_fwhm(n_bins, tcspc_res, fwhm_ns, peak_bin)`** — Generate a Gaussian IRF from FWHM
- **`irf_from_scatter_ptu(path, ptu_ref)`** — Load a scatter/reflection PTU as a measured IRF
- **`irf_from_xlsx(xlsx, n_bins, tcspc_res)`** — Extract IRF from LAS X XLSX export
- **`irf_from_xlsx_analytical(xlsx, n_bins, tcspc_res)`** — Fit the Leica analytical IRF model (Gaussian + exponential tail) to XLSX data. Returns the normalised IRF and fitted parameters (t0, FWHM, tail amplitude, tail decay time).
- **`estimate_irf_from_decay_raw(decay, ...)`** — Non-parametric IRF estimation from the decay rising edge
- **`estimate_irf_from_decay_parametric(decay, ...)`** — Parametric IRF estimation (Gaussian + exponential tail fit)
- **`compare_irfs(irf1, irf2, ...)`** — Visual comparison of two IRF estimates
- **`build_full_irf(irf_prompt, ...)`** — Construct the complete IRF with optional broadening and tail
- **`discover_ptu_xlsx_pairs(folder)`** — Discover paired `<name>.ptu` and `<name>.xlsx` files for machine IRF construction
- **`build_machine_irf_from_folder(folder, ...)`** — Build and optionally save a machine IRF (`.npy/.csv/.json`) from paired PTU/XLSX data

---

### `flimkit.phasor` — Phasor Analysis

#### `signal.py`
- **`return_phasor_from_PTUFile(ptu_file)`** — Read a PTU file and compute phasor coordinates (mean, real, imaginary)
- **`get_phasor_irf(irf_xlsx)`** — Read IRF data from an Excel file (time + counts arrays)
- **`calibrate_signal_with_irf(signal, real, imag, irf_time_ns, irf_counts, frequency)`** — Calibrate phasor coordinates using the IRF. Computes the IRF phasor and applies phase/modulation correction via `phasorpy.phasor.phasor_transform`.

#### `interactive.py`
- **`phasor_cursor_tool(real_cal, imag_cal, mean, frequency, ...)`** — Launch the interactive phasor cursor widget. Works in both Jupyter notebooks (ipywidgets) and standalone scripts (matplotlib.widgets). Features:
  - Click to place up to 6 elliptical cursors
  - Adjustable ellipse radius, minor radius, and angle mode (`semicircle` or `origin`)
  - Per-cursor apparent lifetime maps (τ_φ, τ_m)
  - Two-component decomposition via semicircle intersection (first two cursors)
  - Undo, Peaks, Export, and Save buttons

#### `peaks.py`
- **`find_phasor_peaks(real_cal, imag_cal, mean, frequency, ...)`** — Automatic peak detection on 2-D phasor histograms. Smooths the (G, S) histogram with a Gaussian kernel, detects local maxima, and converts them to apparent phase/modulation lifetimes.
  - Parameters: `min_photons`, `n_bins`, `sigma`, `neighbourhood`, `threshold_frac`

---

### `flimkit.image` — Image Utilities

#### `tools.py`
- **`make_intensity_image(ptu_path, rotate_90_cw=True, save_image=False)`** — Create a 2-D intensity image by summing all time bins from a PTU file
- **`make_cell_mask(intensity_image, save_mask=False, path=None)`** — Generate a binary cell mask from an intensity image using Otsu thresholding and morphological cleanup (OpenCV)
- **`apply_intensity_threshold(intensity_image, threshold)`** — Create a boolean mask excluding pixels below a photon-count threshold
- **`pick_intensity_threshold(intensity_image)`** — Interactive slider for choosing an intensity threshold visually

#### `stitch.py`
- **`stitch_flim_tiles(...)`** — Stitch FLIM PTU tiles into a single mosaic using XLIF metadata (alternative implementation using the `decode` module directly)

---

### `flimkit.LIF` — LIF Format Support

#### `utils.py`
- LIF metadata helpers for Leica image formats

---

### `flimkit.utils` — Shared Utilities

#### `plotting.py`
- **`plot_summed(decay, summary, ptu, xlsx, n_exp, strategy, out_prefix, irf_prompt)`** — Generate the main summed-fit figure: log-scale decay + model overlay, weighted residuals, residual histogram, and parameter table
- **`plot_pixel_maps(stack, summary, ptu, ...)`** — Per-pixel lifetime and amplitude maps using the FLIM colourmap
- **`plot_lifetime_histogram(summary, ...)`** — Lifetime distribution histograms

#### `enhanced_outputs.py`
- **`save_fit_summary_txt(summary, output_path, n_exp, strategy, metadata)`** — Save fit results to a human-readable text file
- **`save_weighted_tau_images(pixel_maps, output_dir, roi_name, n_exp, ...)`** — Save intensity-weighted and amplitude-weighted τ images as TIFFs. Accepts `tau_display_min`, `tau_display_max`, `intensity_display_min`, and `intensity_display_max` to clip out-of-range pixels (Leica LAS X style).
- **`save_individual_tau_maps(summary, output_path, ...)`** — Save individual component maps (τ₁, τ₂, a₁, a₂, etc.)
- **`create_complete_output_package(summary, output_dir, ...)`** — Generate a complete output package with all plots, images, and text summaries

#### `xlsx_tools.py`
- **`load_xlsx(path, debug=False)`** — Parse a LAS X FLIM export XLSX file. Automatically detects column layout and extracts decay, IRF, fit, and residual data. Returns a dict with keys: `decay_t`, `decay_c`, `irf_t`, `irf_c`, `fit_t`, `fit_c`, `res_t`, `res_c`.

#### `xml_utils.py`
- **`parse_xlif_tile_positions(xlif_path, ptu_basename)`** — Parse tile positions from a Leica XLIF file. Returns a list of dicts with `file`, `field_x`, `pos_x`, `pos_y` (positions in microns).
- **`get_pixel_size_from_xlif(xlif_path)`** — Extract pixel size (meters) and pixel count from the XLIF `DimensionDescription`
- **`compute_tile_pixel_positions(tiles, pixel_size_m, tile_size)`** — Convert physical tile positions to pixel coordinates and compute the canvas size

#### `misc.py`
- **`print_summary(summary, ...)`** — Print fit results to the terminal
- **`setup_loggers(output_dir)`** — Configure logging for stitching/fitting runs

#### `fancy.py`
- **`display_banner()`** — Show the FLIMKit ASCII art banner
- **`flim_fitting_banner()`** — Show the FLIM fitting sub-banner
- **`banner_goodbye()`** — Exit message

---

## Project Structure

```
├── main.py                        # Main menu — guided terminal UI
├── fit_cli.py                     # FLIM reconvolution fitting CLI
├── phasor_cli.py                  # Phasor analysis CLI
├── validate_installation.py       # Installation validation script
├── requirements.txt               # Python dependencies
│
├── flimkit/                        # ── Core library ──────────────────────
│   ├── __init__.py                # Top-level exports (launch_phasor, etc.)
│   ├── _version.py                # Version string & roadmap
│   ├── configs.py                 # Default fitting parameters
│   ├── interactive.py             # Guided FLIM fitting launcher (inquirer)
│   ├── phasor_launcher.py         # Guided phasor analysis launcher
│   │
│   ├── PTU/                       # PTU file I/O
│   │   ├── reader.py              # PTUFile / PTUArray5D classes
│   │   ├── decode.py              # Low-level T3 record decoding
│   │   ├── tools.py               # signal_from_PTUFile (→ xarray)
│   │   └── stitch.py              # Multi-tile PTU stitching
│   │
│   ├── FLIM/                      # Reconvolution fitting
│   │   ├── models.py              # Exponential decay models + cost functions
│   │   ├── fitters.py             # fit_summed / fit_per_pixel
│   │   ├── fit_tools.py           # IRF alignment, bin utilities
│   │   └── irf_tools.py           # IRF extraction & estimation
│   │
│   ├── phasor/                    # Phasor analysis
│   │   ├── signal.py              # Phasor computation & IRF calibration
│   │   ├── interactive.py         # Interactive cursor tool
│   │   └── peaks.py               # Automatic peak detection
│   │
│   ├── image/                     # Image utilities
│   │   ├── tools.py               # Intensity images, cell masking
│   │   └── stitch.py              # Tile image stitching
│   │
│   ├── LIF/                       # LIF format support
│   │   └── utils.py               # LIF metadata helpers
│   │
│   └── utils/                     # Shared utilities
│       ├── plotting.py            # Summed-fit plots, pixel maps, histograms
│       ├── enhanced_outputs.py    # Summary text, weighted-τ images, exports
│       ├── xlsx_tools.py          # LAS X Excel file parsing
│       ├── xml_utils.py           # XLIF tile-position parsing
│       ├── misc.py                # Print helpers, logging
│       └── fancy.py               # Terminal banners & ASCII art
│
└── flimkit_tests/                  # Test suite
    ├── pytest.ini
    ├── requirements_test.txt
    ├── run_tests.py               # Test runner script
    ├── mock_data.py               # Synthetic data generators
    ├── conftest.py                # Shared fixtures
    ├── test_complete_pipeline.py  # End-to-end pipeline test
    ├── TESTING.md                 # Test documentation
    └── tests/
        ├── test_decode.py         # PTU decoding tests
        ├── test_integration.py    # Integration tests
        └── test_xml_utils.py      # XLIF parsing tests
```

---

## Testing

### Install Test Dependencies

```bash
pip install -r flimkit_tests/requirements_test.txt
```

### Run Tests

```bash
cd flimkit_tests
python run_tests.py              # All tests
python run_tests.py -c           # With coverage report
python run_tests.py integration  # Integration tests only
```

### Run Individual Test Modules

```bash
pytest tests/test_xml_utils.py -v       # XLIF parsing
pytest tests/test_decode.py -v          # PTU decoding
pytest tests/test_integration.py -v     # Integration
```

### Test Coverage

| Area | What's tested |
|---|---|
| XML/XLIF parsing | Tile positions, metadata extraction |
| PTU decoding | Histogram extraction, time axis creation |
| Tile stitching | Canvas computation, overlap handling |
| Integration | Complete workflows, error handling |
| Memory efficiency | Memmap usage, large datasets |

---

## Outputs & File Formats

### FLIM Fitting Outputs

When a fit completes, the output directory (default: `flim_out/`) contains:

| File | Description |
|---|---|
| `*_summed_fit.png` | Summed decay plot with model overlay, residuals, and parameters |
| `*_pixel_maps.png` | Per-pixel lifetime and amplitude maps |
| `*_lifetime_hist.png` | Lifetime distribution histogram |
| `*_fit_summary.txt` | Human-readable fit results text file |
| `*_weighted_tau.tif` | Intensity-weighted τ image (TIFF) |
| `*_tau1_map.tif`, etc. | Individual component maps (if requested) |

For the **Reconstruct + Fit** workflow, the ROI name is extracted from the XLIF filename (e.g., `R 2.xlif` → `R_2`) and used to:
- Create a dedicated output subdirectory (e.g., `output/R_2/`)
- Prefix all exported files (e.g., `R_2_fit_summary.txt`, `R_2_weighted_tau.tif`)

### Tile Stitching Outputs

Stitching outputs are also placed in the ROI-named subdirectory, with the ROI name prefixed to every file:

| File | Description |
|---|---|
| `{ROI}_stitched_intensity.tif` | Stitched intensity image (TIFF) |
| `{ROI}_stitched_flim_counts.npy` | FLIM histogram cube (Y × X × H) as NumPy memmap |
| `{ROI}_time_axis_ns.npy` | Time axis in nanoseconds |
| `{ROI}_weight_map.npy` | Tile overlap counts |
| `{ROI}_metadata.json` | Stitching metadata (tile count, canvas size, etc.) |

For example, with `R 2.xlif` the outputs would be `R_2_stitched_intensity.tif`, `R_2_metadata.json`, etc. The load functions (`load_stitched_flim`, `load_flim_for_fitting`) auto-detect prefixed filenames and also fall back to generic names for backward compatibility.

### Phasor Session Files

Phasor sessions are saved as `.npz` archives containing:

| Key | Description |
|---|---|
| `real_cal` | Calibrated G (real) phasor array |
| `imag_cal` | Calibrated S (imaginary) phasor array |
| `mean` | Mean intensity image |
| `frequency` | Modulation frequency (MHz) |
| `cursor_g`, `cursor_s` | Cursor centre coordinates |
| `cursor_colors` | Cursor colour strings |
| `param_radius`, `param_radius_minor`, `param_angle_mode` | Ellipse parameters |
| `ptu_file`, `irf_file` | Original source paths (metadata) |

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'phasorpy'` | Install dependencies: `pip install -r requirements.txt` |
| `ValueError: Not a PTU/PQTTTR file` | Ensure the file is a valid PicoQuant PTU file |
| `No TileScanInfo found in XLIF` | XLIF file may not contain tile metadata; verify it's from a tile scan acquisition |
| `FileNotFoundError: Machine IRF file not found` | Build a machine IRF first in the GUI `Machine IRF Builder` tab or with `build_machine_irf_from_folder(...)`, then point fitting to the saved `.npy` file |
| IRF fitting gives poor results | Try `--irf-xlsx` with a dedicated IRF export from LAS X |
| Per-pixel fitting is very slow | Increase `--binning` (e.g., 2 or 4) to reduce resolution, or reduce `--de-maxiter` |
| Phasor points are scattered | Check IRF calibration; uncalibrated data will not lie on the universal semicircle |
| `interactive` threshold picker not appearing | Ensure matplotlib backend supports interactive windows (not `Agg`) |

### Validating Your Setup

Run the validation script to check for common issues:

```bash
python validate_installation.py
```

If any checks fail, the script will report which components need attention.

---

## Roadmap

- [x] Single FOV fitting
- [x] Batch processing of multiple FOVs
- [x] Reconstruction of multi-tile ROIs
- [x] Phasor analysis with interactive cursors, peak detection, and session save/load
- [~] Fitting multi-tile ROIs and exporting tau-fitted images/data
- [~] Documentation and examples
- [ ] GUI development for easier use

---

## Contact

For questions, contact Alex Hunt at alexander.hunt@ed.ac.uk
