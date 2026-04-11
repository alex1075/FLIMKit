# FLIMKit Documentation

> **Version 1.0.0** — A Python toolkit for Fluorescence Lifetime Imaging Microscopy (FLIM)

> **Warning:** This project is in active development. Please cross-validate results with other software before drawing conclusions.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements & Installation](#requirements--installation)
3. [Quick Start](#quick-start)
4. [Workflows](#workflows)
   - [Desktop GUI](#desktop-gui)
   - [Guided Terminal UI](#guided-terminal-ui-mainpy)
   - [Machine IRF Setup](#machine-irf-setup-required)
   - [FLIM Reconvolution Fitting (CLI)](#flim-reconvolution-fitting-cli)
   - [Phasor Analysis (CLI)](#phasor-analysis-cli)
   - [Python API](#python-api)
5. [Configuration Reference](#configuration-reference)
6. [Module Reference](#module-reference)
7. [Project Structure](#project-structure)
8. [Compiled App](#compiled-app-macos--windows--linux)
9. [Testing](#testing)
10. [Outputs & File Formats](#outputs--file-formats)
11. [Troubleshooting](#troubleshooting)
12. [Roadmap](#roadmap)
13. [Contact](#contact)

---

## Overview

**FLIMKit** is a Python toolkit for analysing FLIM data acquired on a Leica SP8/FALCON (or any PTU-based system). It is designed as a drop-in replacement for Leica LAS X FLIM analysis and provides two complementary workflows:

| Workflow | Description |
|---|---|
| **Reconvolution fitting** | Mono/bi/tri-exponential lifetime fitting with full IRF deconvolution, per-pixel and summed modes, multi-tile ROI stitching, and batch processing. |
| **Phasor analysis** | Calibrated phasor plots with interactive elliptical cursors, automatic peak detection, two-component decomposition, and session save/load. |

Both workflows are accessible through a **desktop GUI**, a **guided terminal UI**, **CLI scripts**, or the **Python API**.

---

## Requirements & Installation

### System Requirements

- **Python ≥ 3.14**
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

Runs 9 checks covering dependencies, module imports, XLIF parsing, stitching, fitting, phasor pipeline, and per-tile fit pipeline. All checks should pass.

---

## Quick Start

### 1. Desktop GUI (Recommended)

```bash
python -m flimkit.UI.gui
```

### 2. Guided Terminal UI

```bash
python main.py
```

### 3. Direct CLI

```bash
# Fit a PTU file with a 2-exponential model
python fit_cli.py --ptu data.ptu --machine-irf machine_irf_default.npy --nexp 2

# Phasor analysis
python phasor_cli.py --ptu data.ptu --irf irf.xlsx
```

### 4. Python API

```python
from flimkit.phasor_launcher import launch_phasor
state = launch_phasor('data.ptu', irf_path='irf.xlsx')
```

Before routine fitting, build a machine IRF for your system once (see [Machine IRF Setup](#machine-irf-setup-required)).

---

## Workflows

### Desktop GUI

```bash
python main.py
```

The GUI provides five tabs. The right-hand panel shows the **FOV Preview** (intensity image + summed decay curve) for all fitting tabs, and switches automatically to the **Phasor panel** when the Phasor Analysis tab is active.

#### Tab overview

| Tab | Description |
|---|---|
| **Single FOV Fit** | Load a single PTU file, select IRF method, run summed and/or per-pixel reconvolution fitting. Export intensity and lifetime maps as PNG or OME-TIFF. Save fitting sessions as NPZ for later restoration. Output files are saved to the same directory as the PTU. |
| **Tile Stitch / Fit** | Select an XLIF file and PTU directory to stitch a multi-tile ROI and run the full fitting pipeline. Export stitched images and results as OME-TIFF and GeoJSON. Save sessions as NPZ. Three pipeline modes: stitch only, stitch + fit, or per-tile fit. |
| **Batch ROI Fit** | Point at an XLIF folder to process all ROIs sequentially. Export fit summaries as CSV and ROI geometries as GeoJSON for QuPath integration. |
| **Machine IRF Builder** | Select a folder of matched PTU/XLSX pairs to build and save a machine IRF for reuse across sessions. |
| **Phasor Analysis** | Load a PTU file (or resume a saved `.npz` session). An embedded phasor histogram and intensity image update live as elliptic cursors are placed. Save sessions for later analysis. |

#### Phasor panel controls

When the Phasor Analysis tab is active, the right panel shows:
- **Top:** FOV intensity image (colourised with cursor selections once cursors are placed)
- **Bottom:** Phasor histogram (click to place an elliptic cursor)

Controls above the figure:
- **Clear all / Undo** — remove all cursors or step back one at a time
- **Save session** — persist phasor arrays + cursor state to `.npz`
- **Radius / Minor:major sliders** — resize the elliptic cursor in real time; analysis updates immediately

Per-cursor statistics (phase lifetime τ_φ, pixel count, 5th–95th percentile range) are printed to the Progress log. When two or more cursors are placed, a two-component decomposition line is drawn and the component lifetimes and mean fraction are reported.

---

### Guided Terminal UI (`main.py`)

```bash
python main.py --cli
```

| Option | Description |
|---|---|
| FLIM FIT a single FOV | Loads a PTU file, builds an IRF, runs summed and/or per-pixel fitting |
| Phasor analysis | Opens the interactive phasor cursor tool |
| Reconstruct a FOV and FLIM FIT | Stitches multi-tile PTU data from XLIF metadata then fits the mosaic |
| Just stitch multiple tiles together | Tile stitching only; produces intensity images and FLIM histogram cubes |
| About | Version info and roadmap |

---

### Machine IRF Setup (Required)

For accurate reconvolution fitting, build a machine IRF once per microscope/session configuration and reuse it.

#### GUI method (recommended)

1. Launch the GUI and open the **Machine IRF Builder** tab.
2. Select a folder containing matched `<name>.ptu` and `<name>.xlsx` pairs.
3. Choose anchor (`peak`) and reducer (`median`).
4. Build and save as `machine_irf_default`.

**Save locations:**

| Context | Save location |
|---|---|
| Running from source | `flimkit/machine_irf/` |
| Compiled app (macOS/Linux) | `~/.flimkit/machine_irf/` |
| Compiled app (Windows) | `C:\Users\<name>\.flimkit\machine_irf\` |

The directory is created automatically. After saving a new machine IRF, restart the app so it is picked up as the default.

#### Python API method

```python
from flimkit.FLIM.irf_tools import build_machine_irf_from_folder

build_machine_irf_from_folder(
    folder="/path/to/pairs",
    align_anchor="peak",
    reducer="median",
    save=True,
    output_name="machine_irf_default",
)
```

#### Minimum pair count guidance

From subsampling analysis:

| Goal | Minimum pairs |
|---|---|
| Peak-placement rule only | 4–6 |
| Stable IRF shape + placement | 10–12 |
| Robust production use (mixed objectives/samples) | 15–20 |

Practical default: avoid fewer than 10 pairs unless your dataset is highly homogeneous.

---

### FLIM Reconvolution Fitting (CLI)

```bash
python fit_cli.py [OPTIONS]
```

#### Required

| Argument | Description |
|---|---|
| `--ptu PATH` | Path to the PTU file |

#### IRF Arguments

| Argument | Description |
|---|---|
| `--machine-irf PATH` | Pre-built machine IRF `.npy` file (recommended) |
| `--irf PATH` | Scatter PTU for a directly measured IRF |
| `--irf-xlsx PATH` | LAS X Excel export for analytical IRF fitting |
| `--xlsx PATH` | LAS X export XLSX for comparison |
| `--no-xlsx-irf` | Use XLSX for comparison only; do not use its IRF |
| `--estimate-irf {raw,parametric,none}` | Estimate IRF from decay rising edge (default: `none`) |
| `--irf-fwhm FLOAT` | IRF FWHM in ns |
| `--irf-bins INT` | Number of bins for the IRF (default: 21) |
| `--irf-fit-width FLOAT` | Region around time zero for IRF fitting in ns (default: 1.5) |

**IRF priority order (highest to lowest):**
1. `--machine-irf` (pre-built, peak-aligned to decay)
2. `--irf` (scatter PTU — directly measured)
3. `--irf-xlsx` (analytical fit to LAS X export)
4. `--xlsx` IRF columns (unless `--no-xlsx-irf`)
5. `--estimate-irf raw` / `parametric`
6. Gaussian fallback from FWHM

#### Fitting Arguments

| Argument | Description |
|---|---|
| `--nexp {1,2,3}` | Number of exponential components (default: 3) |
| `--tau-min FLOAT` | Minimum lifetime bound in ns (default: 0.145) |
| `--tau-max FLOAT` | Maximum lifetime bound in ns (default: 45.0) |
| `--mode {summed,perPixel,both}` | Fitting mode (default: `both`) |
| `--binning INT` | Spatial binning for per-pixel fitting (default: 1) |
| `--min-photons INT` | Minimum photons per pixel (default: 10) |
| `--optimizer {lm_multistart,de}` | Optimiser for summed fit (default: `de`) |
| `--restarts INT` | LM multi-start restarts (default: 8) |
| `--de-population INT` | DE population size (default: 30) |
| `--de-maxiter INT` | DE maximum iterations (default: 5000) |
| `--workers INT` | CPU cores for DE (-1 = all; auto-limited to 1 in compiled app) |
| `--no-polish` | Skip LM polish step after DE |
| `--cost-function {poisson,chi2}` | Cost function (default: `poisson`) |
| `--intensity-threshold INT` | Minimum photons per pixel mask |
| `--tau-display-min FLOAT` | Min lifetime for exported tau images (ns) |
| `--tau-display-max FLOAT` | Max lifetime for exported tau images (ns) |

#### Output Arguments

| Argument | Description |
|---|---|
| `--out NAME` | Output file prefix (default: `flim_out`; anchored to PTU directory if no path given) |
| `--no-plots` | Suppress plot generation |
| `--channel INT` | Detection channel (default: auto-detect) |

---

### Phasor Analysis (CLI)

```bash
python phasor_cli.py [OPTIONS]
```

| Argument | Description |
|---|---|
| `--ptu PATH` | Path to a `.ptu` file |
| `--irf PATH` | IRF calibration Excel file (XLSX) |
| `--machine-irf PATH` | Machine IRF `.npy` file for calibration |
| `--session PATH` | Resume a saved `.npz` session |

With no arguments the CLI enters a fully guided `inquirer` flow.

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
save_session('session.npz',
             real_cal=state['real_cal'], imag_cal=state['imag_cal'],
             mean=state['mean'], frequency=state['frequency'],
             cursors=state['cursors'], params=state['params'])

sess = load_session('session.npz')
```

#### PTU File Reading

```python
from flimkit.PTU.reader import PTUFile

ptu = PTUFile('data.ptu', verbose=True)
decay   = ptu.summed_decay(channel=None)          # auto-detect channel
stack   = ptu.pixel_stack(channel=None, binning=1) # (Y, X, H)
print(ptu.n_bins, ptu.tcspc_res, ptu.time_ns)
```

#### Signal Extraction (xarray)

```python
from flimkit.PTU.tools import signal_from_PTUFile
import numpy as np

signal = signal_from_PTUFile('data.ptu', dtype=np.uint32, binning=4)
# signal.attrs['frequency'] — modulation frequency in MHz
```

#### Phasor Computation

```python
from flimkit.phasor.signal import (
    return_phasor_from_PTUFile,
    get_phasor_irf,
    calibrate_signal_with_irf,
    calibrate_signal_with_machine_irf,
)

mean, real, imag = return_phasor_from_PTUFile('data.ptu')

# Calibrate with XLSX IRF
irf_time_ns, irf_counts = get_phasor_irf('irf.xlsx')
real_cal, imag_cal = calibrate_signal_with_irf(
    signal, real, imag, irf_time_ns, irf_counts, frequency)

# Calibrate with machine IRF
real_cal, imag_cal = calibrate_signal_with_machine_irf(
    signal, real, imag, 'machine_irf_default.npy', frequency)
```

#### Tile Stitching

```python
from flimkit.PTU.stitch import stitch_flim_tiles, load_flim_for_fitting
from pathlib import Path

result = stitch_flim_tiles(
    xlif_path=Path('metadata/R 2.xlif'),
    ptu_dir=Path('PTU_tiles/'),
    output_dir=Path('stitched/R_2/'),
    ptu_basename='R 2',
    rotate_tiles=True,
)

stack, tcspc_res, n_bins = load_flim_for_fitting(
    Path('stitched/R_2/'), load_to_memory=True)
decay = stack.sum(axis=(0, 1))
```

#### Intensity Images & Cell Masking

```python
from flimkit.image.tools import (
    make_intensity_image, make_cell_mask,
    apply_intensity_threshold, pick_intensity_threshold,
)

intensity  = make_intensity_image('data.ptu', rotate_90_cw=True)
mask       = make_cell_mask(intensity, save_mask=True, path='output/')
int_mask   = apply_intensity_threshold(intensity, threshold=50)
threshold  = pick_intensity_threshold(intensity)  # interactive slider
```

---

## Configuration Reference

Default parameters are in `flimkit/configs.py`. All can be overridden via CLI arguments or the GUI.

### Fitting Defaults

| Parameter | Default | Description |
|---|---|---|
| `Tau_min` | 0.145 ns | Lower lifetime bound |
| `Tau_max` | 45.0 ns | Upper lifetime bound |
| `n_exp` | 3 | Number of exponential components |
| `D_mode` | `'both'` | Fitting mode: `'summed'`, `'perPixel'`, or `'both'` |
| `binning_factor` | 1 | Spatial binning for per-pixel fitting |
| `Optimizer` | `'de'` | `'de'` (Differential Evolution) or `'lm_multistart'` |
| `MIN_PHOTONS_PERPIX` | 10 | Minimum photons for per-pixel fitting |
| `OUT_NAME` | `'flim_out'` | Default output prefix |

### Optimiser Settings

| Parameter | Default | Description |
|---|---|---|
| `lm_restarts` | 8 | Levenberg–Marquardt multi-start restarts |
| `de_population` | 30 | DE population size |
| `de_maxiter` | 5000 | DE maximum iterations |
| `n_workers` | -1 (source) / 1 (compiled) | CPU cores for DE; automatically limited to 1 in frozen app to avoid multiprocessing conflicts |

### Display Range Settings

When set, pixel values outside the range are clamped to the boundary (matching Leica LAS X behaviour, not zeroed).

| Parameter | Default | Description |
|---|---|---|
| `TAU_DISPLAY_MIN` | `None` | Min lifetime (ns) for tau images |
| `TAU_DISPLAY_MAX` | `None` | Max lifetime (ns) for tau images |
| `INTENSITY_DISPLAY_MIN` | `None` | Min photon count for intensity images |
| `INTENSITY_DISPLAY_MAX` | `None` | Max photon count for intensity images |

### Machine IRF Settings

| Parameter | Default | Description |
|---|---|---|
| `MACHINE_IRF_DIR` | `flimkit/machine_irf` (source) / `~/.flimkit/machine_irf` (compiled) | Storage directory |
| `MACHINE_IRF_DEFAULT_PATH` | User copy if present, else bundled default | Resolved once at startup |
| `MACHINE_IRF_ALIGN_ANCHOR` | `'peak'` | Landmark for IRF alignment during construction |
| `MACHINE_IRF_REDUCER` | `'median'` | Aggregation across paired IRFs |
| `MACHINE_IRF_FIT_STRATEGY` | `'fixed'` | Runtime fitting strategy |
| `MACHINE_IRF_FIT_BG` | `True` | Fit background offset |
| `MACHINE_IRF_FIT_SIGMA` | `False` | Fit Gaussian broadening |
| `MACHINE_IRF_FIT_TAIL` | `False` | Fit exponential tail |

### Cost Functions

| Function | Description |
|---|---|
| `poisson` | Poisson deviance (C-statistic) — statistically correct for low-count bins. **Recommended.** |
| `chi2` | Neyman chi-squared (legacy) — underweights the decay tail. |

---

## Module Reference

### `flimkit.PTU` — PTU File I/O

#### `reader.py`
- **`PTUFile(path, verbose=False)`** — Parse a PicoQuant PTU file. Extracts TCSPC metadata and T3 photon records.
  - `.summed_decay(channel=None)` — Summed decay histogram
  - `.pixel_stack(channel=None, binning=1)` — (Y, X, H) histogram stack
  - `.raw_pixel_stack(channel=None, binning=1)` — Overflow-corrected pixel stack using nsync timing for accurate pixel positions
  - `.n_bins`, `.tcspc_res`, `.time_ns` — TCSPC metadata

#### `decode.py`
- Low-level T3 record decoding (PicoHarp, HydraHarp formats)
- `create_time_axis()` — Build time axis from PTU metadata

#### `tools.py`
- **`signal_from_PTUFile(path, dtype, binning)`** — Load PTU and return an `xarray.DataArray` with labelled dimensions (`Y`, `X`, `H`) and `frequency` attribute

#### `stitch.py`
- **`stitch_flim_tiles(xlif_path, ptu_dir, output_dir, ...)`** — Stitch multi-tile PTU data into a mosaic using XLIF metadata. Applies three-pass phase-correlation registration (Preibisch et al. 2009) correcting column Y drift, row Y residuals, and row X backlash. Uses nearest-centre ownership (winner-takes-all) for canvas assembly.
- **`fit_flim_tiles(...)`** — Full fitting pipeline on a stitched mosaic (two-pass: pooled DE fit → per-pixel NNLS)
- **`load_flim_for_fitting(output_dir, load_to_memory)`** — Load previously stitched data

---

### `flimkit.FLIM` — Reconvolution Fitting

#### `fitters.py`
- **`fit_summed(decay, tcspc_res, n_bins, irf_prompt, ...)`** — Fit a summed FLIM decay via reconvolution. Pass 1 of the two-pass pipeline: Differential Evolution global search followed by Levenberg–Marquardt polish. Returns `(best_params, summary_dict)`.
- **`fit_per_pixel(stack, tcspc_res, n_bins, irf_prompt, global_popt, n_exp, ...)`** — Per-pixel fitting with τ values fixed from the global fit. Uses NNLS (non-negative least squares) — fast, convex, unique solution. Pass 2 of the two-pass pipeline.

**Two-pass fitting model:**

```
y(t) = [IRF(t + Shift_IRF) + Bkgr_IRF] ⊗ [Σ αᵢ·exp(−t/τᵢ) + Bkgr]
```

Pass 1 (summed): DE global search → LM polish → fixes τ₁…τₙ  
Pass 2 (per-pixel): NNLS fits α₁…αₙ and background with fixed τ values

Output per pixel: `tau_mean_amp` = Σ(fracᵢ × τᵢ) — amplitude-weighted mean lifetime (primary output)

#### `assemble.py`
- **`assemble_tile_maps(tile_results, canvas_h, canvas_w, n_exp)`** — Assemble per-tile fit results into a single canvas using nearest-centre ownership
- **`derive_global_tau(canvas, n_exp)`** — Compute ROI-level lifetime statistics from the assembled canvas
- **`save_assembled_maps(canvas, global_summary, output_dir, roi_name, n_exp, ...)`** — Save canvas arrays as TIFFs and NPY files

#### `irf_tools.py`
- **`build_machine_irf_from_folder(folder, align_anchor, reducer, ...)`** — Build a machine IRF from paired PTU/XLSX files. Aligns each IRF to the decay peak, aggregates by median, and saves as `.npy` + companion `_meta.json`.
- **`irf_from_xlsx_analytical(xlsx, ...)`** — Fit the Leica analytical IRF model (Gaussian + exponential tail) to LAS X XLSX data
- **`gaussian_irf_from_fwhm(n_bins, tcspc_res, fwhm_ns, peak_bin)`** — Generate Gaussian IRF from FWHM

---

### `flimkit.phasor` — Phasor Analysis

#### `signal.py`
- **`return_phasor_from_PTUFile(ptu_file)`** — Compute phasor coordinates from a PTU file
- **`get_phasor_irf(irf_xlsx)`** — Read IRF from LAS X Excel export
- **`calibrate_signal_with_irf(signal, real, imag, irf_time_ns, irf_counts, frequency)`** — Phase/modulation correction via IRF phasor
- **`calibrate_signal_with_machine_irf(signal, real, imag, machine_irf_npy, frequency)`** — Calibrate using a pre-built machine IRF `.npy` file. Reads the companion `_meta.json` for time resolution; interpolates onto the signal time axis.

#### `interactive.py`
- **`phasor_cursor_tool(real_cal, imag_cal, mean, frequency, ...)`** — Interactive phasor cursor widget. Works in Jupyter notebooks (ipywidgets) and standalone scripts (matplotlib.widgets). Features: click-to-place elliptic cursors, adjustable radius/angle, per-cursor τ_φ maps, two-component decomposition, Undo/Peaks/Export/Save.

#### `peaks.py`
- **`find_phasor_peaks(real_cal, imag_cal, mean, frequency, ...)`** — Automatic peak detection on 2-D phasor histograms via Gaussian smoothing and local maxima detection

---

### `flimkit.UI` — Desktop GUI

#### `gui.py`
- **`launch_gui()`** — Entry point for the Tkinter GUI
- **`FLIMKitApp`** — Main application class. Tabs: Single FOV Fit, Tile Stitch/Fit, Batch ROI Fit, Machine IRF Builder, Phasor Analysis
- **`FOVPreviewPanel`** — Right-panel widget showing intensity image and summed decay curve. Switches to `PhasorViewPanel` when the Phasor tab is active.

#### `phasor_panel.py`
- **`PhasorViewPanel(parent, max_cursors=6)`** — Embedded Tkinter widget with `FigureCanvasTkAgg`. Top axes shows the FOV intensity image (colourised with `pseudo_color` once cursors are placed); bottom axes shows the phasor histogram (`PhasorPlot.hist2d`). Controls: Clear, Undo, Save session, Radius slider, Minor/major slider.
  - `.set_data(real_cal, imag_cal, mean, frequency, display_image, min_photons)` — Load phasor data; call on main thread
  - `.load_session(session, min_photons)` — Restore a saved `.npz` session
  - `.get_session_dict()` — Export current state for saving

---

### `flimkit.image` — Image Utilities

#### `tools.py`
- **`make_intensity_image(ptu_path, rotate_90_cw, save_image)`** — 2-D intensity image from PTU
- **`make_cell_mask(intensity_image, ...)`** — Binary cell mask using Otsu thresholding + morphological cleanup
- **`apply_intensity_threshold(intensity_image, threshold)`** — Boolean mask for photon-count gating
- **`pick_intensity_threshold(intensity_image)`** — Interactive slider for visual threshold selection

---

### `flimkit.utils` — Shared Utilities

#### `plotting.py`
- **`plot_summed(...)`** — Main summed-fit figure: log-scale decay + model overlay, weighted residuals, parameter table. Saves to the output prefix directory (anchored to PTU directory when no path is given).
- **`plot_pixel_maps(...)`** — Per-pixel lifetime and amplitude maps
- **`plot_lifetime_histogram(...)`** — Lifetime distribution histogram

#### `enhanced_outputs.py`
- **`save_fit_summary_txt(...)`** — Human-readable fit results text file
- **`save_weighted_tau_images(...)`** — Intensity-weighted and amplitude-weighted τ TIFFs with optional display range clipping

#### `lifetime_image.py`
- **`make_lifetime_image(canvas, output_dir, roi_name, tau_min_ns, tau_max_ns, ...)`** — Colourised lifetime image with NaN-aware smoothing and gamma correction

#### `xlsx_tools.py`
- **`load_xlsx(path, debug=False)`** — Parse a LAS X FLIM export XLSX. Auto-detects column layout; returns `decay_t/c`, `irf_t/c`, `fit_t/c`, `res_t/c`.

#### `xml_utils.py`
- **`parse_xlif_tile_positions(xlif_path, ptu_basename)`** — Tile positions from XLIF (microns)
- **`get_pixel_size_from_xlif(xlif_path)`** — Pixel size (m) and pixel count
- **`compute_tile_pixel_positions(tiles, pixel_size_m, tile_size)`** — Convert physical positions to pixel coordinates and compute canvas size

---

## Project Structure

```
├── main.py                        # Guided terminal UI
├── fit_cli.py                     # FLIM fitting CLI
├── phasor_cli.py                  # Phasor analysis CLI
├── build_and_sign.py              # PyInstaller build + codesign
├── validate_installation.py       # Installation sanity check
├── requirements.txt
│
├── flimkit/
│   ├── configs.py                 # Default fitting parameters
│   ├── interactive.py             # Guided fitting launcher
│   ├── phasor_launcher.py         # Guided phasor launcher
│   ├── machine_irf/               # Machine IRF files — generated per system
│   │
│   ├── UI/
│   │   ├── gui.py                 # Tkinter desktop GUI
│   │   └── phasor_panel.py        # Embedded phasor view panel
│   │
│   ├── PTU/
│   │   ├── reader.py              # PTUFile — T3 record decoding
│   │   ├── decode.py              # Low-level histogram extraction
│   │   ├── tools.py               # signal_from_PTUFile (xarray)
│   │   └── stitch.py              # Multi-tile stitching + registration
│   │
│   ├── FLIM/
│   │   ├── models.py              # Decay models + DE cost functions
│   │   ├── fitters.py             # fit_summed / fit_per_pixel (NNLS)
│   │   ├── fit_tools.py           # IRF alignment, bin utilities
│   │   ├── assemble.py            # Tile map assembly + global tau stats
│   │   └── irf_tools.py           # IRF estimation + machine IRF builder
│   │
│   ├── phasor/
│   │   ├── signal.py              # Phasor computation & calibration
│   │   ├── interactive.py         # Interactive cursor tool
│   │   └── peaks.py               # Automatic peak detection
│   │
│   ├── image/
│   │   └── tools.py               # Intensity images, cell masking
│   │
│   └── utils/
│       ├── plotting.py            # Decay + pixel map plots
│       ├── enhanced_outputs.py    # TIFF exports, summary text
│       ├── lifetime_image.py      # Colourised lifetime images
│       ├── xlsx_tools.py          # LAS X Excel parsing
│       ├── xml_utils.py           # XLIF tile-position parsing
│       ├── misc.py                # Logging helpers
│       └── fancy.py               # Terminal banners
│
└── flimkit_tests/
    ├── run_tests.py
    ├── mock_data.py
    ├── conftest.py
    ├── test_complete_pipeline.py
    └── tests/
        ├── test_decode.py
        ├── test_integration.py
        └── test_xml_utils.py
```

---

## Compiled App (macOS / Windows / Linux)

FLIMKit can be packaged as a standalone executable with no Python installation required.

### Prerequisites

```bash
pip install pyinstaller
python fix_matplotlib_startup.py   # pre-warm matplotlib font cache (run once)
```

### Build

```bash
python build_and_sign.py
```

Output: `dist/FLIMKit.app` (macOS) or `dist/FLIMKit` / `dist/FLIMKit.exe` (Linux/Windows).

### macOS notes

- The app is **self-signed** (ad-hoc). It will open without a Gatekeeper prompt because it was built locally (no quarantine flag).
- For distribution to other machines, a paid Apple Developer ID and notarization via `xcrun notarytool` are required.
- Uses `--onedir` to produce a proper `.app` bundle with a single process (avoids the two-dock-icon issue caused by `--onefile`'s two-stage launcher).

### Output file location

In the compiled app, all output files are saved to the same directory as the input PTU file. The current working directory inside the bundle is read-only and is never used as a save target.

### Machine IRF in compiled app

Machine IRF files are stored in `~/.flimkit/machine_irf/` (created automatically). The app ships with a bundled default IRF that is used until you build and save your own.

---

## Testing

### Install test dependencies

```bash
pip install -r flimkit_tests/requirements_test.txt
```

### Run tests

```bash
cd flimkit_tests
python run_tests.py              # all tests
python run_tests.py -c           # with coverage report
python run_tests.py integration  # integration tests only
```

### Individual modules

```bash
pytest tests/test_xml_utils.py -v
pytest tests/test_decode.py -v
pytest tests/test_integration.py -v
```

### Test coverage

| Area | What's tested |
|---|---|
| XML/XLIF parsing | Tile positions, metadata extraction |
| PTU decoding | Histogram extraction, time axis |
| Tile stitching | Canvas computation, overlap handling |
| Integration | Complete workflows, error handling |
| Per-tile fit pipeline | Assembly, global tau, output files |

---

## Outputs & File Formats

### GUI Export Options

The desktop GUI provides flexible export capabilities for downstream analysis:

#### Image Export Formats

| Format | Description | Use case |
|---|---|---|
| **OME-TIFF** | Lossless, metadata-preserving TIFF format | Fiji/ImageJ, downstream quantitative image analysis, archival |
| **PNG** | High-resolution raster images | Quick visualization, presentations, web sharing |

Intensity and lifetime maps can be exported during or after fitting. OME-TIFF images preserve 16-bit intensity and 32-bit float lifetime data with metadata embedded.

#### ROI & Geometry Exports

| Format | Description | Use case |
|---|---|---|
| **GeoJSON** | Standard geospatial format for ROI boundaries | **QuPath integration**, cross-software ROI transfer, archival |
| **CSV** | Tabular fit results and ROI statistics | Excel, R/Python analysis, publication supplementary data |

#### Session Saves

FLIM fitting workflows can be saved as `.npz` (NumPy compressed archive) files, allowing you to:
- Resume work without re-fitting
- Restore cursor positions and analysis parameters in ROI tabs
- Share reproducible analysis states

Access via: **File → Save NPZ / Load NPZ / Save NPZ As**

**QuPath Integration:**

GeoJSON exports from FLIMKit are fully compatible with **QuPath** (Qupath >= 0.6.0 - tested). Exported ROIs can be imported directly into QuPath's annotation workflow for:
- Spatial correlation of FLIM parameters with tissue morphology
- Cell segmentation and measurement on intensity images
- Multi-modal image analysis combining FLIM and brightfield/fluorescence data

**Export workflow (FLIMKit → QuPath):**
1. Run tile stitching or batch ROI fitting in FLIMKit
2. Use File → Export → Export All ROIs as GeoJSON
3. In QuPath: Automate → Show script editor → paste: `importPathObjects(path/to/export.geojson)`
4. Adjust detection parameters and run segmentation
5. Export results as CSV or annotations

### FLIM Fitting Outputs

| File | Description |
|---|---|
| `*_summed_Nexp.png` | Summed decay plot with model overlay, residuals, and parameter table |
| `*_pixelmaps_Nexp.png` | Per-pixel lifetime and amplitude maps |
| `*_lifetime_hist_Nexp.png` | Lifetime distribution histogram |
| `*_intensity.tif` | Intensity image (uint16) |
| `*_tau_mean_amp.tif` | Amplitude-weighted mean lifetime image (uint16) |
| `*_tau_amp_lifetime_image.tif` | Display-ready colourised lifetime image (NaN-aware, gamma-corrected) |
| `*_component_rgb.tif` | Per-component fraction as R/G/B channels |
| `*_global_summary.txt` | Human-readable fit results |
| `*.npy` | Raw float32 canvas arrays |

For the tile stitch/fit pipeline, all files are prefixed with the ROI name derived from the XLIF filename (e.g. `R_2_tau_mean_amp.tif`).

### Tile Stitching Outputs

| File | Description |
|---|---|
| `{ROI}_stitched_intensity.tif` | Stitched intensity mosaic (TIFF) |
| `{ROI}_stitched_flim_counts.npy` | FLIM histogram cube (Y × X × H) as NumPy memmap |
| `{ROI}_time_axis_ns.npy` | Time axis in nanoseconds |
| `{ROI}_weight_map.npy` | Tile overlap count map |
| `{ROI}_metadata.json` | Stitching metadata (tile count, canvas size, registration results) |

### FLIM Fitting Session Files

Fitting sessions are saved as `.npz` archives. Use **File → Save NPZ / Save NPZ As** to persist:

#### Single FOV Fit Sessions

| Key | Description |
|---|---|
| `ptu_file` | Path to the original PTU file |
| `global_summary_json` | JSON string of global fit parameters (n_exp, tau values, amplitudes, chi2) |
| `global_summary_arr_*` | NumPy arrays reattached (lifetime maps, amplitude maps, etc.) |
| `pixel_maps_json` | Per-pixel fitting results as JSON |
| `intensity_image` | 2D intensity map (uint16) |
| `fitted_decay` | 1D fitted decay curve |
| `irf_used` | IRF source metadata |

**Restore a session:**
1. File → Recent Files or File → Load NPZ
2. Previous fit results, cursor positions, and parameters are restored

#### Tile Stitch Session Workflows

Similar structure to Single FOV, with additional keys:
- `stitched_intensity` — multi-tile mosaic
- `tile_metadata` — stitching registration parameters
- `per_tile_results` — individual tile fit data

### Phasor Session Files

Saved as `.npz` archives containing:

| Key | Description |
|---|---|
| `real_cal`, `imag_cal` | Calibrated G/S phasor arrays |
| `mean` | Mean intensity image |
| `frequency` | Modulation frequency (MHz) |
| `cursor_g`, `cursor_s`, `cursor_colors` | Cursor positions and colours |
| `param_radius`, `param_radius_minor`, `param_angle_mode` | Ellipse parameters |
| `display_image` | Spatially-correct FOV intensity image |
| `ptu_file`, `irf_file` | Source file paths (metadata) |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `ValueError: Not a PTU/PQTTTR file` | Verify the file is a valid PicoQuant PTU |
| `No TileScanInfo found in XLIF` | XLIF may not be from a tile scan acquisition |
| `FileNotFoundError: Machine IRF file not found` | Build a machine IRF in the **Machine IRF Builder** tab first |
| Fit crashes in compiled app with `resource_tracker` error | Already fixed — `multiprocessing.freeze_support()` is called at startup and DE uses `workers=1` in the compiled app |
| `OSError: [Errno 30] Read-only file system` when saving plots | Already fixed — output paths are anchored to the PTU file's directory |
| Two dock icons on macOS | Already fixed — compiled app uses `--onedir` instead of `--onefile` |
| Slow first launch of compiled app | Run `fix_matplotlib_startup.py` before building to pre-warm the font cache |
| Phasor points scattered off semicircle | Check IRF calibration; uncalibrated data will not lie on the universal semicircle |
| Per-pixel fitting is slow | Increase `--binning` (2 or 4) or reduce `--de-maxiter` |
| Fit Summary tab empty after fitting | Ensure you are running the latest GUI — `_extract_summary_rows` was updated to handle both single-FOV and tile-fit output schemas |

---

## Roadmap

- [x] Single FOV fitting
- [x] Batch processing of multiple FOVs
- [x] Reconstruction and fitting of multi-tile ROIs
- [x] Phasor analysis with interactive cursors, peak detection, session save/load
- [x] Desktop GUI with embedded phasor panel and live FOV preview
- [x] Standalone compiled app (macOS, Windows, Linux)
- [x] Documentation and tests
- [ ] Chemical validation of fitting results with known fluorophores
- [ ] Publication

---

## Contact

For questions, contact Alex Hunt at alexander.hunt@ed.ac.uk
