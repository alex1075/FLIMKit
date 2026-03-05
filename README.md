# FLIMKit

> **Warning:** This project is in active development. Please cross-validate results with other software before drawing conclusions.

## Overview

**FLIMKit** is a Python toolkit for Fluorescence Lifetime Imaging Microscopy (FLIM) data acquired on a Leica SP8 (or compatible PTU-based systems). It provides two complementary analysis workflows:

1. **Reconvolution fitting** — mono/bi/tri-exponential lifetime fitting with IRF deconvolution, per-pixel and summed modes, and multi-tile ROI stitching.
2. **Phasor analysis** — calibrated phasor plots with interactive elliptical cursors, automatic peak detection, two-component decomposition, and session save/load.

Both workflows can be launched through a guided terminal UI (`main.py`), standalone CLI scripts (`fit_cli.py`, `phasor_cli.py`), or used programmatically as a library.

## Requirements

- Python ≥ 3.11
- See [requirements.txt](requirements.txt) for the full list. Key dependencies:
  - numpy, scipy, matplotlib, xarray
  - phasorpy 0.9
  - ptufile
  - inquirer (interactive prompts)
  - ipywidgets + ipympl (notebook support)

## Installation

```bash
git clone https://github.com/alex1075/FLIMKit.git
cd FLIMKit
pip install -r requirements.txt
```

### Validate installation

Run the built-in validation script to verify everything is working:

```bash
python validate_installation.py
```

It checks dependencies, module imports, XLIF parsing, stitching, fitting, and the phasor pipeline.
All 8 checks should pass.

### Run the test suite

```bash
cd flimkit_tests
python run_tests.py              # all tests
python run_tests.py -c           # with coverage (text report)
python run_tests.py integration  # integration tests only
```

## Usage

### Guided terminal UI

```bash
python main.py
```

Presents an interactive menu with all available workflows (FLIM fitting, phasor analysis, tile stitching).

### FLIM fitting (CLI)

```bash
python fit_cli.py --ptu path/to/file.ptu --irf-xlsx path/to/irf.xlsx --nexp 2 --optimizer de
```

For all options: `python fit_cli.py --help`

> **Tip:** While FLIMKit can estimate an IRF from the decay curve, using an IRF exported from the LAS X FLIM tail-fit graph (right-click → export) is strongly recommended.

### Phasor analysis (CLI)

```bash
# Fully guided — prompts for PTU file and optional IRF
python phasor_cli.py

# Direct paths
python phasor_cli.py --ptu path/to/file.ptu --irf path/to/irf.xlsx

# Resume a saved session
python phasor_cli.py --session path/to/session.npz
```

### Phasor analysis (Python API)

```python
from flimkit.phasor_launcher import launch_phasor

# Interactive prompts
state = launch_phasor()

# Or pass paths directly
state = launch_phasor('data.ptu', irf_path='irf.xlsx')

# Resume a saved session
state = launch_phasor(session_path='session.npz')
```

The interactive phasor tool provides:
- Click-to-place elliptical cursors with adjustable size and orientation
- Per-cursor apparent-lifetime (τ_φ) maps
- Two-component decomposition via semicircle intersection (first two cursors)
- **Peaks** button — automatic peak detection on the phasor histogram
- **Export** button — save the phasor figure (with all ROIs drawn) as PNG/PDF/SVG
- **Save** button — persist arrays + cursor state to `.npz` for later editing

## Project structure

```
├── main.py                        # Main menu — guided terminal UI (all workflows)
├── fit_cli.py                     # FLIM reconvolution fitting CLI
├── phasor_cli.py                  # Phasor analysis CLI
├── phasor.ipynb                   # Phasor walkthrough notebook
├── validate_installation.py       # Quick import / sanity check
├── requirements.txt
├── Dockerfile
│
├── flimkit/                        # ── Core library ──────────────────────────
│   ├── __init__.py                # Top-level exports (launch_phasor, etc.)
│   ├── _version.py                # Version string & roadmap
│   ├── configs.py                 # Default fitting parameters
│   ├── interactive.py             # Guided FLIM fitting launcher (inquirer)
│   ├── phasor_launcher.py         # Guided phasor analysis launcher (inquirer)
│   │
│   ├── PTU/                       # ── PTU file I/O ─────────────────────────
│   │   ├── __init__.py
│   │   ├── reader.py              # PTUFile / PTUArray5D classes
│   │   ├── decode.py              # Low-level T3 record decoding
│   │   ├── tools.py               # signal_from_PTUFile (→ xarray DataArray)
│   │   └── stitch.py              # Multi-tile PTU stitching
│   │
│   ├── FLIM/                      # ── Reconvolution fitting ─────────────────
│   │   ├── __init__.py
│   │   ├── models.py              # Exponential decay models
│   │   ├── fitters.py             # fit_summed / fit_per_pixel
│   │   ├── fit_tools.py           # IRF alignment, bin utilities
│   │   └── irf_tools.py           # IRF extraction & estimation
│   │
│   ├── phasor/                    # ── Phasor analysis ───────────────────────
│   │   ├── __init__.py            # Exports all phasor functions
│   │   ├── signal.py              # Phasor computation & IRF calibration
│   │   ├── interactive.py         # Interactive cursor tool (notebook + script)
│   │   └── peaks.py               # Automatic peak detection on phasor histograms
│   │
│   ├── image/                     # ── Image utilities ───────────────────────
│   │   ├── tools.py               # Intensity images, cell masking
│   │   └── stitch.py              # Tile image stitching
│   │
│   ├── LIF/                       # ── LIF format support ────────────────────
│   │   └── utils.py               # LIF metadata helpers
│   │
│   └── utils/                     # ── Shared utilities ──────────────────────
│       ├── __init__.py
│       ├── plotting.py            # Summed-fit plots, pixel maps, histograms
│       ├── enhanced_outputs.py    # Summary text, weighted-τ images, exports
│       ├── xlsx_tools.py          # LAS X Excel file parsing
│       ├── xml_utils.py           # XLIF tile-position parsing
│       ├── misc.py                # Print helpers
│       └── fancy.py               # Terminal banners & ASCII art
│
└── flimkit_tests/                  # ── Test suite ────────────────────────────
    ├── pytest.ini
    ├── requirements_test.txt
    ├── run_tests.py               # Test runner script
    ├── mock_data.py               # Synthetic data generators
    ├── test_complete_pipeline.py  # End-to-end pipeline test
    ├── TESTING.md
    └── tests/
        ├── __init__.py
        ├── test_decode.py         # PTU decoding tests
        ├── test_integration.py    # Integration tests
        └── test_xml_utils.py      # XLIF parsing tests
```

## Roadmap

- [x] Single FOV fitting
- [x] Batch processing of multiple FOVs
- [x] Reconstruction of multi-tile ROIs
- [x] Phasor analysis with interactive cursors, peak detection, and save/load
- [~] Fitting multi-tile ROIs and exporting tau-fitted images/data
- [~] Documentation and examples
- [ ] GUI development for easier use

## Contact

For questions, contact Alex Hunt at alexander.hunt@ed.ac.uk