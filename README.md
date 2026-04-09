# FLIMKit

> **Warning:** This project is in active development. Please cross-validate results with other software before drawing conclusions. Additionally, API and file formats may change without deprecation during this early stage.

## Overview

**FLIMKit** is a Python toolkit for Fluorescence Lifetime Imaging Microscopy (FLIM) data acquired on a Leica SP8/FALCON (or any PTU-based system). It is designed as a drop-in replacement for Leica LAS X FLIM analysis and provides two complementary workflows:

1. **Reconvolution fitting** — mono/bi/tri-exponential lifetime fitting with full IRF deconvolution, per-pixel and summed modes, multi-tile ROI stitching, and batch processing.
2. **Phasor analysis** — calibrated phasor plots with interactive elliptical cursors, two-component decomposition, automatic peak detection, and session save/load.

Both workflows are accessible through:
- A **desktop GUI** (`python main.py` or the compiled app)
- A **guided terminal UI** (`python main.py --cli`)
- **Standalone CLI scripts** (`fit_cli.py`, `phasor_cli.py`)
- **Python API** (import `flimkit`)

[FLIMkit-Examples repo](https://github.com/alex1075/FLIMKit-Examples.git)

## Requirements

- Python ≥ 3.14
- See [requirements.txt](requirements.txt) for the full dependency list.

Key dependencies: `numpy`, `scipy`, `matplotlib`, `xarray`, `phasorpy 0.9`, `ptufile`, `opencv-python`, `pandas`, `tifffile`, `inquirer`, `ipywidgets`, `ipympl`

## Installation

```bash
git clone https://github.com/alex1075/FLIMKit.git
cd FLIMKit
pip install -r requirements.txt
```

### Validate installation

```bash
python validate_installation.py
```

Runs 9 checks covering dependencies, module imports, XLIF parsing, stitching, fitting, phasor pipeline, and the per-tile fit pipeline. All checks should pass.

### Run the test suite
Don't have to run the tests, but it's a good sanity check after making code changes. The test suite covers core functionality with unit and integration tests.
```bash
cd flimkit_tests
python run_tests.py              # all tests
python run_tests.py -c           # with coverage report
python run_tests.py integration  # integration tests only
```

## Usage

### Desktop GUI (recommended)

```bash
python main.py
```

The GUI provides five tabs:

| Tab | Description |
|---|---|
| **Single FOV Fit** | Load one PTU file, select IRF method, run summed and/or per-pixel fitting |
| **Tile Stitch / Fit** | Stitch multi-tile ROIs from XLIF metadata and run the full fitting pipeline |
| **Batch ROI Fit** | Process a whole XLIF folder of ROIs sequentially, export CSV summary |
| **Machine IRF Builder** | Build a machine IRF from paired PTU/XLSX files for reuse across sessions |
| **Phasor Analysis** | Load a PTU file and analyse phasor coordinates interactively — embedded image and phasor plot update live as cursors are placed |

The right-hand panel shows an **FOV Preview** (intensity image + summed decay) for all fitting tabs, and switches automatically to the interactive **Phasor panel** when the Phasor Analysis tab is selected.

### Guided terminal UI

```bash
python main.py --cli
```

### Machine IRF setup (required before fitting)

Before routine FLIM fitting, create a machine IRF once for your system/session.

1. Launch the GUI and open the **Machine IRF Builder** tab.
2. Select a folder containing matched `<name>.ptu` and `<name>.xlsx` pairs (10–20 pairs recommended).
3. Build and save as `machine_irf_default.npy`.

When running from the compiled app, the IRF is saved to `~/.flimkit/machine_irf/` (created automatically). When running from source it is saved to `flimkit/machine_irf/`.

```python
# Programmatic alternative
from flimkit.FLIM.irf_tools import build_machine_irf_from_folder

build_machine_irf_from_folder(
    folder="/path/to/pairs",
    align_anchor="peak",
    reducer="median",
    save=True,
    output_name="machine_irf_default",
)
```

### FLIM fitting (CLI)

```bash
python fit_cli.py --ptu path/to/file.ptu --machine-irf path/to/machine_irf.npy --nexp 2 --optimizer de
```

For all options: `python fit_cli.py --help`

### Phasor analysis (CLI)

```bash
# Guided prompts
python phasor_cli.py

# Direct paths
python phasor_cli.py --ptu data.ptu --irf irf.xlsx

# Resume a saved session
python phasor_cli.py --session session.npz
```

### Python API

```python
from flimkit.phasor_launcher import launch_phasor
state = launch_phasor('data.ptu', irf_path='irf.xlsx')
```

## Compiled app (macOS / Windows / Linux)

A standalone compiled app can be built with:

```bash
python build_and_sign.py
```

The compiled app requires no Python installation. Output files are saved to the same directory as the input PTU file.

## Project structure

```
├── main.py                        # Guided terminal UI (all workflows)
├── fit_cli.py                     # FLIM reconvolution fitting CLI
├── phasor_cli.py                  # Phasor analysis CLI
├── build_and_sign.py              # PyInstaller build + codesign script
├── validate_installation.py       # Installation sanity check
├── requirements.txt
│
├── flimkit/
│   ├── configs.py                 # Default fitting parameters
│   ├── interactive.py             # Guided FLIM fitting launcher
│   ├── phasor_launcher.py         # Guided phasor analysis launcher
│   ├── machine_irf/               # Machine IRF files (.npy) — generated per system
│   │
│   ├── UI/
│   │   ├── gui.py                 # Tkinter desktop GUI
│   │   └── phasor_panel.py        # Embedded phasor view panel
│   │
│   ├── PTU/
│   │   ├── reader.py              # PTUFile — T3 record decoding
│   │   ├── decode.py              # Low-level histogram extraction
│   │   ├── tools.py               # signal_from_PTUFile (xarray)
│   │   └── stitch.py              # Multi-tile PTU stitching + registration
│   │
│   ├── FLIM/
│   │   ├── models.py              # Decay models + DE cost functions
│   │   ├── fitters.py             # fit_summed / fit_per_pixel (NNLS)
│   │   ├── fit_tools.py           # IRF alignment, bin utilities
│   │   ├── assemble.py            # Tile map assembly + derive_global_tau
│   │   └── irf_tools.py           # IRF estimation & machine IRF builder
│   │
│   ├── phasor/
│   │   ├── signal.py              # Phasor computation & IRF calibration
│   │   ├── interactive.py         # Interactive cursor tool (notebook + script)
│   │   └── peaks.py               # Automatic peak detection
│   │
│   ├── image/
│   │   └── tools.py               # Intensity images, cell masking
│   │
│   └── utils/
│       ├── plotting.py            # Decay + pixel map plots
│       ├── enhanced_outputs.py    # TIFF exports, summary text
│       ├── lifetime_image.py      # Colourised lifetime image generation
│       ├── xlsx_tools.py          # LAS X Excel file parsing
│       ├── xml_utils.py           # XLIF tile-position parsing
│       ├── misc.py                # Logging helpers
│       └── fancy.py               # Terminal banners
│
└── flimkit_tests/
    ├── run_tests.py
    ├── mock_data.py
    ├── test_complete_pipeline.py
    └── tests/
        ├── test_decode.py
        ├── test_integration.py
        └── test_xml_utils.py
```

## Roadmap

- [x] Single FOV fitting
- [x] Batch processing of multiple ROIs
- [x] Reconstruction and fitting of multi-tile ROIs
- [x] Phasor analysis with interactive cursors, peak detection, session save/load
- [x] Desktop GUI with embedded phasor panel, FOV preview, and fit summary
- [x] Standalone compiled app (macOS, Windows, Linux)
- [x] Documentation and tests
- [X] Session restoration
- [ ] Adding image ROI selection post fitting for spatially resolved lifetime analysis #TBD
- [ ] Chemical validation of fitting results with known fluorophores
- [ ] Publication
- [ ] Batch phasor analysis of multiple FOVs in GUI (Can be done with a notebook in the [FLIMkit-Examples repo in the meantime](https://github.com/alex1075/FLIMKit-Examples.git))
- [ ] Fix object/cell detection from intensity image (hit or miss currently) #TBD

## Contact

For questions, contact Alex Hunt at alexander.hunt@ed.ac.uk
