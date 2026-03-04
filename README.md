# Testing-FLIM

Warning: This project is currently in development and may not be fully functional. Please compare the results with other software and use with caution.

## Overview

This project aims to do FLIM fitting of FOVs aquired with a Leica SP8 microscope. The script is designed to be run from the command line interface (CLI) and provides tools for analyzing FLIM data.

Calling 'main.py' should give an in terminal user interface to guide the user through the process of selecting FOVs, setting fitting parameters, and exporting results. The script is currently in development, so please check back later for updates.

For direct calling in the terminal, use 'fit_cli.py' which provides a more streamlined interface for fitting FLIM data without the need for a GUI.

## Requirements

- Python 3.14
- Numpy
- Scipy
- Matplotlib
- OpenCV-python
- Pandas

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/alex1075/pyFLIM.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the script with:
```bash
./main.py
```

To use the command line interface for fitting:
```bash
python fit_cli.py --ptu path/to/ptu_file.ptu --irf-xlsx path/to/irf_file.xlsx -nexp 2 --optimizer de
```
You will need to provide the path to your PTU file, the IRF Excel file extracted from the tail fit graph in the FLIM window of LAS X FLIM (right click on the graph and export - highly recommeded), the number of exponentials to fit, and the optimizer to use (e.g., 'de' for differential evolution).

Note: while the program can extimate an irf from the tail of the decay curve, it is highly recommended to use an IRF extracted from the FLIM window in LAS X FLIM for more accurate fitting results.

For all cli options, run:
```bash
python fit_cli.py --help
```

## Roadmap
- [X] Single FOV fitting
- [X] Batch processing of multiple FOVs
- [X] Reconstruction of multi-tile ROIs
- [~] Fitting multi-tile ROIs and exporting tau-fitted images/data
- [~] Documentation and examples
- [ ] GUI development for easier use
- [ ] Phasor analysis implementation

## Contact

For questions, contact Alex Hunt at alexander.hunt@ed.ac.uk