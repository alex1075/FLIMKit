# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- Updated .gitignore configuration (3097f45)

## [v0.8.9] - 2026-04-07

### Fixed
- Fixed bugs in image tools, interactive module, and test suite
- Improved test coverage and refactored test mock data
- Fixed test cases in test_decode.py, test_ground_truth.py, and test_integration.py
  - Modified files: flimkit/image/tools.py, flimkit/interactive.py, flimkit_tests/mock_data.py, and test files (13ee496)

### Changed
- Updated version to 0.8.9 (4920c9c)

## [v0.8.8] - 2026-04-06

### Removed
- Deleted fix_missing_tiles.py utility script (2599f9c)

### Changed
- Updated PTU reader implementation (6d59790)
- Updated GitHub build workflow configuration (885cebf, 699c5da)
- Updated VS Code project settings (78d8f6a, 699c5da)
- Refactored test mock data module (699c5da, 885cebf, 6d59790)
- Fixed and refined build pipeline (5fb8468, 7b8d77d, 6be8842)
- Various small fixes and improvements (bf89b46)

## [2026-04-02]

### Changed
- Fixed building and updated documentation (e7589ee)

## [2026-03-30]

### Added
- Added phasor_panel.py module for GUI phasor visualization panel (485 lines) (ae4ecfe)

### Changed
- Updated BUILD_AND_SIGN.md documentation (308e171)
- Major GUI visualization update with enhanced UI components
- Refactored gui.py with improved layout and controls (+218 lines, -146 lines)
- Updated build_and_sign.py build configuration
- Removed old stitch.py module, replaced with integrated functionality
- Updated main.py integration (ae4ecfe)

## [2026-03-28]

### Changed
- Save for future fix (856b3c8)

## [2026-03-24]

### Fixed
- Fixed Windows build compatibility issues (b65ff1b)

### Added
- Extended phasor analysis to support machine IRF input parameters

### Changed
- Updated phasor signal processing to incorporate machine IRF (+54 lines)
- Refactored phasor_launcher.py for improved machine IRF handling
- Simplified and refactored gui.py with updated phasor integration (43ade12)
- Updated build configuration (5e069a5)

## [2026-03-23]

### Added
- Refinements to GUI (bb02868)
- GUI updates (e1a5ca9, eb0ccbf)

### Changed
- Build pipeline attempts (0c06ac5)
- Requirements update (0da423d)
- Version update (f955bb3)
- Cleanup (1d82703)

## [2026-03-20]

### Changed
- Updated GUI components (7e16a6e)
- Major improvements to tile stitching algorithm to remove stitch line artifacts
- Extensively refactored PTU/stitch.py with enhanced stitching logic (+433 lines)
- Simplified assemble.py and interactive.py (efficiency improvements)
- Removed per_tile_machine_irf_fitting.ipynb (2208 lines removed, refactored into core modules)
- Updated lifetime_image.py with new image generation capabilities
- Removed deprecated Phasor peak detection from stitching module (ef5e579)

## [2026-03-19]

### Changed
- Enhanced image generation capabilities from fitted FLIM data
- Updated lifetime_image.py module with improved lifetime image calculation (+123 lines)
- Significant improvements to assemble.py image assembly logic
- Added per_tile_machine_irf_fitting.ipynb notebook for advanced fitting workflows (+2208 lines)
- Updated .gitignore for build artifacts (dc55fc9)

## [2026-03-18]

### Changed
- Updated notebook tutorials and examples - See FLIMKit-Examples git repo (ddb7f91)

## [2026-03-17]

### Added
- Implemented batch ROI processing capabilities
- Added batch_roi_fit_stitch.ipynb notebook for batch processing workflows (357 lines) (71ea390)

## [2026-03-15]

### Fixed
- Fixed ROI fitting algorithm and improved accuracy
- Updated GUI with ROI fitting improvements
- Refined test cases for ROI fitting validation (c2ef0ef, 31e30c3)

### Changed
- Updated README.md with usage instructions (75f7068)

## [v0.8.2] - 2026-03-15

### Fixed
- Fixed ROI fitting and updated GUI (31e30c3)

## [v0.6.2] - 2026-03-13

### Added
- Added GitHub Actions CI/CD build workflow for macOS, Linux, and Windows (4b5c0e9)
- Added GUI components and phasor visualization (f315312)
- Implemented machine IRF (Instrument Response Function) creation from existing fitted data
  - Added irf_tools.py module with IRF reconstruction utilities (+305 lines)
  - Added machine_irf directory for storing calibration data
  - Extensively updated interactive.py for IRF workflows (+134 lines) (c0c87e4)
- Added CI guard for Python >=3.12 (3219e28)

### Fixed
- Fixed Ubuntu build: installed tkinter and libGL dependencies, added hidden-import for tkinter (94a2790)
- Fixed GUI windowing issues for multi-monitor support (0ac4c14, 4786d5b)
- Fixed test suite (2e920a7, b9a93b2)

### Changed
- Switched PyInstaller to --onefile for single-executable builds (699f916)
- Updated Python version support to require Python 3.12+ (4d6e65e)
- Enhanced configs.py for machine IRF configurations (23 lines modified) (c0c87e4)
- Updated gui.py with improved phasor and IRF workflows (123 lines modified)
- Updated build_and_sign.py for release builds (+10 lines)
- Added irf_reconstruction_validation.ipynb for validating IRF reconstruction (+10603 lines) (c0c87e4)
- Updated .gitignore for build artifacts and machine IRF cache files
- Updated documentation and README (003fc28)
- Updated build files (dba92bb, 9903731)

### Documentation
- Preserve macOS app bundle in release artifact (347a228)
- Upload only .app on macOS, skip raw binary (d627b79)
- Document machine IRF file locations for compiled exe vs source (f565427)
- Added note that app restart is required after saving machine IRF in compiled app (262e00f)

### Infrastructure
- Allow manual workflow runs to publish GitHub Release (19f742c)
- Use writable user directory for machine IRF saves when running as compiled executable (0183e2d)
- Enhanced build matrix and CI configuration

### Deprecated
- Attempted Python 3.13 support (092b5bb) - later standardized on 3.12 (4d6e65e)

### Removed
- Attempted Python 3.14 for builds (cc83a28) - removed in favor of 3.12

## [2026-03-12]

### Changed
- Updated gitignore (5fe6a5f)
- Testing reconstruction IRF workflow (4fd7b42)

## [2026-03-11]

### Changed
- General update (6d2bb7a)
- Testing tile fitting before fitting for big ROIs (f1b19ba)

## [2026-03-10]

### Added
- Added chunking support for memory-limited PCs (e5fc56d)
- GUI testing (f198152)

### Changed
- General update (943502e)
- Updated gitignore (7b12666)

## [2026-03-09]

### Fixed
- Corrected visualization of Phasor image selected regions (15d1f3b)

### Added
- Added thresholding for image export (5c31034)

### Changed
- Version update (5da09f3)

## [2026-03-06]

### Added
- Added intensity thresholding for fitting (FOV and rebuilt ROI) (034e3fe)

### Changed
- Documentation updates and small fixes (6f98205)

## [2026-03-05]

### Changed
- Updated for repo name (75d61e7)

## [2026-03-04]

### Added
- Added peak(s) detection from phasor (3d7ae5f)
- Functional Phasor analysis added (cf6a17c)
  - Created phasor module infrastructure with __init__.py and interactive.py (578 lines)
  - Added phasor_launcher.py for launching phasor interactive GUI (334 lines)
  - Added phasor_cli.py command-line interface (43 lines)
  - Created batch processing support in phasor.ipynb
- Added helper Phasor computing functions in phasor/signal.py (08d1d39)

### Fixed
- Fixed tests (846e70f)
- Fixed fitter (4f7a2c7)

### Changed
- Troubleshot fit cost function, Poisson now default (779f964)
- Testing phasor implementations (046a1ed)
- Updated tests and README (09d0e94)
- Updated README.md with phasor instructions (4c62589)
- Updated requirements.txt with phasor dependencies
- Updated gitignore for Jupyter notebook artifacts (cf6a17c)

## [v0.3.1] - 2026-03-03

### Added
- Updated reader to add raw photons for image stitching (5847dbd)
- Added test suite infrastructure (e12bc0d)

### Changed
- Updated interactive and main.py (3d36c45)
- Renamed code folder from pyflim to flimkit (642bb24)

## [2026-03-02]

### Added
- Added versioning system with _version.py (cba33f7)

### Fixed
- Fixed missing lines in source code (68421af)
- Fixed CLI call issues (1ecc039)
- Fixed shebang paths (2206b8c, 7c37af5)

### Changed
- Re-adding stitching functionality with optimizations (4f727e7)
- Removed stitching func, needs optimization with custom PTU reader (9254798)
- Updated README (09ff8e4)
- Cleanup and update (5d7a79a)

## [2026-02-27]

### Changed
- Updated .gitignore (bae3e50)

### Removed
- Deleted .DS_Store OS files (0cca89c)
- Deleted PTUs directory (29b0bf8)

## [2026-02-27] - Initial Commit

### Added
- First commit - initial FLIMKit project structure
