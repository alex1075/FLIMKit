# FLIMKit Development Roadmap

**Last Updated:** April 16, 2026

## High Priority - To Do

### 1. Config File Management
YAML/JSON settings that persist. UI panel to edit colors, default tools, output dirs. Per-project overrides + reset option. (Infrastructure exists; needs persistence layer)

### 2. Automatic Error Log Exporter
On crash: timestamped logs with system info, app version, and what you were doing. Export for debugging. (Basic logging exists; needs crash handler integration)

### 3. Filter by Stats
UI controls to filter/select regions by tau, photon count, and other statistics. Display filtered results.

### 4. Stat Histograms
Histogram visualization for ROI statistical distributions (tau, photon counts, etc.).

## Medium Priority - To Do

### 1. Auto-Detect Regions
Automatic boundary detection for regions of interest based on intensity or lifetime gradients.

## Completed 

**High Priority:**
- Undo/Redo System (Ctrl+Z & Ctrl+Shift+Z with menu & button states)
- Project Tree View (Left sidebar browser for multi-PTU scans)

**Medium Priority:**
- Batch FOV analysis
- Tested full IRF support (6 methods: Leica XLSX, Machine IRF, Scatter PTU, Estimate raw, Estimate parametric, Gaussian)
- Keyboard shortcuts (Undo/redo, zoom, menu accelerators)
- Better error messages (Extensive throughout codebase)

**Core Features:**
- Region drawing (4 tools: rectangle, ellipse, polygon, freehand)
- Per-region stats (tau, photon counts)
- CSV/GeoJSON export-import
- Progress bars
- Auto-save to NPZ
- Multi-panel UI


## Known issues

- **Stitched FLIM image export** — FLIM images from stitched ROIs are currently saving with a larger pixel size than the original PTU. This is a known issue related to how the per pixel lifetime is calculated (binning). For now the workaround is to use batch ROI fit (under tools) to get true per pixel lifetime maps, which are exported correctly. This will be fixed in a future update. - NOW FIXED IN 0.9.4
- **Stitched session restoration** — When loading a saved session from a stitched ROI, only the FOV preview and summed fit will restore correctly. ROIs and fit settings don't restore as they should. This will be fixed in a future update. - NOW FIXED IN 0.9.4
- **Importing ROIs from GeoJSON** — When importing ROIs from GeoJSON, only the exterior geometry is imported. Any donught like shapes with holes will lose the hole geometry and be imported as the outer boundary only. 
