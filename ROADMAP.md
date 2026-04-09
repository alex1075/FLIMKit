# FLIMKit Development Roadmap

**Last Updated:** April 9, 2026

## High Priority

### 1. Undo/Redo System
Take back region edits without losing everything. Ctrl+Z & Ctrl+Shift+Z, with visual button states.

### 2. Project Tree View
Left sidebar browser for multi-PTU scans. Shows .ptu files only (hides duplicate .npy), quick switching between scans.

### 3. Config File Management
YAML/JSON settings that persist. UI panel to edit colors, default tools, output dirs. Per-project overrides + reset option.

### 4. Automatic Error Log Exporter
On crash: timestamped logs with system info, app version, and what you were doing. Export for debugging.

## Medium Priority

Batch FOV analysis • Merge/split regions • Filter by stats • Stat histograms • Auto-detect regions • Full IRF support • Keyboard shortcuts • Better error messages

## Completed

Region drawing (4 tools) • Per-region stats (tau, photon counts) • CSV/GeoJSON export-import • Progress bars • Auto-save to NPZ • Multi-panel UI

## Quick Notes

- **Undo/Redo:** next in line
- **Project Tree:** QOL improvement for multi-file scanning workflows
- **Config:** Keeps settings flexible outside of code defaults
- **Error Logs:** Peace of mind + easier debugging


