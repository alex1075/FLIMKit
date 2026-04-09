import json
from typing import List, Dict, Optional, Tuple
import numpy as np


# Default color palette (6 colors, same as phasor panel)
_COLORS = [
    '#FF6B6B',  # red
    '#4ECDC4',  # teal
    '#FFE66D',  # yellow
    '#95E1D3',  # mint
    '#C7CEEA',  # lavender
    '#FF8C42',  # orange
]


class RoiManager:
    """
    Manage multiple regions of interest drawn on a FLIM image.
    
    Regions are stored as JSON-serializable dicts, supporting:
    - Rectangle (2 points: top-left, bottom-right)
    - Ellipse (2 points: bounding box corners)
    - Polygon (N points: vertices)
    - Freehand (N points: traced path)
    
    All data can be serialized to JSON and saved in NPZ files.
    """
    
    def __init__(self):
        """Initialize empty region list."""
        self.regions: List[Dict] = []
        self._next_id = 0
        self._selected_id: Optional[int] = None
    
    def add_region(self, name: str, tool_type: str, coords: List[List[float]], 
                   color_idx: Optional[int] = None) -> int:
        """
        Add a new region.
        
        Args:
            name: Region name (user-facing label)
            tool_type: One of 'rect', 'ellipse', 'polygon', 'freehand'
            coords: List of [x, y] points defining the region
            color_idx: Color palette index (0-5, auto-cycle if None)
        
        Returns:
            Region ID (int)
        
        Raises:
            ValueError: If tool_type is invalid or coords is empty
        """
        if tool_type not in ('rect', 'ellipse', 'polygon', 'freehand'):
            raise ValueError(f"Invalid tool_type: {tool_type}")
        
        if not coords or len(coords) == 0:
            raise ValueError("coords cannot be empty")
        
        if color_idx is None:
            color_idx = len(self.regions) % len(_COLORS)
        
        region = {
            'id': self._next_id,
            'name': name,
            'tool': tool_type,
            'coords': [[float(x), float(y)] for x, y in coords],
            'color_idx': int(color_idx),
        }
        
        self.regions.append(region)
        self._next_id += 1
        return region['id']
    
    def remove_region(self, region_id: int) -> bool:
        """
        Remove a region by ID.
        
        Args:
            region_id: Region ID
        
        Returns:
            True if removed, False if not found
        """
        for i, r in enumerate(self.regions):
            if r['id'] == region_id:
                self.regions.pop(i)
                if self._selected_id == region_id:
                    self._selected_id = None
                return True
        return False
    
    def get_region(self, region_id: int) -> Optional[Dict]:
        """Get region dict by ID."""
        for r in self.regions:
            if r['id'] == region_id:
                return r
        return None
    
    def update_region(self, region_id: int, **kwargs) -> bool:
        """
        Update region fields (name, coords, color_idx, etc.).
        
        Args:
            region_id: Region ID
            **kwargs: Fields to update (name, coords, color_idx, etc.)
        
        Returns:
            True if updated, False if not found
        """
        for r in self.regions:
            if r['id'] == region_id:
                if 'coords' in kwargs:
                    r['coords'] = [[float(x), float(y)] for x, y in kwargs['coords']]
                if 'name' in kwargs:
                    r['name'] = str(kwargs['name'])
                if 'color_idx' in kwargs:
                    r['color_idx'] = int(kwargs['color_idx'])
                if 'tool' in kwargs:
                    r['tool'] = str(kwargs['tool'])
                return True
        return False
    
    def select_region(self, region_id: Optional[int]) -> None:
        """Select a region for highlighting/editing."""
        self._selected_id = region_id
    
    def get_selected_id(self) -> Optional[int]:
        """Get currently selected region ID."""
        return self._selected_id
    
    def get_all_regions(self) -> List[Dict]:
        """Get all regions as list of dicts."""
        return self.regions
    
    def clear_all(self) -> None:
        """Remove all regions."""
        self.regions = []
        self._selected_id = None
        self._next_id = 0
    
    def to_json(self) -> str:
        """Serialize regions to JSON string (for NPZ storage)."""
        data = {
            'regions': self.regions,
            'next_id': self._next_id,
        }
        return json.dumps(data, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RoiManager':
        """Deserialize regions from JSON string."""
        manager = cls()
        try:
            data = json.loads(json_str)
            manager.regions = data.get('regions', [])
            manager._next_id = data.get('next_id', len(manager.regions))
        except (json.JSONDecodeError, ValueError):
            # Gracefully handle corrupted JSON
            pass
        return manager
    
    def compute_region_mask(self, region_id: int, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Args:
            region_id: Region ID
            image_shape: (height, width) of image
        
        Returns:
            Boolean array mask, or None if region not found or tool_type unsupported
        """
        from matplotlib.path import Path as MplPath
        
        region = self.get_region(region_id)
        if region is None:
            return None
        
        height, width = image_shape
        mask = np.zeros((height, width), dtype=bool)
        coords = np.array(region['coords'], dtype=float)
        
        if region['tool'] == 'rect':
            if len(coords) >= 2:
                x0, y0 = coords[0]
                x1, y1 = coords[1]
                x_min, x_max = int(min(x0, x1)), int(max(x0, x1))
                y_min, y_max = int(min(y0, y1)), int(max(y0, y1))
                mask[y_min:y_max+1, x_min:x_max+1] = True
        
        elif region['tool'] == 'ellipse':
            if len(coords) >= 2:
                x0, y0 = coords[0]
                x1, y1 = coords[1]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                rx, ry = abs(x1 - x0) / 2, abs(y1 - y0) / 2
                
                yy, xx = np.ogrid[:height, :width]
                mask = ((xx - cx)**2 / (rx**2 + 1e-6) + 
                        (yy - cy)**2 / (ry**2 + 1e-6)) <= 1
        
        elif region['tool'] in ('polygon', 'freehand'):
            if len(coords) >= 3:
                # Use matplotlib Path for polygon/freehand
                path = MplPath(coords)
                yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
                points = np.column_stack([xx.ravel(), yy.ravel()])
                mask = path.contains_points(points).reshape((height, width))
        
        return mask
    
    def get_color(self, region_id: int) -> str:
        """Get hex color for a region."""
        region = self.get_region(region_id)
        if region is None:
            return '#999999'
        color_idx = region.get('color_idx', 0) % len(_COLORS)
        return _COLORS[color_idx]
    
    @staticmethod
    def get_color_palette() -> List[str]:
        """Get the full color palette."""
        return _COLORS.copy()


def get_rectangle_patch(coords, edgecolor, facecolor='none', linewidth=2):
    """Create matplotlib Rectangle patch from [top-left, bottom-right] coords."""
    from matplotlib.patches import Rectangle
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    xy = (min(x0, x1), min(y0, y1))
    return Rectangle(xy, width, height, edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth)


def get_ellipse_patch(coords, edgecolor, facecolor='none', linewidth=2):
    """Create matplotlib Ellipse patch from bounding box coords."""
    from matplotlib.patches import Ellipse
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    return Ellipse((cx, cy), width, height, edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth)


def get_polygon_patch(coords, edgecolor, facecolor='none', linewidth=2):
    """Create matplotlib Polygon patch from coords."""
    from matplotlib.patches import Polygon
    return Polygon(coords, edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth, closed=True)


class RoiAnalysisPanel:
    """
    Tab panel for region drawing tools and per-region statistics.
    Coordinates with FOVPreviewPanel's RoiManager to add/display regions.
    """
    
    def __init__(self, parent, fov_preview=None):
        """
        Args:
            parent: Parent tk frame
            fov_preview: Reference to FOVPreviewPanel (set later)
        """
        import tkinter as tk
        from tkinter import ttk
        
        self.frame = ttk.Frame(parent, padding=4)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(2, weight=1)
        
        self.fov_preview = fov_preview  # Set by caller
        self._current_mode = tk.StringVar(value="select")
        self._region_counter = 0
        
        #  Drawing Mode Toolbar 
        toolbar = ttk.LabelFrame(self.frame, text="Drawing Mode", padding=4)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        toolbar.columnconfigure(5, weight=1)  # Spacer
        
        self._btn_select = ttk.Button(toolbar, text="◯ Select", width=10,
                                      command=lambda: self._set_mode("select"))
        self._btn_select.grid(row=0, column=0, sticky="ew", padx=2)
        
        self._btn_rect = ttk.Button(toolbar, text="▭ Rectangle", width=10,
                                    command=lambda: self._set_mode("rect"))
        self._btn_rect.grid(row=0, column=1, sticky="ew", padx=2)
        
        self._btn_ellipse = ttk.Button(toolbar, text="○ Ellipse", width=10,
                                       command=lambda: self._set_mode("ellipse"))
        self._btn_ellipse.grid(row=0, column=2, sticky="ew", padx=2)
        
        self._btn_polygon = ttk.Button(toolbar, text="◇ Polygon", width=10,
                                       command=lambda: self._set_mode("polygon"))
        self._btn_polygon.grid(row=0, column=3, sticky="ew", padx=2)
        
        self._btn_freehand = ttk.Button(toolbar, text="✏ Freehand", width=10,
                                        command=lambda: self._set_mode("freehand"))
        self._btn_freehand.grid(row=0, column=4, sticky="ew", padx=2)
        
        # Column 5 is spacer (configured with weight=1 above)
        
        ttk.Button(toolbar, text="Clear All", width=8,
                   command=self._clear_all_regions).grid(row=0, column=6, sticky="ew", padx=2)
        
        #  Region List 
        list_frame = ttk.LabelFrame(self.frame, text="Regions", padding=4)
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 4))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Treeview for regions
        cols = ("Name", "Type", "τ_med (ns)", "Count", "Color")
        self._tree = ttk.Treeview(list_frame, columns=cols, height=6, show="tree headings")
        self._tree.grid(row=0, column=0, sticky="nsew")
        
        self._tree.column("#0", width=0, stretch=False)
        self._tree.column("Name", anchor="w", width=100)
        self._tree.column("Type", anchor="center", width=70)
        self._tree.column("τ_med (ns)", anchor="center", width=80)
        self._tree.column("Count", anchor="center", width=60)
        self._tree.column("Color", anchor="center", width=50)
        
        self._tree.heading("#0", text="", anchor="w")
        self._tree.heading("Name", text="Name", anchor="w")
        self._tree.heading("Type", text="Type", anchor="center")
        self._tree.heading("τ_med (ns)", text="τ_med (ns)", anchor="center")
        self._tree.heading("Count", text="Photons", anchor="center")
        self._tree.heading("Color", text="Color", anchor="center")
        
        self._tree.bind("<Double-1>", self._on_region_double_click)
        self._tree.bind("<Delete>", self._on_delete_key)
        self._tree.bind("<<TreeviewSelect>>", self._on_region_selection_change)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self._tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscroll=scrollbar.set)
        
        #  Region Actions 
        actions_frame = ttk.Frame(self.frame)
        actions_frame.grid(row=2, column=0, sticky="ew", pady=4)
        actions_frame.columnconfigure(2, weight=1)
        
        ttk.Button(actions_frame, text="Delete Selected", width=15,
                   command=self._delete_selected_region).pack(side="left", padx=2)
        ttk.Button(actions_frame, text="Rename...", width=15,
                   command=self._rename_selected_region).pack(side="left", padx=2)
        ttk.Button(actions_frame, text="Export Region", width=15,
                   command=self._export_selected_region).pack(side="left", padx=2)
        
        # Status label
        self._status = tk.StringVar(value="Ready — Select drawing mode or click regions to add")
        ttk.Label(self.frame, textvariable=self._status, foreground="grey", 
                  font=("Courier", 8)).grid(row=3, column=0, sticky="w", padx=2, pady=2)
    
    def _set_mode(self, mode: str):
        """Set drawing mode and sync with FOVPreviewPanel (Phase 3.3)."""
        self._current_mode.set(mode)
        # Sync with FOVPreviewPanel's drawing mode for event handlers
        if self.fov_preview:
            self.fov_preview._drawing_mode.set(mode)
        self._status.set(f"Mode: {mode.upper()} — Draw on FLIM image")
        print(f"[ROI] Drawing mode: {mode}")
    
    def _clear_all_regions(self):
        """Clear all regions."""
        if self.fov_preview:
            self.fov_preview._roi_manager.clear_all()
            self.fov_preview._redraw_region_overlays()
            self.fov_preview._save_regions_update()
        self._refresh_region_list()
        self._status.set("All regions cleared")
    
    def _on_region_double_click(self, event):
        """Double-click to rename region."""
        selected = self._tree.selection()
        if selected:
            self._rename_selected_region()
    
    def _on_delete_key(self, event):
        """Delete key to remove selected region."""
        self._delete_selected_region()
    
    def _on_region_selection_change(self, event):
        """Handle region selection change — highlight on FLIM image (Phase 4)."""
        selected = self._tree.selection()
        if selected:
            item = selected[0]
            region_id = int(item)
            if self.fov_preview:
                self.fov_preview._roi_manager.select_region(region_id)
                self.fov_preview._redraw_region_overlays()
        else:
            if self.fov_preview:
                self.fov_preview._roi_manager.select_region(None)
                self.fov_preview._redraw_region_overlays()
    
    def _delete_selected_region(self):
        """Delete selected region from tree."""
        selected = self._tree.selection()
        if not selected:
            return
        
        item = selected[0]
        region_id = int(item)  # Item ID is the region_id
        
        if self.fov_preview:
            self.fov_preview._roi_manager.remove_region(region_id)
            self.fov_preview._redraw_region_overlays()
            self.fov_preview._save_regions_update()
        
        self._refresh_region_list()
        self._status.set(f"Deleted region {region_id}")
    
    def _rename_selected_region(self):
        """Rename selected region."""
        import tkinter as tk
        from tkinter import simpledialog
        
        selected = self._tree.selection()
        if not selected:
            return
        
        item = selected[0]
        region_id = int(item)  # Item ID is the region_id
        old_name = self._tree.item(item, "values")[0]  # First value is now name
        
        new_name = simpledialog.askstring("Rename Region", 
                                         f"Enter new name for region:",
                                         initialvalue=old_name)
        if new_name:
            if self.fov_preview:
                self.fov_preview._roi_manager.update_region(region_id, name=new_name)
                self.fov_preview._save_regions_update()
            self._refresh_region_list()
            self._status.set(f"Renamed to '{new_name}'")
    
    def _export_selected_region(self):
        """Export selected region as TIFF (placeholder for Phase 5.2)."""
        import tkinter as tk
        from tkinter import messagebox
        
        selected = self._tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Select a region first")
            return
        
        item = selected[0]
        region_name = self._tree.item(item, "values")[0]  # First value is now name
        
        messagebox.showinfo("Export", f"Export '{region_name}' as TIFF\n(Phase 5.2 — not yet implemented)")
    
    def _refresh_region_list(self):
        """Update region list display from RoiManager."""
        import tkinter as tk
        
        # Clear tree
        for item in self._tree.get_children():
            self._tree.delete(item)
        
        if not self.fov_preview or not self.fov_preview._roi_manager:
            return
        
        # Add regions
        for region in self.fov_preview._roi_manager.get_all_regions():
            region_id = region['id']
            name = region['name']
            tool_type = region['tool'].upper()
            color = self.fov_preview._roi_manager.get_color(region_id)
            
            # Compute statistics if lifetime map available
            tau_med = "—"
            photon_count = "—"
            
            if self.fov_preview._lifetime_map is not None:
                try:
                    mask = self.fov_preview._roi_manager.compute_region_mask(
                        region_id, self.fov_preview._lifetime_map.shape
                    )
                    if mask is not None:
                        lifetime_in_region = self.fov_preview._lifetime_map[mask]
                        valid = lifetime_in_region[~np.isnan(lifetime_in_region)]
                        
                        if valid.size > 0:
                            tau_med = f"{np.median(valid):.2f}"
                            photon_count = str(valid.size)
                except Exception as e:
                    print(f"[ROI] Could not compute stats: {e}")
            
            # Add row with region_id as item ID (iid), not in values
            values = (name, tool_type, tau_med, photon_count, color)
            self._tree.insert("", "end", iid=str(region_id), values=values, tags=(f"color_{region_id}",))
            
            # Color the row by region
            self._tree.tag_configure(f"color_{region_id}", foreground=color)
    
    def add_region_from_drawing(self, tool_type: str, coords: List[List[float]]):
        """Call this when a user finishes drawing a region (Phase 3.2 integration)."""
        if not self.fov_preview:
            return
        
        self._region_counter += 1
        name = f"{tool_type.capitalize()}-{self._region_counter}"
        region_id = self.fov_preview._roi_manager.add_region(name, tool_type, coords)
        
        self.fov_preview._redraw_region_overlays()
        self.fov_preview._save_regions_update()
        self._refresh_region_list()
        
        self._status.set(f"Added region: {name}")
    
    def grid(self, **kw):
        """Grid the frame."""
        self.frame.grid(**kw)
