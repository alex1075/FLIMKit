import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from flimkit.UI.roi_tools import (
    RoiManager,
    get_rectangle_patch,
    get_ellipse_patch,
    get_polygon_patch,
)


#  Patch helpers 

class TestPatchHelpers:
    """Ensure matplotlib patches are created correctly from coords."""

    def test_rectangle_patch(self):
        coords = [[10, 20], [50, 60]]
        p = get_rectangle_patch(coords, edgecolor='red')
        assert p.get_xy() == (10, 20)
        assert p.get_width() == 40
        assert p.get_height() == 40

    def test_rectangle_patch_reversed_coords(self):
        """Coords can be given in any order."""
        coords = [[50, 60], [10, 20]]
        p = get_rectangle_patch(coords, edgecolor='red')
        assert p.get_xy() == (10, 20)

    def test_ellipse_patch_center(self):
        coords = [[0, 0], [100, 50]]
        p = get_ellipse_patch(coords, edgecolor='blue')
        assert p.center == (50, 25)
        assert p.width == 100
        assert p.height == 50

    def test_polygon_patch_closes(self):
        coords = [[0, 0], [10, 0], [10, 10], [0, 10]]
        p = get_polygon_patch(coords, edgecolor='green')
        # Polygon should have the vertices
        xy = p.get_xy()
        assert len(xy) >= 4


#  RoiManager — multi-patch storage ─

class TestRoiManagerPatches:
    """Test that the roi_patches dict now stores lists of patches."""

    def test_region_mask_rect(self):
        mgr = RoiManager()
        rid = mgr.add_region("R1", "rect", [[10, 10], [20, 20]])
        mask = mgr.compute_region_mask(rid, (64, 64))
        assert mask is not None
        assert mask.dtype == bool
        assert mask[15, 15] is np.True_
        assert mask[0, 0] is np.False_

    def test_region_mask_ellipse(self):
        mgr = RoiManager()
        rid = mgr.add_region("E1", "ellipse", [[20, 20], [40, 40]])
        mask = mgr.compute_region_mask(rid, (64, 64))
        assert mask is not None
        # Center of ellipse should be inside
        assert mask[30, 30] is np.True_
        # Far corner should be outside
        assert mask[0, 0] is np.False_

    def test_region_mask_polygon(self):
        mgr = RoiManager()
        coords = [[10, 10], [50, 10], [50, 50], [10, 50]]
        rid = mgr.add_region("P1", "polygon", coords)
        mask = mgr.compute_region_mask(rid, (64, 64))
        assert mask is not None
        assert mask[30, 30] is np.True_

    def test_region_mask_nonexistent(self):
        mgr = RoiManager()
        mask = mgr.compute_region_mask(999, (64, 64))
        assert mask is None

    def test_update_region_coords(self):
        """Dragging an ROI updates coords via update_region."""
        mgr = RoiManager()
        rid = mgr.add_region("R1", "rect", [[10, 10], [20, 20]])
        # Simulate drag: shift by (5, 5)
        region = mgr.get_region(rid)
        new_coords = [[c[0] + 5, c[1] + 5] for c in region['coords']]
        mgr.update_region(rid, coords=new_coords)
        updated = mgr.get_region(rid)
        assert updated['coords'][0] == [15.0, 15.0]
        assert updated['coords'][1] == [25.0, 25.0]


#  RoiManager serialization round-trip 

class TestRoiManagerSerialization:
    def test_json_round_trip(self):
        mgr = RoiManager()
        mgr.add_region("R1", "rect", [[0, 0], [10, 10]])
        mgr.add_region("E1", "ellipse", [[5, 5], [15, 15]])
        json_str = mgr.to_json()

        mgr2 = RoiManager.from_json(json_str)
        assert len(mgr2.regions) == 2
        assert mgr2.regions[0]['name'] == "R1"
        assert mgr2.regions[1]['tool'] == "ellipse"

    def test_from_invalid_json(self):
        """Corrupted JSON should not raise."""
        mgr = RoiManager.from_json("not valid json{{{")
        assert len(mgr.regions) == 0

    def test_from_empty_json(self):
        mgr = RoiManager.from_json('{"regions":[], "next_id": 0}')
        assert len(mgr.regions) == 0


#  Selection and color 

class TestRoiSelection:
    def test_select_and_deselect(self):
        mgr = RoiManager()
        rid = mgr.add_region("R1", "rect", [[0, 0], [10, 10]])
        assert mgr.get_selected_id() is None
        mgr.select_region(rid)
        assert mgr.get_selected_id() == rid
        mgr.select_region(None)
        assert mgr.get_selected_id() is None

    def test_color_cycles(self):
        mgr = RoiManager()
        colors = []
        for i in range(8):
            rid = mgr.add_region(f"R{i}", "rect", [[0, 0], [1, 1]])
            colors.append(mgr.get_color(rid))
        # After 6 regions, colors should cycle
        assert colors[0] == colors[6]

    def test_nonexistent_region_color(self):
        mgr = RoiManager()
        assert mgr.get_color(999) == '#999999'

    def test_remove_selected_clears_selection(self):
        mgr = RoiManager()
        rid = mgr.add_region("R1", "rect", [[0, 0], [10, 10]])
        mgr.select_region(rid)
        mgr.remove_region(rid)
        assert mgr.get_selected_id() is None

    def test_clear_all(self):
        mgr = RoiManager()
        mgr.add_region("R1", "rect", [[0, 0], [10, 10]])
        mgr.add_region("R2", "ellipse", [[5, 5], [15, 15]])
        mgr.clear_all()
        assert len(mgr.regions) == 0
        assert mgr.get_selected_id() is None


#  Validation ─

class TestRoiValidation:
    def test_invalid_tool_type_raises(self):
        mgr = RoiManager()
        with pytest.raises(ValueError, match="Invalid tool_type"):
            mgr.add_region("Bad", "circle", [[0, 0]])

    def test_empty_coords_raises(self):
        mgr = RoiManager()
        with pytest.raises(ValueError, match="coords cannot be empty"):
            mgr.add_region("Empty", "rect", [])
