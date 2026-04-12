import pytest
import numpy as np
from flimkit.UI.roi_tools import RoiManager


class TestRoiManager:
    @pytest.fixture
    def manager(self):
        return RoiManager()

    def test_add_rectangle(self, manager):
        coords = [[10, 20], [50, 60]]
        rid = manager.add_region("Test Rect", "rect", coords)
        assert manager.get_region(rid)['tool'] == "rect"

    def test_invalid_tool_type_raises(self, manager):
        with pytest.raises(ValueError):
            manager.add_region("Bad", "circle", [[0,0]])

    def test_empty_coords_raises(self, manager):
        with pytest.raises(ValueError):
            manager.add_region("Empty", "rect", [])

    def test_remove_region(self, manager):
        id1 = manager.add_region("R1", "rect", [[0,0],[10,10]])
        id2 = manager.add_region("R2", "rect", [[20,20],[30,30]])
        assert manager.remove_region(id1) is True
        assert len(manager.get_all_regions()) == 1

    def test_update_region(self, manager):
        rid = manager.add_region("Old", "rect", [[0,0],[10,10]])
        assert manager.update_region(rid, name="New")
        assert manager.get_region(rid)['name'] == "New"

    def test_serialize_deserialize(self, manager):
        manager.add_region("Rect1", "rect", [[10,20],[50,60]], color_idx=2)
        json_str = manager.to_json()
        manager2 = RoiManager.from_json(json_str)
        assert len(manager2.get_all_regions()) == 1

    def test_clear_all(self, manager):
        manager.add_region("R1", "rect", [[0,0],[10,10]])
        manager.clear_all()
        assert len(manager.get_all_regions()) == 0

    def test_compute_rectangle_mask(self, manager):
        manager.add_region("Rect", "rect", [[2,3],[5,7]])
        mask = manager.compute_region_mask(0, (10,10))
        assert mask[3,2]      # numpy bool works in boolean context

    def test_compute_ellipse_mask(self, manager):
        manager.add_region("Ellipse", "ellipse", [[2,3],[6,7]])
        mask = manager.compute_region_mask(0, (10,10))
        assert mask[5,4]

    def test_get_color(self, manager):
        id1 = manager.add_region("R1", "rect", [[0,0],[1,1]], color_idx=0)
        id2 = manager.add_region("R2", "rect", [[0,0],[1,1]])
        colors = manager.get_color_palette()
        assert manager.get_color(id1) == colors[0]
        assert manager.get_color(id2) == colors[1]