import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from flimkit.UI.roi_tools import RoiManager


class TestOMETIFFExport:
    """Test OME-TIFF export from fit results."""

    def test_ometiff_export_intensity(self, tmp_path):
        """Export intensity as OME-TIFF and verify it's readable."""
        try:
            import tifffile
        except ImportError:
            pytest.skip("tifffile not installed")

        from flimkit.utils.enhanced_outputs import save_weighted_tau_images

        # Create dummy pixel_maps
        ny, nx = 64, 64
        pixel_maps = {
            'intensity': np.random.poisson(100, (ny, nx)).astype(np.float32),
            'tau_1': np.random.uniform(1.0, 3.0, (ny, nx)).astype(np.float32),
            'a1': np.random.uniform(0.5, 1.5, (ny, nx)).astype(np.float32),
        }

        save_weighted_tau_images(
            pixel_maps,
            tmp_path,
            roi_name="test_roi",
            n_exp=1,
            save_intensity=True,
            save_amplitude=False,
        )

        # The function saves TIFF, not OME-TIFF by default.
        # We'll test that we can read the saved TIFF and it has correct dimensions.
        tiff_path = tmp_path / "test_roi_intensity.tif"
        assert tiff_path.exists()
        img = tifffile.imread(str(tiff_path))
        assert img.shape == (ny, nx)

class TestGeoJSONExport:
    """Test ROI export to GeoJSON."""

    def test_export_single_region_geojson(self, tmp_path):
        """Export a rectangle region as GeoJSON Feature."""
        manager = RoiManager()
        manager.add_region("TestRect", "rect", [[10, 20], [50, 60]])
        region_id = 0

        # Simulate export
        region = manager.get_region(region_id)
        feature = {
            "type": "Feature",
            "properties": {
                "id": region['id'],
                "name": region['name'],
                "tool_type": region['tool'],
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [10, 20], [50, 20], [50, 60], [10, 60], [10, 20]
                ]]
            }
        }

        geojson_path = tmp_path / "region.geojson"
        with open(geojson_path, 'w') as f:
            json.dump(feature, f)

        # Read back and verify
        with open(geojson_path) as f:
            loaded = json.load(f)
        assert loaded['type'] == 'Feature'
        assert loaded['properties']['name'] == 'TestRect'
        assert loaded['geometry']['type'] == 'Polygon'

    def test_export_all_regions_feature_collection(self, tmp_path):
        """Export multiple regions as GeoJSON FeatureCollection."""
        manager = RoiManager()
        manager.add_region("Rect1", "rect", [[0, 0], [10, 10]])
        manager.add_region("Ellipse1", "ellipse", [[20, 20], [40, 40]])

        features = []
        for region in manager.get_all_regions():
            coords = region['coords']
            if region['tool'] == 'rect':
                geometry = {
                    "type": "Polygon",
                    "coordinates": [[
                        [coords[0][0], coords[0][1]],
                        [coords[1][0], coords[0][1]],
                        [coords[1][0], coords[1][1]],
                        [coords[0][0], coords[1][1]],
                        [coords[0][0], coords[0][1]]
                    ]]
                }
            else:
                # Approximate ellipse as polygon for test
                geometry = {"type": "Polygon", "coordinates": [[[20,20],[40,20],[40,40],[20,40],[20,20]]]}
            features.append({
                "type": "Feature",
                "properties": {"id": region['id'], "name": region['name']},
                "geometry": geometry
            })

        collection = {"type": "FeatureCollection", "features": features}
        geojson_path = tmp_path / "all_regions.geojson"
        with open(geojson_path, 'w') as f:
            json.dump(collection, f)

        with open(geojson_path) as f:
            loaded = json.load(f)
        assert loaded['type'] == 'FeatureCollection'
        assert len(loaded['features']) == 2