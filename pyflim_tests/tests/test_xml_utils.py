"""Unit Tests for xml_utils Module

Tests XLIF parsing, tile position computation, and metadata extraction.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Import test utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from mock_data import generate_mock_xlif


class TestXMLUtils:
    """Test suite for xml_utils module."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def mock_xlif_2x2(self, temp_dir):
        """Create a 2x2 tile XLIF file."""
        xlif_path = generate_mock_xlif(
            temp_dir / "test.xlif",
            n_tiles=4,
            layout="2x2",
            pixel_size_m=3e-7
        )
        return xlif_path
    
    @pytest.fixture
    def mock_xlif_1x4(self, temp_dir):
        """Create a 1x4 tile XLIF file."""
        xlif_path = generate_mock_xlif(
            temp_dir / "test_1x4.xlif",
            n_tiles=4,
            layout="1x4",
            pixel_size_m=3e-7
        )
        return xlif_path
    
    def test_parse_xlif_tile_positions(self, mock_xlif_2x2):
        """Test parsing tile positions from XLIF."""
        from pyflim.utils.xml_utils import parse_xlif_tile_positions
        
        tiles = parse_xlif_tile_positions(mock_xlif_2x2, ptu_basename="R 2")
        
        # Check we got 4 tiles
        assert len(tiles) == 4
        
        # Check structure
        for tile in tiles:
            assert 'file' in tile
            assert 'field_x' in tile
            assert 'pos_x' in tile
            assert 'pos_y' in tile
        
        # Check filenames are correct (s1, s2, s3, s4)
        filenames = [t['file'] for t in tiles]
        expected = ['R 2_s1.ptu', 'R 2_s2.ptu', 'R 2_s3.ptu', 'R 2_s4.ptu']
        assert filenames == expected
        
        # Check positions are numbers
        for tile in tiles:
            assert isinstance(tile['pos_x'], float)
            assert isinstance(tile['pos_y'], float)
    
    def test_get_pixel_size_from_xlif(self, mock_xlif_2x2):
        """Test extracting pixel size from XLIF."""
        from pyflim.utils.xml_utils import get_pixel_size_from_xlif
        
        pixel_size_m, n_pixels = get_pixel_size_from_xlif(mock_xlif_2x2)
        
        # Check values are reasonable
        assert isinstance(pixel_size_m, float)
        assert isinstance(n_pixels, int)
        assert pixel_size_m > 0
        assert n_pixels > 0
        
        # Check expected values
        assert abs(pixel_size_m - 3e-7) < 1e-9  # Within tolerance
        assert n_pixels == 512
    
    def test_compute_tile_pixel_positions(self, mock_xlif_2x2):
        """Test converting physical positions to pixels."""
        from pyflim.utils.xml_utils import (
            parse_xlif_tile_positions,
            get_pixel_size_from_xlif,
            compute_tile_pixel_positions
        )
        
        tiles = parse_xlif_tile_positions(mock_xlif_2x2, "R 2")
        pixel_size_m, _ = get_pixel_size_from_xlif(mock_xlif_2x2)
        
        tiles, canvas_width, canvas_height = compute_tile_pixel_positions(
            tiles, pixel_size_m, tile_size=512
        )
        
        # Check canvas size (2x2 tiles, 512x512 each)
        assert canvas_width == 1024  # 2 tiles * 512
        assert canvas_height == 1024
        
        # Check all tiles have pixel positions
        for tile in tiles:
            assert 'pixel_x' in tile
            assert 'pixel_y' in tile
            assert isinstance(tile['pixel_x'], int)
            assert isinstance(tile['pixel_y'], int)
        
        # Check positions are within canvas
        for tile in tiles:
            assert 0 <= tile['pixel_x'] < canvas_width
            assert 0 <= tile['pixel_y'] < canvas_height
    
    def test_extract_roi_number(self):
        """Test extracting ROI number from filename."""
        from pyflim.utils.xml_utils import extract_roi_number
        
        assert extract_roi_number("R 2_s1.ptu") == 2
        assert extract_roi_number("R 10_s3.ptu") == 10
        assert extract_roi_number("R123_tile5.ptu") == 123
        assert extract_roi_number("no_roi_here.ptu") is None
    
    def test_match_xml_ptu_sets(self, temp_dir):
        """Test matching XML and PTU files."""
        from pyflim.utils.xml_utils import match_xml_ptu_sets
        
        # Create test structure
        metadata_dir = temp_dir / "Metadata"
        metadata_dir.mkdir()
        
        # Create XLIF files
        generate_mock_xlif(metadata_dir / "R 2.xlif", n_tiles=4)
        generate_mock_xlif(metadata_dir / "R 3.xlif", n_tiles=4)
        
        # Create PTU files
        (temp_dir / "R 2_s1.ptu").touch()
        (temp_dir / "R 2_s2.ptu").touch()
        (temp_dir / "R 3_s1.ptu").touch()
        
        # Match
        matches = match_xml_ptu_sets(temp_dir)
        
        # Should find 2 ROIs
        assert len(matches) == 2
        
        # Check structure
        for match in matches:
            assert 'R_number' in match
            assert 'xml_files' in match
            assert 'ptu_files' in match
            assert 'status' in match
        
        # Check R2 is matched
        r2_match = [m for m in matches if m['R_number'] == 'R2'][0]
        assert r2_match['status'] == 'MATCHED'
        assert r2_match['ptu_count'] == 2
        assert r2_match['xml_count'] == 1


def test_xlif_with_missing_tilescan():
    """Test XLIF without TileScanInfo raises error."""
    from pyflim.utils.xml_utils import parse_xlif_tile_positions
    import xml.etree.ElementTree as ET
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create XLIF without TileScanInfo
        root = ET.Element("LMSDataContainerHeader")
        tree = ET.ElementTree(root)
        xlif_path = Path(temp_dir) / "bad.xlif"
        tree.write(xlif_path)
        
        with pytest.raises(RuntimeError, match="No TileScanInfo"):
            parse_xlif_tile_positions(xlif_path, "R 2")


def test_xlif_different_layouts():
    """Test XLIF parsing with different tile layouts."""
    from pyflim.utils.xml_utils import parse_xlif_tile_positions
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test 3x3 layout
        xlif_3x3 = generate_mock_xlif(temp_path / "3x3.xlif", n_tiles=9, layout="3x3")
        tiles = parse_xlif_tile_positions(xlif_3x3, "R 2")
        assert len(tiles) == 9
        
        # Test 1x5 layout
        xlif_1x5 = generate_mock_xlif(temp_path / "1x5.xlif", n_tiles=5, layout="1x5")
        tiles = parse_xlif_tile_positions(xlif_1x5, "R 2")
        assert len(tiles) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
