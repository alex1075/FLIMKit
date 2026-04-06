#!/usr/bin/env python3
"""
Robustly replace test_missing_tiles in test_complete_pipeline.py.
"""

import sys
import re
import difflib
import pathlib

CHECK_ONLY = "--check" in sys.argv
TEST_FILE = pathlib.Path("flimkit_tests/test_complete_pipeline.py")

# The corrected method (use exactly this)
CORRECTED_METHOD = '''    def test_missing_tiles(self):
        """Test handling of missing PTU tiles."""
        try:
            from flimkit.PTU.stitch import stitch_flim_tiles
            from mock_data import generate_mock_xlif, generate_mock_ptu_tiles

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Create XLIF for 4 tiles
                xlif_path = generate_mock_xlif(
                    temp_path / "test.xlif",
                    n_tiles=4,
                    layout="2x2"
                )

                # Create PTU directory with only 2 tiles
                ptu_dir = temp_path / "PTUs"
                ptu_dir.mkdir()

                # Generate valid PTU files for the first two tiles only
                generate_mock_ptu_tiles(
                    ptu_dir,
                    ptu_basename="R 2",
                    n_tiles=2,          # only s1 and s2
                    tile_shape=(512, 512),
                    n_bins=256
                )
                # Tiles s3 and s4 are intentionally missing

                output_dir = temp_path / "output"

                result = stitch_flim_tiles(
                    xlif_path=xlif_path,
                    ptu_dir=ptu_dir,
                    output_dir=output_dir,
                    ptu_basename="R 2",
                    verbose=False
                )

                assert result['tiles_processed'] == 2
                assert result['tiles_skipped'] == 2

        except ImportError:
            pytest.skip("stitch module not available")
'''

def find_method_block(lines):
    """Return (start_idx, end_idx) for the test_missing_tiles method."""
    start_pattern = re.compile(r'^\s*def test_missing_tiles\(self\):')
    start_idx = None
    for i, line in enumerate(lines):
        if start_pattern.match(line):
            start_idx = i
            break
    if start_idx is None:
        return None, None

    # Determine indentation of the method definition
    indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
    # Look for the next line that starts with 'def' at the same or lower indentation
    end_idx = len(lines) - 1
    for i in range(start_idx + 1, len(lines)):
        if lines[i].strip().startswith('def ') and (len(lines[i]) - len(lines[i].lstrip())) <= indent:
            end_idx = i - 1
            break
    return start_idx, end_idx

def main():
    if not TEST_FILE.exists():
        print(f"ERROR: {TEST_FILE} not found. Run from FLIMKit root.")
        sys.exit(1)

    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()

    start, end = find_method_block(original_lines)
    if start is None:
        print("Could not locate 'def test_missing_tiles(self):'")
        print("The method may already be corrected or renamed.")
        sys.exit(1)

    # Build new file content
    new_lines = original_lines[:start] + CORRECTED_METHOD.splitlines(keepends=True) + original_lines[end+1:]
    patched_text = ''.join(new_lines)
    original_text = ''.join(original_lines)

    if CHECK_ONLY:
        diff = difflib.unified_diff(
            original_text.splitlines(keepends=True),
            patched_text.splitlines(keepends=True),
            fromfile=str(TEST_FILE),
            tofile=str(TEST_FILE) + " (patched)"
        )
        sys.stdout.write("".join(diff))
        print("\nDry-run only — no files written.")
        return

    TEST_FILE.write_text(patched_text, encoding='utf-8')
    print(f"✓ Patched {TEST_FILE}")
    print("  - Replaced test_missing_tiles with the corrected version.")
    print("\nNow run the test:")
    print("  pytest flimkit_tests/test_complete_pipeline.py::TestEdgeCases::test_missing_tiles -v")

if __name__ == "__main__":
    main()