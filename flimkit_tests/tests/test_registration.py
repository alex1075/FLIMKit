import pytest
import numpy as np
from flimkit.PTU.stitch import _register_tile_columns


def create_tile_results(positions, tile_h=512, tile_w=512):
    results = []
    for i, (py, px) in enumerate(positions):
        y, x = np.mgrid[0:tile_h, 0:tile_w]
        intensity = np.exp(-((y - tile_h/2)**2 + (x - tile_w/2)**2) / (2*50**2)) * 1000
        results.append({
            'ptu_name': f'tile_{i}.ptu',
            'pixel_maps': {'intensity': intensity.astype(np.float32)},
            'pixel_y': py,
            'pixel_x': px,
            'tile_h': tile_h,
            'tile_w': tile_w,
        })
    return results


class TestTileRegistration:
    def test_perfect_alignment_no_shift(self):
        tile_h, tile_w = 512, 512
        positions = [(0,0), (0,tile_w), (tile_h,0), (tile_h,tile_w)]
        tile_results = create_tile_results(positions, tile_h, tile_w)
        for i, tr in enumerate(tile_results):
            tr['_orig_row_idx'] = i // 2
            tr['_orig_col_idx'] = i % 2
        registered = _register_tile_columns(tile_results, max_shift_px=80, verbose=False)
        for orig, reg in zip(tile_results, registered):
            assert reg['pixel_y'] == orig['pixel_y']

    def test_single_tile_no_registration_needed(self):
        tile_results = create_tile_results([(0,0)], 512, 512)
        for tr in tile_results:
            tr['_orig_row_idx'] = 0
            tr['_orig_col_idx'] = 0
        registered = _register_tile_columns(tile_results, max_shift_px=80, verbose=False)
        assert registered[0]['pixel_y'] == 0