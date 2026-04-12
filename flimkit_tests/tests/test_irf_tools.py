import pytest
import numpy as np
import tempfile
from pathlib import Path
import json
from unittest.mock import patch

from flimkit.FLIM.irf_tools import (
    build_machine_irf_from_folder,
    reconstruct_irf_from_decay,
    compare_irfs,
    gaussian_irf_from_fwhm,
)
from flimkit_tests.mock_data import MOCK_TCSPC_RES, MOCK_IRF_CENTER, MOCK_IRF_FWHM_BINS


class TestBuildMachineIRF:
    @pytest.fixture
    def paired_folder(self, tmp_path):
        ptu_dir = tmp_path / "pairs"
        ptu_dir.mkdir()
        for i in range(1, 4):
            (ptu_dir / f"sample{i}.ptu").touch()
            (ptu_dir / f"sample{i}.xlsx").touch()
        return ptu_dir

    def test_minimum_pairs_required(self, tmp_path):
        ptu_dir = tmp_path / "single"
        ptu_dir.mkdir()
        (ptu_dir / "sample1.ptu").touch()
        (ptu_dir / "sample1.xlsx").touch()
        with pytest.raises(ValueError, match="at least 2"):
            build_machine_irf_from_folder(ptu_dir)

    @patch('flimkit.FLIM.irf_tools.PTUFile')
    @patch('flimkit.FLIM.irf_tools.load_xlsx')
    def test_build_machine_irf_success(self, mock_load_xlsx, mock_ptu, paired_folder):
        mock_ptu.return_value.n_bins = 256
        mock_ptu.return_value.tcspc_res = 97e-12
        def fake_load_xlsx(path):
            t = np.linspace(0, 5, 21)
            c = np.exp(-((t - 2.5) ** 2) / 0.1)
            return {'irf_t': t, 'irf_c': c}
        mock_load_xlsx.side_effect = fake_load_xlsx
        result = build_machine_irf_from_folder(paired_folder, align_anchor='peak', reducer='median', save=False, confirm_save=False)
        assert 'irf' in result
        assert result['metadata']['n_pairs'] == 3
        assert result['irf'].sum() == pytest.approx(1.0)

    @patch('flimkit.FLIM.irf_tools.PTUFile')
    @patch('flimkit.FLIM.irf_tools.load_xlsx')
    def test_save_requires_confirm(self, mock_load_xlsx, mock_ptu, paired_folder, tmp_path):
        mock_ptu.return_value.n_bins = 256
        mock_ptu.return_value.tcspc_res = 97e-12
        mock_load_xlsx.side_effect = lambda p: {'irf_t': np.linspace(0,5,21), 'irf_c': np.ones(21)}
        out_dir = tmp_path / "machine_irf"
        with pytest.raises(RuntimeError, match="confirm_save=False"):
            build_machine_irf_from_folder(paired_folder, save=True, confirm_save=False, output_dir=out_dir)

    @patch('flimkit.FLIM.irf_tools.PTUFile')
    @patch('flimkit.FLIM.irf_tools.load_xlsx')
    def test_save_writes_files(self, mock_load_xlsx, mock_ptu, paired_folder, tmp_path):
        mock_ptu.return_value.n_bins = 256
        mock_ptu.return_value.tcspc_res = 97e-12
        mock_load_xlsx.side_effect = lambda p: {'irf_t': np.linspace(0,5,21), 'irf_c': np.ones(21)}
        out_dir = tmp_path / "machine_irf"
        out_dir.mkdir()
        result = build_machine_irf_from_folder(paired_folder, save=True, confirm_save=True, output_name="test_irf", output_dir=out_dir)
        assert (out_dir / "test_irf.npy").exists()
        assert (out_dir / "test_irf.csv").exists()
        assert (out_dir / "test_irf_meta.json").exists()


class TestReconstructIRF:
    def test_basic_reconstruction(self):
        n_bins = 256
        tcspc_res = 97e-12
        peak_bin = 50
        decay = np.zeros(n_bins)
        decay[40:51] = np.linspace(10, 1000, 11)
        t = np.arange(n_bins - peak_bin) * tcspc_res
        decay[peak_bin:] = 1000 * np.exp(-t / 2e-9)
        decay += np.random.poisson(5, n_bins)
        irf = reconstruct_irf_from_decay(decay, tcspc_res, n_bins, noise_floor=50, noise_frac=0.001, max_bap=2)
        assert irf.sum() == pytest.approx(1.0)

    def test_zero_decay_raises(self):
        with pytest.raises(ValueError, match="empty or all zeros"):
            reconstruct_irf_from_decay(np.zeros(100), 1e-9, 100)


class TestCompareIRFs:
    def test_alignment_corrects_shift(self):
        n_bins = 256
        tcspc_res = 97e-12
        fwhm_ns = 0.3
        est = gaussian_irf_from_fwhm(n_bins, tcspc_res, fwhm_ns, 50)
        ref = gaussian_irf_from_fwhm(n_bins, tcspc_res, fwhm_ns, 53)
        xlsx = {'irf_t': np.arange(n_bins) * tcspc_res * 1e9, 'irf_c': ref * ref.sum()}
        metrics = compare_irfs(est, xlsx, tcspc_res, n_bins, "test", "out")
        assert metrics['raw']['bhattacharyya'] < 0.9
        assert metrics['aligned']['bhattacharyya'] > 0.99
        assert metrics['peak_shift_bins'] == -3

    def test_missing_xlsx_returns_none(self):
        est = np.ones(100) / 100
        xlsx = {'irf_t': None, 'irf_c': None}
        assert compare_irfs(est, xlsx, 1e-9, 100, "test", "out") is None