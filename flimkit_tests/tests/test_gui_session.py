import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock


class MockApp:
    """Minimal app containing the session methods we want to test."""

    def __init__(self):
        self.sv_ptu = MagicMock()
        self.sv_ptu.get.return_value = "/test/sample.ptu"
        self.sv_xlsx = MagicMock()
        self.sv_xlsx.get.return_value = "/test/sample.xlsx"
        self._irf_fov = MagicMock()
        self._irf_fov.sv_method = MagicMock(get=lambda: "irf_xlsx")
        self._irf_fov.sv_path = MagicMock(get=lambda: "")
        self.iv_nexp_fov = MagicMock(get=lambda: 2)
        self.sv_tau_fit_lo = MagicMock(get=lambda: "0.1")
        self.sv_tau_fit_hi = MagicMock(get=lambda: "10.0")
        self._fov_preview = MagicMock()
        self._fov_preview._ptu_path = None
        self._fov_preview._lifetime_map = None
        self._fov_preview._intensity_map = None
        self._fov_preview._flim_color_scale = {}
        self._fov_preview._n_exp = 2
        self._fov_preview._roi_manager = MagicMock()
        self._fov_preview._roi_manager.to_json.return_value = '{"regions":[]}'
        self._current_form = "fov"
        self._form_buttons = {"fov": MagicMock()}

    def _capture_form_state(self):
        state = {
            "active_form": self._current_form,
            "ptu_file": self.sv_ptu.get(),
            "xlsx_file": self.sv_xlsx.get(),
            "irf_method": self._irf_fov.sv_method.get(),
            "irf_file": self._irf_fov.sv_path.get(),
            "nexp_fov": self.iv_nexp_fov.get(),
            "tau_fit_lo": self.sv_tau_fit_lo.get(),
            "tau_fit_hi": self.sv_tau_fit_hi.get(),
        }
        return state

    def _restore_form_state(self, state):
        if "ptu_file" in state:
            self.sv_ptu.set(state["ptu_file"])
        if "xlsx_file" in state:
            self.sv_xlsx.set(state["xlsx_file"])
        if "irf_method" in state:
            self._irf_fov.sv_method.set(state["irf_method"])
        if "irf_file" in state:
            self._irf_fov.sv_path.set(state["irf_file"])
        if "nexp_fov" in state:
            self.iv_nexp_fov.set(state["nexp_fov"])
        if "active_form" in state:
            self._current_form = state["active_form"]

    def _save_roi_progress(self, path, fit_result, summary_rows):
        from datetime import datetime
        base_path = Path(path)
        session_file = base_path.parent / f"{base_path.stem}.roi_session.npz"

        form_state = self._capture_form_state()
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "source": str(path),
            "form_state_json": json.dumps(form_state, default=str),
        }
        for key, val in fit_result.items():
            if isinstance(val, np.ndarray):
                session_data[key] = val
            elif isinstance(val, dict):
                hoisted = {}
                json_safe = {}
                for k2, v2 in val.items():
                    if isinstance(v2, np.ndarray):
                        arr_key = f"{key}_arr_{k2}"
                        hoisted[arr_key] = v2
                    else:
                        json_safe[k2] = v2
                session_data[f"{key}_json"] = json.dumps(json_safe, default=str)
                session_data.update(hoisted)
        np.savez_compressed(session_file, **session_data)
        return str(session_file)


class TestGUISession:
    """Tests for GUI form state capture and restore."""

    def test_capture_form_state(self):
        app = MockApp()
        state = app._capture_form_state()
        assert state["ptu_file"] == "/test/sample.ptu"
        assert state["xlsx_file"] == "/test/sample.xlsx"
        assert state["irf_method"] == "irf_xlsx"
        assert state["nexp_fov"] == 2
        assert state["active_form"] == "fov"

    def test_restore_form_state(self):
        app = MockApp()
        state = {
            "ptu_file": "/new/test.ptu",
            "nexp_fov": 3,
            "active_form": "stitch",
        }
        app._restore_form_state(state)
        app.sv_ptu.set.assert_called_with("/new/test.ptu")
        app.iv_nexp_fov.set.assert_called_with(3)
        assert app._current_form == "stitch"

    def test_save_roi_progress_roundtrip(self, tmp_path):
        """Save a session with fit_result and reload to verify array integrity."""
        app = MockApp()
        # Override the mock PTU path to use the temp path
        ptu_path = tmp_path / "sample.ptu"
        ptu_path.write_text("dummy")
        app.sv_ptu.get.return_value = str(ptu_path)

        fit_result = {
            'intensity': np.random.rand(64, 64).astype(np.float32),
            'lifetime': np.random.rand(64, 64).astype(np.float32),
            'global_summary': {
                'taus_ns': [2.0, 0.5],
                'model': np.random.rand(256).astype(np.float32),
            },
            'decay': np.random.poisson(100, 256).astype(np.float32),
        }
        summary_rows = [("τ1", "2.00", "ns"), ("τ2", "0.50", "ns")]

        session_file = app._save_roi_progress(str(ptu_path), fit_result, summary_rows)
        assert Path(session_file).exists()

        loaded = np.load(session_file, allow_pickle=True)
        assert 'intensity' in loaded
        assert loaded['intensity'].shape == (64, 64)
        assert 'lifetime' in loaded
        assert 'global_summary_arr_model' in loaded
        assert loaded['global_summary_arr_model'].shape == (256,)

        form_state = json.loads(str(loaded['form_state_json']))
        assert form_state['ptu_file'] == str(ptu_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
