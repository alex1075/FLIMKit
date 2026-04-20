import pytest
import json
import os
import tempfile
from unittest.mock import patch
from flimkit.utils.config_manager import ConfigManager, _DEFAULTS


class TestConfigManagerDefaults:
    """Verify built-in defaults are returned when no config file exists."""

    def test_get_expert_optimizer_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = os.path.join(tmpdir, "config.json")
            with patch('flimkit.utils.config_manager._CONFIG_FILE', cfg_file), \
                 patch('flimkit.utils.config_manager._CONFIG_DIR', tmpdir):
                cm = ConfigManager()
                assert cm.get("expert.optimizer") == "de"

    def test_get_missing_key_returns_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = os.path.join(tmpdir, "config.json")
            with patch('flimkit.utils.config_manager._CONFIG_FILE', cfg_file), \
                 patch('flimkit.utils.config_manager._CONFIG_DIR', tmpdir):
                cm = ConfigManager()
                assert cm.get("expert.nonexistent", "fallback") == "fallback"

    def test_get_whole_section(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = os.path.join(tmpdir, "config.json")
            with patch('flimkit.utils.config_manager._CONFIG_FILE', cfg_file), \
                 patch('flimkit.utils.config_manager._CONFIG_DIR', tmpdir):
                cm = ConfigManager()
                expert = cm.get_section("expert")
                assert "optimizer" in expert
                assert "binning_factor" in expert


class TestConfigManagerSetAndSave:
    """Setting values and persisting to disk."""

    def test_set_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = os.path.join(tmpdir, "config.json")
            with patch('flimkit.utils.config_manager._CONFIG_FILE', cfg_file), \
                 patch('flimkit.utils.config_manager._CONFIG_DIR', tmpdir):
                cm = ConfigManager()
                cm.set("expert.optimizer", "lm_multistart")
                assert cm.get("expert.optimizer") == "lm_multistart"

    def test_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = os.path.join(tmpdir, "config.json")
            with patch('flimkit.utils.config_manager._CONFIG_FILE', cfg_file), \
                 patch('flimkit.utils.config_manager._CONFIG_DIR', tmpdir):
                cm = ConfigManager()
                cm.set("expert.binning_factor", 4)
                cm.save()

                cm2 = ConfigManager()
                assert cm2.get("expert.binning_factor") == 4

    def test_update_section(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = os.path.join(tmpdir, "config.json")
            with patch('flimkit.utils.config_manager._CONFIG_FILE', cfg_file), \
                 patch('flimkit.utils.config_manager._CONFIG_DIR', tmpdir):
                cm = ConfigManager()
                cm.update_section("expert", {"optimizer": "de", "binning_factor": 2})
                assert cm.get("expert.binning_factor") == 2
                # Should have been saved
                assert os.path.exists(cfg_file)


class TestConfigManagerProjectOverrides:
    """Project-level overrides take precedence over global config."""

    def test_project_overrides_global(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = os.path.join(tmpdir, "config.json")
            with patch('flimkit.utils.config_manager._CONFIG_FILE', cfg_file), \
                 patch('flimkit.utils.config_manager._CONFIG_DIR', tmpdir):
                cm = ConfigManager()
                cm.set("expert.optimizer", "de")
                cm.load_project_overrides({"expert": {"optimizer": "lm_multistart"}})
                # Project override wins
                assert cm.get("expert.optimizer") == "lm_multistart"

    def test_clear_project_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = os.path.join(tmpdir, "config.json")
            with patch('flimkit.utils.config_manager._CONFIG_FILE', cfg_file), \
                 patch('flimkit.utils.config_manager._CONFIG_DIR', tmpdir):
                cm = ConfigManager()
                cm.load_project_overrides({"expert": {"optimizer": "lm_multistart"}})
                cm.clear_project_overrides()
                # Falls back to default
                assert cm.get("expert.optimizer") == "de"

    def test_get_project_overrides_returns_copy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = os.path.join(tmpdir, "config.json")
            with patch('flimkit.utils.config_manager._CONFIG_FILE', cfg_file), \
                 patch('flimkit.utils.config_manager._CONFIG_DIR', tmpdir):
                cm = ConfigManager()
                overrides = {"expert": {"optimizer": "de"}}
                cm.load_project_overrides(overrides)
                returned = cm.get_project_overrides()
                # Modifying returned copy shouldn't affect internal state
                returned["expert"]["optimizer"] = "changed"
                assert cm.get("expert.optimizer") == "de"


class TestConfigManagerCorruptFile:
    """Handle corrupt or invalid config files gracefully."""

    def test_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = os.path.join(tmpdir, "config.json")
            with open(cfg_file, "w") as f:
                f.write("not json {{{")
            with patch('flimkit.utils.config_manager._CONFIG_FILE', cfg_file), \
                 patch('flimkit.utils.config_manager._CONFIG_DIR', tmpdir):
                cm = ConfigManager()
                # Should fallback to empty config, defaults still work
                assert cm.get("expert.optimizer") == "de"

    def test_non_dict_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = os.path.join(tmpdir, "config.json")
            with open(cfg_file, "w") as f:
                json.dump([1, 2, 3], f)  # List, not dict
            with patch('flimkit.utils.config_manager._CONFIG_FILE', cfg_file), \
                 patch('flimkit.utils.config_manager._CONFIG_DIR', tmpdir):
                cm = ConfigManager()
                assert cm.get("expert.optimizer") == "de"
