import json
import os
import copy

_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".flimkit")
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "config.json")

# Keys and their defaults — flat sections
_DEFAULTS = {
    "expert": {
        "binning_factor": 1,
        "optimizer": "de",
        "lm_restarts": 8,
        "de_population": 30,
        "de_maxiter": 5000,
        "n_workers": -1,
        "cost_function": "poisson",
        "channels": "",
        "min_photons": 10,
    },
    "preferences": {
        "colormap": "viridis",
        "font_size": 9,
        "default_nexp": 2,
        "export_format": "CSV",
        "output_directory": "",
        "auto_save_npz": True,
    },
}


class ConfigManager:
    """JSON-backed, two-layer configuration store."""

    def __init__(self):
        self._global: dict = {}      # persisted to config.yaml
        self._project: dict = {}     # transient, from project.json
        self._load()

    # ── public API ──────────────────────────────────────────────

    def get(self, dotted_key: str, default=None):
        """
        Retrieve a value by dotted key, e.g. ``"expert.optimizer"``.

        Resolution order: project override → global config → built-in default.
        """
        section, _, key = dotted_key.partition(".")
        if not key:
            # whole-section request
            merged = copy.deepcopy(_DEFAULTS.get(section, {}))
            merged.update(self._global.get(section, {}))
            merged.update(self._project.get(section, {}))
            return merged

        # Per-key lookup with layered fallback
        if section in self._project and key in self._project[section]:
            return self._project[section][key]
        if section in self._global and key in self._global[section]:
            return self._global[section][key]
        if section in _DEFAULTS and key in _DEFAULTS[section]:
            return _DEFAULTS[section][key]
        return default

    def get_section(self, section: str) -> dict:
        """Return a merged dict for an entire section."""
        return self.get(section)

    def set(self, dotted_key: str, value):
        """Set a global config value (does NOT auto-save)."""
        section, _, key = dotted_key.partition(".")
        if section not in self._global:
            self._global[section] = {}
        if key:
            self._global[section][key] = value
        else:
            self._global[section] = value

    def update_section(self, section: str, data: dict):
        """Bulk-update a section in global config and save."""
        if section not in self._global:
            self._global[section] = {}
        self._global[section].update(data)
        self.save()

    def load_project_overrides(self, overrides: dict):
        """Load project-level overrides (transient, not persisted globally)."""
        self._project = overrides or {}

    def clear_project_overrides(self):
        """Remove project-level overrides."""
        self._project = {}

    def get_project_overrides(self) -> dict:
        """Return the current project overrides dict (for saving to project.json)."""
        return copy.deepcopy(self._project)

    # ── persistence ─────────────────────────────────────────────

    def _load(self):
        """Load config from JSON file."""
        # Also migrate old YAML config if it exists
        yaml_path = os.path.join(_CONFIG_DIR, "config.yaml")
        if not os.path.exists(_CONFIG_FILE) and os.path.exists(yaml_path):
            try:
                import yaml
                with open(yaml_path, "r") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    self._global = data
                    self.save()
                    return
            except Exception:
                pass
        if not os.path.exists(_CONFIG_FILE):
            self._global = {}
            return
        try:
            with open(_CONFIG_FILE, "r") as f:
                data = json.load(f)
            self._global = data if isinstance(data, dict) else {}
        except Exception as exc:
            print(f"[Config] Warning: could not read {_CONFIG_FILE}: {exc}")
            self._global = {}

    def save(self):
        """Persist global config to JSON file."""
        os.makedirs(_CONFIG_DIR, exist_ok=True)
        with open(_CONFIG_FILE, "w") as f:
            json.dump(self._global, f, indent=2)

    def reload(self):
        """Re-read from disk."""
        self._load()


# Module-level singleton
cfg = ConfigManager()
