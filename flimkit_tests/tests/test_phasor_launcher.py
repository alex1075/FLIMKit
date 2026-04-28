import numpy as np
import pytest


def test_get_ptu_active_channels(monkeypatch):
    from flimkit import phasor_launcher
    from flimkit.PTU import reader

    class FakePTUFile:
        def __init__(self, path, verbose=False):
            self.path = path
            self.verbose = verbose

        def _load_records(self):
            return np.array([1, 2, 3, 4], dtype=np.uint32)

        def _decode_picoharp_t3(self, records):
            return np.array([3, 0xF, 1, 3], dtype=np.uint32), None, None

    monkeypatch.setattr(reader, "PTUFile", FakePTUFile)

    assert phasor_launcher.get_ptu_active_channels("fake.ptu") == [1, 3]


def test_resolve_ptu_channel_autoselects_single(monkeypatch):
    from flimkit import phasor_launcher

    monkeypatch.setattr(
        phasor_launcher,
        "get_ptu_active_channels",
        lambda path: [2],
    )

    assert phasor_launcher.resolve_ptu_channel("fake.ptu") == 2


def test_resolve_ptu_channel_uses_prompt_for_multi(monkeypatch):
    from flimkit import phasor_launcher

    chosen = []

    monkeypatch.setattr(
        phasor_launcher,
        "get_ptu_active_channels",
        lambda path: [1, 4],
    )

    def fake_prompt(active_channels):
        chosen.append(active_channels)
        return 4

    assert phasor_launcher.resolve_ptu_channel(
        "fake.ptu",
        prompt_fn=fake_prompt,
    ) == 4
    assert chosen == [[1, 4]]


def test_resolve_ptu_channel_rejects_invalid_explicit_channel(monkeypatch):
    from flimkit import phasor_launcher

    monkeypatch.setattr(
        phasor_launcher,
        "get_ptu_active_channels",
        lambda path: [1, 3],
    )

    with pytest.raises(ValueError, match="Available channels: \[1, 3\]"):
        phasor_launcher.resolve_ptu_channel("fake.ptu", channel=2)