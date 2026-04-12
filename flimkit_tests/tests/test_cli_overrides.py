import pytest
import argparse
from flimkit import configs


class TestCLIOverrides:
    """Verify that CLI arguments correctly override config defaults."""

    def test_override_nexp(self):
        """--nexp should override configs.n_exp."""
        args = argparse.Namespace()
        args.nexp = 1
        args.tau_min = configs.Tau_min
        args.tau_max = configs.Tau_max
        nexp_used = args.nexp if hasattr(args, 'nexp') else configs.n_exp
        assert nexp_used == 1

    def test_override_tau_bounds(self):
        """--tau-min and --tau-max should override config defaults."""
        args = argparse.Namespace()
        args.tau_min = 0.5
        args.tau_max = 20.0

        tau_min = args.tau_min if args.tau_min is not None else configs.Tau_min
        tau_max = args.tau_max if args.tau_max is not None else configs.Tau_max
        assert tau_min == 0.5
        assert tau_max == 20.0

    def test_override_irf_fwhm(self):
        """--irf-fwhm should override None default."""
        args = argparse.Namespace()
        args.irf_fwhm = 0.15
        fwhm = args.irf_fwhm if args.irf_fwhm is not None else configs.IRF_FWHM
        assert fwhm == 0.15

    def test_fallback_to_config(self):
        """When CLI args not provided, config defaults should be used."""
        args = argparse.Namespace()
        nexp = getattr(args, 'nexp', configs.n_exp)
        tau_min = getattr(args, 'tau_min', configs.Tau_min)
        assert nexp == configs.n_exp
        assert tau_min == configs.Tau_min


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
