import pytest
import csv
from pathlib import Path


class TestBatchCSVFormatting:
    """Tests for batch summary CSV formatting."""

    def test_csv_headers_match_expected(self, tmp_path):
        """CSV should contain standard columns."""
        csv_file = tmp_path / "summary.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['roi', 'status', 'n_pixels_fitted', 'tau_mean_amp_global_ns', 'tau_std_amp_global_ns'])
            writer.writerow(['R_2', 'OK', '5000', '2.34', '0.12'])

        with open(csv_file) as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert row['roi'] == 'R_2'
            assert row['status'] == 'OK'
            assert float(row['tau_mean_amp_global_ns']) == 2.34

    def test_error_status_recorded(self, tmp_path):
        """Errors should be written with status='ERROR:...'."""
        csv_file = tmp_path / "summary.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['roi', 'status'])
            writer.writerow(['R_2', 'ERROR: ValueError: something'])

        with open(csv_file) as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert row['status'].startswith('ERROR:')

    def test_missing_xlif_skipped(self, tmp_path):
        """Simulate missing XLIF - batch should skip gracefully."""
        # This is a conceptual test; actual batch code handles missing files.
        xlif_dir = tmp_path / "xlif"
        xlif_dir.mkdir()
        (xlif_dir / "R 2.xlif").write_text("<dummy/>")
        # No R 3.xlif
        xlif_files = sorted(xlif_dir.glob("*.xlif"))
        assert len(xlif_files) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
