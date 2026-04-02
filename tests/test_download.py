"""Tests for download utilities."""

from __future__ import annotations

import pytest

from gs_sim2real.common.download import download_dataset


class TestDownloadDataset:
    """Tests for download_dataset."""

    def test_download_unknown_dataset(self, tmp_path: pytest.TempPathFactory) -> None:
        """Requesting an unknown dataset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            download_dataset("totally_unknown_dataset_xyz", output_dir=tmp_path)  # type: ignore[arg-type]
