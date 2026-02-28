"""Tests for src/encode_client.py.

Test strategy
-------------
All tests run without real BigWig files or the pyBigWig library.

- :class:`_MockBigWig` mimics the pyBigWig file handle interface
  (``chroms()``, ``values()``, ``close()``).
- :func:`_open_bigwig` is patched in integration tests so that manifest
  paths can be fake (zero-byte sentinel files in a tmp_path directory).
- Internal helpers (_query_region, _extract_signal_stats, _extract_from_open_bws)
  are tested directly without any patching.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.encode_client import (
    ASSAY_NAMES,
    DEFAULT_WINDOW_BP,
    DNASE_OPEN_THRESHOLD,
    FEATURE_NAMES,
    FILE_TEMPLATE,
    NARROW_WINDOW_BP,
    BigWigManifest,
    _extract_from_open_bws,
    _extract_signal_stats,
    _nan_features,
    _open_bigwig,
    _query_region,
    batch_get_chromatin_features,
    build_manifest,
    get_chromatin_features,
    list_required_files,
)
from src.utils import GenomicInterval


# ---------------------------------------------------------------------------
# Mock BigWig
# ---------------------------------------------------------------------------

class _MockBigWig:
    """Lightweight mock for a pybigtools BigWig file handle.

    Parameters
    ----------
    chrom_lengths : dict[str, int]
        Maps chromosome name → length in bp.
    constant_signal : float | None
        Constant value returned for every position.  ``None`` simulates
        a BigWig where every position has no coverage (pybigtools fills
        those positions with the ``missing`` parameter value).
    """

    def __init__(
        self,
        chrom_lengths: dict[str, int],
        constant_signal: float | None = 5.0,
    ) -> None:
        self._chroms = dict(chrom_lengths)
        self._signal = constant_signal
        self.closed = False

    def chroms(self) -> dict[str, int]:
        return dict(self._chroms)

    def values(
        self,
        chrom: str,
        start: int,
        end: int,
        missing: float = float("nan"),
    ) -> np.ndarray:
        n = end - start
        fill = missing if self._signal is None else self._signal
        return np.full(n, fill, dtype=np.float64)

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_bigwig_dir(tmp_path: Path) -> Path:
    """Return a temporary directory with sentinel BigWig files for K562.

    Files are zero-byte — they exist only for path-existence checks.
    Actual reads are intercepted by patching _open_bigwig.
    """
    for assay in ASSAY_NAMES:
        fname = FILE_TEMPLATE.format(cell_type="K562", assay=assay)
        (tmp_path / fname).write_bytes(b"")
    return tmp_path


@pytest.fixture()
def k562_manifest(tmp_bigwig_dir: Path) -> BigWigManifest:
    """Manifest populated with all K562 assays pointing to sentinel files."""
    return build_manifest(tmp_bigwig_dir)


@pytest.fixture()
def mock_bw_factory():
    """Return a factory that creates _MockBigWig instances on demand."""
    def factory(
        chrom_lengths: dict[str, int] | None = None,
        constant_signal: float | None = 5.0,
    ) -> _MockBigWig:
        if chrom_lengths is None:
            chrom_lengths = {"chr7": 159_345_973}
        return _MockBigWig(chrom_lengths, constant_signal)
    return factory


# ---------------------------------------------------------------------------
# Constants and module-level checks
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_feature_names_length(self) -> None:
        assert len(FEATURE_NAMES) == 8

    def test_feature_names_prefix(self) -> None:
        for name in FEATURE_NAMES:
            assert name.startswith("exp_"), f"{name!r} should start with 'exp_'"

    def test_assay_names(self) -> None:
        assert set(ASSAY_NAMES) == {"DNase", "H3K4me3", "H3K27me3", "H3K9me3", "H3K27ac"}

    def test_default_window_bp(self) -> None:
        assert DEFAULT_WINDOW_BP == 500

    def test_narrow_window_bp(self) -> None:
        assert NARROW_WINDOW_BP == 50

    def test_dnase_threshold(self) -> None:
        assert DNASE_OPEN_THRESHOLD == 1.0

    def test_file_template(self) -> None:
        fname = FILE_TEMPLATE.format(cell_type="K562", assay="DNase")
        assert fname == "K562_DNase_hg38.bigWig"


# ---------------------------------------------------------------------------
# list_required_files
# ---------------------------------------------------------------------------

class TestListRequiredFiles:
    def test_returns_ten_files(self) -> None:
        files = list_required_files()
        assert len(files) == 10  # 2 cell types × 5 assays

    def test_all_k562(self) -> None:
        files = list_required_files()
        k562 = [f for f in files if f.startswith("K562")]
        assert len(k562) == 5

    def test_all_hek293(self) -> None:
        files = list_required_files()
        hek = [f for f in files if f.startswith("HEK293")]
        assert len(hek) == 5

    def test_correct_extension(self) -> None:
        for f in list_required_files():
            assert f.endswith(".bigWig"), f

    def test_contains_dnase(self) -> None:
        files = list_required_files()
        assert "K562_DNase_hg38.bigWig" in files
        assert "HEK293_DNase_hg38.bigWig" in files


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------

class TestBuildManifest:
    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        manifest = build_manifest(tmp_path / "nonexistent")
        assert len(manifest.files) == 0
        assert not manifest.has_any("K562")

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        manifest = build_manifest(tmp_path)
        assert len(manifest.files) == 0

    def test_finds_k562_files(self, tmp_bigwig_dir: Path) -> None:
        manifest = build_manifest(tmp_bigwig_dir)
        assert manifest.has_any("K562")
        assert len(manifest.available_assays("K562")) == 5

    def test_does_not_find_hek293(self, tmp_bigwig_dir: Path) -> None:
        manifest = build_manifest(tmp_bigwig_dir)
        assert not manifest.has_any("HEK293")

    def test_ignores_unknown_files(self, tmp_path: Path) -> None:
        (tmp_path / "random_file.bigWig").write_bytes(b"")
        manifest = build_manifest(tmp_path)
        assert len(manifest.files) == 0

    def test_partial_assays(self, tmp_path: Path) -> None:
        # Only DNase and H3K4me3 present for K562
        for assay in ("DNase", "H3K4me3"):
            (tmp_path / f"K562_{assay}_hg38.bigWig").write_bytes(b"")
        manifest = build_manifest(tmp_path)
        assert set(manifest.available_assays("K562")) == {"DNase", "H3K4me3"}

    def test_get_returns_path(self, k562_manifest: BigWigManifest) -> None:
        path = k562_manifest.get("K562", "DNase")
        assert path is not None
        assert path.name == "K562_DNase_hg38.bigWig"

    def test_get_missing_returns_none(self, k562_manifest: BigWigManifest) -> None:
        assert k562_manifest.get("HEK293", "DNase") is None
        assert k562_manifest.get("K562", "ATAC") is None

    def test_available_cell_types(self, k562_manifest: BigWigManifest) -> None:
        assert k562_manifest.available_cell_types() == ["K562"]

    def test_available_assays_sorted(self, k562_manifest: BigWigManifest) -> None:
        assays = k562_manifest.available_assays("K562")
        assert assays == sorted(assays)


# ---------------------------------------------------------------------------
# _nan_features
# ---------------------------------------------------------------------------

class TestNanFeatures:
    def test_keys_match_feature_names(self) -> None:
        nf = _nan_features()
        assert set(nf.keys()) == set(FEATURE_NAMES)

    def test_all_values_nan(self) -> None:
        for v in _nan_features().values():
            assert math.isnan(v)


# ---------------------------------------------------------------------------
# _query_region
# ---------------------------------------------------------------------------

class TestQueryRegion:
    def _make_bw(
        self,
        chrom_len: int = 1_000_000,
        signal: float | None = 3.0,
    ) -> _MockBigWig:
        return _MockBigWig({"chr1": chrom_len}, signal)

    def test_normal_query(self) -> None:
        bw = self._make_bw(signal=4.0)
        arr = _query_region(bw, "chr1", 1000, 1100)
        assert len(arr) == 100
        assert np.all(arr == 4.0)

    def test_none_becomes_nan(self) -> None:
        bw = self._make_bw(signal=None)
        arr = _query_region(bw, "chr1", 1000, 1100)
        assert len(arr) == 100
        assert np.all(np.isnan(arr))

    def test_missing_chrom_returns_nan(self) -> None:
        bw = self._make_bw()
        arr = _query_region(bw, "chr99", 1000, 2000)
        assert len(arr) == 1000
        assert np.all(np.isnan(arr))

    def test_empty_range_returns_empty(self) -> None:
        bw = self._make_bw()
        arr = _query_region(bw, "chr1", 500, 500)
        assert len(arr) == 0

    def test_start_at_zero(self) -> None:
        bw = self._make_bw(signal=7.0)
        arr = _query_region(bw, "chr1", 0, 200)
        assert len(arr) == 200
        assert np.all(arr == 7.0)

    def test_clamp_near_chrom_start(self) -> None:
        """Site with start < 0 should be NaN-padded on the left."""
        bw = self._make_bw(chrom_len=1_000_000, signal=2.0)
        # Request [-100, 100): only [0, 100) is valid
        arr = _query_region(bw, "chr1", -100, 100)
        assert len(arr) == 200
        # First 100 positions (chrom boundary region) → NaN
        assert np.all(np.isnan(arr[:100]))
        # Last 100 positions (valid) → 2.0
        assert np.all(arr[100:] == 2.0)

    def test_clamp_near_chrom_end(self) -> None:
        """Request that extends past chromosome end should be NaN-padded."""
        chrom_len = 500
        bw = self._make_bw(chrom_len=chrom_len, signal=9.0)
        # Request [400, 600): only [400, 500) is valid
        arr = _query_region(bw, "chr1", 400, 600)
        assert len(arr) == 200
        assert np.all(arr[:100] == 9.0)
        assert np.all(np.isnan(arr[100:]))

    def test_entirely_past_chrom_end(self) -> None:
        bw = self._make_bw(chrom_len=100, signal=1.0)
        arr = _query_region(bw, "chr1", 200, 300)
        assert len(arr) == 100
        assert np.all(np.isnan(arr))

    def test_dtype_is_float64(self) -> None:
        bw = self._make_bw(signal=1.0)
        arr = _query_region(bw, "chr1", 0, 10)
        assert arr.dtype == np.float64


# ---------------------------------------------------------------------------
# _extract_signal_stats
# ---------------------------------------------------------------------------

class TestExtractSignalStats:
    def test_uniform_signal(self) -> None:
        arr = np.full(1000, 5.0)
        mean_v, max_v, at_site = _extract_signal_stats(arr, 500, 200)
        assert math.isclose(mean_v, 5.0)
        assert math.isclose(max_v, 5.0)
        assert math.isclose(at_site, 5.0)

    def test_at_site_value(self) -> None:
        arr = np.zeros(1000)
        arr[500] = 99.0
        _, _, at_site = _extract_signal_stats(arr, 500, 10)
        assert math.isclose(at_site, 99.0)

    def test_max_captures_peak(self) -> None:
        arr = np.full(1000, 1.0)
        arr[600] = 50.0
        mean_v, max_v, _ = _extract_signal_stats(arr, 600, 200)
        assert math.isclose(max_v, 50.0)
        assert mean_v < 50.0

    def test_all_nan_returns_nan(self) -> None:
        arr = np.full(100, np.nan)
        mean_v, max_v, at_site = _extract_signal_stats(arr, 50, 20)
        assert math.isnan(mean_v)
        assert math.isnan(max_v)
        assert math.isnan(at_site)

    def test_empty_array(self) -> None:
        arr = np.array([], dtype=np.float64)
        mean_v, max_v, at_site = _extract_signal_stats(arr, 0, 100)
        assert math.isnan(mean_v)
        assert math.isnan(max_v)
        assert math.isnan(at_site)

    def test_site_offset_out_of_bounds(self) -> None:
        arr = np.full(100, 3.0)
        _, _, at_site = _extract_signal_stats(arr, 200, 50)
        assert math.isnan(at_site)

    def test_site_at_edge_zero(self) -> None:
        arr = np.full(100, 2.0)
        mean_v, max_v, at_site = _extract_signal_stats(arr, 0, 10)
        assert not math.isnan(at_site)
        assert math.isclose(at_site, 2.0)

    def test_window_clamps_to_array(self) -> None:
        """Window larger than array should not raise."""
        arr = np.full(50, 3.0)
        mean_v, max_v, at_site = _extract_signal_stats(arr, 25, 500)
        assert math.isclose(mean_v, 3.0)
        assert math.isclose(max_v, 3.0)

    def test_nan_ignored_in_mean(self) -> None:
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        mean_v, _, _ = _extract_signal_stats(arr, 2, 10)
        assert math.isclose(mean_v, 3.0)  # nanmean of [1, 3, 5]


# ---------------------------------------------------------------------------
# _extract_from_open_bws
# ---------------------------------------------------------------------------

class TestExtractFromOpenBws:
    def _bw(
        self, signal: float | None = 5.0, chrom_len: int = 10_000_000
    ) -> _MockBigWig:
        return _MockBigWig({"chr7": chrom_len}, signal)

    def test_dnase_features_populated(self) -> None:
        open_bws = {"DNase": self._bw(signal=6.0)}
        feats = _extract_from_open_bws(open_bws, "chr7", 500_000, 500)
        assert not math.isnan(feats["exp_dnase_mean"])
        assert math.isclose(feats["exp_dnase_mean"], 6.0)
        assert math.isclose(feats["exp_dnase_max"], 6.0)
        assert math.isclose(feats["exp_dnase_at_site"], 6.0)

    def test_histones_populated(self) -> None:
        open_bws = {
            "H3K4me3": self._bw(signal=3.0),
            "H3K27me3": self._bw(signal=1.0),
            "H3K9me3": self._bw(signal=0.5),
            "H3K27ac": self._bw(signal=4.0),
        }
        feats = _extract_from_open_bws(open_bws, "chr7", 500_000, 500)
        assert math.isclose(feats["exp_h3k4me3_mean"], 3.0)
        assert math.isclose(feats["exp_h3k27me3_mean"], 1.0)
        assert math.isclose(feats["exp_h3k9me3_mean"], 0.5)
        assert math.isclose(feats["exp_h3k27ac_mean"], 4.0)

    def test_missing_assays_are_nan(self) -> None:
        """Only DNase present — histone features should be NaN."""
        open_bws = {"DNase": self._bw(signal=2.0)}
        feats = _extract_from_open_bws(open_bws, "chr7", 500_000, 500)
        assert math.isnan(feats["exp_h3k4me3_mean"])
        assert math.isnan(feats["exp_h3k27me3_mean"])
        assert math.isnan(feats["exp_h3k9me3_mean"])
        assert math.isnan(feats["exp_h3k27ac_mean"])

    def test_empty_bws_all_nan(self) -> None:
        feats = _extract_from_open_bws({}, "chr7", 500_000, 500)
        for v in feats.values():
            assert math.isnan(v)

    def test_accessibility_binary_above_threshold(self) -> None:
        signal = DNASE_OPEN_THRESHOLD + 1.0
        open_bws = {"DNase": self._bw(signal=signal)}
        feats = _extract_from_open_bws(open_bws, "chr7", 500_000, 500)
        assert feats["exp_accessibility_binary"] == 1.0

    def test_accessibility_binary_below_threshold(self) -> None:
        signal = DNASE_OPEN_THRESHOLD - 0.5
        open_bws = {"DNase": self._bw(signal=signal)}
        feats = _extract_from_open_bws(open_bws, "chr7", 500_000, 500)
        assert feats["exp_accessibility_binary"] == 0.0

    def test_accessibility_binary_nan_dnase(self) -> None:
        """All-NaN DNase signal → mean=NaN → accessibility binary = 0 (NaN > thresh is False)."""
        open_bws = {"DNase": self._bw(signal=None)}
        feats = _extract_from_open_bws(open_bws, "chr7", 500_000, 500)
        # nan > DNASE_OPEN_THRESHOLD evaluates to False in Python → 0.0
        assert feats["exp_accessibility_binary"] == 0.0

    def test_accessibility_binary_nan_when_no_dnase(self) -> None:
        """No DNase assay in open_bws → accessibility binary stays NaN."""
        open_bws = {"H3K4me3": self._bw(signal=5.0)}
        feats = _extract_from_open_bws(open_bws, "chr7", 500_000, 500)
        assert math.isnan(feats["exp_accessibility_binary"])

    def test_site_near_chrom_start(self) -> None:
        """Site near position 0 should not raise; boundary handled by _query_region."""
        open_bws = {"DNase": self._bw(signal=2.0)}
        feats = _extract_from_open_bws(open_bws, "chr7", 100, 500)
        # Should produce a value (not crash), though window is truncated
        assert not math.isnan(feats["exp_dnase_mean"])

    def test_result_keys_match_feature_names(self) -> None:
        open_bws = {"DNase": self._bw(), "H3K4me3": self._bw()}
        feats = _extract_from_open_bws(open_bws, "chr7", 500_000, 500)
        assert set(feats.keys()) == set(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# get_chromatin_features (single-site, with patched _open_bigwig)
# ---------------------------------------------------------------------------

class TestGetChromatinFeatures:
    def _make_site(self, pos: int = 5_000_000) -> GenomicInterval:
        return GenomicInterval(chrom="chr7", start=pos, end=pos + 23)

    def test_no_files_returns_nan(self, tmp_path: Path) -> None:
        manifest = build_manifest(tmp_path)  # empty dir
        feats = get_chromatin_features(self._make_site(), "K562", manifest)
        for v in feats.values():
            assert math.isnan(v)

    def test_wrong_cell_type_returns_nan(self, k562_manifest: BigWigManifest) -> None:
        feats = get_chromatin_features(self._make_site(), "U2OS", k562_manifest)
        for v in feats.values():
            assert math.isnan(v)

    def test_correct_values_from_mock(self, k562_manifest: BigWigManifest) -> None:
        mock_bw = _MockBigWig({"chr7": 159_345_973}, constant_signal=4.5)
        with patch("src.encode_client._open_bigwig", return_value=mock_bw):
            feats = get_chromatin_features(self._make_site(), "K562", k562_manifest)
        assert math.isclose(feats["exp_dnase_mean"], 4.5)
        assert math.isclose(feats["exp_h3k4me3_mean"], 4.5)

    def test_chrom_normalisation(self, k562_manifest: BigWigManifest) -> None:
        """Site with '7' (no chr prefix) should work via normalize_chrom."""
        site = GenomicInterval(chrom="7", start=5_000_000, end=5_000_023)
        mock_bw = _MockBigWig({"chr7": 159_345_973}, constant_signal=3.0)
        with patch("src.encode_client._open_bigwig", return_value=mock_bw):
            feats = get_chromatin_features(site, "K562", k562_manifest)
        assert not math.isnan(feats["exp_dnase_mean"])

    def test_all_feature_keys_present(self, k562_manifest: BigWigManifest) -> None:
        mock_bw = _MockBigWig({"chr7": 159_345_973}, constant_signal=1.0)
        with patch("src.encode_client._open_bigwig", return_value=mock_bw):
            feats = get_chromatin_features(self._make_site(), "K562", k562_manifest)
        assert set(feats.keys()) == set(FEATURE_NAMES)

    def test_bw_files_are_closed(self, k562_manifest: BigWigManifest) -> None:
        """BigWig handles must be closed even on success."""
        closed_handles: list[_MockBigWig] = []

        def tracking_open(path: Path) -> _MockBigWig:
            bw = _MockBigWig({"chr7": 159_345_973}, 2.0)
            closed_handles.append(bw)
            return bw

        with patch("src.encode_client._open_bigwig", side_effect=tracking_open):
            get_chromatin_features(self._make_site(), "K562", k562_manifest)

        for bw in closed_handles:
            assert bw.closed, "BigWig handle not closed after get_chromatin_features"

    def test_open_error_returns_nan_for_that_assay(
        self, tmp_path: Path
    ) -> None:
        """If one BigWig file cannot be opened, only that assay is NaN."""
        # Only DNase file exists
        (tmp_path / "K562_DNase_hg38.bigWig").write_bytes(b"")
        manifest = build_manifest(tmp_path)

        mock_bw = _MockBigWig({"chr7": 159_345_973}, constant_signal=5.0)
        with patch("src.encode_client._open_bigwig", return_value=mock_bw):
            feats = get_chromatin_features(self._make_site(), "K562", manifest)

        assert not math.isnan(feats["exp_dnase_mean"])
        assert math.isnan(feats["exp_h3k4me3_mean"])


# ---------------------------------------------------------------------------
# batch_get_chromatin_features
# ---------------------------------------------------------------------------

class TestBatchGetChromatinFeatures:
    def _make_sites(self, n: int = 5) -> pd.DataFrame:
        return pd.DataFrame({
            "chrom": ["chr7"] * n,
            "start": [i * 1_000_000 + 5_000_000 for i in range(n)],
            "guide_id": [f"g{i}" for i in range(n)],
        })

    def test_no_files_returns_nan_columns(self, tmp_path: Path) -> None:
        manifest = build_manifest(tmp_path)  # empty
        sites = self._make_sites()
        result = batch_get_chromatin_features(sites, "K562", manifest)
        for feat in FEATURE_NAMES:
            assert feat in result.columns
            assert result[feat].isna().all()

    def test_wrong_cell_type_all_nan(self, k562_manifest: BigWigManifest) -> None:
        sites = self._make_sites()
        result = batch_get_chromatin_features(sites, "HEK293", k562_manifest)
        for feat in FEATURE_NAMES:
            assert result[feat].isna().all()

    def test_correct_values(self, k562_manifest: BigWigManifest) -> None:
        sites = self._make_sites(3)
        mock_bw = _MockBigWig({"chr7": 159_345_973}, constant_signal=7.0)
        with patch("src.encode_client._open_bigwig", return_value=mock_bw):
            result = batch_get_chromatin_features(sites, "K562", k562_manifest)
        assert (result["exp_dnase_mean"] == 7.0).all()
        assert (result["exp_h3k4me3_mean"] == 7.0).all()

    def test_preserves_input_columns(self, k562_manifest: BigWigManifest) -> None:
        sites = self._make_sites(2)
        mock_bw = _MockBigWig({"chr7": 159_345_973}, constant_signal=1.0)
        with patch("src.encode_client._open_bigwig", return_value=mock_bw):
            result = batch_get_chromatin_features(sites, "K562", k562_manifest)
        assert "guide_id" in result.columns
        assert list(result["guide_id"]) == list(sites["guide_id"])

    def test_row_order_preserved(self, k562_manifest: BigWigManifest) -> None:
        sites = self._make_sites(4)
        mock_bw = _MockBigWig({"chr7": 159_345_973}, constant_signal=3.0)
        with patch("src.encode_client._open_bigwig", return_value=mock_bw):
            result = batch_get_chromatin_features(sites, "K562", k562_manifest)
        assert list(result["start"]) == list(sites["start"])

    def test_output_has_feature_columns(self, k562_manifest: BigWigManifest) -> None:
        sites = self._make_sites(2)
        mock_bw = _MockBigWig({"chr7": 159_345_973}, constant_signal=2.0)
        with patch("src.encode_client._open_bigwig", return_value=mock_bw):
            result = batch_get_chromatin_features(sites, "K562", k562_manifest)
        for feat in FEATURE_NAMES:
            assert feat in result.columns

    def test_bw_files_opened_once(self, k562_manifest: BigWigManifest) -> None:
        """Each BigWig file must be opened exactly once per batch call."""
        open_counts: dict[str, int] = {}

        def counting_open(path: Path) -> _MockBigWig:
            key = path.name
            open_counts[key] = open_counts.get(key, 0) + 1
            return _MockBigWig({"chr7": 159_345_973}, 1.0)

        sites = self._make_sites(10)
        with patch("src.encode_client._open_bigwig", side_effect=counting_open):
            batch_get_chromatin_features(sites, "K562", k562_manifest)

        for count in open_counts.values():
            assert count == 1, f"File opened {count} times; expected 1"

    def test_bw_files_closed_after_batch(self, k562_manifest: BigWigManifest) -> None:
        closed_handles: list[_MockBigWig] = []

        def tracking_open(path: Path) -> _MockBigWig:
            bw = _MockBigWig({"chr7": 159_345_973}, 1.0)
            closed_handles.append(bw)
            return bw

        sites = self._make_sites(3)
        with patch("src.encode_client._open_bigwig", side_effect=tracking_open):
            batch_get_chromatin_features(sites, "K562", k562_manifest)

        assert len(closed_handles) > 0
        for bw in closed_handles:
            assert bw.closed

    def test_single_site(self, k562_manifest: BigWigManifest) -> None:
        sites = self._make_sites(1)
        mock_bw = _MockBigWig({"chr7": 159_345_973}, constant_signal=9.0)
        with patch("src.encode_client._open_bigwig", return_value=mock_bw):
            result = batch_get_chromatin_features(sites, "K562", k562_manifest)
        assert len(result) == 1
        assert math.isclose(result["exp_dnase_mean"].iloc[0], 9.0)

    def test_empty_dataframe(self, k562_manifest: BigWigManifest) -> None:
        sites = pd.DataFrame({"chrom": [], "start": []})
        mock_bw = _MockBigWig({"chr7": 159_345_973}, constant_signal=1.0)
        with patch("src.encode_client._open_bigwig", return_value=mock_bw):
            result = batch_get_chromatin_features(sites, "K562", k562_manifest)
        assert len(result) == 0
        for feat in FEATURE_NAMES:
            assert feat in result.columns

    def test_input_dataframe_not_mutated(self, k562_manifest: BigWigManifest) -> None:
        sites = self._make_sites(3)
        original_cols = list(sites.columns)
        mock_bw = _MockBigWig({"chr7": 159_345_973}, constant_signal=1.0)
        with patch("src.encode_client._open_bigwig", return_value=mock_bw):
            batch_get_chromatin_features(sites, "K562", k562_manifest)
        assert list(sites.columns) == original_cols

    def test_accessibility_binary_threshold(
        self, k562_manifest: BigWigManifest
    ) -> None:
        """Sites above/below threshold should get 1/0 respectively."""
        sites = pd.DataFrame({
            "chrom": ["chr7", "chr7"],
            "start": [5_000_000, 6_000_000],
        })
        signals = [DNASE_OPEN_THRESHOLD + 1.0, DNASE_OPEN_THRESHOLD - 0.5]
        call_idx = [0]

        def signal_bw(path: Path) -> _MockBigWig:
            sig = signals[call_idx[0] % 2]
            call_idx[0] += 1
            return _MockBigWig({"chr7": 159_345_973}, sig)

        # Use a single signal value BigWig for both sites
        mock_bw_high = _MockBigWig({"chr7": 159_345_973}, DNASE_OPEN_THRESHOLD + 1.0)
        mock_bw_low = _MockBigWig({"chr7": 159_345_973}, DNASE_OPEN_THRESHOLD - 0.5)

        # Test high signal → binary = 1
        with patch("src.encode_client._open_bigwig", return_value=mock_bw_high):
            result_high = batch_get_chromatin_features(
                sites.head(1), "K562", k562_manifest, show_progress=False
            )
        assert result_high["exp_accessibility_binary"].iloc[0] == 1.0

        # Test low signal → binary = 0
        with patch("src.encode_client._open_bigwig", return_value=mock_bw_low):
            result_low = batch_get_chromatin_features(
                sites.tail(1), "K562", k562_manifest, show_progress=False
            )
        assert result_low["exp_accessibility_binary"].iloc[0] == 0.0

    def test_mixed_chromosomes(self, k562_manifest: BigWigManifest) -> None:
        sites = pd.DataFrame({
            "chrom": ["chr7", "chr12", "chrX"],
            "start": [5_000_000, 3_000_000, 1_000_000],
        })
        mock_bw = _MockBigWig(
            {"chr7": 159_345_973, "chr12": 133_275_309, "chrX": 156_040_895},
            constant_signal=2.5,
        )
        with patch("src.encode_client._open_bigwig", return_value=mock_bw):
            result = batch_get_chromatin_features(sites, "K562", k562_manifest)
        assert (result["exp_dnase_mean"] == 2.5).all()

    def test_chrom_normalisation_in_batch(
        self, k562_manifest: BigWigManifest
    ) -> None:
        """DataFrame with unnormalised chrom names should work."""
        sites = pd.DataFrame({
            "chrom": ["7", "chr12"],  # "7" missing chr prefix
            "start": [5_000_000, 3_000_000],
        })
        mock_bw = _MockBigWig(
            {"chr7": 159_345_973, "chr12": 133_275_309}, constant_signal=1.0
        )
        with patch("src.encode_client._open_bigwig", return_value=mock_bw):
            result = batch_get_chromatin_features(sites, "K562", k562_manifest)
        assert not result["exp_dnase_mean"].isna().any()
