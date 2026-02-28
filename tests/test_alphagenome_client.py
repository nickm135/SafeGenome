"""Tests for src/alphagenome_client.py.

Test strategy
-------------
All tests in this file run without any network access or API key.  The
AlphaGenome SDK types are mimicked by lightweight mock objects that match
the attribute shapes produced by real API calls (verified on 2026-02-24).

Live integration tests (requiring a real API key) are in a separate block
marked ``@pytest.mark.live`` and skipped by default.  Run them with::

    pytest tests/test_alphagenome_client.py -m live

Mock output structure (confirmed from real K562 call)
------------------------------------------------------
- dnase.values  : (131072, 1)   — 1 bp resolution, 1 track
- atac.values   : (131072, 1)   — 1 bp resolution, 1 track
- chip_histone.values  : (1024, n_histone)  — 128 bp/bin
- chip_histone.metadata: DataFrame with 'name' column
- chip_tf.values       : (1024, n_tf)
- chip_tf.metadata     : DataFrame with 'name' column
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.alphagenome_client import (
    BIN_SIZE,
    COURTESY_SLEEP_S,
    DEFAULT_WINDOW_BP,
    DNASE_OPEN_THRESHOLD,
    FEATURE_NAMES,
    INTERVAL_SIZE,
    N_BINS,
    NARROW_WINDOW_BP,
    TrackAvailability,
    _extract_bin_signal,
    _extract_bp_signal,
    _extract_chromatin_features,
    _nan_features,
    _predict_with_retry,
    _site_to_tile,
    batch_get_chromatin_features,
    get_chromatin_features,
    get_track_availability,
    list_cell_types,
)
from src.utils import GenomicInterval


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _MockTrackData(NamedTuple):
    """Mimics the TrackData object returned by the AlphaGenome SDK."""
    values: np.ndarray
    metadata: pd.DataFrame


class _MockOutput:
    """Picklable mock of an AlphaGenome prediction output.

    Uses plain namedtuple TrackData objects with real numpy arrays and
    DataFrames — these pickle cleanly without the ``MagicMock`` problem.
    """
    __slots__ = ("dnase", "atac", "chip_histone", "chip_tf")

    def __init__(
        self,
        dnase: _MockTrackData,
        atac: _MockTrackData,
        chip_histone: _MockTrackData,
        chip_tf: _MockTrackData,
    ) -> None:
        self.dnase = dnase
        self.atac = atac
        self.chip_histone = chip_histone
        self.chip_tf = chip_tf


def _make_mock_output(
    dnase_shape: tuple[int, int] = (INTERVAL_SIZE, 1),
    atac_shape: tuple[int, int] = (INTERVAL_SIZE, 1),
    histone_marks: list[str] | None = None,
    has_ctcf: bool = True,
    seed: int = 42,
) -> _MockOutput:
    """Build a picklable mock AlphaGenome prediction output.

    Parameters
    ----------
    dnase_shape, atac_shape : tuple[int, int]
        Shape of DNase / ATAC arrays.
    histone_marks : list[str] | None
        Mark names to include; default is K562-like set (no H3K9me3).
    has_ctcf : bool
        Whether to include a CTCF TF track.
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    if histone_marks is None:
        # K562 actual marks — H3K9me3 is NOT present
        histone_marks = [
            "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3",
            "H3K4me1", "H3K4me2", "H3K4me3", "H3K79me2",
            "H3K9ac", "H4K20me1",
        ]

    # DNase / ATAC: 1-bp resolution, values in [0, 15)
    dnase_vals = rng.uniform(0, 15, size=dnase_shape).astype(np.float32)
    atac_vals = rng.uniform(0, 10, size=atac_shape).astype(np.float32)

    # Plant a clearly high-signal spike at position 65536 (middle of interval)
    spike_pos = INTERVAL_SIZE // 2
    dnase_vals[spike_pos, 0] = 40.0
    atac_vals[spike_pos, 0] = 30.0

    # Histone: 128-bp bin resolution
    n_histone = len(histone_marks)
    histone_vals = rng.uniform(0, 8, size=(N_BINS, n_histone)).astype(np.float32)
    histone_meta = pd.DataFrame({
        "name": [f"EFO:0002067 Histone ChIP-seq {m}" for m in histone_marks],
        "ontology_curie": ["EFO:0002067"] * n_histone,
    })

    # TF ChIP: 128-bp bin resolution, include CTCF if requested
    tf_names = ["EFO:0002067 TF ChIP-seq CTCF"] if has_ctcf else ["EFO:0002067 TF ChIP-seq OTHER"]
    n_tf = len(tf_names)
    tf_vals = rng.uniform(0, 5, size=(N_BINS, n_tf)).astype(np.float32)
    tf_meta = pd.DataFrame({
        "name": tf_names,
        "ontology_curie": ["EFO:0002067"] * n_tf,
    })

    return _MockOutput(
        dnase=_MockTrackData(values=dnase_vals, metadata=pd.DataFrame()),
        atac=_MockTrackData(values=atac_vals, metadata=pd.DataFrame()),
        chip_histone=_MockTrackData(values=histone_vals, metadata=histone_meta),
        chip_tf=_MockTrackData(values=tf_vals, metadata=tf_meta),
    )


def _full_avail(ontology_term: str = "EFO:0002067") -> TrackAvailability:
    """Return a TrackAvailability representing K562.

    Matches confirmed K562 track inventory: DNase, ATAC, 10 histone marks
    (H3K9me3 is absent), and CTCF.
    """
    return TrackAvailability(
        ontology_term=ontology_term,
        has_dnase=True,
        has_atac=True,
        histone_marks=[
            # H3K9me3 intentionally absent — not available for K562 in AlphaGenome
            "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3",
            "H3K4me1", "H3K4me2", "H3K4me3", "H3K79me2",
            "H3K9ac", "H4K20me1",
        ],
        has_ctcf=True,
    )


def _no_tracks_avail(ontology_term: str = "UNKNOWN:0000000") -> TrackAvailability:
    """Return a TrackAvailability with no tracks (e.g. U2OS)."""
    return TrackAvailability(
        ontology_term=ontology_term,
        has_dnase=False,
        has_atac=False,
        histone_marks=[],
        has_ctcf=False,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_interval_size_is_valid_power_of_two(self):
        # 131072 = 2**17
        assert INTERVAL_SIZE == 131_072
        assert math.log2(INTERVAL_SIZE) == int(math.log2(INTERVAL_SIZE))

    def test_bin_size_divides_interval_evenly(self):
        assert INTERVAL_SIZE % BIN_SIZE == 0
        assert INTERVAL_SIZE // BIN_SIZE == N_BINS

    def test_feature_names_complete(self):
        assert len(FEATURE_NAMES) == 12
        expected = {
            "ag_dnase_mean", "ag_dnase_max", "ag_dnase_at_site",
            "ag_dnase_mean_narrow", "ag_atac_mean", "ag_atac_at_site",
            "ag_h3k4me3_mean", "ag_h3k27me3_mean", "ag_h3k9me3_mean",
            "ag_h3k27ac_mean", "ag_ctcf_mean", "ag_accessibility_binary",
        }
        assert set(FEATURE_NAMES) == expected

    def test_feature_names_no_duplicates(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


# ---------------------------------------------------------------------------
# TrackAvailability
# ---------------------------------------------------------------------------

class TestTrackAvailability:
    def test_has_any_tracks_true_when_dnase(self):
        avail = TrackAvailability("X", has_dnase=True, has_atac=False,
                                  histone_marks=[], has_ctcf=False)
        assert avail.has_any_tracks is True

    def test_has_any_tracks_true_when_histone(self):
        avail = TrackAvailability("X", has_dnase=False, has_atac=False,
                                  histone_marks=["H3K4me3"], has_ctcf=False)
        assert avail.has_any_tracks is True

    def test_has_any_tracks_false_when_empty(self):
        avail = _no_tracks_avail()
        assert avail.has_any_tracks is False

    def test_requested_output_types_k562(self):
        from alphagenome.models import dna_client
        avail = _full_avail()
        types = avail.requested_output_types
        assert dna_client.OutputType.DNASE in types
        assert dna_client.OutputType.ATAC in types
        assert dna_client.OutputType.CHIP_HISTONE in types
        assert dna_client.OutputType.CHIP_TF in types
        assert len(types) == 4

    def test_requested_output_types_no_dnase_atac(self):
        from alphagenome.models import dna_client
        avail = TrackAvailability("EFO:0001182", has_dnase=False, has_atac=False,
                                  histone_marks=["H3K4me3"], has_ctcf=True)
        types = avail.requested_output_types
        assert dna_client.OutputType.DNASE not in types
        assert dna_client.OutputType.ATAC not in types
        assert dna_client.OutputType.CHIP_HISTONE in types
        assert dna_client.OutputType.CHIP_TF in types

    def test_requested_output_types_empty(self):
        avail = _no_tracks_avail()
        assert avail.requested_output_types == []


# ---------------------------------------------------------------------------
# _site_to_tile
# ---------------------------------------------------------------------------

class TestSiteToTile:
    def test_tile_start_aligned_to_interval_size(self):
        chrom, ts, te = _site_to_tile(200_000, "chr1")
        assert ts % INTERVAL_SIZE == 0

    def test_tile_length_is_interval_size(self):
        _, ts, te = _site_to_tile(0, "chr1")
        assert te - ts == INTERVAL_SIZE

    def test_site_at_zero(self):
        chrom, ts, te = _site_to_tile(0, "chr1")
        assert ts == 0
        assert te == INTERVAL_SIZE

    def test_site_at_tile_boundary(self):
        # Exactly at a boundary → belongs to the NEXT tile
        site_pos = INTERVAL_SIZE
        _, ts, te = _site_to_tile(site_pos, "chr1")
        assert ts == INTERVAL_SIZE
        assert te == 2 * INTERVAL_SIZE

    def test_site_in_middle_of_tile(self):
        site_pos = INTERVAL_SIZE + INTERVAL_SIZE // 4
        _, ts, te = _site_to_tile(site_pos, "chr1")
        assert ts == INTERVAL_SIZE
        assert te == 2 * INTERVAL_SIZE

    def test_site_offset_within_tile(self):
        site_pos = INTERVAL_SIZE + 5000
        _, ts, _ = _site_to_tile(site_pos, "chr1")
        assert site_pos - ts == 5000

    def test_chrom_passthrough(self):
        chrom, _, _ = _site_to_tile(100, "chr7")
        assert chrom == "chr7"

    def test_large_position(self):
        site_pos = 200_000_000
        chrom, ts, te = _site_to_tile(site_pos, "chr1")
        assert ts % INTERVAL_SIZE == 0
        assert ts <= site_pos < te
        assert te - ts == INTERVAL_SIZE


# ---------------------------------------------------------------------------
# _extract_bp_signal
# ---------------------------------------------------------------------------

class TestExtractBpSignal:
    def _uniform_signal(self, value: float, length: int = INTERVAL_SIZE, n_tracks: int = 1) -> np.ndarray:
        return np.full((length, n_tracks), value, dtype=np.float32)

    def test_mean_of_uniform_signal(self):
        signal = self._uniform_signal(3.0)
        mean, _, _, _ = _extract_bp_signal(signal, 65536, DEFAULT_WINDOW_BP, NARROW_WINDOW_BP)
        assert pytest.approx(mean, abs=1e-4) == 3.0

    def test_max_of_uniform_signal(self):
        signal = self._uniform_signal(3.0)
        _, sig_max, _, _ = _extract_bp_signal(signal, 65536, DEFAULT_WINDOW_BP, NARROW_WINDOW_BP)
        assert pytest.approx(sig_max, abs=1e-4) == 3.0

    def test_at_site_exact_value(self):
        signal = np.zeros((INTERVAL_SIZE, 1), dtype=np.float32)
        signal[65536, 0] = 99.0
        _, _, at_site, _ = _extract_bp_signal(signal, 65536, DEFAULT_WINDOW_BP, NARROW_WINDOW_BP)
        assert pytest.approx(at_site, abs=1e-4) == 99.0

    def test_spike_captured_by_max(self):
        signal = np.zeros((INTERVAL_SIZE, 1), dtype=np.float32)
        signal[65536, 0] = 50.0
        _, sig_max, _, _ = _extract_bp_signal(signal, 65536, DEFAULT_WINDOW_BP, NARROW_WINDOW_BP)
        assert sig_max == pytest.approx(50.0, abs=1e-4)

    def test_multi_track_averaging(self):
        # Two tracks with values 2.0 and 4.0 → mean of means = 3.0
        signal = np.stack([
            np.full(INTERVAL_SIZE, 2.0),
            np.full(INTERVAL_SIZE, 4.0),
        ], axis=1).astype(np.float32)
        mean, _, _, _ = _extract_bp_signal(signal, 65536, DEFAULT_WINDOW_BP, NARROW_WINDOW_BP)
        assert pytest.approx(mean, abs=1e-4) == 3.0

    def test_boundary_clamping_near_start(self):
        signal = self._uniform_signal(1.0)
        # site_offset = 10, window_bp = 500 → window clamps at 0
        mean, _, _, _ = _extract_bp_signal(signal, 10, DEFAULT_WINDOW_BP, NARROW_WINDOW_BP)
        assert not math.isnan(mean)

    def test_narrow_mean_smaller_window(self):
        # All zeros except the spike at offset 65536; narrow mean should include it
        signal = np.zeros((INTERVAL_SIZE, 1), dtype=np.float32)
        signal[65536, 0] = 10.0
        wide_mean, _, _, narrow_mean = _extract_bp_signal(
            signal, 65536, DEFAULT_WINDOW_BP, NARROW_WINDOW_BP
        )
        # Spike raises narrow_mean more than wide_mean
        assert narrow_mean > wide_mean

    def test_out_of_bounds_offset_returns_nan(self):
        signal = self._uniform_signal(1.0, length=100, n_tracks=1)
        _, _, at_site, _ = _extract_bp_signal(signal, 200, DEFAULT_WINDOW_BP, NARROW_WINDOW_BP)
        assert math.isnan(at_site)


# ---------------------------------------------------------------------------
# _extract_bin_signal
# ---------------------------------------------------------------------------

class TestExtractBinSignal:
    def _uniform_bins(self, value: float, n_tracks: int = 1) -> np.ndarray:
        return np.full((N_BINS, n_tracks), value, dtype=np.float32)

    def test_uniform_signal_returns_value(self):
        signal = self._uniform_bins(5.0)
        result = _extract_bin_signal(signal, INTERVAL_SIZE // 2, DEFAULT_WINDOW_BP)
        assert pytest.approx(result, abs=1e-4) == 5.0

    def test_track_mask_selects_correct_column(self):
        signal = np.zeros((N_BINS, 3), dtype=np.float32)
        signal[:, 1] = 7.0  # only column 1 has signal
        mask = np.array([False, True, False])
        result = _extract_bin_signal(signal, INTERVAL_SIZE // 2, DEFAULT_WINDOW_BP, mask)
        assert pytest.approx(result, abs=1e-4) == 7.0

    def test_empty_mask_returns_nan(self):
        signal = self._uniform_bins(1.0, n_tracks=2)
        mask = np.array([False, False])
        result = _extract_bin_signal(signal, INTERVAL_SIZE // 2, DEFAULT_WINDOW_BP, mask)
        assert math.isnan(result)

    def test_site_at_tile_start_uses_first_bin(self):
        signal = self._uniform_bins(3.0)
        result = _extract_bin_signal(signal, 0, DEFAULT_WINDOW_BP)
        assert not math.isnan(result)

    def test_site_offset_maps_to_correct_bin(self):
        signal = np.zeros((N_BINS, 1), dtype=np.float32)
        site_offset = 512 * BIN_SIZE  # maps to bin 512
        # Set entire signal to 0 except a narrow region around bin 512
        signal[511:513, 0] = 10.0
        result = _extract_bin_signal(signal, site_offset, DEFAULT_WINDOW_BP)
        assert result > 0.0


# ---------------------------------------------------------------------------
# _nan_features
# ---------------------------------------------------------------------------

class TestNanFeatures:
    def test_all_keys_present(self):
        feats = _nan_features()
        assert set(feats.keys()) == set(FEATURE_NAMES)

    def test_all_values_nan(self):
        feats = _nan_features()
        assert all(math.isnan(v) for v in feats.values())


# ---------------------------------------------------------------------------
# _extract_chromatin_features
# ---------------------------------------------------------------------------

class TestExtractChromatinFeatures:
    def test_k562_output_all_features_present(self):
        output = _make_mock_output()
        avail = _full_avail()
        feats = _extract_chromatin_features(output, 65536, avail)
        assert set(feats.keys()) == set(FEATURE_NAMES)

    def test_k562_no_nan_features_except_h3k9me3(self):
        """K562 mock has all tracks except H3K9me3, which is absent in AlphaGenome."""
        output = _make_mock_output()
        avail = _full_avail()
        feats = _extract_chromatin_features(output, 65536, avail)
        for name, val in feats.items():
            if name == "ag_h3k9me3_mean":
                assert math.isnan(val), "H3K9me3 should be NaN for K562"
            else:
                assert not math.isnan(val), f"{name} is NaN but should not be"

    def test_dnase_at_site_matches_spike(self):
        output = _make_mock_output(seed=99)
        # Spike is planted at position 65536 = INTERVAL_SIZE // 2
        avail = _full_avail()
        feats = _extract_chromatin_features(output, INTERVAL_SIZE // 2, avail)
        assert feats["ag_dnase_at_site"] == pytest.approx(40.0, abs=1e-3)

    def test_no_dnase_atac_returns_nan_dnase(self):
        output = _make_mock_output()
        avail = TrackAvailability(
            "EFO:0001182", has_dnase=False, has_atac=False,
            histone_marks=["H3K4me3"], has_ctcf=False,
        )
        feats = _extract_chromatin_features(output, 65536, avail)
        assert math.isnan(feats["ag_dnase_mean"])
        assert math.isnan(feats["ag_atac_mean"])
        assert math.isnan(feats["ag_accessibility_binary"])
        # Histone should still work
        assert not math.isnan(feats["ag_h3k4me3_mean"])

    def test_no_tracks_returns_all_nan(self):
        output = _make_mock_output()
        avail = _no_tracks_avail()
        feats = _extract_chromatin_features(output, 65536, avail)
        for name, val in feats.items():
            assert math.isnan(val), f"{name} should be NaN for no-tracks cell type"

    def test_accessibility_binary_high_signal(self):
        # Force dnase mean >> threshold by using all-high signal
        signal = np.full((INTERVAL_SIZE, 1), DNASE_OPEN_THRESHOLD * 5, dtype=np.float32)
        output = _make_mock_output()
        output.dnase = _MockTrackData(
            values=signal,
            metadata=pd.DataFrame(),
        )
        avail = _full_avail()
        feats = _extract_chromatin_features(output, 65536, avail)
        assert feats["ag_accessibility_binary"] == 1.0

    def test_accessibility_binary_low_signal(self):
        signal = np.full((INTERVAL_SIZE, 1), 0.0, dtype=np.float32)
        output = _make_mock_output()
        output.dnase = _MockTrackData(values=signal, metadata=pd.DataFrame())
        avail = _full_avail()
        feats = _extract_chromatin_features(output, 65536, avail)
        assert feats["ag_accessibility_binary"] == 0.0

    def test_missing_histone_mark_returns_nan(self):
        """H3K9me3 is absent from K562 — that column should be NaN."""
        output = _make_mock_output()   # K562 mock has no H3K9me3
        avail = _full_avail()
        feats = _extract_chromatin_features(output, 65536, avail)
        assert math.isnan(feats["ag_h3k9me3_mean"])

    def test_no_ctcf_returns_nan(self):
        output = _make_mock_output(has_ctcf=False)
        avail = TrackAvailability(
            "EFO:0002067", has_dnase=True, has_atac=True,
            histone_marks=["H3K4me3"], has_ctcf=False,
        )
        feats = _extract_chromatin_features(output, 65536, avail)
        assert math.isnan(feats["ag_ctcf_mean"])

    def test_narrow_mean_present_and_finite(self):
        output = _make_mock_output()
        avail = _full_avail()
        feats = _extract_chromatin_features(output, 65536, avail)
        assert not math.isnan(feats["ag_dnase_mean_narrow"])


# ---------------------------------------------------------------------------
# _predict_with_retry (mock model)
# ---------------------------------------------------------------------------

class TestPredictWithRetry:
    def _make_interval(self):
        from alphagenome.data import genome as ag_genome
        return ag_genome.Interval(chromosome="chr1", start=0, end=INTERVAL_SIZE)

    def test_succeeds_on_first_attempt(self):
        mock_model = MagicMock()
        sentinel = object()
        mock_model.predict_interval.return_value = sentinel
        interval = self._make_interval()
        result = _predict_with_retry(mock_model, interval, [], ["EFO:0002067"])
        assert result is sentinel
        mock_model.predict_interval.assert_called_once()

    def test_retries_on_rate_limit_and_succeeds(self):
        mock_model = MagicMock()
        sentinel = object()
        mock_model.predict_interval.side_effect = [
            Exception("rate limit exceeded"),
            sentinel,
        ]
        interval = self._make_interval()
        with patch("time.sleep"):
            result = _predict_with_retry(mock_model, interval, [], ["EFO:0002067"],
                                         max_attempts=3)
        assert result is sentinel
        assert mock_model.predict_interval.call_count == 2

    def test_raises_after_max_attempts(self):
        mock_model = MagicMock()
        mock_model.predict_interval.side_effect = Exception("persistent error")
        interval = self._make_interval()
        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="attempts failed"):
                _predict_with_retry(mock_model, interval, [], ["EFO:0002067"],
                                    max_attempts=3)
        assert mock_model.predict_interval.call_count == 3

    def test_429_triggers_rate_limit_path(self):
        mock_model = MagicMock()
        sentinel = object()
        mock_model.predict_interval.side_effect = [
            Exception("HTTP 429 too many requests"),
            sentinel,
        ]
        interval = self._make_interval()
        sleep_calls = []
        with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            _predict_with_retry(mock_model, interval, [], ["EFO:0002067"],
                                max_attempts=3)
        assert len(sleep_calls) >= 1


# ---------------------------------------------------------------------------
# get_track_availability (mock model)
# ---------------------------------------------------------------------------

class TestGetTrackAvailability:
    def _make_mock_model(self):
        """Mock model whose output_metadata returns K562-like metadata."""
        from alphagenome.models import dna_client

        mock = MagicMock()
        meta = MagicMock()
        meta.dnase = pd.DataFrame({
            "name": ["EFO:0002067 DNase-seq"],
            "ontology_curie": ["EFO:0002067"],
            "biosample_name": ["K562"],
        })
        meta.atac = pd.DataFrame({
            "name": ["EFO:0002067 ATAC-seq"],
            "ontology_curie": ["EFO:0002067"],
            "biosample_name": ["K562"],
        })
        meta.chip_histone = pd.DataFrame({
            "name": [
                "EFO:0002067 Histone ChIP-seq H3K4me3",
                "EFO:0002067 Histone ChIP-seq H3K27me3",
            ],
            "ontology_curie": ["EFO:0002067", "EFO:0002067"],
            "biosample_name": ["K562", "K562"],
        })
        meta.chip_tf = pd.DataFrame({
            "name": ["EFO:0002067 TF ChIP-seq CTCF"],
            "ontology_curie": ["EFO:0002067"],
            "biosample_name": ["K562"],
        })
        mock.output_metadata.return_value = meta
        return mock

    def test_k562_has_dnase_and_atac(self):
        model = self._make_mock_model()
        avail = get_track_availability(model, "EFO:0002067")
        assert avail.has_dnase is True
        assert avail.has_atac is True

    def test_k562_histone_marks_extracted(self):
        model = self._make_mock_model()
        avail = get_track_availability(model, "EFO:0002067")
        assert "H3K4me3" in avail.histone_marks
        assert "H3K27me3" in avail.histone_marks

    def test_k562_has_ctcf(self):
        model = self._make_mock_model()
        avail = get_track_availability(model, "EFO:0002067")
        assert avail.has_ctcf is True

    def test_unknown_term_has_no_tracks(self):
        model = self._make_mock_model()
        avail = get_track_availability(model, "UNKNOWN:9999999")
        assert avail.has_any_tracks is False

    def test_returns_track_availability_instance(self):
        model = self._make_mock_model()
        avail = get_track_availability(model, "EFO:0002067")
        assert isinstance(avail, TrackAvailability)


# ---------------------------------------------------------------------------
# get_chromatin_features (mock model + tmp cache)
# ---------------------------------------------------------------------------

class TestGetChromatinFeatures:
    def _make_mock_model_with_output(self, output):
        model = MagicMock()
        model.predict_interval.return_value = output
        return model

    def test_returns_all_features(self, tmp_path):
        output = _make_mock_output()
        model = self._make_mock_model_with_output(output)
        avail = _full_avail()
        site = GenomicInterval("chr2", 73_160_000, 73_160_023)
        with patch("time.sleep"):
            feats = get_chromatin_features(site, "EFO:0002067", model,
                                           tmp_path, avail=avail)
        assert set(feats.keys()) == set(FEATURE_NAMES)

    def test_caches_api_response(self, tmp_path):
        output = _make_mock_output()
        model = self._make_mock_model_with_output(output)
        avail = _full_avail()
        site = GenomicInterval("chr2", 73_160_000, 73_160_023)
        with patch("time.sleep"):
            get_chromatin_features(site, "EFO:0002067", model, tmp_path, avail=avail)
            get_chromatin_features(site, "EFO:0002067", model, tmp_path, avail=avail)
        # Second call should use cache — predict_interval called only once
        assert model.predict_interval.call_count == 1

    def test_no_tracks_returns_nan(self, tmp_path):
        model = MagicMock()
        avail = _no_tracks_avail()
        site = GenomicInterval("chr1", 0, 23)
        feats = get_chromatin_features(site, "UNKNOWN:0", model, tmp_path, avail=avail)
        assert all(math.isnan(v) for v in feats.values())
        model.predict_interval.assert_not_called()

    def test_courtesy_sleep_called_on_api_hit(self, tmp_path):
        output = _make_mock_output()
        model = self._make_mock_model_with_output(output)
        avail = _full_avail()
        site = GenomicInterval("chr3", 1_000_000, 1_000_023)
        sleep_calls = []
        with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            get_chromatin_features(site, "EFO:0002067", model, tmp_path, avail=avail)
        assert any(s == COURTESY_SLEEP_S for s in sleep_calls)

    def test_no_sleep_on_cache_hit(self, tmp_path):
        output = _make_mock_output()
        model = self._make_mock_model_with_output(output)
        avail = _full_avail()
        site = GenomicInterval("chr3", 2_000_000, 2_000_023)
        with patch("time.sleep"):
            get_chromatin_features(site, "EFO:0002067", model, tmp_path, avail=avail)
        # Second call — cache should be warm
        sleep_calls = []
        with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            get_chromatin_features(site, "EFO:0002067", model, tmp_path, avail=avail)
        assert COURTESY_SLEEP_S not in sleep_calls


# ---------------------------------------------------------------------------
# batch_get_chromatin_features
# ---------------------------------------------------------------------------

class TestBatchGetChromatinFeatures:
    def _make_sites(self) -> pd.DataFrame:
        """Three sites: two in the same tile on chr1, one on chr2."""
        return pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [100_000, 200_000, 50_000],
            "guide_id": ["g1", "g1", "g2"],
        })

    def _make_mock_model(self):
        output = _make_mock_output()
        model = MagicMock()
        model.predict_interval.return_value = output
        return model, output

    def test_output_has_all_feature_columns(self, tmp_path):
        model, _ = self._make_mock_model()
        avail = _full_avail()
        sites = self._make_sites()
        with patch("time.sleep"):
            result = batch_get_chromatin_features(
                sites, "EFO:0002067", model, tmp_path, avail=avail
            )
        for feat in FEATURE_NAMES:
            assert feat in result.columns

    def test_output_row_count_matches_input(self, tmp_path):
        model, _ = self._make_mock_model()
        avail = _full_avail()
        sites = self._make_sites()
        with patch("time.sleep"):
            result = batch_get_chromatin_features(
                sites, "EFO:0002067", model, tmp_path, avail=avail
            )
        assert len(result) == len(sites)

    def test_input_columns_preserved(self, tmp_path):
        model, _ = self._make_mock_model()
        avail = _full_avail()
        sites = self._make_sites()
        with patch("time.sleep"):
            result = batch_get_chromatin_features(
                sites, "EFO:0002067", model, tmp_path, avail=avail
            )
        assert "guide_id" in result.columns
        assert list(result["guide_id"]) == ["g1", "g1", "g2"]

    def test_sites_in_same_tile_share_one_api_call(self, tmp_path):
        """Two sites on chr1 at 100k and 200k are in the same 131072-bp tile."""
        model, _ = self._make_mock_model()
        avail = _full_avail()
        # Both within first 131072 bp tile
        sites = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [1_000, 2_000],
        })
        with patch("time.sleep"):
            batch_get_chromatin_features(
                sites, "EFO:0002067", model, tmp_path, avail=avail
            )
        assert model.predict_interval.call_count == 1

    def test_sites_in_different_tiles_make_separate_calls(self, tmp_path):
        """Sites far apart must each trigger their own API call."""
        model, _ = self._make_mock_model()
        avail = _full_avail()
        tile2_start = INTERVAL_SIZE
        sites = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [1_000, tile2_start + 1_000],  # different tiles
        })
        with patch("time.sleep"):
            batch_get_chromatin_features(
                sites, "EFO:0002067", model, tmp_path, avail=avail
            )
        assert model.predict_interval.call_count == 2

    def test_no_tracks_returns_nan_no_api_calls(self, tmp_path):
        model, _ = self._make_mock_model()
        avail = _no_tracks_avail()
        sites = self._make_sites()
        result = batch_get_chromatin_features(
            sites, "UNKNOWN:0", model, tmp_path, avail=avail
        )
        model.predict_interval.assert_not_called()
        for feat in FEATURE_NAMES:
            assert all(math.isnan(v) for v in result[feat])

    def test_row_order_preserved(self, tmp_path):
        model, _ = self._make_mock_model()
        avail = _full_avail()
        sites = pd.DataFrame({
            "chrom": ["chr2", "chr1", "chr1"],
            "start": [50_000, 200_000, 100_000],
            "idx": [0, 1, 2],
        })
        with patch("time.sleep"):
            result = batch_get_chromatin_features(
                sites, "EFO:0002067", model, tmp_path, avail=avail
            )
        assert list(result["idx"]) == [0, 1, 2]

    def test_cache_reused_across_calls(self, tmp_path):
        model, _ = self._make_mock_model()
        avail = _full_avail()
        sites = pd.DataFrame({"chrom": ["chr1"], "start": [5_000]})
        with patch("time.sleep"):
            batch_get_chromatin_features(
                sites, "EFO:0002067", model, tmp_path, avail=avail
            )
            batch_get_chromatin_features(
                sites, "EFO:0002067", model, tmp_path, avail=avail
            )
        assert model.predict_interval.call_count == 1


# ---------------------------------------------------------------------------
# list_cell_types (mock model)
# ---------------------------------------------------------------------------

class TestListCellTypes:
    def _make_mock_model(self):
        model = MagicMock()
        meta = MagicMock()
        meta.dnase = pd.DataFrame({
            "ontology_curie": ["EFO:0002067"],
            "biosample_name": ["K562"],
            "name": ["EFO:0002067 DNase-seq"],
        })
        meta.atac = pd.DataFrame({
            "ontology_curie": ["EFO:0002067"],
            "biosample_name": ["K562"],
            "name": ["EFO:0002067 ATAC-seq"],
        })
        meta.chip_histone = pd.DataFrame({
            "ontology_curie": ["EFO:0001182"],
            "biosample_name": ["HEK293"],
            "name": ["EFO:0001182 Histone ChIP-seq H3K4me3"],
        })
        meta.chip_tf = pd.DataFrame({
            "ontology_curie": ["EFO:0002067"],
            "biosample_name": ["K562"],
            "name": ["EFO:0002067 TF ChIP-seq CTCF"],
        })
        model.output_metadata.return_value = meta
        return model

    def test_returns_dataframe(self):
        model = self._make_mock_model()
        df = list_cell_types(model)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self):
        model = self._make_mock_model()
        df = list_cell_types(model)
        for col in ["ontology_curie", "biosample_name", "has_dnase", "has_atac",
                    "n_histone_tracks", "has_ctcf"]:
            assert col in df.columns

    def test_k562_has_dnase_and_ctcf(self):
        model = self._make_mock_model()
        df = list_cell_types(model)
        k562 = df[df["ontology_curie"] == "EFO:0002067"].iloc[0]
        assert k562["has_dnase"] == True   # noqa: E712 — numpy.bool_ != is True
        assert k562["has_ctcf"] == True    # noqa: E712

    def test_hek293_no_dnase(self):
        model = self._make_mock_model()
        df = list_cell_types(model)
        hek = df[df["ontology_curie"] == "EFO:0001182"].iloc[0]
        assert hek["has_dnase"] == False   # noqa: E712


# ---------------------------------------------------------------------------
# Live integration tests (skipped unless -m live)
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestLiveK562:
    """Real API calls — require valid ALPHAGENOME_API_KEY in config.py."""

    @pytest.fixture(scope="class")
    def model_and_avail(self):
        import config
        from src.alphagenome_client import create_client, get_track_availability
        model = create_client(config.ALPHAGENOME_API_KEY)
        avail = get_track_availability(model, "EFO:0002067")
        return model, avail

    def test_k562_has_dnase_atac(self, model_and_avail):
        _, avail = model_and_avail
        assert avail.has_dnase is True
        assert avail.has_atac is True

    def test_k562_has_ctcf(self, model_and_avail):
        _, avail = model_and_avail
        assert avail.has_ctcf is True

    def test_k562_has_expected_histone_marks(self, model_and_avail):
        _, avail = model_and_avail
        for mark in ("H3K27ac", "H3K27me3", "H3K4me3"):
            assert mark in avail.histone_marks

    def test_k562_no_h3k9me3(self, model_and_avail):
        """H3K9me3 is confirmed absent from K562 AlphaGenome tracks."""
        _, avail = model_and_avail
        assert "H3K9me3" not in avail.histone_marks

    def test_single_site_feature_extraction(self, model_and_avail, tmp_path):
        model, avail = model_and_avail
        # EMX1 locus on chr2 (hg38)
        site = GenomicInterval("chr2", 73_160_000, 73_160_023)
        feats = get_chromatin_features(
            site, "EFO:0002067", model, tmp_path, avail=avail
        )
        assert set(feats.keys()) == set(FEATURE_NAMES)
        # DNase should be non-NaN and finite for K562
        assert not math.isnan(feats["ag_dnase_mean"])
        assert math.isfinite(feats["ag_dnase_mean"])

    def test_dnase_signal_nonzero_at_accessible_locus(self, model_and_avail, tmp_path):
        """K562 should show some DNase signal across a 131kb window on chr2."""
        model, avail = model_and_avail
        site = GenomicInterval("chr2", 73_160_000, 73_160_023)
        feats = get_chromatin_features(
            site, "EFO:0002067", model, tmp_path, avail=avail
        )
        # max should be well above 0 somewhere in the 128kb window
        assert feats["ag_dnase_max"] > 0.0
