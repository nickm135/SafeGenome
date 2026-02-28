"""Tests for src/utils.py"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils import (
    VALID_CHROMS_HG38,
    GenomicInterval,
    get_logger,
    is_valid_hg38_chrom,
    load_cache,
    log_filter,
    liftover_hg19_to_hg38,
    normalize_chrom,
    normalize_guide,
    save_cache,
)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class TestGetLogger:
    def test_returns_named_logger(self) -> None:
        logger = get_logger("test.utils.a")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.utils.a"

    def test_has_stream_handler(self) -> None:
        logger = get_logger("test.utils.b")
        assert any(
            isinstance(h, logging.StreamHandler) for h in logger.handlers
        )

    def test_no_duplicate_handlers_on_repeated_calls(self) -> None:
        get_logger("test.utils.c")
        get_logger("test.utils.c")
        logger = get_logger("test.utils.c")
        assert len(logger.handlers) == 1

    def test_default_level_is_info(self) -> None:
        logger = get_logger("test.utils.d")
        assert logger.level == logging.INFO

    def test_custom_level(self) -> None:
        logger = get_logger("test.utils.e", level=logging.DEBUG)
        assert logger.level == logging.DEBUG


class TestLogFilter:
    def test_logs_dropped_count(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger("test.filter.a")
        with caplog.at_level(logging.INFO, logger="test.filter.a"):
            log_filter(logger, "remove on-target sites", 100, 95)
        assert "5" in caplog.text          # dropped
        assert "100" in caplog.text        # before
        assert "95" in caplog.text         # after
        assert "remove on-target sites" in caplog.text

    def test_zero_before_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger("test.filter.b")
        with caplog.at_level(logging.INFO, logger="test.filter.b"):
            log_filter(logger, "empty dataset", 0, 0)  # no ZeroDivisionError

    def test_all_dropped(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger("test.filter.c")
        with caplog.at_level(logging.INFO, logger="test.filter.c"):
            log_filter(logger, "all filtered", 50, 0)
        assert "50" in caplog.text
        assert "0" in caplog.text


# ---------------------------------------------------------------------------
# GenomicInterval
# ---------------------------------------------------------------------------

class TestGenomicIntervalCreation:
    def test_basic_fields(self) -> None:
        iv = GenomicInterval("chr1", 100, 200, "+")
        assert iv.chrom == "chr1"
        assert iv.start == 100
        assert iv.end == 200
        assert iv.strand == "+"

    def test_default_strand_is_dot(self) -> None:
        iv = GenomicInterval("chr1", 0, 100)
        assert iv.strand == "."

    def test_negative_start_raises(self) -> None:
        with pytest.raises(ValueError, match="start must be >= 0"):
            GenomicInterval("chr1", -1, 100)

    def test_end_less_than_start_raises(self) -> None:
        with pytest.raises(ValueError, match="end.*must be >= start"):
            GenomicInterval("chr1", 200, 100)

    def test_zero_length_interval_is_valid(self) -> None:
        iv = GenomicInterval("chr1", 50, 50)
        assert iv.length == 0

    def test_invalid_strand_raises(self) -> None:
        with pytest.raises(ValueError, match="strand"):
            GenomicInterval("chr1", 0, 100, "F")

    def test_frozen_prevents_mutation(self) -> None:
        iv = GenomicInterval("chr1", 0, 100)
        with pytest.raises(Exception):
            iv.start = 999  # type: ignore[misc]


class TestGenomicIntervalProperties:
    def test_length(self) -> None:
        assert GenomicInterval("chr1", 100, 200).length == 100

    def test_center_even(self) -> None:
        assert GenomicInterval("chr1", 100, 200).center == 150

    def test_center_odd(self) -> None:
        assert GenomicInterval("chr1", 0, 21).center == 10

    def test_str_representation(self) -> None:
        iv = GenomicInterval("chr1", 100, 200, "+")
        assert str(iv) == "chr1:100-200(+)"


class TestGenomicIntervalConversions:
    def test_from_1based_decrements_start(self) -> None:
        iv = GenomicInterval.from_1based("chr1", 1, 100, "+")
        assert iv.start == 0
        assert iv.end == 100

    def test_from_1based_to_1based_round_trip(self) -> None:
        iv = GenomicInterval.from_1based("chrX", 500, 600, "-")
        chrom, s, e, strand = iv.to_1based()
        assert chrom == "chrX"
        assert s == 500
        assert e == 600
        assert strand == "-"

    def test_to_1based_increments_start(self) -> None:
        iv = GenomicInterval("chr1", 0, 50, "+")
        _, s, e, _ = iv.to_1based()
        assert s == 1
        assert e == 50

    def test_expand_symmetric(self) -> None:
        # center of [100, 200] = 150; expand by 50 → [100, 200]
        iv = GenomicInterval("chr1", 100, 200, "+")
        expanded = iv.expand(50)
        assert expanded.start == 100
        assert expanded.end == 200

    def test_expand_larger(self) -> None:
        iv = GenomicInterval("chr1", 100, 200, "+")  # center = 150
        expanded = iv.expand(100)
        assert expanded.start == 50
        assert expanded.end == 250

    def test_expand_clamps_at_zero(self) -> None:
        iv = GenomicInterval("chr1", 0, 10, "+")  # center = 5
        expanded = iv.expand(1000)
        assert expanded.start == 0


# ---------------------------------------------------------------------------
# Chromosome normalisation
# ---------------------------------------------------------------------------

class TestNormalizeChrom:
    @pytest.mark.parametrize("raw, expected", [
        ("chr1",   "chr1"),
        ("1",      "chr1"),
        ("Chr1",   "chr1"),
        ("CHR1",   "chr1"),
        ("chr22",  "chr22"),
        ("22",     "chr22"),
        ("chrX",   "chrX"),
        ("X",      "chrX"),
        ("chrx",   "chrX"),
        ("chrY",   "chrY"),
        ("Y",      "chrY"),
        ("chrM",   "chrM"),
        ("M",      "chrM"),
        ("MT",     "chrM"),
        ("chrMT",  "chrM"),
        ("CHRMT",  "chrM"),
    ])
    def test_normalisation(self, raw: str, expected: str) -> None:
        assert normalize_chrom(raw) == expected

    def test_invalid_chr99_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid hg38"):
            normalize_chrom("chr99")

    def test_scaffold_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_chrom("chr1_random_scaffold")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_chrom("")


class TestIsValidHg38Chrom:
    def test_valid_with_prefix(self) -> None:
        assert is_valid_hg38_chrom("chr1") is True

    def test_valid_without_prefix(self) -> None:
        assert is_valid_hg38_chrom("1") is True

    def test_valid_sex_chroms(self) -> None:
        assert is_valid_hg38_chrom("chrX") is True
        assert is_valid_hg38_chrom("chrY") is True

    def test_invalid_returns_false(self) -> None:
        assert is_valid_hg38_chrom("chr99") is False
        assert is_valid_hg38_chrom("scaffold_1") is False

    def test_all_standard_hg38_chroms_valid(self) -> None:
        for chrom in VALID_CHROMS_HG38:
            assert is_valid_hg38_chrom(chrom) is True, f"Expected {chrom} to be valid"


# ---------------------------------------------------------------------------
# Guide RNA normalisation
# ---------------------------------------------------------------------------

class TestNormalizeGuide:
    VALID_20NT = "GAGTCCGAGCAGAAGAAGAA"

    def test_uppercase_conversion(self) -> None:
        assert normalize_guide(self.VALID_20NT.lower()) == self.VALID_20NT

    def test_rna_to_dna_u_to_t(self) -> None:
        rna = "GAGUCCGAGCAGAAGAAGAA"
        assert normalize_guide(rna) == self.VALID_20NT

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert normalize_guide(f"  {self.VALID_20NT}  ") == self.VALID_20NT

    def test_strips_pam_from_23nt(self) -> None:
        with_pam = self.VALID_20NT + "NGG"  # 23 nt
        result = normalize_guide(with_pam)
        assert len(result) == 20
        assert result == self.VALID_20NT

    def test_21nt_raises(self) -> None:
        with pytest.raises(ValueError, match="20 nt"):
            normalize_guide(self.VALID_20NT + "A")

    def test_19nt_raises(self) -> None:
        with pytest.raises(ValueError, match="20 nt"):
            normalize_guide(self.VALID_20NT[:19])

    def test_n_character_raises(self) -> None:
        seq_with_n = "ACGTACGTACGTACGTACGN"
        with pytest.raises(ValueError, match="invalid characters"):
            normalize_guide(seq_with_n)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_guide("")


# ---------------------------------------------------------------------------
# Disk-based caching
# ---------------------------------------------------------------------------

class TestCache:
    def test_miss_returns_none(self, tmp_path: Path) -> None:
        result = load_cache(tmp_path, "chr1", "1000", "2000", "K562")
        assert result is None

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        data = {"signal": [1.0, 2.0, 3.0], "label": "K562"}
        save_cache(tmp_path, data, "chr1", "1000", "2000", "K562")
        loaded = load_cache(tmp_path, "chr1", "1000", "2000", "K562")
        assert loaded == data

    def test_different_keys_no_collision(self, tmp_path: Path) -> None:
        save_cache(tmp_path, "result_A", "chr1", "100")
        save_cache(tmp_path, "result_B", "chr2", "100")
        assert load_cache(tmp_path, "chr1", "100") == "result_A"
        assert load_cache(tmp_path, "chr2", "100") == "result_B"

    def test_creates_missing_cache_dir(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "subdir" / "cache"
        assert not cache_dir.exists()
        save_cache(cache_dir, {"x": 1}, "mykey")
        assert cache_dir.exists()

    def test_save_returns_existing_pkl_path(self, tmp_path: Path) -> None:
        path = save_cache(tmp_path, {"x": 1}, "mykey")
        assert isinstance(path, Path)
        assert path.exists()
        assert path.suffix == ".pkl"

    def test_overwrite_existing_entry(self, tmp_path: Path) -> None:
        save_cache(tmp_path, "first", "key")
        save_cache(tmp_path, "second", "key")
        assert load_cache(tmp_path, "key") == "second"

    def test_numpy_array_survives_round_trip(self, tmp_path: Path) -> None:
        pytest.importorskip("numpy")
        import numpy as np

        arr = np.array([1.0, 2.0, 3.0])
        save_cache(tmp_path, arr, "arr_key")
        loaded = load_cache(tmp_path, "arr_key")
        assert loaded is not None
        np.testing.assert_array_equal(loaded, arr)


# ---------------------------------------------------------------------------
# hg19 → hg38 liftover  (pyliftover mocked — no network calls in tests)
# ---------------------------------------------------------------------------

class TestLiftover:
    def _make_mock_lo(self, return_value: list) -> MagicMock:
        mock_lo = MagicMock()
        mock_lo.convert_coordinate.return_value = return_value
        return mock_lo

    def test_successful_conversion(self) -> None:
        mock_lo = self._make_mock_lo([("chr1", 925951, "+", 0)])
        with patch("src.utils._get_liftover_instance", return_value=mock_lo):
            result = liftover_hg19_to_hg38("chr1", 930000)
        assert result == ("chr1", 925951)

    def test_unmappable_position_returns_none(self) -> None:
        mock_lo = self._make_mock_lo([])
        with patch("src.utils._get_liftover_instance", return_value=mock_lo):
            result = liftover_hg19_to_hg38("chr1", 0)
        assert result is None

    def test_adds_chr_prefix_before_lookup(self) -> None:
        mock_lo = self._make_mock_lo([("chr1", 100, "+", 0)])
        with patch("src.utils._get_liftover_instance", return_value=mock_lo):
            liftover_hg19_to_hg38("1", 100)  # no "chr" prefix
        mock_lo.convert_coordinate.assert_called_once_with("chr1", 100)

    def test_chr_prefix_not_doubled(self) -> None:
        mock_lo = self._make_mock_lo([("chr1", 100, "+", 0)])
        with patch("src.utils._get_liftover_instance", return_value=mock_lo):
            liftover_hg19_to_hg38("chr1", 100)
        mock_lo.convert_coordinate.assert_called_once_with("chr1", 100)

    def test_import_error_propagates(self) -> None:
        with patch(
            "src.utils._get_liftover_instance",
            side_effect=ImportError("pyliftover"),
        ):
            with pytest.raises(ImportError, match="pyliftover"):
                liftover_hg19_to_hg38("chr1", 100)
