"""Tests for src/feature_matrix.py.

Test strategy
-------------
- No AlphaGenome API calls or BigWig file reads occur.
- AlphaGenome and ENCODE batch functions are patched where needed.
- Sequence features are computed for real (small synthetic guides).
- All label assignment, imputation, and assembly logic is tested against
  controlled DataFrames with known expected outputs.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.feature_matrix import (
    AG_FEATURE_NAMES,
    DEFAULT_MATCH_WINDOW_BP,
    EXP_FEATURE_NAMES,
    METADATA_COLS,
    MISSING_INDICATOR_SUFFIX,
    MM_TYPE_TO_INT,
    MODEL_A_FEATURES,
    MODEL_B_FEATURES,
    MODEL_C_FEATURES,
    SEQUENCE_FEATURE_NAMES,
    build_feature_matrix,
    extract_sequence_features_batch,
    get_model_features,
    impute_chromatin_features,
    label_candidates,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

# Valid 20-nt guide sequences for testing.
_GUIDE = "GAGTCCGAGCAGAAGAAGAA"
_OFFTARGET_1MM = "GAGTCCGAGCAGAAGAACAA"  # 1 mismatch at pos 3 from 3' end
_OFFTARGET_2MM = "GAGTCCGAGCAGAAGAAGCC"  # 2 mismatches
_PAM = "TGG"


def _make_candidates(
    n: int = 5,
    chrom: str = "chr7",
    guide_id: str = "VEGFA_1",
    start_base: int = 5_000_000,
    num_mismatches: int = 1,
) -> pd.DataFrame:
    """Return a small synthetic candidates DataFrame."""
    return pd.DataFrame({
        "guide_id": [guide_id] * n,
        "guide_seq": [_GUIDE] * n,
        "chrom": [chrom] * n,
        "start": [start_base + i * 1000 for i in range(n)],
        "strand": ["+"] * n,
        "off_target_seq": [_OFFTARGET_1MM] * n,
        "pam_seq": [_PAM] * n,
        "num_mismatches": [num_mismatches] * n,
    })


def _make_positives(
    chrom: str = "chr7",
    guide_id: str = "VEGFA_1",
    starts: list[int] | None = None,
) -> pd.DataFrame:
    """Return a small synthetic positives DataFrame."""
    if starts is None:
        starts = [5_000_000]
    return pd.DataFrame({
        "guide_id": [guide_id] * len(starts),
        "chrom": [chrom] * len(starts),
        "start": starts,
        "read_count": [100] * len(starts),
    })


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_sequence_feature_names_count(self) -> None:
        # 3 scalar + 20 mm_pos + 20 mm_type + 6 aggregates = 49
        assert len(SEQUENCE_FEATURE_NAMES) == 49

    def test_mm_type_to_int_match_is_zero(self) -> None:
        assert MM_TYPE_TO_INT[""] == 0

    def test_mm_type_to_int_has_12_mismatch_types(self) -> None:
        mismatch_values = {v for k, v in MM_TYPE_TO_INT.items() if k != ""}
        assert mismatch_values == set(range(1, 13))

    def test_model_a_equals_sequence_features(self) -> None:
        assert MODEL_A_FEATURES == SEQUENCE_FEATURE_NAMES

    def test_model_b_contains_exp_features(self) -> None:
        for feat in EXP_FEATURE_NAMES:
            assert feat in MODEL_B_FEATURES

    def test_model_c_contains_ag_features(self) -> None:
        for feat in AG_FEATURE_NAMES:
            assert feat in MODEL_C_FEATURES

    def test_default_match_window_is_5(self) -> None:
        assert DEFAULT_MATCH_WINDOW_BP == 5

    def test_missing_suffix(self) -> None:
        assert MISSING_INDICATOR_SUFFIX == "_missing"


# ---------------------------------------------------------------------------
# TestGetModelFeatures
# ---------------------------------------------------------------------------

class TestGetModelFeatures:
    def test_model_a(self) -> None:
        feats = get_model_features("A")
        assert feats == SEQUENCE_FEATURE_NAMES

    def test_model_b(self) -> None:
        feats = get_model_features("B")
        assert set(SEQUENCE_FEATURE_NAMES).issubset(feats)
        assert set(EXP_FEATURE_NAMES).issubset(feats)
        assert not any(f in feats for f in AG_FEATURE_NAMES)

    def test_model_c(self) -> None:
        feats = get_model_features("C")
        assert set(SEQUENCE_FEATURE_NAMES).issubset(feats)
        assert set(AG_FEATURE_NAMES).issubset(feats)
        assert not any(f in feats for f in EXP_FEATURE_NAMES)

    def test_case_insensitive(self) -> None:
        assert get_model_features("a") == get_model_features("A")
        assert get_model_features("b") == get_model_features("B")
        assert get_model_features("c") == get_model_features("C")

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="model_type"):
            get_model_features("D")

    def test_model_b_with_indicators(self) -> None:
        feats = get_model_features("B", include_missingness_indicators=True)
        for f in EXP_FEATURE_NAMES:
            assert f + "_missing" in feats

    def test_model_c_with_indicators(self) -> None:
        feats = get_model_features("C", include_missingness_indicators=True)
        for f in AG_FEATURE_NAMES:
            assert f + "_missing" in feats

    def test_model_a_no_indicators_even_if_requested(self) -> None:
        feats = get_model_features("A", include_missingness_indicators=True)
        # Model A has no chromatin features, so no _missing columns added.
        assert not any(f.endswith("_missing") for f in feats)


# ---------------------------------------------------------------------------
# TestLabelCandidates
# ---------------------------------------------------------------------------

class TestLabelCandidates:
    def test_exact_position_match(self) -> None:
        cands = _make_candidates(n=2)
        # starts: 5_000_000, 5_001_000 — positive at exactly 5_000_000
        pos = _make_positives(starts=[5_000_000])
        result = label_candidates(cands, pos)
        assert result["label"].iloc[0] == 1
        assert result["label"].iloc[1] == 0

    def test_within_window_is_positive(self) -> None:
        cands = _make_candidates(n=1)
        cands["start"] = [5_000_003]
        pos = _make_positives(starts=[5_000_000])
        result = label_candidates(cands, pos, match_window_bp=5)
        assert result["label"].iloc[0] == 1

    def test_just_outside_window_is_negative(self) -> None:
        cands = _make_candidates(n=1)
        cands["start"] = [5_000_006]
        pos = _make_positives(starts=[5_000_000])
        result = label_candidates(cands, pos, match_window_bp=5)
        assert result["label"].iloc[0] == 0

    def test_exactly_at_window_boundary_is_positive(self) -> None:
        cands = _make_candidates(n=1)
        cands["start"] = [5_000_005]
        pos = _make_positives(starts=[5_000_000])
        result = label_candidates(cands, pos, match_window_bp=5)
        assert result["label"].iloc[0] == 1

    def test_different_guide_id_no_match(self) -> None:
        cands = _make_candidates(n=1, guide_id="VEGFA_1")
        pos = _make_positives(guide_id="VEGFA_2", starts=[5_000_000])
        result = label_candidates(cands, pos)
        assert result["label"].iloc[0] == 0

    def test_different_chrom_no_match(self) -> None:
        cands = _make_candidates(n=1, chrom="chr7")
        pos = _make_positives(chrom="chr12", starts=[5_000_000])
        result = label_candidates(cands, pos)
        assert result["label"].iloc[0] == 0

    def test_empty_candidates(self) -> None:
        cands = _make_candidates(n=0)
        pos = _make_positives()
        result = label_candidates(cands, pos)
        assert len(result) == 0
        assert "label" in result.columns

    def test_empty_positives(self) -> None:
        cands = _make_candidates(n=3)
        pos = _make_positives(starts=[])
        result = label_candidates(cands, pos)
        assert (result["label"] == 0).all()

    def test_all_candidates_are_negative(self) -> None:
        cands = _make_candidates(n=5, start_base=1_000_000)
        pos = _make_positives(starts=[9_000_000])
        result = label_candidates(cands, pos)
        assert result["label"].sum() == 0

    def test_multiple_positives_any_match(self) -> None:
        cands = _make_candidates(n=1)
        cands["start"] = [5_000_000]
        pos = _make_positives(starts=[4_000_000, 5_000_000, 6_000_000])
        result = label_candidates(cands, pos)
        assert result["label"].iloc[0] == 1

    def test_chrom_normalisation(self) -> None:
        """'7' in candidates and 'chr7' in positives should still match."""
        cands = _make_candidates(n=1, chrom="7")
        pos = _make_positives(chrom="chr7", starts=[5_000_000])
        result = label_candidates(cands, pos)
        assert result["label"].iloc[0] == 1

    def test_input_not_mutated(self) -> None:
        cands = _make_candidates(n=3)
        pos = _make_positives()
        orig_cols = list(cands.columns)
        label_candidates(cands, pos)
        assert list(cands.columns) == orig_cols

    def test_label_counts_logged(self, caplog) -> None:
        cands = _make_candidates(n=4)
        cands["start"] = [5_000_000, 5_000_001, 6_000_000, 7_000_000]
        pos = _make_positives(starts=[5_000_000])
        with caplog.at_level("INFO"):
            result = label_candidates(cands, pos)
        assert result["label"].sum() == 2  # both within default window=5


# ---------------------------------------------------------------------------
# TestExtractSequenceFeaturesBatch
# ---------------------------------------------------------------------------

class TestExtractSequenceFeaturesBatch:
    def test_returns_all_feature_columns(self) -> None:
        cands = _make_candidates(n=2)
        result = extract_sequence_features_batch(cands)
        for feat in SEQUENCE_FEATURE_NAMES:
            assert feat in result.columns, f"Missing: {feat}"

    def test_cfd_score_in_range(self) -> None:
        cands = _make_candidates(n=3)
        result = extract_sequence_features_batch(cands)
        assert (result["cfd_score"] >= 0).all()
        assert (result["cfd_score"] <= 1).all()

    def test_mit_score_nonnegative(self) -> None:
        cands = _make_candidates(n=3)
        result = extract_sequence_features_batch(cands)
        assert (result["mit_score"] >= 0).all()

    def test_mm_type_columns_are_integers(self) -> None:
        cands = _make_candidates(n=2)
        result = extract_sequence_features_batch(cands)
        for i in range(1, 21):
            col = f"mm_type_{i}"
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} not numeric"

    def test_mm_type_match_position_is_zero(self) -> None:
        """Positions with no mismatch should encode as 0."""
        cands = _make_candidates(n=1)
        result = extract_sequence_features_batch(cands)
        # Identify matching positions from mm_pos columns
        for i in range(1, 21):
            if result[f"mm_pos_{i}"].iloc[0] == 0:
                assert result[f"mm_type_{i}"].iloc[0] == 0

    def test_num_mismatches_matches_mm_pos_sum(self) -> None:
        cands = _make_candidates(n=1)
        result = extract_sequence_features_batch(cands)
        mm_pos_sum = sum(result[f"mm_pos_{i}"].iloc[0] for i in range(1, 21))
        assert int(result["num_mismatches"].iloc[0]) == int(mm_pos_sum)

    def test_preserves_input_columns(self) -> None:
        cands = _make_candidates(n=2)
        orig_cols = list(cands.columns)
        result = extract_sequence_features_batch(cands)
        for col in orig_cols:
            assert col in result.columns

    def test_bad_sequence_gets_nan(self) -> None:
        """Rows with sequences that fail validation should get NaN features."""
        cands = _make_candidates(n=2)
        cands.loc[cands.index[0], "off_target_seq"] = "BADSEQ"  # too short
        result = extract_sequence_features_batch(cands)
        # First row should be NaN
        assert math.isnan(result["cfd_score"].iloc[0])
        # Second row should be valid
        assert not math.isnan(result["cfd_score"].iloc[1])

    def test_pam_match_canonical(self) -> None:
        cands = _make_candidates(n=1)
        cands["pam_seq"] = ["TGG"]
        result = extract_sequence_features_batch(cands)
        assert result["pam_match"].iloc[0] == 1

    def test_pam_match_noncanonical(self) -> None:
        cands = _make_candidates(n=1)
        cands["pam_seq"] = ["TAA"]
        result = extract_sequence_features_batch(cands)
        assert result["pam_match"].iloc[0] == 0


# ---------------------------------------------------------------------------
# TestImpute
# ---------------------------------------------------------------------------

class TestImpute:
    def test_partial_nan_filled_with_median(self) -> None:
        df = pd.DataFrame({"feat": [1.0, 2.0, np.nan, 4.0, np.nan]})
        result = impute_chromatin_features(df, ["feat"])
        # Median of [1, 2, 4] = 2.0
        assert result["feat"].isna().sum() == 0
        assert result["feat"].iloc[2] == pytest.approx(2.0)
        assert result["feat"].iloc[4] == pytest.approx(2.0)

    def test_non_nan_values_unchanged(self) -> None:
        df = pd.DataFrame({"feat": [1.0, 2.0, np.nan, 4.0]})
        result = impute_chromatin_features(df, ["feat"])
        assert result["feat"].iloc[0] == pytest.approx(1.0)
        assert result["feat"].iloc[1] == pytest.approx(2.0)
        assert result["feat"].iloc[3] == pytest.approx(4.0)

    def test_missingness_indicator_added(self) -> None:
        df = pd.DataFrame({"feat": [1.0, np.nan, 3.0]})
        result = impute_chromatin_features(df, ["feat"])
        assert "feat_missing" in result.columns
        assert result["feat_missing"].iloc[0] == 0
        assert result["feat_missing"].iloc[1] == 1
        assert result["feat_missing"].iloc[2] == 0

    def test_no_nan_values_unchanged(self) -> None:
        df = pd.DataFrame({"feat": [1.0, 2.0, 3.0]})
        result = impute_chromatin_features(df, ["feat"])
        assert list(result["feat"]) == pytest.approx([1.0, 2.0, 3.0])
        assert (result["feat_missing"] == 0).all()

    def test_all_nan_filled_with_zero(self) -> None:
        df = pd.DataFrame({"feat": [np.nan, np.nan, np.nan]})
        result = impute_chromatin_features(df, ["feat"])
        assert result["feat"].isna().sum() == 0
        assert (result["feat"] == 0.0).all()

    def test_missing_column_skipped(self) -> None:
        df = pd.DataFrame({"feat": [1.0, 2.0]})
        # Should not raise even though "nonexistent" is not in df.
        result = impute_chromatin_features(df, ["nonexistent"])
        assert list(result.columns) == ["feat"]

    def test_multiple_features(self) -> None:
        df = pd.DataFrame({
            "feat_a": [1.0, np.nan, 3.0],
            "feat_b": [np.nan, 2.0, np.nan],
        })
        result = impute_chromatin_features(df, ["feat_a", "feat_b"])
        assert result["feat_a"].isna().sum() == 0
        assert result["feat_b"].isna().sum() == 0
        assert "feat_a_missing" in result.columns
        assert "feat_b_missing" in result.columns

    def test_input_not_mutated(self) -> None:
        df = pd.DataFrame({"feat": [1.0, np.nan, 3.0]})
        orig = df.copy()
        impute_chromatin_features(df, ["feat"])
        pd.testing.assert_frame_equal(df, orig)


# ---------------------------------------------------------------------------
# TestBuildFeatureMatrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    """Tests for build_feature_matrix — sequence-only path (no API calls)."""

    def _base_candidates(self) -> pd.DataFrame:
        """5 candidates, guide_id VEGFA_1, positions 5M–5.004M, 1 mismatch each."""
        return _make_candidates(n=5, num_mismatches=1)

    def _base_positives(self) -> pd.DataFrame:
        return _make_positives(starts=[5_000_000])

    def _add_on_target(self, cands: pd.DataFrame) -> pd.DataFrame:
        """Append a row with num_mismatches=0 (on-target)."""
        on_target = cands.iloc[[0]].copy()
        on_target["num_mismatches"] = 0
        return pd.concat([cands, on_target], ignore_index=True)

    # ---- Validation -------------------------------------------------------

    def test_missing_required_column_raises(self) -> None:
        cands = self._base_candidates().drop(columns=["guide_seq"])
        pos = self._base_positives()
        with pytest.raises(ValueError, match="guide_seq"):
            build_feature_matrix(cands, pos, cell_type="K562")

    def test_ag_model_without_ontology_raises(self) -> None:
        cands = self._base_candidates()
        pos = self._base_positives()
        mock_model = MagicMock()
        with pytest.raises(ValueError, match="ag_ontology_term"):
            build_feature_matrix(
                cands, pos, cell_type="K562",
                ag_model=mock_model, ag_cache_dir=Path("/tmp"),
            )

    def test_ag_model_without_cache_dir_raises(self) -> None:
        cands = self._base_candidates()
        pos = self._base_positives()
        mock_model = MagicMock()
        with pytest.raises(ValueError, match="ag_cache_dir"):
            build_feature_matrix(
                cands, pos, cell_type="K562",
                ag_model=mock_model, ag_ontology_term="EFO:0002067",
            )

    # ---- On-target removal ------------------------------------------------

    def test_on_target_removed(self) -> None:
        cands = self._add_on_target(self._base_candidates())
        pos = self._base_positives()
        result = build_feature_matrix(cands, pos, cell_type="K562")
        assert (result["num_mismatches"] > 0).all()

    def test_on_target_count_logged(self, caplog) -> None:
        cands = self._add_on_target(self._base_candidates())
        pos = self._base_positives()
        with caplog.at_level("INFO"):
            build_feature_matrix(cands, pos, cell_type="K562")
        assert any("on-target" in msg.lower() for msg in caplog.messages)

    # ---- Label assignment -------------------------------------------------

    def test_label_column_present(self) -> None:
        cands = self._base_candidates()
        pos = self._base_positives()
        result = build_feature_matrix(cands, pos, cell_type="K562")
        assert "label" in result.columns

    def test_correct_positive_count(self) -> None:
        cands = self._base_candidates()
        pos = _make_positives(starts=[5_000_000, 5_001_000])
        result = build_feature_matrix(cands, pos, cell_type="K562")
        # Two candidates fall within window of two positives
        assert result["label"].sum() == 2

    # ---- Sequence features ------------------------------------------------

    def test_sequence_features_present(self) -> None:
        cands = self._base_candidates()
        pos = self._base_positives()
        result = build_feature_matrix(cands, pos, cell_type="K562")
        for feat in SEQUENCE_FEATURE_NAMES:
            assert feat in result.columns, f"Missing: {feat}"

    def test_cfd_score_in_range(self) -> None:
        cands = self._base_candidates()
        pos = self._base_positives()
        result = build_feature_matrix(cands, pos, cell_type="K562")
        assert (result["cfd_score"] >= 0).all()
        assert (result["cfd_score"] <= 1).all()

    # ---- Metadata ---------------------------------------------------------

    def test_cell_type_column(self) -> None:
        cands = self._base_candidates()
        pos = self._base_positives()
        result = build_feature_matrix(cands, pos, cell_type="K562")
        assert (result["cell_type"] == "K562").all()

    def test_dataset_column_default(self) -> None:
        cands = self._base_candidates()
        pos = self._base_positives()
        result = build_feature_matrix(cands, pos, cell_type="K562")
        assert "dataset" in result.columns

    def test_preserves_extra_candidate_columns(self) -> None:
        cands = self._base_candidates()
        cands["custom_col"] = "extra"
        pos = self._base_positives()
        result = build_feature_matrix(cands, pos, cell_type="K562")
        assert "custom_col" in result.columns

    # ---- AlphaGenome integration (mocked) ---------------------------------

    def test_ag_features_added_when_model_provided(self, tmp_path: Path) -> None:
        cands = self._base_candidates()
        pos = self._base_positives()
        mock_model = MagicMock()

        # Build a fake AG output: input DataFrame + ag_* columns
        def fake_ag_batch(sites, cell_type, model, cache_dir, **kwargs):
            out = sites.copy()
            for feat in AG_FEATURE_NAMES:
                out[feat] = 1.0
            return out

        with patch("src.feature_matrix._ag_mod.batch_get_chromatin_features", side_effect=fake_ag_batch):
            result = build_feature_matrix(
                cands, pos, cell_type="K562",
                ag_model=mock_model,
                ag_ontology_term="EFO:0002067",
                ag_cache_dir=tmp_path,
            )

        for feat in AG_FEATURE_NAMES:
            assert feat in result.columns
            assert (result[feat] == 1.0).all()

    def test_ag_missingness_indicators_added(self, tmp_path: Path) -> None:
        cands = self._base_candidates()
        pos = self._base_positives()
        mock_model = MagicMock()

        def fake_ag_batch(sites, cell_type, model, cache_dir, **kwargs):
            out = sites.copy()
            for feat in AG_FEATURE_NAMES:
                out[feat] = np.nan  # all NaN → imputed
            return out

        with patch("src.feature_matrix._ag_mod.batch_get_chromatin_features", side_effect=fake_ag_batch):
            result = build_feature_matrix(
                cands, pos, cell_type="K562",
                ag_model=mock_model,
                ag_ontology_term="EFO:0002067",
                ag_cache_dir=tmp_path,
            )

        for feat in AG_FEATURE_NAMES:
            assert feat + "_missing" in result.columns
            # All were NaN → all indicators should be 1
            assert (result[feat + "_missing"] == 1).all()

    # ---- ENCODE integration (mocked) --------------------------------------

    def test_encode_features_added_when_manifest_provided(self) -> None:
        cands = self._base_candidates()
        pos = self._base_positives()
        mock_manifest = MagicMock()

        def fake_enc_batch(sites, cell_type, manifest, **kwargs):
            out = sites.copy()
            for feat in EXP_FEATURE_NAMES:
                out[feat] = 2.0
            return out

        with patch("src.feature_matrix._enc_mod.batch_get_chromatin_features", side_effect=fake_enc_batch):
            result = build_feature_matrix(
                cands, pos, cell_type="K562",
                encode_manifest=mock_manifest,
            )

        for feat in EXP_FEATURE_NAMES:
            assert feat in result.columns
            assert (result[feat] == 2.0).all()

    def test_encode_missingness_indicators_added(self) -> None:
        cands = self._base_candidates()
        pos = self._base_positives()
        mock_manifest = MagicMock()

        def fake_enc_batch(sites, cell_type, manifest, **kwargs):
            out = sites.copy()
            for feat in EXP_FEATURE_NAMES:
                out[feat] = np.nan
            return out

        with patch("src.feature_matrix._enc_mod.batch_get_chromatin_features", side_effect=fake_enc_batch):
            result = build_feature_matrix(
                cands, pos, cell_type="K562",
                encode_manifest=mock_manifest,
            )

        for feat in EXP_FEATURE_NAMES:
            assert feat + "_missing" in result.columns

    # ---- Parquet save -----------------------------------------------------

    def test_saves_parquet(self, tmp_path: Path) -> None:
        cands = self._base_candidates()
        pos = self._base_positives()
        out_path = tmp_path / "fm.parquet"
        build_feature_matrix(cands, pos, cell_type="K562", output_path=out_path)
        assert out_path.exists()
        loaded = pd.read_parquet(out_path)
        assert "label" in loaded.columns
        assert "cfd_score" in loaded.columns

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        cands = self._base_candidates()
        pos = self._base_positives()
        out_path = tmp_path / "nested" / "dir" / "fm.parquet"
        build_feature_matrix(cands, pos, cell_type="K562", output_path=out_path)
        assert out_path.exists()

    # ---- Edge cases -------------------------------------------------------

    def test_all_on_target_returns_empty(self) -> None:
        cands = _make_candidates(n=3, num_mismatches=0)
        pos = self._base_positives()
        result = build_feature_matrix(cands, pos, cell_type="K562")
        assert len(result) == 0

    def test_zero_positives_all_negative(self) -> None:
        cands = self._base_candidates()
        pos = _make_positives(starts=[])
        result = build_feature_matrix(cands, pos, cell_type="K562")
        assert (result["label"] == 0).all()
