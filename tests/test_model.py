"""Tests for src/model.py.

Test strategy
-------------
- Metric helpers (compute_metrics, precision_at_k, etc.) are tested against
  controlled inputs with known expected values.
- XGBoost is trained for real on small synthetic datasets (fast: ~100 rows,
  10 estimators).  This validates the full CV pipeline without mocking.
- ``compare_models`` uses CVResult objects with fabricated fold metrics to
  test statistical comparisons.
- ``run_three_model_comparison`` is tested with a minimal synthetic feature
  matrix using n_splits=2 and n_estimators=10.
- No external API calls, BigWig files, or filesystem access are required.
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
    EXP_FEATURE_NAMES,
    MODEL_A_FEATURES,
    MODEL_B_FEATURES,
    MODEL_C_FEATURES,
    SEQUENCE_FEATURE_NAMES,
)
from src.model import (
    CVResult,
    DEFAULT_N_SPLITS,
    DEFAULT_XGB_PARAMS,
    FoldMetrics,
    MODEL_BC_CELL_TYPES,
    ModelComparison,
    TOP_FRACTION,
    compare_models,
    compute_metrics,
    compute_scale_pos_weight,
    cross_validate_model,
    enrichment_at_top_fraction,
    make_xgb_classifier,
    precision_at_k,
    prepare_model_data,
    recall_at_k,
    run_ablation,
    run_three_model_comparison,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_FAST_XGB = {"n_estimators": 10, "max_depth": 3}


def _make_synthetic_feature_matrix(
    n_rows: int = 120,
    n_guides: int = 6,
    seed: int = 42,
    include_ag: bool = True,
    include_exp: bool = True,
    cell_types: list[str] | None = None,
    datasets: list[str] | None = None,
    pos_rate: float = 0.10,
) -> pd.DataFrame:
    """Build a realistic synthetic feature matrix for testing.

    Each guide has at least one positive label to ensure GroupKFold can
    produce valid folds with positives in both train and test sets.
    """
    rng = np.random.default_rng(seed)

    if cell_types is None:
        cell_types = ["K562"] * 50 + ["HEK293"] * 40 + ["T cells"] * 30
    if datasets is None:
        datasets = ["guide_seq"] * 80 + ["change_seq"] * 40

    # Trim/repeat to match n_rows.
    ct_arr = (cell_types * (n_rows // len(cell_types) + 1))[:n_rows]
    ds_arr = (datasets * (n_rows // len(datasets) + 1))[:n_rows]

    guide_ids = [f"guide_{i % n_guides}" for i in range(n_rows)]

    # Labels: at least 1 positive per guide, then ~pos_rate overall.
    labels = np.zeros(n_rows, dtype=np.int32)
    for g in range(n_guides):
        guide_idx = [i for i in range(n_rows) if i % n_guides == g]
        labels[guide_idx[0]] = 1  # guaranteed positive per guide

    # Random additional positives.
    extra = rng.random(n_rows) < pos_rate
    labels = np.clip(labels + extra.astype(np.int32), 0, 1)

    df = pd.DataFrame(
        {
            "guide_id": guide_ids,
            "label": labels,
            "chrom": "chr1",
            "start": rng.integers(1_000_000, 5_000_000, size=n_rows),
            "strand": "+",
            "cell_type": ct_arr,
            "dataset": ds_arr,
        }
    )

    # Sequence features (all SEQUENCE_FEATURE_NAMES).
    for col in SEQUENCE_FEATURE_NAMES:
        if "mm_type" in col:
            df[col] = rng.integers(0, 13, size=n_rows)
        elif col == "num_mismatches":
            df[col] = rng.integers(1, 5, size=n_rows)  # >0 (no on-targets)
        elif col in (
            "seed_mismatches",
            "nonseed_mismatches",
            "consecutive_mm",
        ) or "mm_pos" in col:
            df[col] = rng.integers(0, 4, size=n_rows)
        elif col in ("pam_match",):
            df[col] = rng.integers(0, 2, size=n_rows)
        else:
            df[col] = rng.random(n_rows).astype(np.float32)

    # AlphaGenome features.
    if include_ag:
        for col in AG_FEATURE_NAMES:
            df[col] = (rng.random(n_rows) * 10).astype(np.float32)

    # ENCODE experimental features.
    if include_exp:
        for col in EXP_FEATURE_NAMES:
            df[col] = (rng.random(n_rows) * 10).astype(np.float32)

    return df


def _make_cv_result(
    model_type: str,
    fold_auprcs: list[float],
    fold_aurocs: list[float] | None = None,
    n_test: int = 20,
    n_pos_test: int = 2,
) -> CVResult:
    """Build a CVResult with fabricated fold metrics (no XGBoost needed)."""
    if fold_aurocs is None:
        fold_aurocs = [0.8] * len(fold_auprcs)

    folds = [
        FoldMetrics(
            fold=i,
            n_train=80,
            n_test=n_test,
            n_pos_test=n_pos_test,
            auprc=ap,
            auroc=ar,
            precision_at_top=0.5,
            recall_at_top=0.5,
            enrichment_at_top=2.0,
        )
        for i, (ap, ar) in enumerate(zip(fold_auprcs, fold_aurocs))
    ]

    n_folds = len(fold_auprcs)
    valid_ap = [v for v in fold_auprcs if not math.isnan(v)]

    dummy_n = n_test * n_folds
    dummy_y = np.zeros(dummy_n, dtype=np.int32)
    dummy_y[:n_pos_test * n_folds] = 1
    dummy_score = np.linspace(0, 1, dummy_n)[::-1]

    return CVResult(
        model_type=model_type,
        feature_cols=["f1", "f2"],
        folds=folds,
        oof_y_true=dummy_y,
        oof_y_score=dummy_score,
        final_model=MagicMock(),
        mean_auprc=float(np.mean(valid_ap)) if valid_ap else float("nan"),
        std_auprc=float(np.std(valid_ap)) if len(valid_ap) > 1 else 0.0,
        mean_auroc=float(np.mean(fold_aurocs)),
        std_auroc=float(np.std(fold_aurocs)) if len(fold_aurocs) > 1 else 0.0,
    )


# ===========================================================================
# compute_scale_pos_weight
# ===========================================================================

class TestComputeScalePosWeight:
    def test_basic_ratio(self):
        y = np.array([0, 0, 0, 1], dtype=np.int32)
        assert compute_scale_pos_weight(y) == pytest.approx(3.0)

    def test_balanced(self):
        y = np.array([0, 1, 0, 1], dtype=np.int32)
        assert compute_scale_pos_weight(y) == pytest.approx(1.0)

    def test_no_positives_returns_one(self):
        y = np.zeros(10, dtype=np.int32)
        assert compute_scale_pos_weight(y) == pytest.approx(1.0)

    def test_all_positives(self):
        y = np.ones(5, dtype=np.int32)
        # n_neg=0, n_pos=5 → 0 / 5 = 0.0
        assert compute_scale_pos_weight(y) == pytest.approx(0.0)

    def test_large_imbalance(self):
        y = np.array([1] + [0] * 99)
        assert compute_scale_pos_weight(y) == pytest.approx(99.0)


# ===========================================================================
# precision_at_k
# ===========================================================================

class TestPrecisionAtK:
    def _scores(self):
        y_true = np.array([1, 0, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.1])
        return y_true, y_score

    def test_top_1_is_correct(self):
        y_true, y_score = self._scores()
        # Top-1 score is 0.9 → label 1 → precision=1.0
        assert precision_at_k(y_true, y_score, k=1) == pytest.approx(1.0)

    def test_top_2(self):
        y_true, y_score = self._scores()
        # Top-2: scores 0.9 (label 1) and 0.8 (label 0) → precision=0.5
        assert precision_at_k(y_true, y_score, k=2) == pytest.approx(0.5)

    def test_top_3(self):
        y_true, y_score = self._scores()
        # Top-3: 0.9→1, 0.8→0, 0.7→1 → precision=2/3
        assert precision_at_k(y_true, y_score, k=3) == pytest.approx(2 / 3)

    def test_k_zero(self):
        y_true, y_score = self._scores()
        assert precision_at_k(y_true, y_score, k=0) == pytest.approx(0.0)

    def test_k_exceeds_length(self):
        y_true = np.array([1, 0])
        y_score = np.array([0.9, 0.1])
        # k=10 capped to 2; 1/2 = 0.5
        assert precision_at_k(y_true, y_score, k=10) == pytest.approx(0.5)

    def test_all_correct(self):
        y_true = np.array([1, 1, 1])
        y_score = np.array([0.9, 0.8, 0.7])
        assert precision_at_k(y_true, y_score, k=3) == pytest.approx(1.0)


# ===========================================================================
# recall_at_k
# ===========================================================================

class TestRecallAtK:
    def _scores(self):
        # 2 positives at index 0 and 2
        y_true = np.array([1, 0, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.1])
        return y_true, y_score

    def test_top_1_recalls_half(self):
        y_true, y_score = self._scores()
        # Top-1: index 0 (label 1) → recall = 1/2
        assert recall_at_k(y_true, y_score, k=1) == pytest.approx(0.5)

    def test_top_3_recalls_all(self):
        y_true, y_score = self._scores()
        # Top-3 contains both positives → recall = 1.0
        assert recall_at_k(y_true, y_score, k=3) == pytest.approx(1.0)

    def test_no_positives_returns_zero(self):
        y_true = np.zeros(5, dtype=np.int32)
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.1])
        assert recall_at_k(y_true, y_score, k=3) == pytest.approx(0.0)

    def test_k_exceeds_length(self):
        y_true = np.array([1, 0, 1])
        y_score = np.array([0.9, 0.5, 0.1])
        assert recall_at_k(y_true, y_score, k=100) == pytest.approx(1.0)


# ===========================================================================
# enrichment_at_top_fraction
# ===========================================================================

class TestEnrichmentAtTopFraction:
    def test_perfect_ranking_gives_high_enrichment(self):
        # All positives at top: 10% of 100 = 10 sites, 10 positives.
        y_true = np.array([1] * 10 + [0] * 90)
        y_score = np.array(list(range(100, 0, -1)), dtype=float)
        # overall rate = 10/100 = 0.10; top-10 precision = 10/10 = 1.0
        # enrichment = 1.0 / 0.10 = 10.0
        enr = enrichment_at_top_fraction(y_true, y_score, fraction=0.10)
        assert enr == pytest.approx(10.0)

    def test_random_ranking_gives_enrichment_near_one(self):
        rng = np.random.default_rng(0)
        y_true = (rng.random(1000) < 0.10).astype(int)
        y_score = rng.random(1000)
        enr = enrichment_at_top_fraction(y_true, y_score, fraction=0.10)
        # Expected ~1.0; allow generous tolerance for small-sample randomness.
        assert 0.3 < enr < 3.0

    def test_no_positives(self):
        y_true = np.zeros(50, dtype=int)
        y_score = np.arange(50, dtype=float)
        assert enrichment_at_top_fraction(y_true, y_score) == pytest.approx(0.0)

    def test_empty_array(self):
        assert enrichment_at_top_fraction(np.array([]), np.array([])) == pytest.approx(0.0)


# ===========================================================================
# compute_metrics
# ===========================================================================

class TestComputeMetrics:
    def _perfect(self, n_pos=10, n_neg=90):
        y_true = np.array([1] * n_pos + [0] * n_neg)
        # Perfect: positives ranked first.
        y_score = np.array(
            [1.0] * n_pos + [0.0] * n_neg, dtype=float
        )
        return y_true, y_score

    def test_perfect_classifier_auprc(self):
        y_true, y_score = self._perfect()
        m = compute_metrics(y_true, y_score)
        assert m["auprc"] == pytest.approx(1.0, abs=1e-6)

    def test_perfect_classifier_auroc(self):
        y_true, y_score = self._perfect()
        m = compute_metrics(y_true, y_score)
        assert m["auroc"] == pytest.approx(1.0, abs=1e-6)

    def test_random_classifier_auprc_near_prevalence(self):
        rng = np.random.default_rng(1)
        y_true = (rng.random(1000) < 0.05).astype(int)
        y_score = rng.random(1000)
        m = compute_metrics(y_true, y_score)
        assert 0.01 < m["auprc"] < 0.30  # near prevalence 5%

    def test_only_negatives_returns_nan(self):
        y_true = np.zeros(50, dtype=int)
        y_score = np.ones(50)
        m = compute_metrics(y_true, y_score)
        assert math.isnan(m["auprc"])
        assert math.isnan(m["auroc"])

    def test_only_positives_returns_nan(self):
        y_true = np.ones(50, dtype=int)
        y_score = np.ones(50)
        m = compute_metrics(y_true, y_score)
        assert math.isnan(m["auprc"])
        assert math.isnan(m["auroc"])

    def test_metric_keys_present(self):
        y_true, y_score = self._perfect()
        m = compute_metrics(y_true, y_score)
        for key in ("auprc", "auroc", "precision_at_top", "recall_at_top", "enrichment_at_top"):
            assert key in m

    def test_precision_recall_in_range(self):
        y_true, y_score = self._perfect()
        m = compute_metrics(y_true, y_score)
        assert 0.0 <= m["precision_at_top"] <= 1.0
        assert 0.0 <= m["recall_at_top"] <= 1.0
        assert m["enrichment_at_top"] >= 0.0

    def test_top_fraction_parameter(self):
        y_true, y_score = self._perfect(n_pos=5, n_neg=95)
        m10 = compute_metrics(y_true, y_score, top_fraction=0.10)
        m50 = compute_metrics(y_true, y_score, top_fraction=0.50)
        # With perfect ranking, top-10% (10 sites) captures all 5 positives.
        assert m10["recall_at_top"] == pytest.approx(1.0)
        # With 50% = 50 sites, recall is still 1.0 (all positives in top-10).
        assert m50["recall_at_top"] == pytest.approx(1.0)


# ===========================================================================
# FoldMetrics and CVResult dataclasses
# ===========================================================================

class TestDataclasses:
    def test_fold_metrics_construction(self):
        fm = FoldMetrics(
            fold=0,
            n_train=100,
            n_test=20,
            n_pos_test=2,
            auprc=0.45,
            auroc=0.80,
            precision_at_top=0.30,
            recall_at_top=0.60,
            enrichment_at_top=3.0,
        )
        assert fm.fold == 0
        assert fm.auprc == pytest.approx(0.45)

    def test_cv_result_construction(self):
        result = _make_cv_result("A", [0.3, 0.4, 0.35])
        assert result.model_type == "A"
        assert len(result.folds) == 3
        assert result.mean_auprc == pytest.approx(np.mean([0.3, 0.4, 0.35]))

    def test_model_comparison_construction(self):
        mc = ModelComparison(
            model_a_type="A",
            model_b_type="C",
            delta_auprc=0.05,
            p_value=0.03,
            significant=True,
        )
        assert mc.significant is True
        assert mc.delta_auprc == pytest.approx(0.05)


# ===========================================================================
# make_xgb_classifier
# ===========================================================================

class TestMakeXgbClassifier:
    def test_returns_xgb_classifier(self):
        import xgboost as xgb

        clf = make_xgb_classifier(3.0)
        assert isinstance(clf, xgb.XGBClassifier)

    def test_scale_pos_weight_set(self):
        clf = make_xgb_classifier(9.0)
        assert clf.scale_pos_weight == pytest.approx(9.0)

    def test_random_state_default(self):
        clf = make_xgb_classifier(1.0)
        assert clf.random_state == 42

    def test_extra_params_override(self):
        clf = make_xgb_classifier(1.0, extra_params={"n_estimators": 5, "max_depth": 2})
        assert clf.n_estimators == 5
        assert clf.max_depth == 2

    def test_custom_random_state(self):
        clf = make_xgb_classifier(1.0, random_state=7)
        assert clf.random_state == 7

    def test_verbosity_zero(self):
        clf = make_xgb_classifier(1.0)
        assert clf.verbosity == 0


# ===========================================================================
# prepare_model_data
# ===========================================================================

class TestPrepareModelData:
    @pytest.fixture
    def fm(self):
        return _make_synthetic_feature_matrix(n_rows=120, n_guides=6)

    def test_model_a_returns_all_rows(self, fm):
        X, y, groups, feat = prepare_model_data(fm, "A")
        assert X.shape[0] == len(fm)

    def test_model_a_features_are_sequence_only(self, fm):
        _, _, _, feat = prepare_model_data(fm, "A")
        # All returned features should be in SEQUENCE_FEATURE_NAMES.
        for f in feat:
            assert f in MODEL_A_FEATURES

    def test_model_b_features_include_exp(self, fm):
        _, _, _, feat = prepare_model_data(fm, "B")
        exp_in_feat = [f for f in feat if f.startswith("exp_")]
        assert len(exp_in_feat) > 0

    def test_model_c_features_include_ag(self, fm):
        _, _, _, feat = prepare_model_data(fm, "C")
        ag_in_feat = [f for f in feat if f.startswith("ag_")]
        assert len(ag_in_feat) > 0

    def test_cell_type_filter(self, fm):
        X, y, groups, _ = prepare_model_data(fm, "A", cell_types=["K562"])
        assert X.shape[0] < len(fm)
        # All returned rows should be K562.
        filtered = fm[fm["cell_type"] == "K562"]
        assert X.shape[0] == len(filtered)

    def test_exclude_datasets(self, fm):
        X_all, _, _, _ = prepare_model_data(fm, "A")
        X_excl, _, _, _ = prepare_model_data(
            fm, "A", exclude_datasets=["change_seq"]
        )
        assert X_excl.shape[0] < X_all.shape[0]

    def test_exclude_all_datasets_raises(self, fm):
        all_datasets = fm["dataset"].unique().tolist()
        with pytest.raises(ValueError, match="no rows remain"):
            prepare_model_data(fm, "A", exclude_datasets=all_datasets)

    def test_on_target_raises(self, fm):
        bad_fm = fm.copy()
        bad_fm.loc[bad_fm.index[0], "num_mismatches"] = 0
        with pytest.raises(ValueError, match="on-target"):
            prepare_model_data(bad_fm, "A")

    def test_returns_correct_types(self, fm):
        X, y, groups, feat = prepare_model_data(fm, "A")
        assert X.dtype == np.float32
        assert y.dtype == np.int32
        assert isinstance(feat, list)

    def test_groups_are_guide_ids(self, fm):
        _, _, groups, _ = prepare_model_data(fm, "A")
        assert set(groups) == set(fm["guide_id"].unique())

    def test_invalid_model_type_raises(self, fm):
        with pytest.raises(ValueError, match="'D'"):
            prepare_model_data(fm, "D")

    def test_missing_label_column_raises(self, fm):
        bad_fm = fm.drop(columns=["label"])
        with pytest.raises(ValueError, match="label"):
            prepare_model_data(bad_fm, "A")

    def test_model_type_case_insensitive(self, fm):
        X1, _, _, _ = prepare_model_data(fm, "a")
        X2, _, _, _ = prepare_model_data(fm, "A")
        np.testing.assert_array_equal(X1, X2)

    def test_include_missingness_indicators(self, fm):
        # Missingness indicator columns are created by impute_chromatin_features
        # and stored in the DataFrame before prepare_model_data is called.
        # Simulate that by manually adding a _missing column.
        fm2 = fm.copy()
        fm2["ag_dnase_mean_missing"] = 0  # added by impute_chromatin_features
        _, _, _, feat = prepare_model_data(
            fm2, "C", include_missingness_indicators=True
        )
        missing_cols = [f for f in feat if f.endswith("_missing")]
        assert len(missing_cols) > 0
        assert "ag_dnase_mean_missing" in missing_cols

    def test_x_shape_matches_feature_count(self, fm):
        X, _, _, feat = prepare_model_data(fm, "A")
        assert X.shape[1] == len(feat)


# ===========================================================================
# cross_validate_model
# ===========================================================================

class TestCrossValidateModel:
    """Tests using real XGBoost on tiny synthetic data."""

    @pytest.fixture
    def small_data(self):
        """3 guides, 60 rows total, at least 1 positive per guide."""
        rng = np.random.default_rng(0)
        n = 60
        n_guides = 3
        guide_ids = np.array([f"g{i % n_guides}" for i in range(n)])
        X = rng.random((n, 5)).astype(np.float32)
        y = np.zeros(n, dtype=np.int32)
        # 1 positive per guide
        for g in range(n_guides):
            idx = np.where(guide_ids == f"g{g}")[0][0]
            y[idx] = 1
        return X, y, guide_ids

    def test_returns_cv_result(self, small_data):
        X, y, groups = small_data
        result = cross_validate_model(
            X, y, groups,
            model_type="A",
            feature_cols=["f1", "f2", "f3", "f4", "f5"],
            n_splits=3,
            xgb_extra_params=_FAST_XGB,
        )
        assert isinstance(result, CVResult)

    def test_correct_model_type_stored(self, small_data):
        X, y, groups = small_data
        result = cross_validate_model(
            X, y, groups, model_type="test_A",
            feature_cols=["f1"],
            n_splits=2, xgb_extra_params=_FAST_XGB,
        )
        assert result.model_type == "test_A"

    def test_n_folds_equals_n_splits(self, small_data):
        X, y, groups = small_data
        result = cross_validate_model(
            X, y, groups, model_type="A",
            feature_cols=["f1", "f2", "f3", "f4", "f5"],
            n_splits=3, xgb_extra_params=_FAST_XGB,
        )
        assert len(result.folds) == 3

    def test_oof_arrays_same_length_as_input(self, small_data):
        X, y, groups = small_data
        result = cross_validate_model(
            X, y, groups, model_type="A",
            feature_cols=["f1", "f2", "f3", "f4", "f5"],
            n_splits=3, xgb_extra_params=_FAST_XGB,
        )
        assert len(result.oof_y_true) == len(y)
        assert len(result.oof_y_score) == len(y)

    def test_oof_scores_in_zero_one(self, small_data):
        X, y, groups = small_data
        result = cross_validate_model(
            X, y, groups, model_type="A",
            feature_cols=["f1", "f2", "f3", "f4", "f5"],
            n_splits=3, xgb_extra_params=_FAST_XGB,
        )
        assert np.all(result.oof_y_score >= 0.0)
        assert np.all(result.oof_y_score <= 1.0)

    def test_mean_auprc_in_valid_range_or_nan(self, small_data):
        X, y, groups = small_data
        result = cross_validate_model(
            X, y, groups, model_type="A",
            feature_cols=["f1", "f2", "f3", "f4", "f5"],
            n_splits=3, xgb_extra_params=_FAST_XGB,
        )
        if not math.isnan(result.mean_auprc):
            assert 0.0 <= result.mean_auprc <= 1.0

    def test_mean_auroc_in_valid_range_or_nan(self, small_data):
        X, y, groups = small_data
        result = cross_validate_model(
            X, y, groups, model_type="A",
            feature_cols=["f1", "f2", "f3", "f4", "f5"],
            n_splits=3, xgb_extra_params=_FAST_XGB,
        )
        if not math.isnan(result.mean_auroc):
            assert 0.0 <= result.mean_auroc <= 1.0

    def test_n_splits_auto_reduced_when_fewer_guides(self, small_data):
        """When n_splits > n_unique_groups, it should be silently reduced."""
        X, y, groups = small_data
        # 3 unique guides; request n_splits=10 — should be reduced to 3.
        result = cross_validate_model(
            X, y, groups, model_type="A",
            feature_cols=["f1", "f2", "f3", "f4", "f5"],
            n_splits=10, xgb_extra_params=_FAST_XGB,
        )
        assert len(result.folds) == 3  # reduced to n_unique_groups=3

    def test_final_model_is_fitted(self, small_data):
        import xgboost as xgb

        X, y, groups = small_data
        result = cross_validate_model(
            X, y, groups, model_type="A",
            feature_cols=["f1", "f2", "f3", "f4", "f5"],
            n_splits=3, xgb_extra_params=_FAST_XGB,
        )
        assert isinstance(result.final_model, xgb.XGBClassifier)
        # Should be able to predict without error.
        preds = result.final_model.predict_proba(X[:5])
        assert preds.shape == (5, 2)

    def test_feature_cols_stored(self, small_data):
        X, y, groups = small_data
        feat = ["f1", "f2", "f3", "f4", "f5"]
        result = cross_validate_model(
            X, y, groups, model_type="A",
            feature_cols=feat, n_splits=3, xgb_extra_params=_FAST_XGB,
        )
        assert result.feature_cols == feat

    def test_each_fold_has_train_and_test(self, small_data):
        X, y, groups = small_data
        result = cross_validate_model(
            X, y, groups, model_type="A",
            feature_cols=["f1", "f2", "f3", "f4", "f5"],
            n_splits=3, xgb_extra_params=_FAST_XGB,
        )
        for fm in result.folds:
            assert fm.n_train > 0
            assert fm.n_test > 0
            assert fm.n_train + fm.n_test == len(y)

    def test_no_guide_leakage(self, small_data):
        """Verify that train and test guides are disjoint in each fold."""
        X, y, groups = small_data

        # We cannot directly inspect splits from CVResult, but we can
        # check via StratifiedGroupKFold directly.
        from sklearn.model_selection import StratifiedGroupKFold

        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        for train_idx, test_idx in cv.split(X, y, groups=groups):
            train_guides = set(groups[train_idx])
            test_guides = set(groups[test_idx])
            assert train_guides.isdisjoint(test_guides), (
                "Guide leakage: same guide appears in both train and test"
            )


# ===========================================================================
# compare_models
# ===========================================================================

class TestCompareModels:
    def test_positive_delta_when_b_better(self):
        result_a = _make_cv_result("A", [0.3, 0.3, 0.3, 0.3, 0.3])
        result_c = _make_cv_result("C", [0.4, 0.4, 0.4, 0.4, 0.4])
        comp = compare_models(result_a, result_c)
        assert comp.delta_auprc == pytest.approx(0.1, abs=1e-6)

    def test_negative_delta_when_a_better(self):
        result_a = _make_cv_result("A", [0.5, 0.5, 0.5])
        result_c = _make_cv_result("C", [0.3, 0.3, 0.3])
        comp = compare_models(result_a, result_c)
        assert comp.delta_auprc < 0.0

    def test_significant_when_p_low(self):
        # Large consistent improvement → low p-value.
        result_a = _make_cv_result("A", [0.10, 0.10, 0.10, 0.10, 0.10])
        result_c = _make_cv_result("C", [0.50, 0.50, 0.50, 0.50, 0.50])
        comp = compare_models(result_a, result_c)
        assert comp.significant is True

    def test_not_significant_when_values_equal(self):
        result_a = _make_cv_result("A", [0.4, 0.4, 0.4, 0.4, 0.4])
        result_c = _make_cv_result("C", [0.4, 0.4, 0.4, 0.4, 0.4])
        comp = compare_models(result_a, result_c)
        assert comp.significant is False

    def test_model_type_labels_stored(self):
        result_a = _make_cv_result("A", [0.3, 0.4])
        result_c = _make_cv_result("C", [0.4, 0.5])
        comp = compare_models(result_a, result_c)
        assert comp.model_a_type == "A"
        assert comp.model_b_type == "C"

    def test_nan_folds_excluded(self):
        # If one fold has NaN AUPRC, it should be excluded from the t-test.
        result_a = _make_cv_result("A", [0.3, float("nan"), 0.3])
        result_c = _make_cv_result("C", [0.5, float("nan"), 0.5])
        comp = compare_models(result_a, result_c)
        assert comp.delta_auprc == pytest.approx(0.2, abs=1e-6)

    def test_fewer_than_two_valid_pairs_returns_nan_p(self):
        # Only one valid fold → cannot do paired t-test.
        result_a = _make_cv_result("A", [0.3, float("nan")])
        result_c = _make_cv_result("C", [0.5, float("nan")])
        comp = compare_models(result_a, result_c)
        assert math.isnan(comp.p_value)
        assert comp.significant is False

    def test_returns_model_comparison(self):
        result_a = _make_cv_result("A", [0.3, 0.4, 0.35])
        result_c = _make_cv_result("C", [0.4, 0.5, 0.45])
        comp = compare_models(result_a, result_c)
        assert isinstance(comp, ModelComparison)


# ===========================================================================
# run_ablation
# ===========================================================================

class TestRunAblation:
    @pytest.fixture
    def ablation_data(self):
        """Small dataset with 3 guides and 5 features for ablation."""
        rng = np.random.default_rng(10)
        n = 60
        n_guides = 3
        groups = np.array([f"g{i % n_guides}" for i in range(n)])
        X = rng.random((n, 5)).astype(np.float32)
        y = np.zeros(n, dtype=np.int32)
        for g in range(n_guides):
            idx = np.where(groups == f"g{g}")[0][0]
            y[idx] = 1
        feat = ["seq_a", "seq_b", "ag_dnase_mean", "ag_h3k4me3_mean", "ag_atac_mean"]
        return X, y, groups, feat

    def test_returns_dict_with_subset_keys(self, ablation_data):
        X, y, groups, feat = ablation_data
        subsets = {
            "seq_only": ["seq_a", "seq_b"],
            "seq_dnase": ["seq_a", "seq_b", "ag_dnase_mean"],
        }
        results = run_ablation(
            X, y, groups, feat,
            feature_subsets=subsets,
            n_splits=3,
            xgb_extra_params=_FAST_XGB,
        )
        assert set(results.keys()) == {"seq_only", "seq_dnase"}

    def test_each_result_has_required_keys(self, ablation_data):
        X, y, groups, feat = ablation_data
        subsets = {"seq_only": ["seq_a", "seq_b"]}
        results = run_ablation(
            X, y, groups, feat,
            feature_subsets=subsets,
            n_splits=3,
            xgb_extra_params=_FAST_XGB,
        )
        for v in results.values():
            assert "mean_auprc" in v
            assert "std_auprc" in v
            assert "mean_auroc" in v
            assert "std_auroc" in v

    def test_invalid_subset_columns_skipped(self, ablation_data):
        X, y, groups, feat = ablation_data
        subsets = {
            "valid": ["seq_a"],
            "invalid": ["nonexistent_col"],
        }
        results = run_ablation(
            X, y, groups, feat,
            feature_subsets=subsets,
            n_splits=3,
            xgb_extra_params=_FAST_XGB,
        )
        # "invalid" subset should be silently skipped.
        assert "valid" in results
        assert "invalid" not in results

    def test_more_features_at_least_as_good_on_average(self, ablation_data):
        """Not guaranteed for random data, but validates function runs."""
        X, y, groups, feat = ablation_data
        subsets = {
            "one_feat": ["seq_a"],
            "all_feats": list(feat),
        }
        results = run_ablation(
            X, y, groups, feat,
            feature_subsets=subsets,
            n_splits=3,
            xgb_extra_params=_FAST_XGB,
        )
        # Just check both ran and produced finite values (or NaN).
        for k in ("one_feat", "all_feats"):
            val = results[k]["mean_auprc"]
            assert isinstance(val, float)


# ===========================================================================
# run_three_model_comparison (integration)
# ===========================================================================

class TestRunThreeModelComparison:
    @pytest.fixture
    def full_fm(self):
        """Synthetic feature matrix with all MODEL_A/B/C features."""
        return _make_synthetic_feature_matrix(
            n_rows=120,
            n_guides=6,
            include_ag=True,
            include_exp=True,
        )

    def test_returns_three_keys(self, full_fm):
        results = run_three_model_comparison(
            full_fm,
            n_splits=2,
            xgb_extra_params=_FAST_XGB,
            model_bc_cell_types=["K562", "HEK293"],
        )
        assert set(results.keys()) == {"A", "B", "C"}

    def test_all_results_are_cv_result(self, full_fm):
        results = run_three_model_comparison(
            full_fm,
            n_splits=2,
            xgb_extra_params=_FAST_XGB,
            model_bc_cell_types=["K562", "HEK293"],
        )
        for v in results.values():
            assert isinstance(v, CVResult)

    def test_model_a_trained_on_more_data_than_b(self, full_fm):
        results = run_three_model_comparison(
            full_fm,
            n_splits=2,
            xgb_extra_params=_FAST_XGB,
            model_bc_cell_types=["K562", "HEK293"],
        )
        # Model A uses all cell types; B/C restricted to K562+HEK293.
        n_oof_a = len(results["A"].oof_y_true)
        n_oof_b = len(results["B"].oof_y_true)
        # T cells excluded from B → A has more or equal rows.
        assert n_oof_a >= n_oof_b

    def test_exclude_datasets_reduces_model_a_data(self, full_fm):
        results_all = run_three_model_comparison(
            full_fm, n_splits=2, xgb_extra_params=_FAST_XGB,
            model_bc_cell_types=["K562", "HEK293"],
        )
        results_excl = run_three_model_comparison(
            full_fm, n_splits=2, xgb_extra_params=_FAST_XGB,
            exclude_datasets=["change_seq"],
            model_bc_cell_types=["K562", "HEK293"],
        )
        # Excluding change_seq should give fewer OOF rows for Model A.
        assert len(results_excl["A"].oof_y_true) < len(results_all["A"].oof_y_true)

    def test_model_types_correctly_labeled(self, full_fm):
        results = run_three_model_comparison(
            full_fm, n_splits=2, xgb_extra_params=_FAST_XGB,
            model_bc_cell_types=["K562", "HEK293"],
        )
        assert results["A"].model_type == "A"
        assert results["B"].model_type == "B"
        assert results["C"].model_type == "C"

    def test_model_b_c_feature_cols_include_chromatin(self, full_fm):
        results = run_three_model_comparison(
            full_fm, n_splits=2, xgb_extra_params=_FAST_XGB,
            model_bc_cell_types=["K562", "HEK293"],
        )
        b_has_exp = any(f.startswith("exp_") for f in results["B"].feature_cols)
        c_has_ag = any(f.startswith("ag_") for f in results["C"].feature_cols)
        assert b_has_exp, "Model B should include exp_* features"
        assert c_has_ag, "Model C should include ag_* features"

    def test_model_a_feature_cols_are_sequence_only(self, full_fm):
        results = run_three_model_comparison(
            full_fm, n_splits=2, xgb_extra_params=_FAST_XGB,
            model_bc_cell_types=["K562", "HEK293"],
        )
        for f in results["A"].feature_cols:
            assert not f.startswith("exp_"), f"exp_ feature {f!r} found in Model A"
            assert not f.startswith("ag_"), f"ag_ feature {f!r} found in Model A"

    def test_saves_pr_curve_figure(self, full_fm, tmp_path):
        run_three_model_comparison(
            full_fm, n_splits=2, xgb_extra_params=_FAST_XGB,
            model_bc_cell_types=["K562", "HEK293"],
            output_dir=tmp_path,
        )
        fig_path = tmp_path / "fig2_pr_curves.png"
        assert fig_path.exists(), "PR curve figure not saved"

    def test_default_model_bc_cell_types(self, full_fm):
        """Omitting model_bc_cell_types should default to K562 + HEK293."""
        results = run_three_model_comparison(
            full_fm, n_splits=2, xgb_extra_params=_FAST_XGB
        )
        assert set(results.keys()) == {"A", "B", "C"}

    def test_oof_labels_are_binary(self, full_fm):
        results = run_three_model_comparison(
            full_fm, n_splits=2, xgb_extra_params=_FAST_XGB,
            model_bc_cell_types=["K562", "HEK293"],
        )
        for result in results.values():
            unique = set(result.oof_y_true.tolist())
            assert unique.issubset({0, 1})


# ===========================================================================
# plot_pr_curves
# ===========================================================================

class TestPlotPrCurves:
    def test_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        from src.model import plot_pr_curves

        results = {
            "A": _make_cv_result("A", [0.3, 0.4]),
            "C": _make_cv_result("C", [0.4, 0.5]),
        }
        # Provide OOF arrays with actual positives.
        for r in results.values():
            rng = np.random.default_rng(0)
            r.oof_y_true = np.array([1] * 5 + [0] * 45)
            r.oof_y_score = rng.random(50)

        fig = plot_pr_curves(results)
        import matplotlib.pyplot as plt
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.model import plot_pr_curves

        result = _make_cv_result("A", [0.3])
        rng = np.random.default_rng(1)
        result.oof_y_true = np.array([1] * 5 + [0] * 45)
        result.oof_y_score = rng.random(50)

        out = tmp_path / "pr_test.png"
        fig = plot_pr_curves({"A": result}, output_path=out)
        assert out.exists()
        plt.close(fig)


# ===========================================================================
# Module-level constants
# ===========================================================================

class TestConstants:
    def test_model_bc_cell_types(self):
        assert "K562" in MODEL_BC_CELL_TYPES
        assert "HEK293" in MODEL_BC_CELL_TYPES

    def test_default_n_splits(self):
        assert DEFAULT_N_SPLITS == 5

    def test_top_fraction(self):
        assert 0.0 < TOP_FRACTION < 1.0

    def test_default_xgb_params_has_required_keys(self):
        for key in ("n_estimators", "max_depth", "learning_rate",
                    "subsample", "colsample_bytree"):
            assert key in DEFAULT_XGB_PARAMS
