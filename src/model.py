"""Model training, cross-validation, and evaluation for SafeGenome.

Three XGBoost models are compared using leave-one-guide-out cross-validation:

- **Model A** (sequence-only baseline): trained on all datasets using only
  sequence-derived features (CFD score, mismatch positions/types, GC content,
  etc.).  Uses the full dataset for maximum baseline strength.
- **Model B** (sequence + experimental chromatin — gold standard): trained on
  K562 + HEK293 only, using sequence features plus ENCODE experimental
  chromatin signals (DNase-seq, ChIP-seq histone marks).
- **Model C** (sequence + AlphaGenome predicted chromatin — our method): same
  cell-type subset as Model B, using sequence features plus AlphaGenome-
  predicted chromatin features.

Critical protocol enforced here
---------------------------------
1. **Leave-one-guide-out CV** via ``StratifiedGroupKFold`` grouped by
   ``guide_id``.  Sites from the same guide must never split across train/test
   — they share sequence-level properties that would cause data leakage.
2. **Average Precision (AUPRC)** is the primary metric.  ROC-AUC is computed
   as a secondary metric but is misleadingly optimistic given the severe class
   imbalance (typically 1–5% positives).
3. **scale_pos_weight** is set to ``n_neg / n_pos`` in each XGBoost model to
   compensate for class imbalance.
4. **random_state=42** used everywhere for reproducibility.
5. **On-target sites must already be removed** from the feature matrix before
   calling any function here (enforced via a ValueError in
   ``prepare_model_data``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb

from src.feature_matrix import get_model_features
from src.utils import log_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default cell types for Models B and C (where chromatin data exists).
MODEL_BC_CELL_TYPES: list[str] = ["K562", "HEK293", "CD4_Tcell"]

#: Default number of CV folds.
DEFAULT_N_SPLITS: int = 5

#: Top fraction used for precision/recall/enrichment metrics.
TOP_FRACTION: float = 0.10

#: Default XGBoost parameters shared by all three models.
DEFAULT_XGB_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "verbosity": 0,
    "random_state": 42,
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Compute XGBoost ``scale_pos_weight`` as ``n_neg / n_pos``.

    Parameters
    ----------
    y : np.ndarray
        Binary label array (0 or 1).

    Returns
    -------
    float
        ``n_neg / n_pos``.  Returns 1.0 if there are no positives.
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        logger.warning("compute_scale_pos_weight: no positives found; returning 1.0")
        return 1.0
    return float(n_neg) / float(n_pos)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Precision among the top-k scored predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels.
    y_score : np.ndarray
        Predicted scores (higher = more likely positive).
    k : int
        Number of top predictions to evaluate.

    Returns
    -------
    float
        Fraction of top-k predictions that are true positives.
    """
    k = min(k, len(y_true))
    if k == 0:
        return 0.0
    top_k_idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[top_k_idx].sum()) / k


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Recall among the top-k scored predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels.
    y_score : np.ndarray
        Predicted scores (higher = more likely positive).
    k : int
        Number of top predictions to evaluate.

    Returns
    -------
    float
        Fraction of all true positives recovered in the top-k predictions.
        Returns 0.0 if there are no positives.
    """
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    k = min(k, len(y_true))
    top_k_idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[top_k_idx].sum()) / n_pos


def enrichment_at_top_fraction(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fraction: float = TOP_FRACTION,
) -> float:
    """Enrichment of true positives in the top-fraction of scored predictions.

    Enrichment = (observed positive rate in top fraction) / (overall positive
    rate).  A value of 1.0 indicates no enrichment (random).  2.0 means 2×
    enrichment.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels.
    y_score : np.ndarray
        Predicted scores (higher = more likely positive).
    fraction : float
        Top fraction to evaluate (default 0.10 = top 10%).

    Returns
    -------
    float
        Enrichment factor.  Returns 0.0 if overall positive rate is 0.
    """
    n_total = len(y_true)
    if n_total == 0:
        return 0.0
    overall_rate = float(y_true.mean())
    if overall_rate == 0.0:
        return 0.0
    k = max(1, int(np.round(fraction * n_total)))
    top_rate = precision_at_k(y_true, y_score, k)
    return top_rate / overall_rate


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    top_fraction: float = TOP_FRACTION,
) -> dict[str, float]:
    """Compute the full set of evaluation metrics for one fold.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels (0 or 1).
    y_score : np.ndarray
        Predicted scores (higher = more likely positive).
    top_fraction : float
        Fraction of top predictions for precision/recall/enrichment metrics.

    Returns
    -------
    dict[str, float]
        Keys: ``auprc``, ``auroc``, ``precision_at_top``, ``recall_at_top``,
        ``enrichment_at_top``.  Any metric that cannot be computed (e.g., only
        one class present) is returned as ``float("nan")``.
    """
    metrics: dict[str, float] = {}
    k = max(1, int(np.round(top_fraction * len(y_true))))

    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        logger.warning(
            "compute_metrics: only one class present (n_pos=%d, n_total=%d); "
            "AUPRC and AUROC set to NaN",
            n_pos, len(y_true),
        )
        metrics["auprc"] = float("nan")
        metrics["auroc"] = float("nan")
    else:
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))

    metrics["precision_at_top"] = precision_at_k(y_true, y_score, k)
    metrics["recall_at_top"] = recall_at_k(y_true, y_score, k)
    metrics["enrichment_at_top"] = enrichment_at_top_fraction(y_true, y_score, top_fraction)
    return metrics


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FoldMetrics:
    """Metrics from a single cross-validation fold."""

    fold: int
    n_train: int
    n_test: int
    n_pos_test: int
    auprc: float
    auroc: float
    precision_at_top: float
    recall_at_top: float
    enrichment_at_top: float
    test_guide_ids: list[str] = field(default_factory=list)


@dataclass
class CVResult:
    """Results from leave-one-guide-out cross-validation for one model type.

    Attributes
    ----------
    model_type : str
        One of ``"A"``, ``"B"``, or ``"C"`` (or an ablation label).
    feature_cols : list[str]
        Feature columns used for this model.
    folds : list[FoldMetrics]
        Per-fold evaluation metrics.
    oof_y_true : np.ndarray
        Concatenated out-of-fold ground truth labels.
    oof_y_score : np.ndarray
        Concatenated out-of-fold predicted scores.
    final_model : object
        XGBClassifier trained on all data (for SHAP / cell-type transfer).
    mean_auprc : float
        Mean AUPRC across valid (non-NaN) folds.
    std_auprc : float
        Standard deviation of AUPRC across valid folds.
    mean_auroc : float
        Mean AUROC across valid folds.
    std_auroc : float
        Standard deviation of AUROC across valid folds.
    """

    model_type: str
    feature_cols: list[str]
    folds: list[FoldMetrics]
    oof_y_true: np.ndarray
    oof_y_score: np.ndarray
    final_model: object  # xgb.XGBClassifier
    mean_auprc: float
    std_auprc: float
    mean_auroc: float
    std_auroc: float


@dataclass
class ModelComparison:
    """Statistical comparison between two CVResult objects.

    Attributes
    ----------
    model_a_type : str
        Label for the reference model.
    model_b_type : str
        Label for the model being compared.
    delta_auprc : float
        ``mean_auprc(b) − mean_auprc(a)``; positive means model B is better.
    p_value : float
        Two-sided paired t-test p-value (backward-compat alias).
    significant : bool
        True if any two-sided p-value < 0.05.
    n_paired : int
        Number of valid (non-NaN) paired folds used in the tests.
    ttest_p_one_sided : float
        One-sided paired t-test p-value (H1: model_b > model_a).
    wilcoxon_p_two_sided : float
        Two-sided Wilcoxon signed-rank p-value.
    wilcoxon_p_one_sided : float
        One-sided Wilcoxon signed-rank p-value (H1: model_b > model_a).
    """

    model_a_type: str
    model_b_type: str
    delta_auprc: float
    p_value: float
    significant: bool
    n_paired: int = 0
    ttest_p_one_sided: float = float("nan")
    wilcoxon_p_two_sided: float = float("nan")
    wilcoxon_p_one_sided: float = float("nan")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_model_data(
    feature_matrix: pd.DataFrame,
    model_type: str,
    *,
    cell_types: Optional[list[str]] = None,
    exclude_datasets: Optional[list[str]] = None,
    include_missingness_indicators: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Filter the feature matrix and return arrays ready for model training.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Full feature matrix from ``build_feature_matrix``.  Must contain
        ``guide_id``, ``label``, and all relevant feature columns.
    model_type : str
        One of ``"A"``, ``"B"``, ``"C"`` (case-insensitive).
    cell_types : list[str] | None
        If provided, keep only rows where ``cell_type`` is in this list.
        ``None`` keeps all rows.
    exclude_datasets : list[str] | None
        If provided, drop rows where ``dataset`` is in this list.  Useful for
        sensitivity analysis (e.g. ``exclude_datasets=["deepcrispr_k562"]``).
    include_missingness_indicators : bool
        If ``True``, append ``{feature}_missing`` indicator columns to the
        feature set for chromatin features in Models B and C.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix (float32).
    y : np.ndarray of shape (n_samples,)
        Binary labels (int32).
    groups : np.ndarray of shape (n_samples,)
        Guide IDs for ``StratifiedGroupKFold`` splitting.
    feature_cols : list[str]
        Feature column names in the same order as X columns.

    Raises
    ------
    ValueError
        If on-target sites (``num_mismatches == 0``) are detected, if
        required columns are missing, or if no rows remain after filtering.
    """
    mt = model_type.strip().upper()
    if mt not in {"A", "B", "C"}:
        raise ValueError(f"model_type must be 'A', 'B', or 'C', got {model_type!r}")

    # Guard: on-targets must have been removed upstream.
    if "num_mismatches" in feature_matrix.columns:
        n_ontarget = int((feature_matrix["num_mismatches"] == 0).sum())
        if n_ontarget > 0:
            raise ValueError(
                f"prepare_model_data: {n_ontarget} on-target site(s) "
                "(num_mismatches == 0) detected. Remove them with "
                "build_feature_matrix before calling this function."
            )

    for col in ("guide_id", "label"):
        if col not in feature_matrix.columns:
            raise ValueError(
                f"Feature matrix is missing required column {col!r}."
            )

    df = feature_matrix.copy()
    n_start = len(df)

    # --- Exclude specified datasets ---
    if exclude_datasets:
        if "dataset" in df.columns:
            mask = df["dataset"].isin(exclude_datasets)
            n_excluded = int(mask.sum())
            log_filter(
                logger,
                f"exclude_datasets={exclude_datasets}",
                n_start,
                n_start - n_excluded,
            )
            df = df[~mask]
        else:
            logger.warning(
                "exclude_datasets specified but 'dataset' column not present; ignoring"
            )

    # --- Filter by cell type ---
    if cell_types is not None:
        if "cell_type" in df.columns:
            n_before = len(df)
            mask = df["cell_type"].isin(cell_types)
            log_filter(
                logger,
                f"cell_types={cell_types}",
                n_before,
                int(mask.sum()),
            )
            df = df[mask]
        else:
            logger.warning(
                "cell_types specified but 'cell_type' column not present; ignoring"
            )

    if len(df) == 0:
        raise ValueError(
            f"prepare_model_data: no rows remain after filtering for model {mt}. "
            "Check cell_types and exclude_datasets arguments."
        )

    # --- Select feature columns ---
    feature_cols = get_model_features(
        mt, include_missingness_indicators=include_missingness_indicators
    )

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.warning(
            "Model %s: %d feature column(s) not found in matrix and will be "
            "excluded: %s%s",
            mt,
            len(missing_cols),
            missing_cols[:5],
            " ..." if len(missing_cols) > 5 else "",
        )
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        raise ValueError(
            f"prepare_model_data: no feature columns found for model {mt}. "
            "Ensure the feature matrix contains the appropriate feature columns."
        )

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    groups = df["guide_id"].to_numpy()

    n_pos = int(y.sum())
    logger.info(
        "prepare_model_data (Model %s): %d rows, %d features, "
        "%d positives (%.1f%%), %d guides",
        mt, len(df), len(feature_cols),
        n_pos, 100.0 * n_pos / len(df),
        len(np.unique(groups)),
    )

    return X, y, groups, feature_cols


# ---------------------------------------------------------------------------
# XGBoost helpers
# ---------------------------------------------------------------------------

def make_xgb_classifier(
    scale_pos_weight: float,
    *,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    extra_params: Optional[dict] = None,
) -> xgb.XGBClassifier:
    """Construct an ``XGBClassifier`` with SafeGenome defaults.

    Parameters
    ----------
    scale_pos_weight : float
        Class imbalance weight (``n_neg / n_pos``).
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Step-size shrinkage.
    subsample : float
        Subsample ratio of training instances per tree.
    colsample_bytree : float
        Subsample ratio of features per tree.
    random_state : int
        Random seed (``random_state=42`` everywhere for reproducibility).
    extra_params : dict | None
        Additional keyword arguments passed to ``XGBClassifier``, overriding
        any of the above defaults.

    Returns
    -------
    xgb.XGBClassifier
    """
    params: dict = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "verbosity": 0,
        "random_state": random_state,
    }
    if extra_params:
        params.update(extra_params)
    return xgb.XGBClassifier(**params)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    model_type: str,
    feature_cols: list[str],
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = 42,
    xgb_extra_params: Optional[dict] = None,
    top_fraction: float = TOP_FRACTION,
) -> CVResult:
    """Run leave-one-guide-out cross-validation for one model type.

    Uses ``StratifiedGroupKFold`` to ensure:

    1. Sites from the same guide are never split across train/test.
    2. Label distribution is approximately preserved across folds.

    A final model is also trained on all data for SHAP analysis and cell-type
    transfer predictions.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y : np.ndarray
        Binary labels of shape ``(n_samples,)``.
    groups : np.ndarray
        Guide IDs (same length as ``y``) for group-based splitting.
    model_type : str
        Label for this model (``"A"``, ``"B"``, or ``"C"``).
    feature_cols : list[str]
        Names of columns in ``X`` (same order).
    n_splits : int
        Number of CV folds.  Automatically reduced if fewer unique guides
        are present.
    random_state : int
        Random seed.
    xgb_extra_params : dict | None
        Additional parameters forwarded to ``make_xgb_classifier``.
    top_fraction : float
        Fraction of top-scored sites for precision/recall/enrichment.

    Returns
    -------
    CVResult
    """
    n_unique_groups = len(np.unique(groups))
    if n_splits > n_unique_groups:
        logger.warning(
            "cross_validate_model (Model %s): n_splits=%d > n_unique_guides=%d; "
            "reducing n_splits to %d",
            model_type, n_splits, n_unique_groups, n_unique_groups,
        )
        n_splits = n_unique_groups

    scale_pos_weight = compute_scale_pos_weight(y)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    logger.info(
        "cross_validate_model (Model %s): %d samples, %d features, "
        "%d positives, %d negatives, scale_pos_weight=%.2f, n_splits=%d",
        model_type, len(y), len(feature_cols),
        n_pos, n_neg, scale_pos_weight, n_splits,
    )

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics_list: list[FoldMetrics] = []
    oof_y_true_parts: list[np.ndarray] = []
    oof_y_score_parts: list[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_guide_ids_in_fold = sorted(set(str(g) for g in groups[test_idx]))

        fold_spw = compute_scale_pos_weight(y_train)
        clf = make_xgb_classifier(
            fold_spw,
            random_state=random_state,
            extra_params=xgb_extra_params,
        )
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:, 1]

        fold_m = compute_metrics(y_test, y_score, top_fraction=top_fraction)
        n_pos_test = int(y_test.sum())

        fold_metrics_list.append(FoldMetrics(
            fold=fold_idx,
            n_train=len(y_train),
            n_test=len(y_test),
            n_pos_test=n_pos_test,
            auprc=fold_m["auprc"],
            auroc=fold_m["auroc"],
            precision_at_top=fold_m["precision_at_top"],
            recall_at_top=fold_m["recall_at_top"],
            enrichment_at_top=fold_m["enrichment_at_top"],
            test_guide_ids=test_guide_ids_in_fold,
        ))
        oof_y_true_parts.append(y_test)
        oof_y_score_parts.append(y_score)

        logger.info(
            "  Fold %d [%s]: n_test=%d (%d pos), AUPRC=%.4f, AUROC=%.4f",
            fold_idx, ",".join(test_guide_ids_in_fold),
            len(y_test), n_pos_test,
            fold_m["auprc"], fold_m["auroc"],
        )

    # Aggregate across non-NaN folds.
    auprcs = [fm.auprc for fm in fold_metrics_list if not np.isnan(fm.auprc)]
    aurocs = [fm.auroc for fm in fold_metrics_list if not np.isnan(fm.auroc)]
    mean_auprc = float(np.mean(auprcs)) if auprcs else float("nan")
    std_auprc = float(np.std(auprcs)) if len(auprcs) > 1 else 0.0
    mean_auroc = float(np.mean(aurocs)) if aurocs else float("nan")
    std_auroc = float(np.std(aurocs)) if len(aurocs) > 1 else 0.0

    logger.info(
        "Model %s CV complete: mean_AUPRC=%.4f±%.4f, mean_AUROC=%.4f±%.4f",
        model_type, mean_auprc, std_auprc, mean_auroc, std_auroc,
    )

    # Train final model on all data for downstream use.
    final_clf = make_xgb_classifier(
        scale_pos_weight, random_state=random_state, extra_params=xgb_extra_params
    )
    final_clf.fit(X, y)

    return CVResult(
        model_type=model_type,
        feature_cols=list(feature_cols),
        folds=fold_metrics_list,
        oof_y_true=np.concatenate(oof_y_true_parts),
        oof_y_score=np.concatenate(oof_y_score_parts),
        final_model=final_clf,
        mean_auprc=mean_auprc,
        std_auprc=std_auprc,
        mean_auroc=mean_auroc,
        std_auroc=std_auroc,
    )


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def compare_models(result_a: CVResult, result_b: CVResult) -> ModelComparison:
    """Statistical comparison of per-fold AUPRC between two CV results.

    When both results have ``test_guide_ids`` populated (LOGO CV), folds are
    matched by guide ID so that models trained on different subsets can be
    compared on their overlapping guides.  Otherwise falls back to fold-index
    pairing.

    Computes paired t-test and Wilcoxon signed-rank test, each two-sided and
    one-sided (H1: result_b > result_a).

    Parameters
    ----------
    result_a : CVResult
        Reference model (typically Model A or Model B).
    result_b : CVResult
        Model to compare (typically Model C).

    Returns
    -------
    ModelComparison
    """
    # --- Build guide → auprc mapping when available ---
    a_has_guides = all(fm.test_guide_ids for fm in result_a.folds)
    b_has_guides = all(fm.test_guide_ids for fm in result_b.folds)

    if a_has_guides and b_has_guides:
        a_map: dict[str, float] = {}
        for fm in result_a.folds:
            for gid in fm.test_guide_ids:
                a_map[gid] = fm.auprc
        b_map: dict[str, float] = {}
        for fm in result_b.folds:
            for gid in fm.test_guide_ids:
                b_map[gid] = fm.auprc
        shared = sorted(set(a_map) & set(b_map))
        auprcs_a = np.array([a_map[g] for g in shared])
        auprcs_b = np.array([b_map[g] for g in shared])
    else:
        auprcs_a = np.array([fm.auprc for fm in result_a.folds])
        auprcs_b = np.array([fm.auprc for fm in result_b.folds])

    # Drop NaN pairs
    valid = ~(np.isnan(auprcs_a) | np.isnan(auprcs_b))
    a_valid = auprcs_a[valid]
    b_valid = auprcs_b[valid]
    n_paired = len(a_valid)

    delta_auprc = (
        float(np.mean(b_valid) - np.mean(a_valid)) if n_paired > 0 else float("nan")
    )

    # --- Paired t-test ---
    if n_paired < 2:
        logger.warning(
            "compare_models: fewer than 2 valid fold pairs (%d); tests unreliable",
            n_paired,
        )
        ttest_two = ttest_one = float("nan")
    else:
        _, ttest_two = stats.ttest_rel(a_valid, b_valid)
        _, ttest_one = stats.ttest_rel(a_valid, b_valid, alternative="less")
        ttest_two = float(ttest_two)
        ttest_one = float(ttest_one)

    # --- Wilcoxon signed-rank test ---
    diffs = b_valid - a_valid
    n_nonzero = int(np.count_nonzero(diffs))
    if n_nonzero < 6:
        logger.info(
            "compare_models (%s vs %s): only %d non-zero differences; "
            "Wilcoxon unreliable — reporting NaN",
            result_a.model_type, result_b.model_type, n_nonzero,
        )
        wilcoxon_two = wilcoxon_one = float("nan")
    else:
        _, wilcoxon_two = stats.wilcoxon(diffs)
        _, wilcoxon_one = stats.wilcoxon(diffs, alternative="greater")
        wilcoxon_two = float(wilcoxon_two)
        wilcoxon_one = float(wilcoxon_one)

    significant = any(
        (not np.isnan(p)) and p < 0.05
        for p in [ttest_two, wilcoxon_two]
    )

    logger.info(
        "compare_models (%s vs %s): delta=%.4f, n_paired=%d | "
        "ttest 2s=%.4f 1s=%.4f | wilcoxon 2s=%.4f 1s=%.4f | sig=%s",
        result_a.model_type, result_b.model_type,
        delta_auprc, n_paired,
        ttest_two, ttest_one,
        wilcoxon_two, wilcoxon_one,
        significant,
    )

    return ModelComparison(
        model_a_type=result_a.model_type,
        model_b_type=result_b.model_type,
        delta_auprc=delta_auprc,
        p_value=ttest_two,
        significant=significant,
        n_paired=n_paired,
        ttest_p_one_sided=ttest_one,
        wilcoxon_p_two_sided=wilcoxon_two,
        wilcoxon_p_one_sided=wilcoxon_one,
    )


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    feature_cols: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values for the trained model.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Trained XGBoost classifier.
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    feature_cols : list[str]
        Feature column names (same order as X columns).

    Returns
    -------
    shap_values : np.ndarray of shape (n_samples, n_features)
        SHAP values for each sample and feature (positive-class).
    mean_abs_shap : pd.DataFrame
        DataFrame with columns ``feature`` and ``mean_abs_shap``, sorted
        descending by mean absolute SHAP value.

    Raises
    ------
    ImportError
        If the ``shap`` package is not installed.
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP is required for compute_shap_values. "
            "Install with: pip install shap>=0.42"
        )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle both legacy (list of arrays) and modern (single array) shap APIs.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive-class SHAP values

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return shap_values, importance_df


# ---------------------------------------------------------------------------
# PR curve plotting
# ---------------------------------------------------------------------------

def plot_pr_curves(
    cv_results: dict[str, CVResult],
    *,
    output_path: Optional[Path] = None,
    title: str = "Precision-Recall Curves — SafeGenome Three-Model Comparison",
    figsize: tuple[float, float] = (7.0, 6.0),
):
    """Plot pooled out-of-fold PR curves for Models A, B, and C.

    Uses concatenated OOF predictions across all folds.  Also plots a dashed
    random-classifier baseline at the dataset's positive rate.

    Parameters
    ----------
    cv_results : dict[str, CVResult]
        Mapping from model label (``"A"``, ``"B"``, ``"C"``) to CVResult.
    output_path : Path | None
        If provided, save the figure here (PNG or SVG based on suffix).
    title : str
        Figure title.
    figsize : tuple[float, float]
        Figure size in inches (width, height).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_pr_curves. "
            "Install with: pip install matplotlib>=3.7"
        )

    fig, ax = plt.subplots(figsize=figsize)

    _COLORS = {"A": "#1f77b4", "B": "#2ca02c", "C": "#ff7f0e"}
    _LINESTYLES = {"A": "--", "B": "-", "C": "-"}
    _LABELS = {
        "A": "Model A — Sequence only",
        "B": "Model B — Sequence + Experimental chromatin",
        "C": "Model C — Sequence + AlphaGenome (ours)",
    }

    for model_label, result in cv_results.items():
        y_true = result.oof_y_true
        y_score = result.oof_y_score
        if len(y_true) == 0 or int(y_true.sum()) == 0:
            logger.warning(
                "plot_pr_curves: Model %s has no positives in OOF — skipping", model_label
            )
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auprc = float(average_precision_score(y_true, y_score))
        color = _COLORS.get(model_label, "gray")
        ls = _LINESTYLES.get(model_label, "-")
        label = _LABELS.get(model_label, f"Model {model_label}")
        ax.plot(
            recall, precision,
            color=color, linestyle=ls, linewidth=2,
            label=f"{label} (AUPRC={auprc:.3f})",
        )

    # Random baseline.
    if cv_results:
        any_result = next(iter(cv_results.values()))
        baseline = float(any_result.oof_y_true.mean())
        ax.axhline(
            baseline, color="gray", linestyle=":", linewidth=1.0,
            label=f"Random (P={baseline:.3f})",
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("PR curve figure saved → %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Ablation analysis
# ---------------------------------------------------------------------------

def run_ablation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
    *,
    feature_subsets: dict[str, list[str]],
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = 42,
    xgb_extra_params: Optional[dict] = None,
) -> dict[str, dict[str, float]]:
    """Run ablation analysis across a set of feature subsets.

    Trains a separate leave-one-guide-out CV for each feature subset and
    returns mean AUPRC and AUROC per subset.  Useful for understanding which
    AlphaGenome chromatin feature types drive performance (DNase vs histone
    marks vs CTCF etc.).

    Parameters
    ----------
    X : np.ndarray
        Full feature matrix ``(n_samples, n_features)``.
    y : np.ndarray
        Binary labels.
    groups : np.ndarray
        Guide IDs for GroupKFold.
    feature_cols : list[str]
        Full list of feature column names (same order as X columns).
    feature_subsets : dict[str, list[str]]
        Mapping from subset name to the feature names to include.  Names not
        present in *feature_cols* are silently ignored.
    n_splits : int
        Number of CV folds.
    random_state : int
        Random seed.
    xgb_extra_params : dict | None
        Extra XGBoost parameters.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping from subset name to
        ``{"mean_auprc", "std_auprc", "mean_auroc", "std_auroc"}``.
    """
    col_to_idx = {col: i for i, col in enumerate(feature_cols)}
    results: dict[str, dict[str, float]] = {}

    for subset_name, subset_cols in feature_subsets.items():
        valid_cols = [c for c in subset_cols if c in col_to_idx]
        if not valid_cols:
            logger.warning(
                "run_ablation: subset %r has no valid columns — skipping", subset_name
            )
            continue

        col_indices = [col_to_idx[c] for c in valid_cols]
        X_sub = X[:, col_indices]

        cv_result = cross_validate_model(
            X_sub, y, groups,
            model_type=f"ablation_{subset_name}",
            feature_cols=valid_cols,
            n_splits=n_splits,
            random_state=random_state,
            xgb_extra_params=xgb_extra_params,
        )

        results[subset_name] = {
            "mean_auprc": cv_result.mean_auprc,
            "std_auprc": cv_result.std_auprc,
            "mean_auroc": cv_result.mean_auroc,
            "std_auroc": cv_result.std_auroc,
        }
        logger.info(
            "Ablation [%s]: mean_AUPRC=%.4f±%.4f",
            subset_name, cv_result.mean_auprc, cv_result.std_auprc,
        )

    return results


# ---------------------------------------------------------------------------
# Three-model pipeline entry point
# ---------------------------------------------------------------------------

def run_three_model_comparison(
    feature_matrix: pd.DataFrame,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    exclude_datasets: Optional[list[str]] = None,
    model_bc_cell_types: Optional[list[str]] = None,
    include_missingness_indicators: bool = False,
    xgb_extra_params: Optional[dict] = None,
    random_state: int = 42,
    output_dir: Optional[Path] = None,
    run_shap: bool = False,
    run_ablation_flag: bool = False,
    top_fraction: float = TOP_FRACTION,
) -> dict[str, CVResult]:
    """Run the full three-model comparison pipeline.

    Trains and cross-validates Models A, B, and C, prints a statistical
    comparison, and optionally saves figures and ablation results.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Full assembled feature matrix (from ``build_feature_matrix``).
        On-target sites (``num_mismatches == 0``) must already be removed.
    n_splits : int
        Number of leave-one-guide-out CV folds.
    exclude_datasets : list[str] | None
        Dataset names to exclude for sensitivity analysis
        (e.g. ``["deepcrispr_k562"]``).
    model_bc_cell_types : list[str] | None
        Cell types included in Models B and C.  Defaults to
        ``["K562", "HEK293"]``.
    include_missingness_indicators : bool
        Whether to include ``{feature}_missing`` indicator columns in
        Models B and C.
    xgb_extra_params : dict | None
        Extra XGBoost hyperparameters to override defaults.
    random_state : int
        Global random seed.
    output_dir : Path | None
        Directory for saving PR curve figure, SHAP CSVs, and ablation CSV.
    run_shap : bool
        If ``True``, compute SHAP feature importances for all three models
        and save to *output_dir* (requires the ``shap`` package).
    run_ablation_flag : bool
        If ``True``, run feature-subset ablation analysis for Model C and
        save results to *output_dir*.
    top_fraction : float
        Fraction of top predictions for precision/recall/enrichment metrics.

    Returns
    -------
    dict[str, CVResult]
        Keys ``"A"``, ``"B"``, ``"C"`` → their :class:`CVResult` objects.

    Notes
    -----
    **Model A** is trained on ALL rows not excluded by *exclude_datasets*
    (no cell-type restriction).  This maximises baseline strength.

    **Models B and C** are restricted to *model_bc_cell_types* (default
    K562 + HEK293) where both experimental and AlphaGenome chromatin features
    are available.
    """
    if model_bc_cell_types is None:
        model_bc_cell_types = list(MODEL_BC_CELL_TYPES)

    logger.info("=== SafeGenome Three-Model Comparison ===")
    logger.info(
        "  n_splits=%d, exclude_datasets=%s, model_bc_cell_types=%s",
        n_splits, exclude_datasets, model_bc_cell_types,
    )

    results: dict[str, CVResult] = {}

    # --- Model A: all data, sequence features only ---
    logger.info("--- Model A: sequence-only baseline (all cell types) ---")
    X_a, y_a, groups_a, feat_a = prepare_model_data(
        feature_matrix, "A",
        cell_types=None,
        exclude_datasets=exclude_datasets,
        include_missingness_indicators=False,
    )
    results["A"] = cross_validate_model(
        X_a, y_a, groups_a,
        model_type="A",
        feature_cols=feat_a,
        n_splits=n_splits,
        random_state=random_state,
        xgb_extra_params=xgb_extra_params,
        top_fraction=top_fraction,
    )

    # --- Model B: K562 + HEK293, sequence + experimental chromatin ---
    logger.info("--- Model B: sequence + experimental chromatin ---")
    X_b, y_b, groups_b, feat_b = prepare_model_data(
        feature_matrix, "B",
        cell_types=model_bc_cell_types,
        exclude_datasets=exclude_datasets,
        include_missingness_indicators=include_missingness_indicators,
    )
    results["B"] = cross_validate_model(
        X_b, y_b, groups_b,
        model_type="B",
        feature_cols=feat_b,
        n_splits=n_splits,
        random_state=random_state,
        xgb_extra_params=xgb_extra_params,
        top_fraction=top_fraction,
    )

    # --- Model C: K562 + HEK293, sequence + AlphaGenome chromatin ---
    logger.info("--- Model C: sequence + AlphaGenome predicted chromatin ---")
    X_c, y_c, groups_c, feat_c = prepare_model_data(
        feature_matrix, "C",
        cell_types=model_bc_cell_types,
        exclude_datasets=exclude_datasets,
        include_missingness_indicators=include_missingness_indicators,
    )
    results["C"] = cross_validate_model(
        X_c, y_c, groups_c,
        model_type="C",
        feature_cols=feat_c,
        n_splits=n_splits,
        random_state=random_state,
        xgb_extra_params=xgb_extra_params,
        top_fraction=top_fraction,
    )

    # --- Statistical comparisons ---
    comp_ac = compare_models(results["A"], results["C"])
    comp_bc = compare_models(results["B"], results["C"])
    comp_ab = compare_models(results["A"], results["B"])

    # --- PR curve figure ---
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            plot_pr_curves(results, output_path=output_dir / "fig2_pr_curves.png")
        except Exception as exc:
            logger.warning("PR curve plotting failed: %s", exc)

    # --- SHAP analysis ---
    if run_shap:
        _data_for_shap: dict[str, tuple[np.ndarray, list[str]]] = {
            "A": (X_a, feat_a),
            "B": (X_b, feat_b),
            "C": (X_c, feat_c),
        }
        for model_label, (X_shap, feat_shap) in _data_for_shap.items():
            try:
                _, importance_df = compute_shap_values(
                    results[model_label].final_model, X_shap, feat_shap
                )
                if output_dir is not None:
                    shap_path = output_dir / f"shap_importance_model_{model_label}.csv"
                    importance_df.to_csv(shap_path, index=False)
                    logger.info("SHAP importance saved → %s", shap_path)
            except Exception as exc:
                logger.warning("SHAP analysis for Model %s failed: %s", model_label, exc)

    # --- Ablation for Model C ---
    if run_ablation_flag:
        from src.feature_matrix import AG_FEATURE_NAMES, SEQUENCE_FEATURE_NAMES as SEQ_FEATS

        ag_subsets: dict[str, list[str]] = {
            "seq_only": list(SEQ_FEATS),
            "seq_dnase": list(SEQ_FEATS) + [f for f in AG_FEATURE_NAMES if "dnase" in f],
            "seq_atac": list(SEQ_FEATS) + [f for f in AG_FEATURE_NAMES if "atac" in f],
            "seq_histone": list(SEQ_FEATS) + [
                f for f in AG_FEATURE_NAMES
                if any(m in f for m in ("h3k4me3", "h3k27me3", "h3k9me3", "h3k27ac"))
            ],
            "seq_ctcf": list(SEQ_FEATS) + [f for f in AG_FEATURE_NAMES if "ctcf" in f],
            "seq_full_ag": list(feat_c),
        }

        ablation_results = run_ablation(
            X_c, y_c, groups_c, feat_c,
            feature_subsets=ag_subsets,
            n_splits=n_splits,
            random_state=random_state,
            xgb_extra_params=xgb_extra_params,
        )

        if output_dir is not None:
            ablation_df = pd.DataFrame(
                [{"subset": k, **v} for k, v in ablation_results.items()]
            )
            ablation_path = output_dir / "ablation_results.csv"
            ablation_df.to_csv(ablation_path, index=False)
            logger.info("Ablation results saved → %s", ablation_path)

    return results
