"""
scripts/08_benchmarks.py — External tool benchmarks and held-out validation.

Two analyses:
1. CFD/MIT score baselines: compute AUPRC using raw CFD and MIT scores as
   predictors (no model), add to PR curve alongside Models A, B, C.
2. Held-out validation: train Models A, B, C on GUIDE-seq HEK293 only,
   test on DISCOVER-seq K562 as an independent held-out set.

Requires
--------
* data/features/feature_matrix_combined.parquet  (from script 04)
* data/models/cv_results.pkl                     (from script 05)

Outputs
-------
    figures/fig2_pr_curves.png       Updated PR curve with CFD/MIT baselines
    data/results/baseline_scores.csv CFD + MIT AUPRC results
    data/results/held_out_validation.csv  Held-out DISCOVER-seq results
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_matrix import get_model_features
from src.model import (
    compute_metrics,
    compute_scale_pos_weight,
    make_xgb_classifier,
    prepare_model_data,
)
from src.utils import get_logger

logger = get_logger("benchmarks")

_ROOT = Path(__file__).resolve().parents[1]
_FEATURES = _ROOT / "data" / "features"
_MODELS = _ROOT / "data" / "models"
_RESULTS = _ROOT / "data" / "results"
_FIGURES = _ROOT / "figures"


# ---------------------------------------------------------------------------
# Task 2: CFD / MIT score baselines
# ---------------------------------------------------------------------------

def compute_baseline_auprc(
    feature_matrix: pd.DataFrame,
    *,
    cell_types: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute AUPRC for raw CFD and MIT scores as off-target predictors.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Must contain ``cfd_score``, ``mit_score``, ``label``, ``cell_type``.
    cell_types : list[str] | None
        If provided, filter to these cell types (to match Model B/C data).

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping from score name to {auprc, n_sites, n_positives}.
    """
    df = feature_matrix.copy()
    if cell_types:
        df = df[df["cell_type"].isin(cell_types)]

    y = df["label"].to_numpy()
    n_pos = int(y.sum())
    logger.info(
        "Baseline scoring: %d sites, %d positives (%.1f%%)",
        len(df), n_pos, 100.0 * n_pos / len(df),
    )

    results: dict[str, dict[str, float]] = {}
    for score_name in ("cfd_score", "mit_score"):
        scores = df[score_name].to_numpy(dtype=np.float64)
        # Handle any NaN by filling with 0 (conservative)
        nan_mask = np.isnan(scores)
        if nan_mask.any():
            logger.warning(
                "%s: %d NaN values filled with 0.0", score_name, nan_mask.sum()
            )
            scores = np.where(nan_mask, 0.0, scores)

        auprc = float(average_precision_score(y, scores))
        metrics = compute_metrics(y, scores)
        results[score_name] = {
            "auprc": auprc,
            "auroc": metrics["auroc"],
            "precision_at_top": metrics["precision_at_top"],
            "recall_at_top": metrics["recall_at_top"],
            "n_sites": len(df),
            "n_positives": n_pos,
        }
        logger.info("  %s: AUPRC=%.4f, AUROC=%.4f", score_name, auprc, metrics["auroc"])

    return results


def plot_pr_curves_with_baselines(
    cv_results: dict,
    feature_matrix: pd.DataFrame,
    *,
    cell_types: list[str] | None = None,
    output_path: Path | None = None,
    title: str = "SafeGenome: Precision-Recall Curves (LOGO CV)",
    figsize: tuple[float, float] = (7.5, 6.5),
):
    """Plot PR curves for Models A, B, C + CFD/MIT baselines.

    Parameters
    ----------
    cv_results : dict[str, CVResult]
        From script 05.
    feature_matrix : pd.DataFrame
        Feature matrix with cfd_score, mit_score, label.
    cell_types : list[str] | None
        Filter feature matrix to match Model B/C data for baselines.
    output_path : Path | None
        Where to save the figure.
    title : str
        Figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # --- Models A, B, C (pooled OOF) ---
    _COLORS = {"A": "#1f77b4", "B": "#2ca02c", "C": "#ff7f0e"}
    _LINESTYLES = {"A": "-", "B": "-", "C": "-"}
    _LINEWIDTHS = {"A": 1.8, "B": 1.8, "C": 2.5}
    _LABELS = {
        "A": "Model A \u2014 Sequence only",
        "B": "Model B \u2014 Seq + Experimental",
        "C": "Model C \u2014 Seq + AlphaGenome",
    }

    for model_label, result in cv_results.items():
        y_true = result.oof_y_true
        y_score = result.oof_y_score
        if len(y_true) == 0 or int(y_true.sum()) == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auprc = float(average_precision_score(y_true, y_score))
        ax.plot(
            recall, precision,
            color=_COLORS.get(model_label, "gray"),
            linestyle=_LINESTYLES.get(model_label, "-"),
            linewidth=_LINEWIDTHS.get(model_label, 1.8),
            label=f"{_LABELS.get(model_label, model_label)} (AUPRC={auprc:.3f})",
        )

    # --- CFD and MIT baselines ---
    df = feature_matrix.copy()
    if cell_types:
        df = df[df["cell_type"].isin(cell_types)]
    y_all = df["label"].to_numpy()

    _BASELINE_COLORS = {"cfd_score": "#d62728", "mit_score": "#9467bd"}
    _BASELINE_LABELS = {"cfd_score": "CFD score (raw)", "mit_score": "MIT score (raw)"}

    for score_name in ("cfd_score", "mit_score"):
        scores = df[score_name].to_numpy(dtype=np.float64)
        scores = np.where(np.isnan(scores), 0.0, scores)
        precision, recall, _ = precision_recall_curve(y_all, scores)
        auprc = float(average_precision_score(y_all, scores))
        ax.plot(
            recall, precision,
            color=_BASELINE_COLORS[score_name],
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"{_BASELINE_LABELS[score_name]} (AUPRC={auprc:.3f})",
        )

    # --- Random baseline ---
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
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("PR curve figure saved \u2192 %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Task 3: Held-out DISCOVER-seq validation
# ---------------------------------------------------------------------------

def run_held_out_validation(
    feature_matrix: pd.DataFrame,
    *,
    train_dataset: str = "guide_seq_tsai2015",
    test_dataset: str = "discover_seq_k562",
    include_missingness: bool = True,
    top_fraction: float = 0.10,
) -> pd.DataFrame:
    """Train on GUIDE-seq, test on DISCOVER-seq K562.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Combined feature matrix with ``dataset`` column.
    train_dataset : str
        Dataset name for training data.
    test_dataset : str
        Dataset name for test data.
    include_missingness : bool
        Include missingness indicator features for Models B/C.
    top_fraction : float
        Top fraction for precision/recall metrics.

    Returns
    -------
    pd.DataFrame
        Results table with columns: model, auprc, auroc, precision_at_top,
        recall_at_top, n_train, n_test, n_pos_train, n_pos_test.
    """
    train_df = feature_matrix[feature_matrix["dataset"] == train_dataset].copy()
    test_df = feature_matrix[feature_matrix["dataset"] == test_dataset].copy()

    logger.info(
        "Held-out split: train=%s (%d rows, %d pos), test=%s (%d rows, %d pos)",
        train_dataset, len(train_df), int(train_df["label"].sum()),
        test_dataset, len(test_df), int(test_df["label"].sum()),
    )

    # Report guide composition
    train_guides = sorted(train_df["guide_id"].unique())
    test_guides = sorted(test_df["guide_id"].unique())
    logger.info("  Train guides (%d): %s", len(train_guides), train_guides)
    logger.info("  Test guides (%d): %s", len(test_guides), test_guides)

    # Check for guide overlap (should be zero)
    overlap = set(train_guides) & set(test_guides)
    if overlap:
        logger.warning("Guide overlap between train and test: %s", overlap)
    else:
        logger.info("  No guide overlap between train and test (good)")

    results_rows: list[dict] = []

    for model_type in ("A", "B", "C"):
        feature_cols = get_model_features(
            model_type,
            include_missingness_indicators=include_missingness,
        )
        # Filter to columns that exist
        feature_cols = [c for c in feature_cols if c in feature_matrix.columns]

        if not feature_cols:
            logger.warning("Model %s: no feature columns found, skipping", model_type)
            continue

        X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_train = train_df["label"].to_numpy(dtype=np.int32)
        X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
        y_test = test_df["label"].to_numpy(dtype=np.int32)

        n_pos_train = int(y_train.sum())
        n_pos_test = int(y_test.sum())

        if n_pos_train == 0:
            logger.warning("Model %s: no positives in training data, skipping", model_type)
            continue
        if n_pos_test == 0:
            logger.warning("Model %s: no positives in test data, skipping", model_type)
            continue

        spw = compute_scale_pos_weight(y_train)
        clf = make_xgb_classifier(spw, random_state=42)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_score, top_fraction=top_fraction)
        results_rows.append({
            "model": f"Model {model_type}",
            "auprc": metrics["auprc"],
            "auroc": metrics["auroc"],
            "precision_at_top": metrics["precision_at_top"],
            "recall_at_top": metrics["recall_at_top"],
            "enrichment_at_top": metrics["enrichment_at_top"],
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_pos_train": n_pos_train,
            "n_pos_test": n_pos_test,
            "n_features": len(feature_cols),
            "train_guides": ",".join(train_guides),
            "test_guides": ",".join(test_guides),
        })
        logger.info(
            "  Model %s: AUPRC=%.4f, AUROC=%.4f, P@10%%=%.4f, R@10%%=%.4f",
            model_type, metrics["auprc"], metrics["auroc"],
            metrics["precision_at_top"], metrics["recall_at_top"],
        )

    # Also add CFD/MIT baselines on the held-out test set
    for score_name in ("cfd_score", "mit_score"):
        scores = test_df[score_name].to_numpy(dtype=np.float64)
        scores = np.where(np.isnan(scores), 0.0, scores)
        y_test = test_df["label"].to_numpy(dtype=np.int32)
        n_pos_test = int(y_test.sum())
        metrics = compute_metrics(y_test, scores)
        results_rows.append({
            "model": f"{score_name} (raw)",
            "auprc": metrics["auprc"],
            "auroc": metrics["auroc"],
            "precision_at_top": metrics["precision_at_top"],
            "recall_at_top": metrics["recall_at_top"],
            "enrichment_at_top": metrics["enrichment_at_top"],
            "n_train": 0,
            "n_test": len(y_test),
            "n_pos_train": 0,
            "n_pos_test": n_pos_test,
            "n_features": 1,
            "train_guides": "",
            "test_guides": ",".join(test_guides),
        })

    return pd.DataFrame(results_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _RESULTS.mkdir(parents=True, exist_ok=True)
    _FIGURES.mkdir(parents=True, exist_ok=True)

    # ---- Load data -----------------------------------------------------------
    fm_path = _FEATURES / "feature_matrix_combined.parquet"
    cv_path = _MODELS / "cv_results.pkl"

    logger.info("Loading feature matrix: %s", fm_path)
    feature_matrix = pd.read_parquet(fm_path)
    logger.info("Feature matrix: %d rows x %d cols", len(feature_matrix), len(feature_matrix.columns))

    logger.info("Loading CV results: %s", cv_path)
    with open(cv_path, "rb") as f:
        cv_results = pickle.load(f)
    logger.info("CV results loaded: models %s", list(cv_results.keys()))

    # ---- Task 2: CFD/MIT baselines -------------------------------------------
    print("\n" + "=" * 72)
    print("Task 2: External Tool Benchmarks (CFD / MIT Score Baselines)")
    print("=" * 72)

    cell_types = ["K562", "HEK293"]
    baseline_results = compute_baseline_auprc(feature_matrix, cell_types=cell_types)

    print(f"\n{'Score':<15} {'AUPRC':>10} {'AUROC':>10} {'P@10%':>10} {'R@10%':>10}")
    print("-" * 60)
    for name, res in baseline_results.items():
        print(
            f"{name:<15} {res['auprc']:>10.4f} {res['auroc']:>10.4f} "
            f"{res['precision_at_top']:>10.4f} {res['recall_at_top']:>10.4f}"
        )
    for mt, cvr in cv_results.items():
        print(f"Model {mt:<9} {cvr.mean_auprc:>10.4f}")
    print("-" * 60)
    print(f"  Evaluated on {baseline_results['cfd_score']['n_sites']} sites, "
          f"{baseline_results['cfd_score']['n_positives']} positives")
    print("  Note: CFD/MIT use ALL data (no train/test split needed)")
    print("  Models A/B/C use pooled out-of-fold predictions from LOGO CV")

    # Save baseline CSV
    baseline_df = pd.DataFrame([
        {"score": name, **res} for name, res in baseline_results.items()
    ])
    baseline_path = _RESULTS / "baseline_scores.csv"
    baseline_df.to_csv(baseline_path, index=False)
    logger.info("Baseline scores saved -> %s", baseline_path)

    # Plot updated PR curves
    fig = plot_pr_curves_with_baselines(
        cv_results,
        feature_matrix,
        cell_types=cell_types,
        output_path=_FIGURES / "fig2_pr_curves.png",
        title="SafeGenome: Precision-Recall Curves (LOGO CV)",
    )
    plt.close(fig)
    print(f"\n  Updated PR curve saved -> {_FIGURES / 'fig2_pr_curves.png'}")

    # ---- Task 3: Held-out DISCOVER-seq validation ----------------------------
    print("\n" + "=" * 72)
    print("Task 3: Held-Out Dataset Validation")
    print("     Train: GUIDE-seq HEK293 | Test: DISCOVER-seq K562")
    print("=" * 72)

    held_out_df = run_held_out_validation(
        feature_matrix,
        train_dataset="guide_seq_tsai2015",
        test_dataset="discover_seq_k562",
    )

    print(f"\n{'Model':<20} {'AUPRC':>8} {'AUROC':>8} {'P@10%':>8} {'R@10%':>8} "
          f"{'N_train':>8} {'N_test':>7} {'Pos_test':>9}")
    print("-" * 90)
    for _, row in held_out_df.iterrows():
        print(
            f"{row['model']:<20} {row['auprc']:>8.4f} {row['auroc']:>8.4f} "
            f"{row['precision_at_top']:>8.4f} {row['recall_at_top']:>8.4f} "
            f"{int(row['n_train']):>8} {int(row['n_test']):>7} {int(row['n_pos_test']):>9}"
        )
    print("-" * 90)
    print(f"  Train guides: {held_out_df.iloc[0]['train_guides']}")
    print(f"  Test guides:  {held_out_df.iloc[0]['test_guides']}")
    print("  Note: Small test set — interpret with caution")

    # Save
    held_out_path = _RESULTS / "held_out_validation.csv"
    held_out_df.to_csv(held_out_path, index=False)
    logger.info("Held-out validation saved -> %s", held_out_path)
    print(f"  Saved -> {held_out_path}")

    print("\n" + "=" * 72)
    print("Script 08 complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
