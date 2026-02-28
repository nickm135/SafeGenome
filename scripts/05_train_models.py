"""
scripts/05_train_models.py — Train and evaluate the three-model comparison.

Loads the combined feature matrix from script 04 and runs full
leave-one-GUIDE-out (LOGO) cross-validation for all three model types:

* Model A — sequence-only baseline (all cell types)
* Model B — sequence + ENCODE experimental chromatin (K562 + HEK293)
* Model C — sequence + AlphaGenome predicted chromatin (K562 + HEK293)

n_splits is set to the number of unique guides (LOGO CV): each fold holds
out exactly one guide.  Model A uses ~23 folds; Models B/C auto-reduce to
~14 folds (guides present in K562 + HEK293).

Saves the trained Model C for use in the cell-type transfer demo (script 06).

Requires
--------
* data/features/feature_matrix_combined.parquet  (from script 04)

Outputs
-------
    data/models/model_c.pkl              Trained Model C (XGBClassifier + feature_cols)
    data/models/cv_results.pkl           Full CVResult objects for A, B, C
    data/models/metrics_summary.csv      Per-fold and mean metrics for all models
    data/models/ablation_results.csv     Feature ablation (seq+DNase/ATAC, +histone, +CTCF, full)
    data/results/per_guide_auprc.csv     Per-guide AUPRC for Models A, B, C
    data/results/statistical_tests.csv   Paired t-test + Wilcoxon for A-C, B-C, A-B
    figures/fig2_pr_curves.png           PR curve comparison figure

Critical rules enforced
-----------------------
* Leave-one-GUIDE-out CV (StratifiedGroupKFold by guide_id)
* Primary metric = AUPRC, not ROC-AUC
* scale_pos_weight = n_neg/n_pos for class imbalance
* random_state=42 everywhere for reproducibility
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- path setup so src imports work when run as a script ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import (
    CVResult,
    ModelComparison,
    compare_models,
    prepare_model_data,
    run_ablation,
    run_three_model_comparison,
)
from src.utils import get_logger

logger = get_logger("train_models")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]
_FEATURES = _ROOT / "data" / "features"
_MODELS = _ROOT / "data" / "models"
_RESULTS = _ROOT / "data" / "results"
_FIGURES = _ROOT / "figures"

_FEATURE_MATRIX_DEFAULT = _FEATURES / "feature_matrix_combined.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_metrics_csv(cv_results: dict[str, CVResult], output_path: Path) -> None:
    """Export per-fold and mean metrics for all models to CSV."""
    rows: list[dict] = []

    for model_type, result in cv_results.items():
        for fold in result.folds:
            guide_str = ",".join(fold.test_guide_ids) if fold.test_guide_ids else ""
            rows.append({
                "model_type": model_type,
                "row_type": "fold",
                "fold": fold.fold,
                "guide_id": guide_str,
                "n_train": fold.n_train,
                "n_test": fold.n_test,
                "n_pos_test": fold.n_pos_test,
                "auprc": fold.auprc,
                "auroc": fold.auroc,
                "precision_at_top": fold.precision_at_top,
                "recall_at_top": fold.recall_at_top,
                "enrichment_at_top": fold.enrichment_at_top,
            })
        rows.append({
            "model_type": model_type,
            "row_type": "mean",
            "fold": "mean",
            "guide_id": "",
            "n_train": None,
            "n_test": None,
            "n_pos_test": None,
            "auprc": result.mean_auprc,
            "auroc": result.mean_auroc,
            "precision_at_top": None,
            "recall_at_top": None,
            "enrichment_at_top": None,
        })
        rows.append({
            "model_type": model_type,
            "row_type": "std",
            "fold": "std",
            "guide_id": "",
            "n_train": None,
            "n_test": None,
            "n_pos_test": None,
            "auprc": result.std_auprc,
            "auroc": result.std_auroc,
            "precision_at_top": None,
            "recall_at_top": None,
            "enrichment_at_top": None,
        })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Metrics saved → %s", output_path)


def _build_per_guide_table(cv_results: dict[str, CVResult]) -> pd.DataFrame:
    """Build per-guide AUPRC table from LOGO CV results.

    For each guide, extracts AUPRC from the fold that held it out.
    Includes summary rows for median and IQR (25th/75th percentile).
    """
    # Gather guide → (auprc, n_sites, n_pos) per model
    model_maps: dict[str, dict[str, dict]] = {}
    for model_type, result in cv_results.items():
        guide_map: dict[str, dict] = {}
        for fm in result.folds:
            for gid in fm.test_guide_ids:
                guide_map[gid] = {
                    "n_sites": fm.n_test,
                    "n_positives": fm.n_pos_test,
                    "auprc": fm.auprc,
                }
        model_maps[model_type] = guide_map

    # Collect all guide IDs across all models
    all_guides = sorted(
        set().union(*(m.keys() for m in model_maps.values()))
    )

    rows: list[dict] = []
    for gid in all_guides:
        row: dict = {"guide_id": gid}
        # Use the first model that has this guide for n_sites / n_positives
        for mt in cv_results:
            if gid in model_maps[mt]:
                row["n_sites"] = model_maps[mt][gid]["n_sites"]
                row["n_positives"] = model_maps[mt][gid]["n_positives"]
                break
        for mt in cv_results:
            val = model_maps[mt].get(gid, {}).get("auprc", float("nan"))
            row[f"auprc_{mt}"] = val
        rows.append(row)

    df = pd.DataFrame(rows)

    # Summary rows
    for label, func in [("median", np.nanmedian), ("IQR_25", lambda x: np.nanpercentile(x, 25)),
                         ("IQR_75", lambda x: np.nanpercentile(x, 75))]:
        summary: dict = {"guide_id": label, "n_sites": None, "n_positives": None}
        for mt in cv_results:
            col = f"auprc_{mt}"
            vals = df[col].dropna().values
            summary[col] = float(func(vals)) if len(vals) > 0 else float("nan")
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    return df


def _save_statistical_tests(
    cv_results: dict[str, CVResult],
    output_path: Path,
) -> list[ModelComparison]:
    """Compute and save all pairwise model comparisons."""
    comparisons = [
        ("A", "C"),
        ("B", "C"),
        ("A", "B"),
    ]
    results: list[ModelComparison] = []
    rows: list[dict] = []

    for a_key, b_key in comparisons:
        if a_key not in cv_results or b_key not in cv_results:
            continue
        comp = compare_models(cv_results[a_key], cv_results[b_key])
        results.append(comp)
        rows.append({
            "comparison": f"{a_key}_vs_{b_key}",
            "delta_auprc": comp.delta_auprc,
            "n_paired": comp.n_paired,
            "ttest_p_two_sided": comp.p_value,
            "ttest_p_one_sided": comp.ttest_p_one_sided,
            "wilcoxon_p_two_sided": comp.wilcoxon_p_two_sided,
            "wilcoxon_p_one_sided": comp.wilcoxon_p_one_sided,
            "significant": comp.significant,
        })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Statistical tests saved → %s", output_path)
    return results


def _run_ablation_analysis(
    feature_matrix: pd.DataFrame,
    n_splits: int,
    *,
    include_missingness: bool,
    exclude_datasets: list[str] | None,
    output_path: Path,
) -> dict[str, dict[str, float]]:
    """Run ablation with the four requested feature subsets for Model C."""
    from src.feature_matrix import AG_FEATURE_NAMES, SEQUENCE_FEATURE_NAMES

    # Prepare Model C data
    X_c, y_c, groups_c, feat_c = prepare_model_data(
        feature_matrix, "C",
        cell_types=["K562", "HEK293"],
        exclude_datasets=exclude_datasets,
        include_missingness_indicators=include_missingness,
    )

    seq_feats = list(SEQUENCE_FEATURE_NAMES)

    # DNase/ATAC/accessibility AG features + their _missing indicators
    dnase_atac_ag = [f for f in AG_FEATURE_NAMES
                     if any(k in f for k in ("dnase", "atac", "accessibility"))]
    dnase_atac_missing = [f + "_missing" for f in dnase_atac_ag]

    # Histone AG features
    histone_ag = [f for f in AG_FEATURE_NAMES
                  if any(k in f for k in ("h3k4me3", "h3k27me3", "h3k9me3", "h3k27ac"))]
    histone_missing = [f + "_missing" for f in histone_ag]

    # CTCF AG features
    ctcf_ag = [f for f in AG_FEATURE_NAMES if "ctcf" in f]
    ctcf_missing = [f + "_missing" for f in ctcf_ag]

    subsets: dict[str, list[str]] = {
        "seq + DNase/ATAC": seq_feats + dnase_atac_ag + (dnase_atac_missing if include_missingness else []),
        "seq + histone": seq_feats + histone_ag + (histone_missing if include_missingness else []),
        "seq + CTCF": seq_feats + ctcf_ag + (ctcf_missing if include_missingness else []),
        "full Model C": list(feat_c),
    }

    logger.info("Running ablation with %d subsets, n_splits=%d ...", len(subsets), n_splits)
    ablation_results = run_ablation(
        X_c, y_c, groups_c, feat_c,
        feature_subsets=subsets,
        n_splits=n_splits,
        random_state=42,
    )

    # Save
    rows = []
    for name, metrics in ablation_results.items():
        n_feats = len([c for c in subsets[name] if c in feat_c])
        rows.append({"subset": name, "n_features": n_feats, **metrics})
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Ablation results saved → %s", output_path)
    return ablation_results


def _print_summary(
    cv_results: dict[str, CVResult],
    comparisons: list[ModelComparison],
    ablation: dict[str, dict[str, float]] | None = None,
) -> None:
    """Print comprehensive results to stdout."""
    n_folds = {mt: len(r.folds) for mt, r in cv_results.items()}

    print("\n" + "=" * 72)
    print("SafeGenome — Leave-One-Guide-Out CV Results")
    print("=" * 72)
    print(f"{'Model':<12} {'Folds':>6} {'Mean AUPRC':>12} {'Std AUPRC':>12} {'Mean AUROC':>12}")
    print("-" * 72)
    for model_type, result in cv_results.items():
        print(
            f"Model {model_type:<6} "
            f"{n_folds[model_type]:>6} "
            f"{result.mean_auprc:>12.4f} "
            f"{result.std_auprc:>12.4f} "
            f"{result.mean_auroc:>12.4f}"
        )
    print("=" * 72)

    # Statistical tests
    print("\n" + "=" * 100)
    print("Statistical Comparisons (paired by guide ID)")
    print("=" * 100)
    print(f"{'Pair':<10} {'Delta':>8} {'N':>4} "
          f"{'t 2-sided':>10} {'t 1-sided':>10} "
          f"{'W 2-sided':>10} {'W 1-sided':>10} {'Sig':>5}")
    print("-" * 100)
    for comp in comparisons:
        print(
            f"{comp.model_a_type} vs {comp.model_b_type:<4} "
            f"{comp.delta_auprc:>+8.4f} "
            f"{comp.n_paired:>4} "
            f"{comp.p_value:>10.4f} "
            f"{comp.ttest_p_one_sided:>10.4f} "
            f"{comp.wilcoxon_p_two_sided:>10.4f} "
            f"{comp.wilcoxon_p_one_sided:>10.4f} "
            f"{'*' if comp.significant else '':>5}"
        )
    print("=" * 100)
    print("  t = paired t-test | W = Wilcoxon signed-rank")
    print("  1-sided: H1 = second model > first model")
    print("  * = any two-sided p < 0.05")

    # Ablation
    if ablation:
        print("\n" + "=" * 60)
        print("Feature Ablation (Model C data, LOGO CV)")
        print("=" * 60)
        print(f"{'Subset':<25} {'Mean AUPRC':>12} {'Std AUPRC':>12}")
        print("-" * 60)
        for name, metrics in ablation.items():
            print(f"{name:<25} {metrics['mean_auprc']:>12.4f} {metrics['std_auprc']:>12.4f}")
        print("=" * 60)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train three-model comparison and save Model C for cell-type transfer"
    )
    parser.add_argument(
        "--feature-matrix", type=Path, default=_FEATURE_MATRIX_DEFAULT,
        help="Combined feature matrix parquet (default: data/features/feature_matrix_combined.parquet)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_MODELS,
        help="Output directory for model artifacts (default: data/models)",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=_FIGURES,
        help="Output directory for figures (default: figures)",
    )
    parser.add_argument(
        "--exclude-datasets", type=str, default=None,
        help="Comma-separated dataset names to exclude (e.g. deepcrispr_k562)",
    )
    parser.add_argument(
        "--no-missingness", action="store_true",
        help="Do not include missingness indicator features (_missing columns)",
    )
    args = parser.parse_args()

    # ---- Load feature matrix ------------------------------------------------
    if not args.feature_matrix.exists():
        logger.error(
            "Feature matrix not found: %s\n"
            "Run script 04 first: python scripts/04_extract_features.py",
            args.feature_matrix,
        )
        sys.exit(1)

    logger.info("Loading feature matrix: %s", args.feature_matrix)
    feature_matrix = pd.read_parquet(args.feature_matrix)
    logger.info(
        "Feature matrix: %d rows × %d columns",
        len(feature_matrix), len(feature_matrix.columns),
    )

    if "label" not in feature_matrix.columns:
        logger.error(
            "Feature matrix lacks 'label' column.  "
            "Re-run script 04 — the matrix may be from a partial run."
        )
        sys.exit(1)

    n_pos = int(feature_matrix["label"].sum())
    n_neg = len(feature_matrix) - n_pos
    logger.info(
        "Class balance: %d positive (%.1f%%), %d negative",
        n_pos, 100.0 * n_pos / len(feature_matrix) if len(feature_matrix) else 0.0, n_neg,
    )

    # ---- LOGO CV: n_splits = number of unique guides ------------------------
    n_unique_guides = feature_matrix["guide_id"].nunique()
    n_splits = n_unique_guides
    logger.info(
        "LOGO CV: %d unique guides total → n_splits=%d (B/C will auto-reduce "
        "to the subset of guides present in K562+HEK293)",
        n_unique_guides, n_splits,
    )

    # Parse exclude_datasets
    exclude_datasets: list[str] | None = None
    if args.exclude_datasets:
        exclude_datasets = [d.strip() for d in args.exclude_datasets.split(",")]
        logger.info("Excluding datasets: %s", exclude_datasets)

    include_missingness = not args.no_missingness

    # ---- Create output directories ------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    _RESULTS.mkdir(parents=True, exist_ok=True)

    # ---- Run three-model comparison -----------------------------------------
    logger.info(
        "Starting three-model comparison — LOGO CV (%d folds) ...",
        n_splits,
    )
    logger.info("Missingness indicators: %s", include_missingness)

    cv_results = run_three_model_comparison(
        feature_matrix,
        n_splits=n_splits,
        exclude_datasets=exclude_datasets,
        include_missingness_indicators=include_missingness,
        output_dir=args.output_dir,
        run_shap=False,
        run_ablation_flag=False,  # we run ablation ourselves below
    )

    # ---- Statistical tests (t-test + Wilcoxon, A-C / B-C / A-B) ------------
    comparisons = _save_statistical_tests(
        cv_results,
        _RESULTS / "statistical_tests.csv",
    )

    # ---- Ablation analysis --------------------------------------------------
    ablation = _run_ablation_analysis(
        feature_matrix,
        n_splits=n_splits,
        include_missingness=include_missingness,
        exclude_datasets=exclude_datasets,
        output_path=args.output_dir / "ablation_results.csv",
    )

    # ---- Print summary -------------------------------------------------------
    _print_summary(cv_results, comparisons, ablation)

    # ---- Save metrics CSV ---------------------------------------------------
    metrics_path = args.output_dir / "metrics_summary.csv"
    _save_metrics_csv(cv_results, metrics_path)

    # ---- Per-guide AUPRC table -----------------------------------------------
    per_guide_df = _build_per_guide_table(cv_results)
    per_guide_path = _RESULTS / "per_guide_auprc.csv"
    per_guide_df.to_csv(per_guide_path, index=False)
    logger.info("Per-guide AUPRC saved → %s", per_guide_path)

    # Print it
    print("\n" + "=" * 85)
    print("Per-Guide AUPRC (LOGO CV)")
    print("=" * 85)
    print(per_guide_df.to_string(index=False, float_format="%.4f"))
    print("=" * 85)

    # ---- Save PR curve figure -----------------------------------------------
    try:
        from src.model import plot_pr_curves
        import matplotlib.pyplot as plt

        fig = plot_pr_curves(
            cv_results,
            output_path=args.figures_dir / "fig2_pr_curves.png",
            title="SafeGenome: Model A vs B vs C — Precision-Recall Curves (LOGO CV)",
        )
        plt.close(fig)
        logger.info("PR curve figure saved → %s", args.figures_dir / "fig2_pr_curves.png")
    except Exception as exc:
        logger.warning("Could not save PR curve figure: %s", exc)

    # ---- Save all CVResult objects (for analysis notebooks) -----------------
    cv_results_path = args.output_dir / "cv_results.pkl"
    with open(cv_results_path, "wb") as f:
        pickle.dump(cv_results, f)
    logger.info("Full CVResult objects saved → %s", cv_results_path)

    # ---- Save Model C for cell-type transfer (script 06) --------------------
    if "C" not in cv_results:
        logger.error("Model C was not trained — check that AlphaGenome features are present.")
        sys.exit(1)

    result_c = cv_results["C"]
    model_c_artifact = {
        "model": result_c.final_model,
        "feature_cols": result_c.feature_cols,
        "model_type": "C",
        "mean_auprc": result_c.mean_auprc,
        "std_auprc": result_c.std_auprc,
    }

    model_c_path = args.output_dir / "model_c.pkl"
    with open(model_c_path, "wb") as f:
        pickle.dump(model_c_artifact, f)
    logger.info(
        "Model C (AUPRC=%.4f±%.4f) saved → %s",
        result_c.mean_auprc, result_c.std_auprc, model_c_path,
    )

    logger.info("Script 05 complete.  Next: python scripts/06_cell_type_transfer.py")


if __name__ == "__main__":
    main()
