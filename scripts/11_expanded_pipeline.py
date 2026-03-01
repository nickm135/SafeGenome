"""
scripts/11_expanded_pipeline.py — Retrain models on expanded dataset (K562 +
HEK293 + CD4 T cells) and run comprehensive statistical analysis.

This is the most important script in the project. It:

1. Loads the expanded feature matrix (K562 + HEK293 + Lazzarotto T cells)
2. Runs LOGO CV excluding DISCOVER-seq on ~39 guides (16 existing + 23 T cell)
3. Trains Models A, B, C + ablation variant (seq+DNase/ATAC)
4. Runs ALL statistical tests:
   - Paired t-test (per-guide)
   - Wilcoxon signed-rank (per-guide)
   - Guide-level permutation test (10k iterations)
   - Paired bootstrap on pooled OOF (10k resamples)
   - Mixed-effects linear model (site-level with guide as random effect)
5. Per-cell-type AUPRC breakdowns (K562, HEK293, CD4_Tcell)
6. Side-by-side comparison with old (16-guide) results

Requires
--------
* data/features/feature_matrix_expanded.parquet  (from script 04b)
* data/results/consolidated_statistics.csv        (old results for comparison)
* data/results/per_guide_auprc_no_discoverseq.csv (old results for comparison)

Outputs (all with _expanded suffix)
-------
    data/models/cv_results_expanded.pkl
    data/models/metrics_summary_expanded.csv
    data/results/per_guide_auprc_expanded.csv
    data/results/statistical_tests_expanded.csv
    data/models/ablation_results_expanded.csv
    data/results/permutation_test_expanded.csv
    data/results/bootstrap_test_expanded.csv
    data/results/auprc_by_celltype_expanded.csv
    data/results/consolidated_statistics_expanded.csv
    data/results/all_statistical_tests_expanded.csv
    data/results/comparison_old_vs_expanded.csv
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import (
    CVResult,
    ModelComparison,
    compare_models,
    cross_validate_model,
    prepare_model_data,
    run_ablation,
    run_three_model_comparison,
)
from src.feature_matrix import AG_FEATURE_NAMES, SEQUENCE_FEATURE_NAMES
from src.utils import get_logger

logger = get_logger("expanded_pipeline")

_ROOT = Path(__file__).resolve().parents[1]
_FEATURES = _ROOT / "data" / "features"
_MODELS = _ROOT / "data" / "models"
_RESULTS = _ROOT / "data" / "results"

EXCLUDE_DATASETS = ["discover_seq_k562"]
MODEL_BC_CELL_TYPES = ["K562", "HEK293", "CD4_Tcell"]


# ===================================================================
# Helpers (reused from script 10, with improvements)
# ===================================================================

def _build_per_guide_table(cv_results: dict[str, CVResult]) -> pd.DataFrame:
    """Build per-guide AUPRC table from LOGO CV results."""
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

    all_guides = sorted(set().union(*(m.keys() for m in model_maps.values())))
    rows: list[dict] = []
    for gid in all_guides:
        row: dict = {"guide_id": gid}
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
    for label, func in [("median", np.nanmedian),
                         ("IQR_25", lambda x: np.nanpercentile(x, 25)),
                         ("IQR_75", lambda x: np.nanpercentile(x, 75)),
                         ("mean", np.nanmean),
                         ("std", np.nanstd)]:
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
    """Compute and save pairwise model comparisons."""
    comparisons_pairs = [("A", "C"), ("B", "C"), ("A", "B")]
    results: list[ModelComparison] = []
    rows: list[dict] = []
    for a_key, b_key in comparisons_pairs:
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
    logger.info("Statistical tests saved -> %s", output_path)
    return results


def _save_metrics_csv(cv_results: dict[str, CVResult], output_path: Path) -> None:
    """Export per-fold and mean metrics to CSV."""
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
            })
        rows.append({
            "model_type": model_type,
            "row_type": "mean",
            "fold": "mean",
            "guide_id": "",
            "auprc": result.mean_auprc,
            "auroc": result.mean_auroc,
        })
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def _run_ablation_analysis(
    feature_matrix: pd.DataFrame,
    n_splits: int,
    *,
    exclude_datasets: list[str] | None,
    cell_types: list[str],
    output_path: Path,
) -> dict[str, dict[str, float]]:
    """Run ablation with the four feature subsets for Model C."""
    X_c, y_c, groups_c, feat_c = prepare_model_data(
        feature_matrix, "C",
        cell_types=cell_types,
        exclude_datasets=exclude_datasets,
        include_missingness_indicators=True,
    )

    seq_feats = list(SEQUENCE_FEATURE_NAMES)
    dnase_atac_ag = [f for f in AG_FEATURE_NAMES
                     if any(k in f for k in ("dnase", "atac", "accessibility"))]
    dnase_atac_missing = [f + "_missing" for f in dnase_atac_ag]
    histone_ag = [f for f in AG_FEATURE_NAMES
                  if any(k in f for k in ("h3k4me3", "h3k27me3", "h3k9me3", "h3k27ac",
                                           "h3k36me3", "h3k4me1"))]
    histone_missing = [f + "_missing" for f in histone_ag]
    ctcf_ag = [f for f in AG_FEATURE_NAMES if "ctcf" in f]
    ctcf_missing = [f + "_missing" for f in ctcf_ag]

    subsets: dict[str, list[str]] = {
        "seq + DNase/ATAC": seq_feats + dnase_atac_ag + dnase_atac_missing,
        "seq + histone": seq_feats + histone_ag + histone_missing,
        "seq + CTCF": seq_feats + ctcf_ag + ctcf_missing,
        "full Model C": list(feat_c),
    }

    ablation_results = run_ablation(
        X_c, y_c, groups_c, feat_c,
        feature_subsets=subsets,
        n_splits=n_splits,
        random_state=42,
    )

    rows = []
    for name, metrics in ablation_results.items():
        n_feats = len([c for c in subsets[name] if c in feat_c])
        rows.append({"subset": name, "n_features": n_feats, **metrics})
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Ablation results saved -> %s", output_path)
    return ablation_results


# ===================================================================
# Statistical test helpers
# ===================================================================

def guide_level_permutation_test(
    per_guide_df: pd.DataFrame,
    col_x: str,
    col_y: str,
    n_iter: int = 10_000,
    seed: int = 42,
) -> dict:
    """Permutation test: for each guide, 50/50 swap col_x and col_y."""
    mask = per_guide_df[col_x].notna() & per_guide_df[col_y].notna()
    x = per_guide_df.loc[mask, col_x].values
    y = per_guide_df.loc[mask, col_y].values
    n = len(x)

    observed = np.mean(y - x)

    rng = np.random.RandomState(seed)
    count_ge = 0
    for _ in range(n_iter):
        swap = rng.randint(0, 2, size=n).astype(bool)
        perm_x = np.where(swap, y, x)
        perm_y = np.where(swap, x, y)
        if np.mean(perm_y - perm_x) >= observed:
            count_ge += 1

    return {
        "n_guides": n,
        "observed_mean_diff": observed,
        "n_permutations": n_iter,
        "count_ge_observed": count_ge,
        "p_value": count_ge / n_iter,
    }


def paired_bootstrap_auprc(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap: resample sites, compute AUPRC difference."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    observed_a = average_precision_score(y_true, scores_a)
    observed_b = average_precision_score(y_true, scores_b)
    observed_delta = observed_b - observed_a

    count_ge = 0
    n_valid = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_boot = y_true[idx]
        if y_boot.sum() == 0 or y_boot.sum() == len(y_boot):
            continue
        n_valid += 1
        ap_a = average_precision_score(y_boot, scores_a[idx])
        ap_b = average_precision_score(y_boot, scores_b[idx])
        if ap_a >= ap_b:
            count_ge += 1

    return {
        "observed_auprc_inferior": observed_a,
        "observed_auprc_superior": observed_b,
        "observed_delta": observed_delta,
        "n_bootstrap": n_bootstrap,
        "n_valid": n_valid,
        "count_inferior_ge_superior": count_ge,
        "p_value": count_ge / n_valid if n_valid > 0 else float("nan"),
    }


def run_mixed_effects_model(
    cv_results: dict[str, CVResult],
    feature_matrix: pd.DataFrame,
    exclude_datasets: list[str] | None = None,
) -> dict[str, dict]:
    """Run mixed-effects linear model on per-site predicted scores.

    Uses a linear mixed-effects model (LMM):
        score ~ model_type (fixed effect) + (1 | guide_id) (random effect)

    For each pair (A vs C, B vs C), fits a model on the stacked OOF scores
    and tests whether the fixed effect of model_type is significant.

    Falls back to a simpler approach if statsmodels fails.

    Returns
    -------
    dict[str, dict]
        Keys: 'A_vs_C', 'B_vs_C'. Each contains p_value, coefficient, etc.
    """
    results = {}

    # Get guide_id for each site from the feature matrix
    # OOF predictions are indexed in the same order as prepare_model_data output
    for label, inferior_key, superior_key in [
        ("A_vs_C", "A", "C"),
        ("B_vs_C", "B", "C"),
    ]:
        try:
            # Use OOF scores and true labels
            y_true_inf = cv_results[inferior_key].oof_y_true
            y_score_inf = cv_results[inferior_key].oof_y_score
            y_true_sup = cv_results[superior_key].oof_y_true
            y_score_sup = cv_results[superior_key].oof_y_score

            # We need guide_ids for the OOF predictions.
            # Extract from the fold metadata.
            # Each fold has test_guide_ids and n_test. Build guide_id array.
            guide_ids_inf = []
            for fold in cv_results[inferior_key].folds:
                for gid in fold.test_guide_ids:
                    # Each guide's sites in this fold
                    # The fold holds out all sites for these guides
                    # n_test is total for the fold (may be multiple guides)
                    pass

            # Alternative: get guide_ids from prepare_model_data
            # Since both models see the same sites (for B/C with cell_types),
            # we can extract from the feature matrix directly.
            # For the inferior model, reconstruct.
            if inferior_key == "A":
                # Model A uses all cell types
                _, _, groups_inf, _ = prepare_model_data(
                    feature_matrix, "A",
                    cell_types=None,
                    exclude_datasets=exclude_datasets,
                )
            else:
                _, _, groups_inf, _ = prepare_model_data(
                    feature_matrix, inferior_key,
                    cell_types=MODEL_BC_CELL_TYPES,
                    exclude_datasets=exclude_datasets,
                    include_missingness_indicators=True,
                )

            _, _, groups_sup, _ = prepare_model_data(
                feature_matrix, superior_key,
                cell_types=MODEL_BC_CELL_TYPES,
                exclude_datasets=exclude_datasets,
                include_missingness_indicators=True,
            )

            # For A vs C comparison, the OOF arrays may have different lengths
            # (A trains on all cell types, C on BC cell types only).
            # Use paired sites where both models have predictions.
            # The safest approach: use the C model's site set (subset of A).

            n_inf = len(y_score_inf)
            n_sup = len(y_score_sup)

            if n_inf != n_sup:
                logger.info(
                    "  %s: different OOF sizes (inf=%d, sup=%d). "
                    "Using superior model's site set for LMM.",
                    label, n_inf, n_sup,
                )
                # Model C (superior) has fewer sites. Use those.
                # The guide_ids from groups_sup correspond to y_score_sup.
                guide_ids = groups_sup
                scores_sup = y_score_sup
                y_true = y_true_sup

                # For the inferior model, we need to get scores for the same sites.
                # Since Model A was trained on a superset, its OOF predictions
                # include the C sites but also extra ones. We need to align.
                # The simplest approach: skip LMM for mismatched pairs.
                logger.warning(
                    "  %s: Cannot align OOF predictions across different "
                    "site sets for LMM. Using per-guide average difference instead.",
                    label,
                )
                # Fallback: per-guide mean score difference with clustered SE
                results[label] = _fallback_clustered_test(
                    cv_results[inferior_key], cv_results[superior_key]
                )
                continue
            else:
                guide_ids = groups_sup
                scores_inf_arr = y_score_inf
                scores_sup_arr = y_score_sup
                y_true = y_true_sup

            # Build stacked dataframe for LMM
            n = len(scores_sup_arr)
            df_lmm = pd.DataFrame({
                "score": np.concatenate([scores_inf_arr, scores_sup_arr]),
                "model": [inferior_key] * n + [superior_key] * n,
                "guide_id": np.concatenate([guide_ids, guide_ids]),
                "site_idx": list(range(n)) + list(range(n)),
                "y_true": np.concatenate([y_true, y_true]),
            })

            # Encode model as numeric (0 = inferior, 1 = superior)
            df_lmm["model_num"] = (df_lmm["model"] == superior_key).astype(int)

            try:
                import statsmodels.formula.api as smf

                # Linear mixed-effects model
                md = smf.mixedlm(
                    "score ~ model_num",
                    df_lmm,
                    groups=df_lmm["guide_id"],
                )
                mdf = md.fit(reml=True)

                coef = mdf.params.get("model_num", float("nan"))
                p_val = mdf.pvalues.get("model_num", float("nan"))
                se = mdf.bse.get("model_num", float("nan"))
                ci = mdf.conf_int().loc["model_num"] if "model_num" in mdf.conf_int().index else [float("nan"), float("nan")]

                results[label] = {
                    "test": "Linear mixed-effects model",
                    "coefficient": float(coef),
                    "std_error": float(se),
                    "p_value": float(p_val),
                    "ci_lower": float(ci[0]) if not isinstance(ci, float) else float("nan"),
                    "ci_upper": float(ci[1]) if not isinstance(ci, float) else float("nan"),
                    "n_sites": n,
                    "n_guides": len(np.unique(guide_ids)),
                    "formula": "score ~ model_num + (1 | guide_id)",
                }

                logger.info(
                    "  LMM %s: coef=%.4f, SE=%.4f, p=%.6f, n_sites=%d, n_guides=%d",
                    label, coef, se, p_val, n, len(np.unique(guide_ids)),
                )

            except ImportError:
                logger.warning("statsmodels not available. Falling back to clustered test.")
                results[label] = _fallback_clustered_test(
                    cv_results[inferior_key], cv_results[superior_key]
                )
            except Exception as exc:
                logger.warning("LMM failed for %s: %s. Falling back.", label, exc)
                results[label] = _fallback_clustered_test(
                    cv_results[inferior_key], cv_results[superior_key]
                )

        except Exception as exc:
            logger.error("Mixed-effects test failed for %s: %s", label, exc)
            results[label] = {"test": "failed", "error": str(exc), "p_value": float("nan")}

    return results


def _fallback_clustered_test(
    result_a: CVResult,
    result_b: CVResult,
) -> dict:
    """Fallback: paired test on per-guide AUPRC with cluster-robust reasoning.

    When OOF arrays have different lengths (Model A vs C), use per-guide
    AUPRC from fold metrics as paired observations.
    """
    guide_auprc_a = {}
    for fold in result_a.folds:
        for gid in fold.test_guide_ids:
            guide_auprc_a[gid] = fold.auprc

    guide_auprc_b = {}
    for fold in result_b.folds:
        for gid in fold.test_guide_ids:
            guide_auprc_b[gid] = fold.auprc

    common = sorted(set(guide_auprc_a.keys()) & set(guide_auprc_b.keys()))
    if len(common) < 3:
        return {
            "test": "fallback_paired (insufficient guides)",
            "p_value": float("nan"),
            "n_guides": len(common),
        }

    a_vals = np.array([guide_auprc_a[g] for g in common])
    b_vals = np.array([guide_auprc_b[g] for g in common])
    diffs = b_vals - a_vals

    t_stat, p_val = stats.ttest_rel(b_vals, a_vals)

    return {
        "test": "fallback_paired_t (guide-level)",
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs, ddof=1)),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "p_value_one_sided": float(p_val / 2) if t_stat > 0 else float(1 - p_val / 2),
        "n_guides": len(common),
    }


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    _RESULTS.mkdir(parents=True, exist_ok=True)
    _MODELS.mkdir(parents=True, exist_ok=True)

    # Load expanded feature matrix
    fm_path = _FEATURES / "feature_matrix_expanded.parquet"
    logger.info("Loading expanded feature matrix: %s", fm_path)
    if not fm_path.exists():
        logger.error(
            "Expanded feature matrix not found: %s\n"
            "Run script 04b first.",
            fm_path,
        )
        sys.exit(1)

    feature_matrix = pd.read_parquet(fm_path)
    logger.info(
        "Feature matrix: %d rows x %d columns",
        len(feature_matrix), len(feature_matrix.columns),
    )

    # Show dataset composition
    if "cell_type" in feature_matrix.columns:
        print("\n" + "=" * 80)
        print("DATASET COMPOSITION")
        print("=" * 80)
        for ct in sorted(feature_matrix["cell_type"].unique()):
            ct_data = feature_matrix[feature_matrix["cell_type"] == ct]
            ct_pos = int(ct_data["label"].sum())
            ct_guides = ct_data["guide_id"].nunique()
            print(f"  {ct:15s}: {len(ct_data):6d} rows, {ct_pos:4d} positive ({100*ct_pos/len(ct_data):.1f}%), {ct_guides:3d} guides")

    # Show what we're excluding
    disc_rows = feature_matrix[feature_matrix["dataset"] == "discover_seq_k562"]
    if len(disc_rows) > 0:
        logger.info(
            "Excluding discover_seq_k562: %d rows, %d positives, guides=%s",
            len(disc_rows), int(disc_rows["label"].sum()),
            sorted(disc_rows["guide_id"].unique()),
        )

    remaining = feature_matrix[feature_matrix["dataset"] != "discover_seq_k562"]
    n_guides_remaining = remaining["guide_id"].nunique()
    n_splits = n_guides_remaining
    logger.info(
        "Remaining after DISCOVER-seq exclusion: %d rows, %d positives, %d guides -> n_splits=%d",
        len(remaining), int(remaining["label"].sum()),
        n_guides_remaining, n_splits,
    )

    # ==================================================================
    # TASK 1: LOGO CV on expanded dataset
    # ==================================================================
    print("\n" + "=" * 80)
    print("TASK 1: LOGO CV on expanded dataset (excluding DISCOVER-seq)")
    print(f"  Cell types: {MODEL_BC_CELL_TYPES}")
    print(f"  Expected ~{n_splits} LOGO folds")
    print("=" * 80)

    cv_results = run_three_model_comparison(
        feature_matrix,
        n_splits=n_splits,
        exclude_datasets=EXCLUDE_DATASETS,
        model_bc_cell_types=MODEL_BC_CELL_TYPES,
        include_missingness_indicators=True,
        run_shap=False,
        run_ablation_flag=False,
    )

    # Report
    for mt in ("A", "B", "C"):
        r = cv_results[mt]
        guides = set()
        for fold in r.folds:
            for gid in fold.test_guide_ids:
                guides.add(gid)
        oof_auprc = average_precision_score(r.oof_y_true, r.oof_y_score)
        print(
            f"  Model {mt}: {len(r.folds)} folds, {len(guides)} guides, "
            f"{len(r.oof_y_true)} sites, {int(r.oof_y_true.sum())} positives"
        )
        print(
            f"    Mean AUPRC = {r.mean_auprc:.4f} +/- {r.std_auprc:.4f}  |  "
            f"OOF pooled AUPRC = {oof_auprc:.4f}"
        )

    # Save cv_results
    cv_path = _MODELS / "cv_results_expanded.pkl"
    with open(cv_path, "wb") as f:
        pickle.dump(cv_results, f)
    print(f"\n  Saved -> {cv_path}")

    # Save metrics CSV
    _save_metrics_csv(cv_results, _MODELS / "metrics_summary_expanded.csv")

    # Per-guide AUPRC table
    per_guide_df = _build_per_guide_table(cv_results)
    per_guide_path = _RESULTS / "per_guide_auprc_expanded.csv"
    per_guide_df.to_csv(per_guide_path, index=False)
    print(f"  Saved -> {per_guide_path}")

    # Print per-guide table
    print("\nPer-guide AUPRC (expanded):")
    print(per_guide_df.to_string(index=False, float_format="%.4f"))

    # Statistical tests (t-test, Wilcoxon)
    comparisons = _save_statistical_tests(
        cv_results,
        _RESULTS / "statistical_tests_expanded.csv",
    )
    print("\nStatistical tests (paired by guide):")
    for comp in comparisons:
        print(
            f"  {comp.model_a_type} vs {comp.model_b_type}: "
            f"delta={comp.delta_auprc:+.4f}, n={comp.n_paired}, "
            f"t_2s={comp.p_value:.4f}, t_1s={comp.ttest_p_one_sided:.4f}, "
            f"W_2s={comp.wilcoxon_p_two_sided:.4f}, W_1s={comp.wilcoxon_p_one_sided:.4f}"
        )

    # Ablation
    ablation = _run_ablation_analysis(
        feature_matrix,
        n_splits=n_splits,
        exclude_datasets=EXCLUDE_DATASETS,
        cell_types=MODEL_BC_CELL_TYPES,
        output_path=_MODELS / "ablation_results_expanded.csv",
    )
    print("\nAblation (Model C, expanded):")
    for name, metrics in ablation.items():
        print(f"  {name:<25s} AUPRC = {metrics['mean_auprc']:.4f} +/- {metrics['std_auprc']:.4f}")

    # ==================================================================
    # TASK 2: Permutation + Bootstrap tests
    # ==================================================================
    print("\n" + "=" * 80)
    print("TASK 2: Permutation + Bootstrap tests (expanded)")
    print("=" * 80)

    # Filter per_guide to actual guides (not summary rows)
    summary_labels = {"median", "IQR_25", "IQR_75", "mean", "std"}
    guides_only = per_guide_df[
        ~per_guide_df["guide_id"].isin(summary_labels)
    ].copy()

    perm_results = []
    for label, col_x, col_y in [("A_vs_C", "auprc_A", "auprc_C"),
                                  ("B_vs_C", "auprc_B", "auprc_C"),
                                  ("A_vs_B", "auprc_A", "auprc_B")]:
        res = guide_level_permutation_test(guides_only, col_x, col_y)
        res["comparison"] = label
        perm_results.append(res)
        print(
            f"  Permutation {label}: observed diff={res['observed_mean_diff']:+.4f}, "
            f"p={res['p_value']:.4f} ({res['count_ge_observed']}/{res['n_permutations']})"
        )

    perm_df = pd.DataFrame(perm_results)
    perm_path = _RESULTS / "permutation_test_expanded.csv"
    perm_df.to_csv(perm_path, index=False)
    print(f"  Saved -> {perm_path}")

    # Bootstrap on pooled OOF
    print("\n  Bootstrap (pooled OOF, 10k resamples):")
    boot_results = []
    y_true_ref = cv_results["C"].oof_y_true
    for label, inferior_key, superior_key in [
        ("A_vs_C", "A", "C"),
        ("B_vs_C", "B", "C"),
        ("A_vs_B", "A", "B"),
    ]:
        y_inf = cv_results[inferior_key].oof_y_score
        y_sup = cv_results[superior_key].oof_y_score
        # For A vs B/C, A may have different OOF length. Check.
        if len(y_inf) != len(y_true_ref):
            # Use the smaller set's true labels
            y_true_use = cv_results[superior_key].oof_y_true
            # Skip if sizes don't match
            if len(y_inf) != len(y_true_use):
                print(f"    {label}: SKIPPED (OOF size mismatch: {len(y_inf)} vs {len(y_sup)})")
                boot_results.append({
                    "comparison": label,
                    "p_value": float("nan"),
                    "observed_delta": float("nan"),
                    "note": f"OOF size mismatch: {len(y_inf)} vs {len(y_sup)}",
                })
                continue
        else:
            y_true_use = y_true_ref

        res = paired_bootstrap_auprc(y_true_use, y_inf, y_sup)
        res["comparison"] = label
        boot_results.append(res)
        print(
            f"    {label}: delta={res['observed_delta']:+.4f}, "
            f"p={res['p_value']:.4f} ({res['count_inferior_ge_superior']}/{res['n_valid']} valid)"
        )

    boot_df = pd.DataFrame(boot_results)
    boot_path = _RESULTS / "bootstrap_test_expanded.csv"
    boot_df.to_csv(boot_path, index=False)
    print(f"  Saved -> {boot_path}")

    # ==================================================================
    # TASK 3: Mixed-effects model
    # ==================================================================
    print("\n" + "=" * 80)
    print("TASK 3: Mixed-effects model (site-level, guide as random effect)")
    print("=" * 80)

    lmm_results = run_mixed_effects_model(
        cv_results, feature_matrix, exclude_datasets=EXCLUDE_DATASETS
    )

    for label, res in lmm_results.items():
        p_val = res.get("p_value", float("nan"))
        test_type = res.get("test", "unknown")
        if "coefficient" in res:
            print(
                f"  {label}: {test_type}, coef={res['coefficient']:.4f}, "
                f"SE={res.get('std_error', float('nan')):.4f}, p={p_val:.6f}, "
                f"n_sites={res.get('n_sites', '?')}, n_guides={res.get('n_guides', '?')}"
            )
        else:
            print(f"  {label}: {test_type}, p={p_val:.4f}")

    # ==================================================================
    # TASK 4: Per-cell-type AUPRC breakdown
    # ==================================================================
    print("\n" + "=" * 80)
    print("TASK 4: Per-cell-type AUPRC breakdown (expanded)")
    print("=" * 80)

    fm_no_disc = feature_matrix[feature_matrix["dataset"] != "discover_seq_k562"]
    guide_celltype = fm_no_disc.groupby("guide_id")["cell_type"].first().to_dict()

    ct_rows = []
    for ct in ["K562", "HEK293", "CD4_Tcell"]:
        ct_guides = [g for g, c in guide_celltype.items() if c == ct]
        ct_data = guides_only[guides_only["guide_id"].isin(ct_guides)]
        print(f"\n  {ct} ({len(ct_guides)} guides):")
        for mt in ("A", "B", "C"):
            col = f"auprc_{mt}"
            vals = ct_data[col].dropna()
            if len(vals) > 0:
                mean_v = vals.mean()
                std_v = vals.std()
                med_v = vals.median()
                print(f"    Model {mt}: mean={mean_v:.4f}+/-{std_v:.4f}, median={med_v:.4f}, n={len(vals)}")
            else:
                mean_v = std_v = med_v = float("nan")
                print(f"    Model {mt}: no data")

            ct_rows.append({
                "cell_type": ct,
                "model": mt,
                "n_guides": len(vals),
                "mean_auprc": round(mean_v, 4) if not np.isnan(mean_v) else None,
                "std_auprc": round(std_v, 4) if not np.isnan(std_v) else None,
                "median_auprc": round(med_v, 4) if not np.isnan(med_v) else None,
            })

    ct_df = pd.DataFrame(ct_rows)
    ct_path = _RESULTS / "auprc_by_celltype_expanded.csv"
    ct_df.to_csv(ct_path, index=False)
    print(f"\n  Saved -> {ct_path}")

    # ==================================================================
    # TASK 5: Consolidated statistical comparison table
    # ==================================================================
    print("\n" + "=" * 80)
    print("TASK 5: Consolidated statistical comparison table (expanded)")
    print("=" * 80)

    consolidated_rows = []

    # Wilcoxon
    for comp in comparisons:
        pair = f"{comp.model_a_type} vs {comp.model_b_type}"
        consolidated_rows.append({
            "comparison": pair,
            "test": "Wilcoxon signed-rank (per-guide, two-sided)",
            "p_value": comp.wilcoxon_p_two_sided,
            "n": comp.n_paired,
            "effect_size": comp.delta_auprc,
            "significant_005": comp.wilcoxon_p_two_sided < 0.05 if comp.wilcoxon_p_two_sided is not None else None,
        })
        consolidated_rows.append({
            "comparison": pair,
            "test": "Wilcoxon signed-rank (per-guide, one-sided)",
            "p_value": comp.wilcoxon_p_one_sided,
            "n": comp.n_paired,
            "effect_size": comp.delta_auprc,
            "significant_005": comp.wilcoxon_p_one_sided < 0.05 if comp.wilcoxon_p_one_sided is not None else None,
        })

    # Paired t-test
    for comp in comparisons:
        pair = f"{comp.model_a_type} vs {comp.model_b_type}"
        consolidated_rows.append({
            "comparison": pair,
            "test": "Paired t-test (per-guide, two-sided)",
            "p_value": comp.p_value,
            "n": comp.n_paired,
            "effect_size": comp.delta_auprc,
            "significant_005": comp.p_value < 0.05,
        })
        consolidated_rows.append({
            "comparison": pair,
            "test": "Paired t-test (per-guide, one-sided)",
            "p_value": comp.ttest_p_one_sided,
            "n": comp.n_paired,
            "effect_size": comp.delta_auprc,
            "significant_005": comp.ttest_p_one_sided < 0.05,
        })

    # Guide-level permutation test
    for res in perm_results:
        consolidated_rows.append({
            "comparison": res["comparison"].replace("_", " "),
            "test": "Guide-level permutation (10k iter)",
            "p_value": res["p_value"],
            "n": res["n_guides"],
            "effect_size": res["observed_mean_diff"],
            "significant_005": res["p_value"] < 0.05,
        })

    # Paired bootstrap on pooled OOF
    for res in boot_results:
        p = res.get("p_value", float("nan"))
        consolidated_rows.append({
            "comparison": res["comparison"].replace("_", " "),
            "test": "Paired bootstrap on pooled OOF (10k resamples)",
            "p_value": p,
            "n": int(res.get("n_valid", 0)),
            "effect_size": res.get("observed_delta", float("nan")),
            "significant_005": p < 0.05 if not np.isnan(p) else None,
        })

    # Mixed-effects model
    for label, res in lmm_results.items():
        p = res.get("p_value", float("nan"))
        consolidated_rows.append({
            "comparison": label.replace("_", " "),
            "test": f"Mixed-effects model ({res.get('test', 'LMM')})",
            "p_value": p,
            "n": res.get("n_sites", res.get("n_guides", None)),
            "effect_size": res.get("coefficient", res.get("mean_diff", float("nan"))),
            "significant_005": p < 0.05 if not np.isnan(p) else None,
        })

    consol_df = pd.DataFrame(consolidated_rows)
    consol_path = _RESULTS / "consolidated_statistics_expanded.csv"
    consol_df.to_csv(consol_path, index=False)

    # Also save as all_statistical_tests_expanded.csv
    consol_df.to_csv(_RESULTS / "all_statistical_tests_expanded.csv", index=False)

    # Pretty print
    print(f"\n{'Comparison':<12} {'Test':<52} {'p-value':>10} {'Effect':>10} {'n':>6} {'Sig':>5}")
    print("-" * 100)
    for _, row in consol_df.iterrows():
        sig = "*" if row.get("significant_005") else ""
        p_str = f"{row['p_value']:.4f}" if pd.notna(row["p_value"]) else "N/A"
        e_str = f"{row['effect_size']:+.4f}" if pd.notna(row["effect_size"]) else "N/A"
        n_str = str(int(row["n"])) if pd.notna(row.get("n")) else ""
        test_str = str(row["test"])[:50]
        print(f"{row['comparison']:<12} {test_str:<52} {p_str:>10} {e_str:>10} {n_str:>6} {sig:>5}")

    print(f"\n  Saved -> {consol_path}")

    # ==================================================================
    # TASK 6: Side-by-side comparison with old results
    # ==================================================================
    print("\n" + "=" * 80)
    print("TASK 6: Side-by-side comparison — Old (16 guides) vs Expanded")
    print("=" * 80)

    # Try to load old results
    old_per_guide_path = _RESULTS / "per_guide_auprc_no_discoverseq.csv"
    old_consol_path = _RESULTS / "consolidated_statistics.csv"

    comparison_rows = []

    if old_per_guide_path.exists():
        old_pg = pd.read_csv(old_per_guide_path)
        old_guides = old_pg[~old_pg["guide_id"].isin(["median", "IQR_25", "IQR_75", "mean", "std"])]
        new_guides = guides_only

        n_old = len(old_guides)
        n_new = len(new_guides)

        print(f"\n  {'':35s} {'Old (16 guides)':>20s} {'Expanded ({} guides)':>20s}".format(n_new))
        print(f"  {'-'*75}")

        for mt in ("A", "B", "C"):
            col = f"auprc_{mt}"
            if col in old_pg.columns:
                old_vals = old_guides[col].dropna()
                old_mean = old_vals.mean()
                old_std = old_vals.std()
            else:
                old_mean = old_std = float("nan")

            if col in new_guides.columns:
                new_vals = new_guides[col].dropna()
                new_mean = new_vals.mean()
                new_std = new_vals.std()
            else:
                new_mean = new_std = float("nan")

            print(
                f"  Model {mt} Mean AUPRC:                "
                f"{old_mean:.4f}±{old_std:.4f}      "
                f"{new_mean:.4f}±{new_std:.4f}"
            )

            comparison_rows.append({
                "metric": f"Model {mt} Mean AUPRC",
                "old_16_guides": f"{old_mean:.4f}±{old_std:.4f}",
                "expanded": f"{new_mean:.4f}±{new_std:.4f}",
                "old_value": old_mean,
                "new_value": new_mean,
                "delta": new_mean - old_mean if not np.isnan(old_mean) else float("nan"),
            })

        # OOF AUPRC
        print()
        for mt in ("A", "B", "C"):
            new_oof = average_precision_score(
                cv_results[mt].oof_y_true, cv_results[mt].oof_y_score
            )
            comparison_rows.append({
                "metric": f"Model {mt} OOF AUPRC",
                "old_16_guides": "see old results",
                "expanded": f"{new_oof:.4f}",
                "new_value": new_oof,
            })
            print(f"  Model {mt} OOF AUPRC:                                     {new_oof:.4f}")

        # Dataset size comparison
        print(f"\n  Dataset size:")
        old_n_sites = int(old_guides["n_sites"].sum()) if "n_sites" in old_guides.columns else "?"
        new_n_sites = len(remaining)
        old_n_pos = int(old_guides["n_positives"].sum()) if "n_positives" in old_guides.columns else "?"
        new_n_pos = int(remaining["label"].sum()) if "label" in remaining.columns else "?"
        print(f"    Guides:    {n_old:>6}        ->    {n_new:>6}")
        print(f"    Sites:     {old_n_sites:>6}        ->    {new_n_sites:>6}")
        print(f"    Positives: {old_n_pos:>6}        ->    {new_n_pos:>6}")

        comparison_rows.extend([
            {"metric": "N guides", "old_16_guides": str(n_old), "expanded": str(n_new)},
            {"metric": "N sites", "old_16_guides": str(old_n_sites), "expanded": str(new_n_sites)},
            {"metric": "N positives", "old_16_guides": str(old_n_pos), "expanded": str(new_n_pos)},
        ])
    else:
        print("  Old results not found — showing expanded results only.")
        for mt in ("A", "B", "C"):
            new_vals = guides_only[f"auprc_{mt}"].dropna()
            new_oof = average_precision_score(
                cv_results[mt].oof_y_true, cv_results[mt].oof_y_score
            )
            print(
                f"  Model {mt}: mean={new_vals.mean():.4f}±{new_vals.std():.4f}, "
                f"OOF={new_oof:.4f}, n_folds={len(cv_results[mt].folds)}"
            )

    # Save comparison
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        comp_path = _RESULTS / "comparison_old_vs_expanded.csv"
        comp_df.to_csv(comp_path, index=False)
        print(f"\n  Saved -> {comp_path}")

    # Statistical test comparison
    if old_consol_path.exists():
        old_consol = pd.read_csv(old_consol_path)
        print(f"\n  Statistical test comparison (old vs expanded):")
        print(f"  {'Comparison':<12} {'Test':<45} {'Old p':>10} {'New p':>10} {'Change':>10}")
        print(f"  {'-'*90}")

        for _, new_row in consol_df.iterrows():
            # Find matching old row
            old_match = old_consol[
                (old_consol["comparison"] == new_row["comparison"]) &
                (old_consol["test"] == new_row["test"])
            ]
            if len(old_match) > 0:
                old_p = old_match.iloc[0]["p_value"]
                new_p = new_row["p_value"]
                if pd.notna(old_p) and pd.notna(new_p):
                    change = "IMPROVED" if new_p < old_p else "weaker"
                    if new_p < 0.05 and old_p >= 0.05:
                        change = "NOW SIG*"
                    print(
                        f"  {new_row['comparison']:<12} {str(new_row['test'])[:43]:<45} "
                        f"{old_p:>10.4f} {new_p:>10.4f} {change:>10}"
                    )

    print("\n" + "=" * 80)
    print("Script 11 (expanded pipeline) complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
