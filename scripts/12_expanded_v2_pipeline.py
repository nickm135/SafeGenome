"""
scripts/12_expanded_v2_pipeline.py — Expanded pipeline WITH real T cell
AlphaGenome chromatin features.

Differences from script 11 (expanded, no AG for T cells):
- T cell sites now have 12/12 real AlphaGenome chromatin features from
  CD4+ T cell ontology (CL:0000624) instead of all-NaN.
- All outputs use _expanded_v2 suffix.
- Includes three-way comparison: original (16 guides) vs expanded-no-chrom
  vs expanded-with-chrom.
- Includes held-out DISCOVER-seq validation training on ALL 60 GUIDE-seq
  guides (not just 4 HEK293 guides as in script 08).

Outputs (all with _expanded_v2 suffix)
-------
    data/models/cv_results_expanded_v2.pkl
    data/models/metrics_summary_expanded_v2.csv
    data/models/ablation_results_expanded_v2.csv
    data/results/per_guide_auprc_expanded_v2.csv
    data/results/statistical_tests_expanded_v2.csv
    data/results/permutation_test_expanded_v2.csv
    data/results/bootstrap_test_expanded_v2.csv
    data/results/auprc_by_celltype_expanded_v2.csv
    data/results/consolidated_statistics_expanded_v2.csv
    data/results/all_statistical_tests_expanded_v2.csv
    data/results/comparison_three_way.csv
    data/results/held_out_validation_expanded_v2.csv
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
    compute_metrics,
    compute_scale_pos_weight,
    cross_validate_model,
    make_xgb_classifier,
    prepare_model_data,
    run_ablation,
    run_three_model_comparison,
)
from src.feature_matrix import (
    AG_FEATURE_NAMES,
    SEQUENCE_FEATURE_NAMES,
    get_model_features,
)
from src.utils import get_logger

logger = get_logger("expanded_v2_pipeline")

_ROOT = Path(__file__).resolve().parents[1]
_FEATURES = _ROOT / "data" / "features"
_MODELS = _ROOT / "data" / "models"
_RESULTS = _ROOT / "data" / "results"

EXCLUDE_DATASETS = ["discover_seq_k562"]
MODEL_BC_CELL_TYPES = ["K562", "HEK293", "CD4_Tcell"]
SUFFIX = "_expanded_v2"


# ===================================================================
# Helpers
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

    Uses LMM: score ~ model_num + (1 | guide_id)
    """
    results = {}

    for label, inferior_key, superior_key in [
        ("A_vs_C", "A", "C"),
        ("B_vs_C", "B", "C"),
        ("A_vs_B", "A", "B"),
    ]:
        try:
            y_true_inf = cv_results[inferior_key].oof_y_true
            y_score_inf = cv_results[inferior_key].oof_y_score
            y_true_sup = cv_results[superior_key].oof_y_true
            y_score_sup = cv_results[superior_key].oof_y_score

            if inferior_key == "A":
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

            n_inf = len(y_score_inf)
            n_sup = len(y_score_sup)

            if n_inf != n_sup:
                logger.info(
                    "  %s: different OOF sizes (inf=%d, sup=%d). Using fallback.",
                    label, n_inf, n_sup,
                )
                results[label] = _fallback_clustered_test(
                    cv_results[inferior_key], cv_results[superior_key]
                )
                continue

            guide_ids = groups_sup
            scores_inf_arr = y_score_inf
            scores_sup_arr = y_score_sup
            y_true = y_true_sup

            n = len(scores_sup_arr)
            df_lmm = pd.DataFrame({
                "score": np.concatenate([scores_inf_arr, scores_sup_arr]),
                "model": [inferior_key] * n + [superior_key] * n,
                "guide_id": np.concatenate([guide_ids, guide_ids]),
                "site_idx": list(range(n)) + list(range(n)),
                "y_true": np.concatenate([y_true, y_true]),
            })
            df_lmm["model_num"] = (df_lmm["model"] == superior_key).astype(int)

            try:
                import statsmodels.formula.api as smf

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
                logger.warning("statsmodels not available. Falling back.")
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
    """Fallback: paired test on per-guide AUPRC."""
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
# Held-out DISCOVER-seq validation
# ===================================================================

def run_held_out_validation(
    feature_matrix: pd.DataFrame,
    *,
    train_datasets: list[str],
    test_dataset: str = "discover_seq_k562",
    include_missingness: bool = True,
) -> pd.DataFrame:
    """Train on ALL GUIDE-seq data, test on DISCOVER-seq K562.

    Unlike script 08 which trained on only HEK293 (4 guides), this trains
    on all 60 GUIDE-seq guides (K562 + HEK293 + T cells).
    """
    train_df = feature_matrix[feature_matrix["dataset"].isin(train_datasets)].copy()
    test_df = feature_matrix[feature_matrix["dataset"] == test_dataset].copy()

    logger.info(
        "Held-out split: train=%s (%d rows, %d pos), test=%s (%d rows, %d pos)",
        train_datasets, len(train_df), int(train_df["label"].sum()),
        test_dataset, len(test_df), int(test_df["label"].sum()),
    )

    train_guides = sorted(train_df["guide_id"].unique())
    test_guides = sorted(test_df["guide_id"].unique())
    logger.info("  Train guides (%d): %s", len(train_guides), train_guides[:10])
    logger.info("  Test guides (%d): %s", len(test_guides), test_guides)

    overlap = set(train_guides) & set(test_guides)
    if overlap:
        logger.warning("Guide overlap between train and test: %s", overlap)
    else:
        logger.info("  No guide overlap (good)")

    results_rows: list[dict] = []

    for model_type in ("A", "B", "C"):
        feature_cols = get_model_features(
            model_type,
            include_missingness_indicators=include_missingness,
        )
        feature_cols = [c for c in feature_cols if c in feature_matrix.columns]

        if not feature_cols:
            logger.warning("Model %s: no feature columns, skipping", model_type)
            continue

        X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_train = train_df["label"].to_numpy(dtype=np.int32)
        X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
        y_test = test_df["label"].to_numpy(dtype=np.int32)

        n_pos_train = int(y_train.sum())
        n_pos_test = int(y_test.sum())

        if n_pos_train == 0 or n_pos_test == 0:
            logger.warning("Model %s: no positives, skipping", model_type)
            continue

        spw = compute_scale_pos_weight(y_train)
        clf = make_xgb_classifier(spw, random_state=42)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_score, top_fraction=0.10)
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
            "n_train_guides": len(train_guides),
            "n_features": len(feature_cols),
            "train_datasets": ",".join(train_datasets),
            "test_dataset": test_dataset,
        })
        logger.info(
            "  Model %s: AUPRC=%.4f, AUROC=%.4f",
            model_type, metrics["auprc"], metrics["auroc"],
        )

    # CFD/MIT baselines on test set
    for score_name in ("cfd_score", "mit_score"):
        scores = test_df[score_name].to_numpy(dtype=np.float64)
        scores = np.where(np.isnan(scores), 0.0, scores)
        y_test = test_df["label"].to_numpy(dtype=np.int32)
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
            "n_pos_test": int(y_test.sum()),
            "n_train_guides": 0,
            "n_features": 1,
            "train_datasets": "",
            "test_dataset": test_dataset,
        })

    return pd.DataFrame(results_rows)


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    _RESULTS.mkdir(parents=True, exist_ok=True)
    _MODELS.mkdir(parents=True, exist_ok=True)

    # ---- Load and verify expanded feature matrix ----
    fm_path = _FEATURES / "feature_matrix_expanded.parquet"
    logger.info("Loading expanded feature matrix: %s", fm_path)
    if not fm_path.exists():
        logger.error("Feature matrix not found: %s. Run script 04b first.", fm_path)
        sys.exit(1)

    feature_matrix = pd.read_parquet(fm_path)
    logger.info(
        "Feature matrix: %d rows x %d columns",
        len(feature_matrix), len(feature_matrix.columns),
    )

    # Verify T cell AG features are populated
    ag_cols = [c for c in feature_matrix.columns if c.startswith("ag_") and not c.endswith("_missing")]
    tcell = feature_matrix[feature_matrix["cell_type"] == "CD4_Tcell"]
    if len(tcell) > 0:
        for col in ("ag_dnase_mean", "ag_atac_mean"):
            non_nan = tcell[col].notna().sum()
            pct = 100 * non_nan / len(tcell)
            if pct < 90:
                logger.error(
                    "ABORT: %s has only %.1f%% non-NaN for T cell sites. "
                    "Re-run 04b without --no-ag first.",
                    col, pct,
                )
                sys.exit(1)
        logger.info(
            "Verified: T cell AG features populated (%d/%d non-NaN for ag_dnase_mean)",
            tcell["ag_dnase_mean"].notna().sum(), len(tcell),
        )

    # Dataset composition
    print("\n" + "=" * 80)
    print("DATASET COMPOSITION (expanded, with real T cell AlphaGenome)")
    print("=" * 80)
    for ct in sorted(feature_matrix["cell_type"].unique()):
        ct_data = feature_matrix[feature_matrix["cell_type"] == ct]
        ct_pos = int(ct_data["label"].sum())
        ct_guides = ct_data["guide_id"].nunique()
        ag_nonnull = ct_data[ag_cols].notna().all(axis=1).sum()
        print(
            f"  {ct:15s}: {len(ct_data):6d} rows, {ct_pos:4d} positive "
            f"({100*ct_pos/len(ct_data):.1f}%), {ct_guides:3d} guides, "
            f"{ag_nonnull}/{len(ct_data)} AG complete"
        )

    disc_rows = feature_matrix[feature_matrix["dataset"] == "discover_seq_k562"]
    if len(disc_rows) > 0:
        logger.info(
            "Excluding discover_seq_k562: %d rows, %d pos, guides=%s",
            len(disc_rows), int(disc_rows["label"].sum()),
            sorted(disc_rows["guide_id"].unique()),
        )

    remaining = feature_matrix[feature_matrix["dataset"] != "discover_seq_k562"]
    n_guides_remaining = remaining["guide_id"].nunique()
    n_splits = n_guides_remaining
    logger.info(
        "After DISCOVER-seq exclusion: %d rows, %d pos, %d guides",
        len(remaining), int(remaining["label"].sum()), n_guides_remaining,
    )

    # ==================================================================
    # TASK 1: LOGO CV on expanded dataset (with real T cell AG)
    # ==================================================================
    print("\n" + "=" * 80)
    print(f"TASK 1: LOGO CV on expanded dataset WITH real T cell chromatin")
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

    for mt in ("A", "B", "C"):
        r = cv_results[mt]
        guides = set()
        for fold in r.folds:
            for gid in fold.test_guide_ids:
                guides.add(gid)
        oof_auprc = average_precision_score(r.oof_y_true, r.oof_y_score)
        print(
            f"  Model {mt}: {len(r.folds)} folds, {len(guides)} guides, "
            f"{len(r.oof_y_true)} sites, {int(r.oof_y_true.sum())} pos"
        )
        print(
            f"    Mean AUPRC = {r.mean_auprc:.4f} +/- {r.std_auprc:.4f}  |  "
            f"OOF pooled AUPRC = {oof_auprc:.4f}"
        )

    # Save cv_results
    cv_path = _MODELS / f"cv_results{SUFFIX}.pkl"
    with open(cv_path, "wb") as f:
        pickle.dump(cv_results, f)
    print(f"\n  Saved -> {cv_path}")

    _save_metrics_csv(cv_results, _MODELS / f"metrics_summary{SUFFIX}.csv")

    # Per-guide AUPRC table
    per_guide_df = _build_per_guide_table(cv_results)
    per_guide_path = _RESULTS / f"per_guide_auprc{SUFFIX}.csv"
    per_guide_df.to_csv(per_guide_path, index=False)
    print(f"  Saved -> {per_guide_path}")

    print("\nPer-guide AUPRC (top 10 + summary):")
    display_df = pd.concat([per_guide_df.head(10), per_guide_df.tail(5)])
    print(display_df.to_string(index=False, float_format="%.4f"))

    # Statistical tests (t-test, Wilcoxon)
    comparisons = _save_statistical_tests(
        cv_results,
        _RESULTS / f"statistical_tests{SUFFIX}.csv",
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
        output_path=_MODELS / f"ablation_results{SUFFIX}.csv",
    )
    print(f"\nAblation (Model C, expanded v2):")
    for name, metrics in ablation.items():
        print(f"  {name:<25s} AUPRC = {metrics['mean_auprc']:.4f} +/- {metrics['std_auprc']:.4f}")

    # ==================================================================
    # TASK 2: Permutation + Bootstrap tests
    # ==================================================================
    print("\n" + "=" * 80)
    print("TASK 2: Permutation + Bootstrap tests")
    print("=" * 80)

    summary_labels = {"median", "IQR_25", "IQR_75", "mean", "std"}
    guides_only = per_guide_df[~per_guide_df["guide_id"].isin(summary_labels)].copy()

    perm_results = []
    for label, col_x, col_y in [("A_vs_C", "auprc_A", "auprc_C"),
                                  ("B_vs_C", "auprc_B", "auprc_C"),
                                  ("A_vs_B", "auprc_A", "auprc_B")]:
        res = guide_level_permutation_test(guides_only, col_x, col_y)
        res["comparison"] = label
        perm_results.append(res)
        print(
            f"  Permutation {label}: diff={res['observed_mean_diff']:+.4f}, "
            f"p={res['p_value']:.4f}"
        )

    perm_df = pd.DataFrame(perm_results)
    perm_df.to_csv(_RESULTS / f"permutation_test{SUFFIX}.csv", index=False)

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
        if len(y_inf) != len(y_true_ref):
            y_true_use = cv_results[superior_key].oof_y_true
            if len(y_inf) != len(y_true_use):
                print(f"    {label}: SKIPPED (OOF size mismatch)")
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
            f"p={res['p_value']:.4f}"
        )

    boot_df = pd.DataFrame(boot_results)
    boot_df.to_csv(_RESULTS / f"bootstrap_test{SUFFIX}.csv", index=False)

    # ==================================================================
    # TASK 3: Mixed-effects model
    # ==================================================================
    print("\n" + "=" * 80)
    print("TASK 3: Mixed-effects model")
    print("=" * 80)

    lmm_results = run_mixed_effects_model(
        cv_results, feature_matrix, exclude_datasets=EXCLUDE_DATASETS
    )

    for label, res in lmm_results.items():
        p_val = res.get("p_value", float("nan"))
        test_type = res.get("test", "unknown")
        if "coefficient" in res:
            print(
                f"  {label}: coef={res['coefficient']:.4f}, "
                f"SE={res.get('std_error', float('nan')):.4f}, p={p_val:.6f}"
            )
        else:
            print(f"  {label}: {test_type}, p={p_val:.4f}")

    # ==================================================================
    # TASK 4: Per-cell-type AUPRC breakdown
    # ==================================================================
    print("\n" + "=" * 80)
    print("TASK 4: Per-cell-type AUPRC breakdown")
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
    ct_df.to_csv(_RESULTS / f"auprc_by_celltype{SUFFIX}.csv", index=False)

    # ==================================================================
    # TASK 5: Consolidated statistical comparison table
    # ==================================================================
    print("\n" + "=" * 80)
    print("TASK 5: All statistical tests consolidated")
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

    # Permutation
    for res in perm_results:
        consolidated_rows.append({
            "comparison": res["comparison"].replace("_", " "),
            "test": "Guide-level permutation (10k iter)",
            "p_value": res["p_value"],
            "n": res["n_guides"],
            "effect_size": res["observed_mean_diff"],
            "significant_005": res["p_value"] < 0.05,
        })

    # Bootstrap
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

    # LMM
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
    consol_df.to_csv(_RESULTS / f"consolidated_statistics{SUFFIX}.csv", index=False)
    consol_df.to_csv(_RESULTS / f"all_statistical_tests{SUFFIX}.csv", index=False)

    print(f"\n{'Comparison':<12} {'Test':<52} {'p-value':>10} {'Effect':>10} {'n':>6} {'Sig':>5}")
    print("-" * 100)
    for _, row in consol_df.iterrows():
        sig = "*" if row.get("significant_005") else ""
        p_str = f"{row['p_value']:.4f}" if pd.notna(row["p_value"]) else "N/A"
        e_str = f"{row['effect_size']:+.4f}" if pd.notna(row["effect_size"]) else "N/A"
        n_str = str(int(row["n"])) if pd.notna(row.get("n")) else ""
        test_str = str(row["test"])[:50]
        print(f"{row['comparison']:<12} {test_str:<52} {p_str:>10} {e_str:>10} {n_str:>6} {sig:>5}")

    # ==================================================================
    # TASK 6: Three-way comparison
    # ==================================================================
    print("\n" + "=" * 80)
    print("TASK 6: Three-way comparison")
    print("  (1) Original 16 guides  (2) Expanded 60 no-chrom  (3) Expanded 60 with-chrom")
    print("=" * 80)

    three_way_rows = []

    # Load old results
    old_pg_path = _RESULTS / "per_guide_auprc_no_discoverseq.csv"
    exp_pg_path = _RESULTS / "per_guide_auprc_expanded.csv"

    # Current v2 results
    v2_guides = guides_only

    for label_name, pg_path in [
        ("original_16", old_pg_path),
        ("expanded_no_chrom", exp_pg_path),
    ]:
        if pg_path.exists():
            pg = pd.read_csv(pg_path)
            pg_data = pg[~pg["guide_id"].isin(summary_labels)]
            n_guides = len(pg_data)
            for mt in ("A", "B", "C"):
                col = f"auprc_{mt}"
                if col in pg_data.columns:
                    vals = pg_data[col].dropna()
                    three_way_rows.append({
                        "version": label_name,
                        "model": mt,
                        "n_guides": n_guides,
                        "mean_auprc": vals.mean(),
                        "std_auprc": vals.std(),
                        "median_auprc": vals.median(),
                    })
        else:
            logger.warning("File not found for three-way comparison: %s", pg_path)

    # V2 (with chrom)
    for mt in ("A", "B", "C"):
        col = f"auprc_{mt}"
        vals = v2_guides[col].dropna()
        three_way_rows.append({
            "version": "expanded_with_chrom",
            "model": mt,
            "n_guides": len(v2_guides),
            "mean_auprc": vals.mean(),
            "std_auprc": vals.std(),
            "median_auprc": vals.median(),
        })

    three_way_df = pd.DataFrame(three_way_rows)
    three_way_path = _RESULTS / "comparison_three_way.csv"
    three_way_df.to_csv(three_way_path, index=False)

    # Pretty print
    print(f"\n{'':25s} {'Original (16)':>18s} {'Exp no-chrom':>18s} {'Exp with-chrom':>18s}")
    print("-" * 80)
    for mt in ("A", "B", "C"):
        parts = []
        for version in ("original_16", "expanded_no_chrom", "expanded_with_chrom"):
            row = three_way_df[(three_way_df["version"] == version) & (three_way_df["model"] == mt)]
            if len(row) > 0:
                m = row.iloc[0]["mean_auprc"]
                s = row.iloc[0]["std_auprc"]
                parts.append(f"{m:.4f}+/-{s:.4f}")
            else:
                parts.append("N/A")
        print(f"  Model {mt} Mean AUPRC:     {parts[0]:>18s} {parts[1]:>18s} {parts[2]:>18s}")

    # Per-cell-type C vs A gap comparison
    print(f"\n  Per-cell-type Model C vs A gap (expanded with-chrom):")
    for ct in ["K562", "HEK293", "CD4_Tcell"]:
        ct_guides_list = [g for g, c in guide_celltype.items() if c == ct]
        ct_data = v2_guides[v2_guides["guide_id"].isin(ct_guides_list)]
        a_vals = ct_data["auprc_A"].dropna()
        c_vals = ct_data["auprc_C"].dropna()
        if len(a_vals) > 0 and len(c_vals) > 0:
            gap = c_vals.mean() - a_vals.mean()
            print(f"    {ct:15s}: C-A gap = {gap:+.4f} (A={a_vals.mean():.4f}, C={c_vals.mean():.4f})")

    print(f"\n  Saved -> {three_way_path}")

    # ==================================================================
    # TASK 7: Held-out DISCOVER-seq validation
    # ==================================================================
    print("\n" + "=" * 80)
    print("TASK 7: Held-out DISCOVER-seq validation")
    print("  Train: ALL GUIDE-seq (60 guides) | Test: DISCOVER-seq K562 (2 guides)")
    print("=" * 80)

    train_datasets = [
        "guide_seq_tsai2015",
        "deepcrispr_k562",
        "guide_seq_lazzarotto2020",
    ]

    held_out_df = run_held_out_validation(
        feature_matrix,
        train_datasets=train_datasets,
        test_dataset="discover_seq_k562",
    )

    print(f"\n{'Model':<20} {'AUPRC':>8} {'AUROC':>8} {'N_train':>8} {'N_test':>7} {'Pos_test':>9}")
    print("-" * 65)
    for _, row in held_out_df.iterrows():
        print(
            f"{row['model']:<20} {row['auprc']:>8.4f} {row['auroc']:>8.4f} "
            f"{int(row['n_train']):>8} {int(row['n_test']):>7} {int(row['n_pos_test']):>9}"
        )

    # Compare with old held-out (script 08, 4 HEK293 guides only)
    old_heldout_path = _RESULTS / "held_out_validation.csv"
    if old_heldout_path.exists():
        old_ho = pd.read_csv(old_heldout_path)
        print(f"\n  Comparison with old held-out (4 HEK293 guides):")
        print(f"  {'Model':<20} {'Old AUPRC':>10} {'New AUPRC':>10} {'Delta':>10}")
        print(f"  {'-'*55}")
        for _, new_row in held_out_df.iterrows():
            old_match = old_ho[old_ho["model"] == new_row["model"]]
            if len(old_match) > 0:
                old_auprc = old_match.iloc[0]["auprc"]
                new_auprc = new_row["auprc"]
                delta = new_auprc - old_auprc
                print(
                    f"  {new_row['model']:<20} {old_auprc:>10.4f} {new_auprc:>10.4f} {delta:>+10.4f}"
                )

    held_out_path = _RESULTS / f"held_out_validation{SUFFIX}.csv"
    held_out_df.to_csv(held_out_path, index=False)
    print(f"\n  Saved -> {held_out_path}")

    # ==================================================================
    # Final summary
    # ==================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY — Expanded v2 (with real T cell AlphaGenome chromatin)")
    print("=" * 80)

    for mt in ("A", "B", "C"):
        r = cv_results[mt]
        oof = average_precision_score(r.oof_y_true, r.oof_y_score)
        print(f"  Model {mt}: LOGO Mean AUPRC = {r.mean_auprc:.4f}+/-{r.std_auprc:.4f}, OOF = {oof:.4f}")

    print(f"\n  Key statistical tests (A vs C):")
    for _, row in consol_df[consol_df["comparison"].str.contains("A vs C")].iterrows():
        sig = "*" if row.get("significant_005") else ""
        p_str = f"{row['p_value']:.4f}" if pd.notna(row["p_value"]) else "N/A"
        print(f"    {str(row['test'])[:55]}: p={p_str} {sig}")

    print(f"\n  All output files saved with '{SUFFIX}' suffix.")
    print("=" * 80)
    print("Script 12 (expanded v2 pipeline) complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
