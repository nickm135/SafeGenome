"""
scripts/10_rerun_no_discoverseq.py — Re-run LOGO CV excluding DISCOVER-seq.

Reserves DISCOVER-seq K562 as a clean held-out test set by removing it from
the main cross-validation. Runs all four verification tasks:

Task 1: LOGO CV without DISCOVER-seq -> updated cv_results, per-guide AUPRC,
        ablation, statistical tests (all with _no_discoverseq suffix).
Task 2: Permutation test + bootstrap test on updated results.
Task 3: Per-cell-type AUPRC breakdown (K562 vs HEK293 guides).
Task 4: Consolidated statistical comparison table.

Requires
--------
* data/features/feature_matrix_combined.parquet  (from script 04)

Outputs (all in data/results/ or data/models/ with _no_discoverseq suffix)
-------
    data/models/cv_results_no_discoverseq.pkl
    data/models/metrics_summary_no_discoverseq.csv
    data/results/per_guide_auprc_no_discoverseq.csv
    data/results/statistical_tests_no_discoverseq.csv
    data/models/ablation_results_no_discoverseq.csv
    data/results/permutation_test_no_discoverseq.csv
    data/results/bootstrap_test_no_discoverseq.csv
    data/results/auprc_by_celltype.csv
    data/results/consolidated_statistics.csv
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

logger = get_logger("rerun_no_discoverseq")

_ROOT = Path(__file__).resolve().parents[1]
_FEATURES = _ROOT / "data" / "features"
_MODELS = _ROOT / "data" / "models"
_RESULTS = _ROOT / "data" / "results"

EXCLUDE_DATASETS = ["discover_seq_k562"]


# ===================================================================
# Helpers (adapted from script 05)
# ===================================================================

def _build_per_guide_table(cv_results: dict[str, CVResult]) -> pd.DataFrame:
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
    output_path: Path,
) -> dict[str, dict[str, float]]:
    X_c, y_c, groups_c, feat_c = prepare_model_data(
        feature_matrix, "C",
        cell_types=["K562", "HEK293"],
        exclude_datasets=exclude_datasets,
        include_missingness_indicators=True,
    )

    seq_feats = list(SEQUENCE_FEATURE_NAMES)
    dnase_atac_ag = [f for f in AG_FEATURE_NAMES
                     if any(k in f for k in ("dnase", "atac", "accessibility"))]
    dnase_atac_missing = [f + "_missing" for f in dnase_atac_ag]
    histone_ag = [f for f in AG_FEATURE_NAMES
                  if any(k in f for k in ("h3k4me3", "h3k27me3", "h3k9me3", "h3k27ac"))]
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
# Task 2 helpers: permutation + bootstrap
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
    observed_delta = observed_b - observed_a  # b is "superior"

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
        if ap_a >= ap_b:  # inferior >= superior
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


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    _RESULTS.mkdir(parents=True, exist_ok=True)
    _MODELS.mkdir(parents=True, exist_ok=True)

    # Load feature matrix
    fm_path = _FEATURES / "feature_matrix_combined.parquet"
    logger.info("Loading feature matrix: %s", fm_path)
    feature_matrix = pd.read_parquet(fm_path)
    logger.info("Feature matrix: %d rows x %d columns", len(feature_matrix), len(feature_matrix.columns))

    # Show what we're excluding
    disc_rows = feature_matrix[feature_matrix["dataset"] == "discover_seq_k562"]
    logger.info(
        "Excluding discover_seq_k562: %d rows, %d positives, guides=%s",
        len(disc_rows), int(disc_rows["label"].sum()),
        sorted(disc_rows["guide_id"].unique()),
    )

    remaining = feature_matrix[feature_matrix["dataset"] != "discover_seq_k562"]
    n_guides_remaining = remaining["guide_id"].nunique()
    n_splits = n_guides_remaining  # true LOGO
    logger.info(
        "Remaining: %d rows, %d positives, %d guides -> n_splits=%d",
        len(remaining), int(remaining["label"].sum()),
        n_guides_remaining, n_splits,
    )

    # ==================================================================
    # TASK 1: Re-run LOGO CV without DISCOVER-seq
    # ==================================================================
    print("\n" + "=" * 80)
    print("TASK 1: Re-run LOGO CV excluding DISCOVER-seq")
    print("=" * 80)

    cv_results = run_three_model_comparison(
        feature_matrix,
        n_splits=n_splits,
        exclude_datasets=EXCLUDE_DATASETS,
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
    cv_path = _MODELS / "cv_results_no_discoverseq.pkl"
    with open(cv_path, "wb") as f:
        pickle.dump(cv_results, f)
    print(f"\n  Saved -> {cv_path}")

    # Save metrics CSV
    _save_metrics_csv(cv_results, _MODELS / "metrics_summary_no_discoverseq.csv")

    # Per-guide AUPRC table
    per_guide_df = _build_per_guide_table(cv_results)
    per_guide_path = _RESULTS / "per_guide_auprc_no_discoverseq.csv"
    per_guide_df.to_csv(per_guide_path, index=False)
    print(f"  Saved -> {per_guide_path}")

    # Print per-guide table
    print("\nPer-guide AUPRC:")
    print(per_guide_df.to_string(index=False, float_format="%.4f"))

    # Statistical tests (t-test, Wilcoxon)
    comparisons = _save_statistical_tests(
        cv_results,
        _RESULTS / "statistical_tests_no_discoverseq.csv",
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
        output_path=_MODELS / "ablation_results_no_discoverseq.csv",
    )
    print("\nAblation (Model C):")
    for name, metrics in ablation.items():
        print(f"  {name:<25s} AUPRC = {metrics['mean_auprc']:.4f} +/- {metrics['std_auprc']:.4f}")

    # ==================================================================
    # TASK 2: Permutation test + Bootstrap test
    # ==================================================================
    print("\n" + "=" * 80)
    print("TASK 2: Permutation test + Bootstrap test (no DISCOVER-seq)")
    print("=" * 80)

    # Filter per_guide to actual guides (not summary rows)
    guides_only = per_guide_df[
        ~per_guide_df["guide_id"].isin(["median", "IQR_25", "IQR_75"])
    ].copy()

    perm_results = []
    for label, col_x, col_y in [("A_vs_C", "auprc_A", "auprc_C"),
                                  ("B_vs_C", "auprc_B", "auprc_C")]:
        res = guide_level_permutation_test(guides_only, col_x, col_y)
        res["comparison"] = label
        perm_results.append(res)
        print(
            f"  Permutation {label}: observed diff={res['observed_mean_diff']:+.4f}, "
            f"p={res['p_value']:.4f} ({res['count_ge_observed']}/{res['n_permutations']})"
        )

    perm_df = pd.DataFrame(perm_results)
    perm_path = _RESULTS / "permutation_test_no_discoverseq.csv"
    perm_df.to_csv(perm_path, index=False)
    print(f"  Saved -> {perm_path}")

    # Bootstrap on pooled OOF
    print("\n  Bootstrap (pooled OOF, 10k resamples):")
    boot_results = []
    y_true_ref = cv_results["C"].oof_y_true
    for label, inferior_key, superior_key in [
        ("A_vs_C", "A", "C"),
        ("B_vs_C", "B", "C"),
    ]:
        y_inf = cv_results[inferior_key].oof_y_score
        y_sup = cv_results[superior_key].oof_y_score
        res = paired_bootstrap_auprc(y_true_ref, y_inf, y_sup)
        res["comparison"] = label
        boot_results.append(res)
        print(
            f"    {label}: delta={res['observed_delta']:+.4f}, "
            f"p={res['p_value']:.4f} ({res['count_inferior_ge_superior']}/{res['n_valid']} valid)"
        )

    boot_df = pd.DataFrame(boot_results)
    boot_path = _RESULTS / "bootstrap_test_no_discoverseq.csv"
    boot_df.to_csv(boot_path, index=False)
    print(f"  Saved -> {boot_path}")

    # ==================================================================
    # TASK 3: Per-cell-type AUPRC breakdown
    # ==================================================================
    print("\n" + "=" * 80)
    print("TASK 3: Per-cell-type AUPRC breakdown")
    print("=" * 80)

    # Load the feature matrix to identify which guides are K562 vs HEK293
    fm_no_disc = feature_matrix[feature_matrix["dataset"] != "discover_seq_k562"]
    guide_celltype = fm_no_disc.groupby("guide_id")["cell_type"].first().to_dict()

    print(f"\n  Guide -> cell type mapping:")
    for gid in sorted(guide_celltype.keys()):
        print(f"    {gid}: {guide_celltype[gid]}")

    ct_rows = []
    for ct in ["K562", "HEK293"]:
        ct_guides = [g for g, c in guide_celltype.items() if c == ct]
        ct_data = guides_only[guides_only["guide_id"].isin(ct_guides)]
        print(f"\n  {ct} guides ({len(ct_guides)}): {sorted(ct_guides)}")
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
                "guide_ids": ",".join(sorted(ct_guides)),
            })

    ct_df = pd.DataFrame(ct_rows)
    ct_path = _RESULTS / "auprc_by_celltype.csv"
    ct_df.to_csv(ct_path, index=False)
    print(f"\n  Saved -> {ct_path}")

    # ==================================================================
    # TASK 4: Consolidated statistical comparison table
    # ==================================================================
    print("\n" + "=" * 80)
    print("TASK 4: Consolidated statistical comparison table")
    print("=" * 80)

    consolidated_rows = []

    # Wilcoxon signed-rank (from _save_statistical_tests)
    for comp in comparisons:
        pair = f"{comp.model_a_type} vs {comp.model_b_type}"
        consolidated_rows.append({
            "comparison": pair,
            "test": "Wilcoxon signed-rank (per-guide, two-sided)",
            "statistic": comp.wilcoxon_statistic if hasattr(comp, "wilcoxon_statistic") else None,
            "p_value": comp.wilcoxon_p_two_sided,
            "n": comp.n_paired,
            "effect_size": comp.delta_auprc,
            "significant_005": comp.wilcoxon_p_two_sided < 0.05 if comp.wilcoxon_p_two_sided is not None else None,
        })

    # Paired t-test (from _save_statistical_tests)
    for comp in comparisons:
        pair = f"{comp.model_a_type} vs {comp.model_b_type}"
        consolidated_rows.append({
            "comparison": pair,
            "test": "Paired t-test (per-guide, two-sided)",
            "statistic": comp.t_statistic if hasattr(comp, "t_statistic") else None,
            "p_value": comp.p_value,
            "n": comp.n_paired,
            "effect_size": comp.delta_auprc,
            "significant_005": comp.p_value < 0.05,
        })

    # Guide-level permutation test
    for res in perm_results:
        consolidated_rows.append({
            "comparison": res["comparison"].replace("_", " "),
            "test": "Guide-level permutation (10k iter)",
            "statistic": None,
            "p_value": res["p_value"],
            "n": res["n_guides"],
            "effect_size": res["observed_mean_diff"],
            "significant_005": res["p_value"] < 0.05,
        })

    # Paired bootstrap on pooled OOF
    for res in boot_results:
        consolidated_rows.append({
            "comparison": res["comparison"].replace("_", " "),
            "test": "Paired bootstrap on pooled OOF (10k resamples)",
            "statistic": None,
            "p_value": res["p_value"],
            "n": int(res["n_valid"]),
            "effect_size": res["observed_delta"],
            "significant_005": res["p_value"] < 0.05,
        })

    consol_df = pd.DataFrame(consolidated_rows)
    consol_path = _RESULTS / "consolidated_statistics.csv"
    consol_df.to_csv(consol_path, index=False)
    print(f"\n  Saved -> {consol_path}")

    # Pretty print
    print(f"\n{'Comparison':<12} {'Test':<48} {'p-value':>10} {'Effect':>10} {'n':>5} {'Sig':>5}")
    print("-" * 95)
    for _, row in consol_df.iterrows():
        sig = "*" if row["significant_005"] else ""
        p_str = f"{row['p_value']:.4f}" if pd.notna(row["p_value"]) else "N/A"
        e_str = f"{row['effect_size']:+.4f}" if pd.notna(row["effect_size"]) else "N/A"
        n_str = str(int(row["n"])) if pd.notna(row["n"]) else ""
        print(f"{row['comparison']:<12} {row['test']:<48} {p_str:>10} {e_str:>10} {n_str:>5} {sig:>5}")

    print("\n" + "=" * 80)
    print("Script 10 complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
