"""
scripts/13_final_verification.py — Final verification run for SafeGenome.

Tasks:
1. SHAP values for expanded_v2 Model C (TreeExplainer on final model)
2. Model C-no-missing ablation (drop all _missing columns, full LOGO CV)
3. Per-cell-type breakdown for C vs C-no-missing
4. Export clean summary files with full statistical tests

Outputs:
    data/results/shap_importance_expanded_v2.csv
    data/results/per_guide_auprc_expanded_v2_no_missing.csv
    data/results/final_verification_summary.csv
    data/results/final_statistical_tests.csv
"""

from __future__ import annotations

import pickle
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import (
    CVResult,
    FoldMetrics,
    compare_models,
    compute_metrics,
    compute_scale_pos_weight,
    cross_validate_model,
    make_xgb_classifier,
    prepare_model_data,
)
from src.feature_matrix import (
    AG_FEATURE_NAMES,
    SEQUENCE_FEATURE_NAMES,
    get_model_features,
    MISSING_INDICATOR_SUFFIX,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("final_verification")

_ROOT = Path(__file__).resolve().parents[1]
_FEATURES = _ROOT / "data" / "features"
_MODELS = _ROOT / "data" / "models"
_RESULTS = _ROOT / "data" / "results"

EXCLUDE_DATASETS = ["discover_seq_k562"]
MODEL_BC_CELL_TYPES = ["K562", "HEK293", "CD4_Tcell"]
N_SPLITS_LOGO = 60  # One per guide


def main() -> None:
    # Load feature matrix
    fm_path = _FEATURES / "feature_matrix_expanded.parquet"
    logger.info("Loading feature matrix: %s", fm_path)
    fm = pd.read_parquet(fm_path)
    logger.info("Feature matrix: %d rows x %d cols", len(fm), len(fm.columns))

    # Load existing cv_results for Models A, B, C
    cv_path = _MODELS / "cv_results_expanded_v2.pkl"
    logger.info("Loading CV results: %s", cv_path)
    with open(cv_path, "rb") as f:
        cv_results: dict[str, CVResult] = pickle.load(f)

    # ===================================================================
    # TASK 1: SHAP values for expanded_v2 Model C
    # ===================================================================
    print("\n" + "=" * 70)
    print("TASK 1: SHAP values for expanded_v2 Model C")
    print("=" * 70)

    import shap

    result_c = cv_results["C"]
    final_model = result_c.final_model
    feature_cols = result_c.feature_cols

    # Get the data used for Model C
    X_c, y_c, groups_c, feat_c = prepare_model_data(
        fm, "C",
        cell_types=MODEL_BC_CELL_TYPES,
        exclude_datasets=EXCLUDE_DATASETS,
        include_missingness_indicators=True,
    )

    logger.info("Computing SHAP values with TreeExplainer on %d samples, %d features...",
                X_c.shape[0], X_c.shape[1])

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_c)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feat_c,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_path = _RESULTS / "shap_importance_expanded_v2.csv"
    shap_df.to_csv(shap_path, index=False)
    logger.info("SHAP values saved -> %s", shap_path)

    print("\nTop 30 features by mean |SHAP| (expanded_v2 Model C):")
    print(f"{'Rank':<6} {'Feature':<40} {'Mean |SHAP|':<15}")
    print("-" * 61)
    missing_in_top30 = []
    for i, row in shap_df.head(30).iterrows():
        flag = ""
        if row["feature"].endswith("_missing"):
            flag = " *** _missing FLAG ***"
            missing_in_top30.append((i + 1, row["feature"], row["mean_abs_shap"]))
        print(f"{i+1:<6} {row['feature']:<40} {row['mean_abs_shap']:<15.6f}{flag}")

    if missing_in_top30:
        print(f"\n*** WARNING: {len(missing_in_top30)} _missing feature(s) in top 30: ***")
        for rank, feat, val in missing_in_top30:
            print(f"  Rank {rank}: {feat} = {val:.6f}")
    else:
        print("\nNo _missing features in top 30.")

    # ===================================================================
    # TASK 2: Model C without missingness indicators (full LOGO CV)
    # ===================================================================
    print("\n" + "=" * 70)
    print("TASK 2: Model C without missingness indicators")
    print("=" * 70)

    # Get Model C features WITHOUT missingness indicators
    c_features_no_missing = get_model_features("C", include_missingness_indicators=False)
    logger.info("Model C features (no _missing): %d features", len(c_features_no_missing))
    logger.info("Model C features (with _missing): %d features", len(feat_c))
    logger.info("Dropped features: %s",
                [f for f in feat_c if f not in c_features_no_missing])

    # Filter feature matrix and extract arrays manually
    df_filtered = fm.copy()

    # Exclude datasets
    if "dataset" in df_filtered.columns:
        mask = df_filtered["dataset"].isin(EXCLUDE_DATASETS)
        df_filtered = df_filtered[~mask]

    # Filter cell types
    if "cell_type" in df_filtered.columns:
        mask = df_filtered["cell_type"].isin(MODEL_BC_CELL_TYPES)
        df_filtered = df_filtered[mask]

    # Verify no on-targets
    n_ontarget = int((df_filtered["num_mismatches"] == 0).sum())
    assert n_ontarget == 0, f"On-target sites found: {n_ontarget}"

    # Select features (no _missing)
    available_feats = [c for c in c_features_no_missing if c in df_filtered.columns]
    logger.info("Available C-no-missing features: %d / %d",
                len(available_feats), len(c_features_no_missing))

    X_c_nm = df_filtered[available_feats].to_numpy(dtype=np.float32)
    y_c_nm = df_filtered["label"].to_numpy(dtype=np.int32)
    groups_c_nm = df_filtered["guide_id"].to_numpy()

    n_guides = len(np.unique(groups_c_nm))
    n_pos = int(y_c_nm.sum())
    logger.info("C-no-missing data: %d samples, %d features, %d positives, %d guides",
                len(y_c_nm), len(available_feats), n_pos, n_guides)

    # Run full LOGO CV
    logger.info("Running LOGO CV for C-no-missing (n_splits=%d)...", n_guides)
    cv_c_no_missing = cross_validate_model(
        X_c_nm, y_c_nm, groups_c_nm,
        model_type="C_no_missing",
        feature_cols=available_feats,
        n_splits=n_guides,
        random_state=42,
    )

    print(f"\nModel C-no-missing LOGO CV: mean AUPRC = {cv_c_no_missing.mean_auprc:.6f} "
          f"± {cv_c_no_missing.std_auprc:.6f}")
    print(f"Model A              LOGO CV: mean AUPRC = {cv_results['A'].mean_auprc:.6f} "
          f"± {cv_results['A'].std_auprc:.6f}")
    print(f"Model C (with missing) LOGO CV: mean AUPRC = {cv_results['C'].mean_auprc:.6f} "
          f"± {cv_results['C'].std_auprc:.6f}")

    # Build per-guide AUPRC for C-no-missing
    cnm_guide_map: dict[str, dict] = {}
    for fold in cv_c_no_missing.folds:
        for gid in fold.test_guide_ids:
            cnm_guide_map[gid] = {
                "n_sites": fold.n_test,
                "n_positives": fold.n_pos_test,
                "auprc": fold.auprc,
            }

    cnm_rows = []
    for gid in sorted(cnm_guide_map.keys()):
        info = cnm_guide_map[gid]
        cnm_rows.append({
            "guide_id": gid,
            "n_sites": info["n_sites"],
            "n_positives": info["n_positives"],
            "auprc_C_no_missing": info["auprc"],
        })
    cnm_df = pd.DataFrame(cnm_rows)
    cnm_path = _RESULTS / "per_guide_auprc_expanded_v2_no_missing.csv"
    cnm_df.to_csv(cnm_path, index=False)
    logger.info("Per-guide AUPRC (C-no-missing) saved -> %s", cnm_path)

    # Wilcoxon A vs C-no-missing
    # Build paired arrays matched by guide_id
    a_map: dict[str, float] = {}
    for fold in cv_results["A"].folds:
        for gid in fold.test_guide_ids:
            a_map[gid] = fold.auprc
    cnm_auprc_map: dict[str, float] = {}
    for fold in cv_c_no_missing.folds:
        for gid in fold.test_guide_ids:
            cnm_auprc_map[gid] = fold.auprc

    shared_guides = sorted(set(a_map.keys()) & set(cnm_auprc_map.keys()))
    a_vals = np.array([a_map[g] for g in shared_guides])
    cnm_vals = np.array([cnm_auprc_map[g] for g in shared_guides])
    valid = ~(np.isnan(a_vals) | np.isnan(cnm_vals))
    a_valid = a_vals[valid]
    cnm_valid = cnm_vals[valid]

    diffs = cnm_valid - a_valid
    n_nonzero = int(np.count_nonzero(diffs))

    print(f"\nWilcoxon signed-rank test: A vs C-no-missing")
    print(f"  n_paired = {len(a_valid)}, n_nonzero_diffs = {n_nonzero}")

    if n_nonzero >= 6:
        _, wilcoxon_p = stats.wilcoxon(diffs)
        print(f"  Wilcoxon p (two-sided) = {wilcoxon_p:.6f}")
    else:
        wilcoxon_p = float("nan")
        print(f"  Wilcoxon p = NaN (fewer than 6 nonzero diffs)")

    _, ttest_p = stats.ttest_rel(a_valid, cnm_valid)
    print(f"  Paired t-test p (two-sided) = {ttest_p:.6f}")
    print(f"  Mean delta (C_nm - A) = {np.mean(diffs):.6f}")

    # ===================================================================
    # TASK 3: Per-cell-type breakdown for both C variants
    # ===================================================================
    print("\n" + "=" * 70)
    print("TASK 3: Per-cell-type breakdown")
    print("=" * 70)

    # We need cell_type for each guide. Get from feature matrix.
    guide_cell_type = (
        df_filtered.groupby("guide_id")["cell_type"]
        .first()
        .to_dict()
    )

    # Build per-guide maps for all models
    model_maps: dict[str, dict[str, float]] = {}
    for model_key in ["A", "B", "C"]:
        m: dict[str, float] = {}
        for fold in cv_results[model_key].folds:
            for gid in fold.test_guide_ids:
                m[gid] = fold.auprc
        model_maps[model_key] = m

    model_maps["C_no_missing"] = cnm_auprc_map

    cell_type_stats: list[dict] = []
    for ct in ["K562", "HEK293", "CD4_Tcell"]:
        ct_guides = [g for g in shared_guides if guide_cell_type.get(g) == ct]
        ct_guides_valid_a = [g for g in ct_guides
                             if not np.isnan(model_maps["A"].get(g, float("nan")))]

        a_ct = [model_maps["A"][g] for g in ct_guides_valid_a
                if g in model_maps["A"] and not np.isnan(model_maps["A"][g])]
        c_ct = [model_maps["C"][g] for g in ct_guides_valid_a
                if g in model_maps["C"] and not np.isnan(model_maps["C"][g])]
        cnm_ct = [model_maps["C_no_missing"][g] for g in ct_guides_valid_a
                  if g in model_maps["C_no_missing"]
                  and not np.isnan(model_maps["C_no_missing"][g])]

        cell_type_stats.append({
            "cell_type": ct,
            "n_guides": len(ct_guides_valid_a),
            "model_A": np.mean(a_ct) if a_ct else float("nan"),
            "model_C_with_missing": np.mean(c_ct) if c_ct else float("nan"),
            "model_C_no_missing": np.mean(cnm_ct) if cnm_ct else float("nan"),
        })

    print(f"\n{'Cell Type':<15} {'N Guides':<10} {'Model A':<12} "
          f"{'C (w/ miss)':<14} {'C (no miss)':<14}")
    print("-" * 65)
    for row in cell_type_stats:
        print(f"{row['cell_type']:<15} {row['n_guides']:<10} "
              f"{row['model_A']:<12.6f} "
              f"{row['model_C_with_missing']:<14.6f} "
              f"{row['model_C_no_missing']:<14.6f}")

    # ===================================================================
    # TASK 4: Export clean summary files
    # ===================================================================
    print("\n" + "=" * 70)
    print("TASK 4: Export clean summary files")
    print("=" * 70)

    # --- 4a: final_verification_summary.csv ---
    all_guides = sorted(set().union(
        model_maps["A"].keys(),
        model_maps["B"].keys(),
        model_maps["C"].keys(),
        model_maps["C_no_missing"].keys(),
    ))

    summary_rows: list[dict] = []
    for gid in all_guides:
        ct = guide_cell_type.get(gid, "unknown")

        # Get n_sites and n_positives from any model's fold data
        n_sites = None
        n_pos_guide = None
        for model_key in ["A", "B", "C"]:
            for fold in cv_results[model_key].folds:
                if gid in fold.test_guide_ids:
                    n_sites = fold.n_test
                    n_pos_guide = fold.n_pos_test
                    break
            if n_sites is not None:
                break

        summary_rows.append({
            "guide_id": gid,
            "cell_type": ct,
            "n_sites": n_sites,
            "n_positives": n_pos_guide,
            "auprc_A": model_maps["A"].get(gid, float("nan")),
            "auprc_B": model_maps["B"].get(gid, float("nan")),
            "auprc_C": model_maps["C"].get(gid, float("nan")),
            "auprc_C_no_missing": model_maps["C_no_missing"].get(gid, float("nan")),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = _RESULTS / "final_verification_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Final verification summary saved -> %s (%d rows)", summary_path, len(summary_df))
    print(f"Saved: {summary_path} ({len(summary_df)} guides)")

    # --- 4b: final_statistical_tests.csv ---
    # Run full statistical battery for all comparisons, overall and per cell type
    test_rows: list[dict] = []

    def run_tests(
        name: str,
        a_arr: np.ndarray,
        b_arr: np.ndarray,
        scope: str,
    ) -> None:
        """Compute Wilcoxon, t-test, permutation test and append to test_rows."""
        valid_mask = ~(np.isnan(a_arr) | np.isnan(b_arr))
        a_v = a_arr[valid_mask]
        b_v = b_arr[valid_mask]
        n = len(a_v)
        delta = float(np.mean(b_v) - np.mean(a_v)) if n > 0 else float("nan")

        # Paired t-test
        if n >= 2:
            _, tt_p = stats.ttest_rel(a_v, b_v)
            tt_p = float(tt_p)
        else:
            tt_p = float("nan")

        # Wilcoxon
        diffs_local = b_v - a_v
        n_nz = int(np.count_nonzero(diffs_local))
        if n_nz >= 6:
            _, wil_p = stats.wilcoxon(diffs_local)
            wil_p = float(wil_p)
        else:
            wil_p = float("nan")

        # Permutation test (10k)
        if n >= 2:
            rng = np.random.RandomState(42)
            observed_delta = float(np.mean(b_v) - np.mean(a_v))
            count_extreme = 0
            for _ in range(10000):
                signs = rng.choice([-1, 1], size=n)
                perm_delta = float(np.mean(diffs_local * signs))
                if abs(perm_delta) >= abs(observed_delta):
                    count_extreme += 1
            perm_p = (count_extreme + 1) / (10000 + 1)
        else:
            perm_p = float("nan")

        test_rows.append({
            "comparison": name,
            "scope": scope,
            "n_paired": n,
            "delta_mean_auprc": delta,
            "wilcoxon_p_two_sided": wil_p,
            "ttest_p_two_sided": tt_p,
            "permutation_p_two_sided_10k": perm_p,
        })

    # Build arrays for each comparison
    comparisons = [
        ("A_vs_C", "A", "C"),
        ("A_vs_C_no_missing", "A", "C_no_missing"),
        ("B_vs_C", "B", "C"),
        ("B_vs_C_no_missing", "B", "C_no_missing"),
    ]

    for comp_name, key_a, key_b in comparisons:
        # Overall
        shared = sorted(set(model_maps[key_a].keys()) & set(model_maps[key_b].keys()))
        arr_a = np.array([model_maps[key_a][g] for g in shared])
        arr_b = np.array([model_maps[key_b][g] for g in shared])
        run_tests(comp_name, arr_a, arr_b, "overall")

        # Per cell type
        for ct in ["K562", "CD4_Tcell"]:
            ct_shared = [g for g in shared if guide_cell_type.get(g) == ct]
            if len(ct_shared) < 2:
                continue
            arr_a_ct = np.array([model_maps[key_a][g] for g in ct_shared])
            arr_b_ct = np.array([model_maps[key_b][g] for g in ct_shared])
            run_tests(comp_name, arr_a_ct, arr_b_ct, ct)

    tests_df = pd.DataFrame(test_rows)
    tests_path = _RESULTS / "final_statistical_tests.csv"
    tests_df.to_csv(tests_path, index=False)
    logger.info("Final statistical tests saved -> %s (%d rows)", tests_path, len(tests_df))

    print(f"\nSaved: {tests_path} ({len(tests_df)} rows)")
    print("\nFull statistical tests table:")
    print(f"{'Comparison':<25} {'Scope':<12} {'N':<5} {'Delta':<10} "
          f"{'Wilcoxon p':<12} {'T-test p':<12} {'Perm p':<12}")
    print("-" * 88)
    for _, row in tests_df.iterrows():
        delta_str = f"{row['delta_mean_auprc']:.6f}" if not np.isnan(row['delta_mean_auprc']) else "NaN"
        wil_str = f"{row['wilcoxon_p_two_sided']:.6f}" if not np.isnan(row['wilcoxon_p_two_sided']) else "NaN"
        tt_str = f"{row['ttest_p_two_sided']:.6f}" if not np.isnan(row['ttest_p_two_sided']) else "NaN"
        perm_str = f"{row['permutation_p_two_sided_10k']:.6f}" if not np.isnan(row['permutation_p_two_sided_10k']) else "NaN"
        print(f"{row['comparison']:<25} {row['scope']:<12} {row['n_paired']:<5} "
              f"{delta_str:<10} {wil_str:<12} {tt_str:<12} {perm_str:<12}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
