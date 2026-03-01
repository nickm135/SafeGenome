"""
scripts/14_two_stage_residual.py — Two-stage residual model variants.

Forces AlphaGenome chromatin features to explain what sequence features miss
by training on the residuals of a sequence-only first stage.

Three Stage 2 configurations test overfitting vs absence of signal:

- **D-ridge**: Ridge regression (alpha=1.0) — simplest linear model.
- **D-stump**: XGBoost with max_depth=1, n_estimators=10, reg_lambda=10 —
  single-split stumps, heavily constrained.
- **D-lasso**: Lasso regression (alpha=0.01) — L1 sparsity reveals which
  AG features (if any) carry residual signal.

All share the same Stage 1 (XGBoost classifier on sequence features) and the
same LOGO CV folds as the canonical expanded_v2 pipeline.

Outputs
-------
    data/results/per_guide_auprc_two_stage.csv
    data/results/two_stage_statistical_tests.csv
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import (
    compute_scale_pos_weight,
    make_xgb_classifier,
    prepare_model_data,
    cross_validate_model,
)
from src.feature_matrix import (
    MODEL_A_FEATURES,
    get_model_features,
)
from src.alphagenome_client import FEATURE_NAMES as AG_FEATURE_NAMES
from src.utils import get_logger

logger = get_logger("two_stage_residual")

_ROOT = Path(__file__).resolve().parents[1]
_FEATURES = _ROOT / "data" / "features"
_RESULTS = _ROOT / "data" / "results"

EXCLUDE_DATASETS: list[str] = ["discover_seq_k562"]
MODEL_BC_CELL_TYPES: list[str] = ["K562", "HEK293", "CD4_Tcell"]
RANDOM_STATE: int = 42

#: The 12 AlphaGenome chromatin features for Stage 2 (no _missing indicators).
STAGE2_AG_FEATURES: list[str] = list(AG_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Stage 2 model factories
# ---------------------------------------------------------------------------

@dataclass
class Stage2Config:
    """Configuration for a Stage 2 residual model."""

    name: str
    short_label: str

    def make_model(self) -> Any:
        """Create a fresh Stage 2 model instance."""
        raise NotImplementedError

    def get_coefficients(self, model: Any, feature_names: list[str]) -> dict[str, float]:
        """Extract per-feature coefficients/importances from a fitted model.

        Returns
        -------
        dict[str, float]
            Feature name → coefficient (or importance). Empty dict if not
            applicable.
        """
        return {}


class RidgeConfig(Stage2Config):
    """Ridge regression (alpha=1.0)."""

    name: str = "D-ridge"
    short_label: str = "D_ridge"

    def make_model(self) -> Ridge:
        return Ridge(alpha=1.0)

    def get_coefficients(self, model: Ridge, feature_names: list[str]) -> dict[str, float]:
        return dict(zip(feature_names, model.coef_.tolist()))


class StumpConfig(Stage2Config):
    """XGBoost stumps (max_depth=1, n_estimators=10, reg_lambda=10)."""

    name: str = "D-stump"
    short_label: str = "D_stump"

    def make_model(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            n_estimators=10,
            max_depth=1,
            learning_rate=0.1,
            reg_lambda=10,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            verbosity=0,
            random_state=RANDOM_STATE,
        )

    def get_coefficients(
        self, model: xgb.XGBRegressor, feature_names: list[str],
    ) -> dict[str, float]:
        imp = model.feature_importances_
        return dict(zip(feature_names, imp.tolist()))


class LassoConfig(Stage2Config):
    """Lasso regression (alpha=0.01)."""

    name: str = "D-lasso"
    short_label: str = "D_lasso"

    def make_model(self) -> Lasso:
        return Lasso(alpha=0.01, max_iter=10_000)

    def get_coefficients(self, model: Lasso, feature_names: list[str]) -> dict[str, float]:
        return dict(zip(feature_names, model.coef_.tolist()))


STAGE2_CONFIGS: list[Stage2Config] = [
    RidgeConfig(name="D-ridge", short_label="D_ridge"),
    StumpConfig(name="D-stump", short_label="D_stump"),
    LassoConfig(name="D-lasso", short_label="D_lasso"),
]


# ---------------------------------------------------------------------------
# Two-stage residual CV (multi-variant)
# ---------------------------------------------------------------------------

def run_two_stage_logo_cv(
    df: pd.DataFrame,
    seq_features: list[str],
    ag_features: list[str],
    configs: list[Stage2Config],
    *,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, list[dict]], dict[str, list[dict[str, float]]]]:
    """Run LOGO CV for all two-stage variants sharing the same Stage 1.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix filtered to relevant cell types, DISCOVER-seq
        excluded, on-targets removed.
    seq_features : list[str]
        Sequence feature column names (Stage 1).
    ag_features : list[str]
        AlphaGenome feature column names (Stage 2).
    configs : list[Stage2Config]
        Stage 2 model configurations to evaluate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    all_results : dict[str, list[dict]]
        ``{config.short_label: [per-guide dicts]}``.
    all_coefficients : dict[str, list[dict[str, float]]]
        ``{config.short_label: [per-fold coefficient dicts]}``.
    """
    X_seq = df[seq_features].to_numpy(dtype=np.float32)
    X_ag = df[ag_features].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    groups = df["guide_id"].to_numpy()

    n_guides = len(np.unique(groups))
    logger.info(
        "Two-stage LOGO CV: %d samples, %d seq features, %d AG features, "
        "%d positives, %d guides, %d Stage 2 variants",
        len(y), len(seq_features), len(ag_features),
        int(y.sum()), n_guides, len(configs),
    )

    cv = StratifiedGroupKFold(
        n_splits=n_guides, shuffle=True, random_state=random_state,
    )

    all_results: dict[str, list[dict]] = {cfg.short_label: [] for cfg in configs}
    all_coefficients: dict[str, list[dict[str, float]]] = {
        cfg.short_label: [] for cfg in configs
    }

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_seq, y, groups=groups)):
        X_seq_train, X_seq_test = X_seq[train_idx], X_seq[test_idx]
        X_ag_train, X_ag_test = X_ag[train_idx], X_ag[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_guides = sorted(set(str(g) for g in groups[test_idx]))

        n_pos_test = int(y_test.sum())
        cell_type = df.iloc[test_idx[0]]["cell_type"]

        # --- Stage 1: sequence-only classifier (shared across variants) ---
        spw = compute_scale_pos_weight(y_train)
        clf_s1 = make_xgb_classifier(spw, random_state=random_state)
        clf_s1.fit(X_seq_train, y_train)

        p_hat_train = clf_s1.predict_proba(X_seq_train)[:, 1]
        p_hat_test = clf_s1.predict_proba(X_seq_test)[:, 1]

        # Residuals
        residuals_train = y_train.astype(np.float64) - p_hat_train.astype(np.float64)

        # --- Stage 2: one model per config ---
        fold_auprcs: list[str] = []
        for cfg in configs:
            s2_model = cfg.make_model()
            s2_model.fit(X_ag_train, residuals_train)

            correction_test = s2_model.predict(X_ag_test)
            final_score = np.clip(
                p_hat_test.astype(np.float64) + correction_test, 0.0, 1.0,
            )

            if n_pos_test == 0 or n_pos_test == len(y_test):
                auprc = float("nan")
            else:
                auprc = float(average_precision_score(y_test, final_score))

            for gid in test_guides:
                all_results[cfg.short_label].append({
                    "guide_id": gid,
                    "cell_type": cell_type,
                    "n_sites": len(y_test),
                    "n_positives": n_pos_test,
                    "auprc": auprc,
                })

            coefs = cfg.get_coefficients(s2_model, ag_features)
            all_coefficients[cfg.short_label].append(coefs)

            fold_auprcs.append(f"{cfg.name}={auprc:.4f}")

        logger.info(
            "  Fold %d [%s]: n_test=%d (%d pos), %s",
            fold_idx, ",".join(test_guides),
            len(y_test), n_pos_test, ", ".join(fold_auprcs),
        )

    return all_results, all_coefficients


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_permutations: int = 10_000,
    random_state: int = RANDOM_STATE,
) -> float:
    """Two-sided permutation test on paired differences.

    Parameters
    ----------
    a, b : np.ndarray
        Paired AUPRC arrays.
    n_permutations : int
        Number of random sign flips.
    random_state : int
        Random seed.

    Returns
    -------
    float
        Two-sided p-value.
    """
    diffs = b - a
    observed = float(np.abs(np.mean(diffs)))
    rng = np.random.RandomState(random_state)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        if abs(float(np.mean(diffs * signs))) >= observed:
            count += 1
    return (count + 1) / (n_permutations + 1)


def run_statistical_tests(
    map_a: dict[str, float],
    map_b: dict[str, float],
    guide_cell_types: dict[str, str],
    comparison_name: str,
) -> list[dict]:
    """Compute Wilcoxon, t-test, permutation test for a model pair.

    Parameters
    ----------
    map_a, map_b : dict[str, float]
        Guide ID -> AUPRC mappings for each model.
    guide_cell_types : dict[str, str]
        Guide ID -> cell type.
    comparison_name : str
        Label for this comparison.

    Returns
    -------
    list[dict]
        Rows for the output CSV (one per scope: overall + per cell type).
    """
    rows: list[dict] = []

    def _test(guides: list[str], scope: str) -> None:
        arr_a = np.array([map_a[g] for g in guides])
        arr_b = np.array([map_b[g] for g in guides])
        valid = ~(np.isnan(arr_a) | np.isnan(arr_b))
        a_v, b_v = arr_a[valid], arr_b[valid]
        n = len(a_v)
        delta = float(np.mean(b_v) - np.mean(a_v)) if n > 0 else float("nan")

        tt_p = float(stats.ttest_rel(a_v, b_v).pvalue) if n >= 2 else float("nan")

        diffs = b_v - a_v
        n_nz = int(np.count_nonzero(diffs))
        wil_p = float(stats.wilcoxon(diffs).pvalue) if n_nz >= 6 else float("nan")

        perm_p = run_permutation_test(a_v, b_v) if n >= 2 else float("nan")

        rows.append({
            "comparison": comparison_name,
            "scope": scope,
            "n_paired": n,
            "delta_mean_auprc": delta,
            "wilcoxon_p_two_sided": wil_p,
            "ttest_p_two_sided": tt_p,
            "permutation_p_two_sided_10k": perm_p,
        })

    shared = sorted(set(map_a) & set(map_b))
    _test(shared, "overall")

    for ct in ["K562", "HEK293", "CD4_Tcell"]:
        ct_guides = [g for g in shared if guide_cell_types.get(g) == ct]
        if len(ct_guides) >= 2:
            _test(ct_guides, ct)

    return rows


def _fmt(v: float) -> str:
    """Format a float or NaN for display."""
    return f"{v:.6f}" if not np.isnan(v) else "NaN"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the two-stage residual pipeline with three Stage 2 variants."""
    # Load feature matrix
    fm_path = _FEATURES / "feature_matrix_expanded.parquet"
    logger.info("Loading feature matrix: %s", fm_path)
    fm = pd.read_parquet(fm_path)
    logger.info("Shape: %d rows x %d cols", len(fm), len(fm.columns))

    # --- Prepare data (same filtering as script 12) ---
    df = fm.copy()

    if "dataset" in df.columns:
        mask = df["dataset"].isin(EXCLUDE_DATASETS)
        n_excl = int(mask.sum())
        logger.info("Excluding %d DISCOVER-seq rows", n_excl)
        df = df[~mask]

    if "cell_type" in df.columns:
        mask = df["cell_type"].isin(MODEL_BC_CELL_TYPES)
        n_drop = int((~mask).sum())
        logger.info("Dropping %d rows outside cell types %s", n_drop, MODEL_BC_CELL_TYPES)
        df = df[mask]

    n_ontarget = int((df["num_mismatches"] == 0).sum())
    if n_ontarget > 0:
        raise ValueError(f"On-target sites found: {n_ontarget}")

    n_guides = len(df["guide_id"].unique())
    n_pos = int(df["label"].sum())
    logger.info(
        "Final data: %d sites, %d guides, %d positives (%.1f%%)",
        len(df), n_guides, n_pos, 100.0 * n_pos / len(df),
    )

    seq_features = [c for c in MODEL_A_FEATURES if c in df.columns]
    ag_features = [c for c in STAGE2_AG_FEATURES if c in df.columns]
    logger.info("Seq features: %d/%d", len(seq_features), len(MODEL_A_FEATURES))
    logger.info("AG features: %d/%d", len(ag_features), len(STAGE2_AG_FEATURES))

    # --- Run Model A (standard LOGO CV for comparison) ---
    logger.info("=" * 60)
    logger.info("Running Model A (sequence-only) LOGO CV...")
    X_a, y_a, groups_a, feat_a = prepare_model_data(
        fm, "A",
        cell_types=MODEL_BC_CELL_TYPES,
        exclude_datasets=EXCLUDE_DATASETS,
        include_missingness_indicators=False,
    )
    cv_a = cross_validate_model(
        X_a, y_a, groups_a,
        model_type="A",
        feature_cols=feat_a,
        n_splits=n_guides,
        random_state=RANDOM_STATE,
    )

    # --- Run Model C (standard LOGO CV, no _missing, for comparison) ---
    logger.info("=" * 60)
    logger.info("Running Model C (seq+AG, no _missing) LOGO CV...")
    c_features = get_model_features("C", include_missingness_indicators=False)
    c_feats_available = [c for c in c_features if c in fm.columns]
    X_c = df[c_feats_available].to_numpy(dtype=np.float32)
    y_c = df["label"].to_numpy(dtype=np.int32)
    groups_c = df["guide_id"].to_numpy()
    cv_c = cross_validate_model(
        X_c, y_c, groups_c,
        model_type="C",
        feature_cols=c_feats_available,
        n_splits=n_guides,
        random_state=RANDOM_STATE,
    )

    # --- Run two-stage variants ---
    logger.info("=" * 60)
    logger.info("Running two-stage residual models (3 variants)...")
    d_all_results, d_all_coefficients = run_two_stage_logo_cv(
        df, seq_features, ag_features, STAGE2_CONFIGS,
        random_state=RANDOM_STATE,
    )

    # --- Build per-guide AUPRC maps ---
    guide_cell_type: dict[str, str] = (
        df.groupby("guide_id")["cell_type"].first().to_dict()
    )

    a_map: dict[str, float] = {}
    for fold in cv_a.folds:
        for gid in fold.test_guide_ids:
            a_map[gid] = fold.auprc

    c_map: dict[str, float] = {}
    for fold in cv_c.folds:
        for gid in fold.test_guide_ids:
            c_map[gid] = fold.auprc

    d_maps: dict[str, dict[str, float]] = {}
    for cfg in STAGE2_CONFIGS:
        d_maps[cfg.short_label] = {
            r["guide_id"]: r["auprc"] for r in d_all_results[cfg.short_label]
        }

    # --- Per-guide AUPRC table ---
    all_guides = sorted(
        set(a_map) | set(c_map)
        | set().union(*(dm.keys() for dm in d_maps.values()))
    )

    per_guide_rows: list[dict] = []
    for gid in all_guides:
        row: dict[str, Any] = {
            "guide_id": gid,
            "cell_type": guide_cell_type.get(gid, ""),
            "n_sites": next(
                (r["n_sites"] for r in d_all_results["D_ridge"]
                 if r["guide_id"] == gid),
                None,
            ),
            "n_positives": next(
                (r["n_positives"] for r in d_all_results["D_ridge"]
                 if r["guide_id"] == gid),
                None,
            ),
            "auprc_A": a_map.get(gid, float("nan")),
            "auprc_C": c_map.get(gid, float("nan")),
        }
        for cfg in STAGE2_CONFIGS:
            row[f"auprc_{cfg.short_label}"] = d_maps[cfg.short_label].get(
                gid, float("nan"),
            )
        per_guide_rows.append(row)

    per_guide_df = pd.DataFrame(per_guide_rows)
    per_guide_path = _RESULTS / "per_guide_auprc_two_stage.csv"
    _RESULTS.mkdir(parents=True, exist_ok=True)
    per_guide_df.to_csv(per_guide_path, index=False)
    logger.info("Per-guide AUPRC saved -> %s", per_guide_path)

    # ===================================================================
    # Console output
    # ===================================================================

    print("\n" + "=" * 80)
    print("TWO-STAGE RESIDUAL MODEL RESULTS (3 Stage 2 variants)")
    print("=" * 80)

    # --- Overall summary ---
    def _mean_std(m: dict[str, float]) -> tuple[float, float]:
        vals = np.array([m[g] for g in all_guides if g in m])
        vals = vals[~np.isnan(vals)]
        return float(np.mean(vals)), float(np.std(vals))

    print("\nOverall mean AUPRC:")
    a_mean, a_std = _mean_std(a_map)
    c_mean, c_std = _mean_std(c_map)
    print(f"  Model A (seq-only):        {a_mean:.6f} +/- {a_std:.6f}")
    print(f"  Model C (seq+AG, no miss): {c_mean:.6f} +/- {c_std:.6f}")
    for cfg in STAGE2_CONFIGS:
        dm, ds = _mean_std(d_maps[cfg.short_label])
        print(f"  {cfg.name:<29s}{dm:.6f} +/- {ds:.6f}")

    # --- Per cell type ---
    header = f"{'Cell Type':<15} {'N':<5} {'A':<10} {'C':<10}"
    for cfg in STAGE2_CONFIGS:
        header += f" {cfg.name:<10}"
    print(f"\n{header}")
    print("-" * len(header))

    for ct in ["K562", "HEK293", "CD4_Tcell"]:
        ct_guides = [g for g in all_guides if guide_cell_type.get(g) == ct]
        a_ct = np.array([a_map.get(g, float("nan")) for g in ct_guides])
        a_ct = a_ct[~np.isnan(a_ct)]
        c_ct = np.array([c_map.get(g, float("nan")) for g in ct_guides])
        c_ct = c_ct[~np.isnan(c_ct)]
        line = f"{ct:<15} {len(a_ct):<5} {np.mean(a_ct):<10.6f} {np.mean(c_ct):<10.6f}"
        for cfg in STAGE2_CONFIGS:
            d_ct = np.array([
                d_maps[cfg.short_label].get(g, float("nan")) for g in ct_guides
            ])
            d_ct = d_ct[~np.isnan(d_ct)]
            line += f" {np.mean(d_ct):<10.6f}"
        print(line)

    # --- Statistical tests ---
    test_rows: list[dict] = []
    for cfg in STAGE2_CONFIGS:
        test_rows.extend(run_statistical_tests(
            a_map, d_maps[cfg.short_label], guide_cell_type, f"A_vs_{cfg.short_label}",
        ))
        test_rows.extend(run_statistical_tests(
            c_map, d_maps[cfg.short_label], guide_cell_type, f"C_vs_{cfg.short_label}",
        ))

    tests_df = pd.DataFrame(test_rows)
    tests_path = _RESULTS / "two_stage_regularized_tests.csv"
    tests_df.to_csv(tests_path, index=False)
    logger.info("Statistical tests saved -> %s", tests_path)

    print(f"\n{'Comparison':<20} {'Scope':<12} {'N':<5} {'Delta':<10} "
          f"{'Wilcoxon p':<12} {'T-test p':<12} {'Perm p':<12}")
    print("-" * 83)
    for _, row in tests_df.iterrows():
        print(
            f"{row['comparison']:<20} {row['scope']:<12} {row['n_paired']:<5} "
            f"{_fmt(row['delta_mean_auprc']):<10} "
            f"{_fmt(row['wilcoxon_p_two_sided']):<12} "
            f"{_fmt(row['ttest_p_two_sided']):<12} "
            f"{_fmt(row['permutation_p_two_sided_10k']):<12}"
        )

    # --- Coefficient / feature importance analysis ---
    print("\n" + "=" * 80)
    print("AG FEATURE COEFFICIENTS (mean across CV folds)")
    print("=" * 80)

    for cfg in STAGE2_CONFIGS:
        coef_dicts = d_all_coefficients[cfg.short_label]
        if not coef_dicts or not coef_dicts[0]:
            continue

        # Average coefficients across folds
        coef_df = pd.DataFrame(coef_dicts)
        mean_coefs = coef_df.mean().sort_values(key=lambda s: s.abs(), ascending=False)

        print(f"\n--- {cfg.name} ---")
        print(f"{'Feature':<30} {'Mean coef/imp':<15} {'Std':<15} {'Non-zero folds'}")
        print("-" * 75)
        for feat in mean_coefs.index:
            vals = coef_df[feat].values
            n_nonzero = int(np.count_nonzero(vals))
            print(
                f"{feat:<30} {mean_coefs[feat]:<15.6f} "
                f"{coef_df[feat].std():<15.6f} {n_nonzero}/{len(vals)}"
            )

    # --- Lasso non-zero features ---
    print("\n" + "=" * 80)
    print("LASSO NON-ZERO AG FEATURES (per fold)")
    print("=" * 80)

    lasso_coefs = d_all_coefficients["D_lasso"]
    if lasso_coefs:
        lasso_df = pd.DataFrame(lasso_coefs)
        nonzero_counts = (lasso_df.abs() > 1e-10).sum()
        print(f"\n{'Feature':<30} {'Folds with nonzero coef':<25} {'Mean coef (when nonzero)'}")
        print("-" * 80)
        for feat in nonzero_counts.sort_values(ascending=False).index:
            n_nz = int(nonzero_counts[feat])
            if n_nz > 0:
                nz_vals = lasso_df[feat][lasso_df[feat].abs() > 1e-10]
                mean_nz = float(nz_vals.mean())
            else:
                mean_nz = 0.0
            marker = " ***" if n_nz == len(lasso_coefs) else ""
            print(f"{feat:<30} {n_nz}/{len(lasso_coefs):<22} {mean_nz:<+.8f}{marker}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
