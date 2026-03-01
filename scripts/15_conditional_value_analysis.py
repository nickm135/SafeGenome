"""
scripts/15_conditional_value_analysis.py — Conditional value analysis.

Asks: "Among sites that look similar by sequence alone, can chromatin break
ties correctly?" rather than testing overall AUPRC improvement.

Three analyses:
    1. Within-CFD-bin discrimination (quintiles + fixed bins)
    2. Hard vs easy guide analysis
    3. Site-level rank improvement for true positives

Requires site-level OOF predictions for Models A and C (no _missing).
Reruns LOGO CV to obtain predictions aligned with the feature matrix.

Outputs:
    data/results/within_bin_analysis.csv
    data/results/hard_vs_easy_guide_analysis.csv
    data/results/rank_improvement_positives.csv
    figures/fig_within_bin.png
    figures/fig_rank_improvement.png
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import (
    compute_scale_pos_weight,
    make_xgb_classifier,
)
from src.feature_matrix import (
    get_model_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("conditional_value")

_ROOT = Path(__file__).resolve().parents[1]
_FEATURES = _ROOT / "data" / "features"
_RESULTS = _ROOT / "data" / "results"
_FIGURES = _ROOT / "figures"

EXCLUDE_DATASETS = ["discover_seq_k562"]
MODEL_BC_CELL_TYPES = ["K562", "HEK293", "CD4_Tcell"]
RANDOM_STATE = 42
MIN_POS_FOR_AUPRC = 5  # Skip bins with < 5 positives


# -----------------------------------------------------------------------
# Helper: run LOGO CV returning site-level aligned OOF predictions
# -----------------------------------------------------------------------

def run_logo_cv_site_aligned(
    df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
) -> np.ndarray:
    """Run LOGO CV and return OOF predictions aligned with df rows.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered feature matrix with ``guide_id``, ``label``, and all
        ``feature_cols`` present.
    feature_cols : list[str]
        Feature columns to use.
    label : str
        Model label for logging.

    Returns
    -------
    np.ndarray
        OOF predicted probabilities, same length as df, aligned by index.
    """
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    groups = df["guide_id"].to_numpy()

    n_guides = len(np.unique(groups))
    oof_preds = np.full(len(y), np.nan, dtype=np.float64)

    cv = StratifiedGroupKFold(
        n_splits=n_guides, shuffle=True, random_state=RANDOM_STATE
    )

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        spw = compute_scale_pos_weight(y_train)
        clf = make_xgb_classifier(spw, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:, 1]

        oof_preds[test_idx] = y_score

        test_gids = sorted(set(str(g) for g in groups[test_idx]))
        n_pos = int(y[test_idx].sum())
        logger.info(
            "  %s fold %d [%s]: n_test=%d, n_pos=%d",
            label, fold_idx, ",".join(test_gids), len(test_idx), n_pos,
        )

    assert not np.any(np.isnan(oof_preds)), "Some sites were not predicted"
    logger.info(
        "%s LOGO CV complete: %d sites, %d positives",
        label, len(y), int(y.sum()),
    )
    return oof_preds


# -----------------------------------------------------------------------
# Analysis 1: Within-CFD-bin discrimination
# -----------------------------------------------------------------------

def within_bin_analysis(
    df: pd.DataFrame,
    pred_a: np.ndarray,
    pred_c: np.ndarray,
) -> pd.DataFrame:
    """Compute within-CFD-bin AUPRC for Models A and C.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with ``cfd_score``, ``label``, ``guide_id``.
    pred_a, pred_c : np.ndarray
        Site-level OOF predictions.

    Returns
    -------
    pd.DataFrame
        One row per bin with AUPRC_A, AUPRC_C, delta, etc.
    """
    work = df[["cfd_score", "label", "guide_id", "cell_type"]].copy()
    work["pred_a"] = pred_a
    work["pred_c"] = pred_c

    # --- Fixed bins ---
    fixed_edges = [0.0, 0.05, 0.1, 0.2, 0.5, 1.001]
    fixed_labels = ["[0, 0.05)", "[0.05, 0.1)", "[0.1, 0.2)", "[0.2, 0.5)", "[0.5, 1.0]"]
    work["fixed_bin"] = pd.cut(
        work["cfd_score"], bins=fixed_edges, labels=fixed_labels,
        right=False, include_lowest=True,
    )

    # --- Quintile bins ---
    work["quintile_bin"] = pd.qcut(
        work["cfd_score"], q=5, duplicates="drop",
    )

    rows: list[dict] = []

    for bin_type, bin_col in [("fixed", "fixed_bin"), ("quintile", "quintile_bin")]:
        for bin_label, grp in work.groupby(bin_col, observed=True):
            n_sites = len(grp)
            n_pos = int(grp["label"].sum())
            pos_rate = n_pos / n_sites if n_sites > 0 else 0.0
            cfd_min = float(grp["cfd_score"].min())
            cfd_max = float(grp["cfd_score"].max())
            cfd_median = float(grp["cfd_score"].median())

            logger.info(
                "Bin %s [%s]: %d sites, %d positives (%.1f%%), CFD range [%.4f, %.4f]",
                bin_type, bin_label, n_sites, n_pos, 100 * pos_rate, cfd_min, cfd_max,
            )

            row: dict = {
                "bin_type": bin_type,
                "bin_label": str(bin_label),
                "n_sites": n_sites,
                "n_positives": n_pos,
                "positive_rate": pos_rate,
                "cfd_min": cfd_min,
                "cfd_max": cfd_max,
                "cfd_median": cfd_median,
            }

            if n_pos < MIN_POS_FOR_AUPRC:
                logger.info("  Skipping AUPRC (n_pos=%d < %d)", n_pos, MIN_POS_FOR_AUPRC)
                row.update({
                    "auprc_A": np.nan, "auprc_C": np.nan, "delta_C_minus_A": np.nan,
                    "C_better": np.nan,
                    "wilcoxon_p": np.nan, "n_guides_tested": 0,
                })
                rows.append(row)
                continue

            y_bin = grp["label"].to_numpy()
            auprc_a = float(average_precision_score(y_bin, grp["pred_a"].to_numpy()))
            auprc_c = float(average_precision_score(y_bin, grp["pred_c"].to_numpy()))
            delta = auprc_c - auprc_a

            row.update({
                "auprc_A": auprc_a,
                "auprc_C": auprc_c,
                "delta_C_minus_A": delta,
                "C_better": int(delta > 0),
            })

            # Per-guide AUPRC within this bin (for Wilcoxon)
            if n_pos >= 10:
                guide_a_vals: list[float] = []
                guide_c_vals: list[float] = []
                for gid, ggrp in grp.groupby("guide_id"):
                    g_n_pos = int(ggrp["label"].sum())
                    g_n_total = len(ggrp)
                    if g_n_pos == 0 or g_n_pos == g_n_total:
                        continue  # AUPRC undefined
                    g_y = ggrp["label"].to_numpy()
                    ga = float(average_precision_score(g_y, ggrp["pred_a"].to_numpy()))
                    gc = float(average_precision_score(g_y, ggrp["pred_c"].to_numpy()))
                    guide_a_vals.append(ga)
                    guide_c_vals.append(gc)

                n_tested = len(guide_a_vals)
                diffs = np.array(guide_c_vals) - np.array(guide_a_vals)
                n_nonzero = int(np.count_nonzero(diffs))

                if n_nonzero >= 6:
                    _, wil_p = stats.wilcoxon(diffs)
                    wil_p = float(wil_p)
                else:
                    wil_p = np.nan

                row.update({
                    "wilcoxon_p": wil_p,
                    "n_guides_tested": n_tested,
                })
                logger.info(
                    "  AUPRC A=%.4f, C=%.4f, delta=%.4f, Wilcoxon p=%.4f (n_guides=%d)",
                    auprc_a, auprc_c, delta,
                    wil_p if not np.isnan(wil_p) else -1, n_tested,
                )
            else:
                row.update({"wilcoxon_p": np.nan, "n_guides_tested": 0})
                logger.info(
                    "  AUPRC A=%.4f, C=%.4f, delta=%.4f (n_pos=%d, skipping Wilcoxon)",
                    auprc_a, auprc_c, delta, n_pos,
                )

            rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Analysis 2: Hard vs easy guide analysis
# -----------------------------------------------------------------------

def hard_vs_easy_analysis(
    df: pd.DataFrame,
    pred_a: np.ndarray,
    pred_c: np.ndarray,
) -> pd.DataFrame:
    """Compare C-A delta between hard and easy guides.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with ``guide_id``, ``label``, ``cell_type``.
    pred_a, pred_c : np.ndarray
        Site-level OOF predictions.

    Returns
    -------
    pd.DataFrame
        Summary with hard/easy split statistics.
    """
    work = df[["guide_id", "label", "cell_type"]].copy()
    work["pred_a"] = pred_a
    work["pred_c"] = pred_c

    # Per-guide AUPRC
    guide_stats: list[dict] = []
    for gid, ggrp in work.groupby("guide_id"):
        n_pos = int(ggrp["label"].sum())
        n_total = len(ggrp)
        ct = ggrp["cell_type"].iloc[0]
        if n_pos == 0 or n_pos == n_total:
            continue
        g_y = ggrp["label"].to_numpy()
        ga = float(average_precision_score(g_y, ggrp["pred_a"].to_numpy()))
        gc = float(average_precision_score(g_y, ggrp["pred_c"].to_numpy()))
        guide_stats.append({
            "guide_id": gid,
            "cell_type": ct,
            "n_sites": n_total,
            "n_positives": n_pos,
            "auprc_A": ga,
            "auprc_C": gc,
            "delta_C_A": gc - ga,
        })

    guide_df = pd.DataFrame(guide_stats)

    # Hard = A < 0.5, Easy = A >= 0.5
    guide_df["difficulty"] = np.where(guide_df["auprc_A"] < 0.5, "hard", "easy")

    rows: list[dict] = []
    for diff_label in ["hard", "easy", "all"]:
        if diff_label == "all":
            sub = guide_df
        else:
            sub = guide_df[guide_df["difficulty"] == diff_label]

        if len(sub) == 0:
            continue

        n_guides = len(sub)
        mean_a = float(sub["auprc_A"].mean())
        mean_c = float(sub["auprc_C"].mean())
        mean_delta = float(sub["delta_C_A"].mean())
        median_delta = float(sub["delta_C_A"].median())
        n_c_better = int((sub["delta_C_A"] > 0).sum())

        diffs = sub["delta_C_A"].to_numpy()
        n_nonzero = int(np.count_nonzero(diffs))
        if n_nonzero >= 6:
            _, wil_p = stats.wilcoxon(diffs)
            wil_p = float(wil_p)
        else:
            wil_p = np.nan

        rows.append({
            "group": diff_label,
            "n_guides": n_guides,
            "mean_auprc_A": mean_a,
            "mean_auprc_C": mean_c,
            "mean_delta_C_A": mean_delta,
            "median_delta_C_A": median_delta,
            "n_C_better": n_c_better,
            "frac_C_better": n_c_better / n_guides,
            "wilcoxon_p": wil_p,
        })

        logger.info(
            "  %s: n=%d, A=%.4f, C=%.4f, delta=%.4f, C>A=%d/%d, Wilcoxon p=%.4f",
            diff_label, n_guides, mean_a, mean_c, mean_delta,
            n_c_better, n_guides, wil_p if not np.isnan(wil_p) else -1,
        )

    # Per-cell-type hard/easy
    for ct in ["K562", "HEK293", "CD4_Tcell"]:
        ct_sub = guide_df[guide_df["cell_type"] == ct]
        if len(ct_sub) == 0:
            continue
        for diff_label in ["hard", "easy"]:
            sub = ct_sub[ct_sub["difficulty"] == diff_label]
            if len(sub) == 0:
                continue
            n_guides = len(sub)
            mean_delta = float(sub["delta_C_A"].mean())
            n_c_better = int((sub["delta_C_A"] > 0).sum())
            diffs = sub["delta_C_A"].to_numpy()
            n_nonzero = int(np.count_nonzero(diffs))
            if n_nonzero >= 6:
                _, wil_p = stats.wilcoxon(diffs)
                wil_p = float(wil_p)
            else:
                wil_p = np.nan

            rows.append({
                "group": f"{ct}_{diff_label}",
                "n_guides": n_guides,
                "mean_auprc_A": float(sub["auprc_A"].mean()),
                "mean_auprc_C": float(sub["auprc_C"].mean()),
                "mean_delta_C_A": mean_delta,
                "median_delta_C_A": float(sub["delta_C_A"].median()),
                "n_C_better": n_c_better,
                "frac_C_better": n_c_better / n_guides,
                "wilcoxon_p": wil_p,
            })

    return pd.DataFrame(rows), guide_df


# -----------------------------------------------------------------------
# Analysis 3: Site-level rank improvement for positives
# -----------------------------------------------------------------------

def rank_improvement_analysis(
    df: pd.DataFrame,
    pred_a: np.ndarray,
    pred_c: np.ndarray,
) -> pd.DataFrame:
    """Compare ranks of true positives under Models A and C.

    Within each guide's candidate set, rank sites by predicted score
    (rank 1 = highest risk). For true positives only, test whether
    Model C ranks them higher (lower rank number) than Model A.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with ``guide_id``, ``label``, ``cell_type``.
    pred_a, pred_c : np.ndarray
        Site-level OOF predictions.

    Returns
    -------
    pd.DataFrame
        Rank comparison statistics, overall and per cell type.
    """
    work = df[["guide_id", "label", "cell_type"]].copy()
    work["pred_a"] = pred_a
    work["pred_c"] = pred_c

    # Compute within-guide ranks (1 = highest predicted risk)
    work["rank_a"] = work.groupby("guide_id")["pred_a"].rank(
        ascending=False, method="average"
    )
    work["rank_c"] = work.groupby("guide_id")["pred_c"].rank(
        ascending=False, method="average"
    )

    positives = work[work["label"] == 1].copy()
    positives["rank_diff"] = positives["rank_a"] - positives["rank_c"]
    # positive rank_diff = C ranks it higher (better)

    rows: list[dict] = []

    for scope in ["overall", "K562", "HEK293", "CD4_Tcell"]:
        if scope == "overall":
            sub = positives
        else:
            sub = positives[positives["cell_type"] == scope]

        if len(sub) == 0:
            continue

        n = len(sub)
        mean_rank_a = float(sub["rank_a"].mean())
        mean_rank_c = float(sub["rank_c"].mean())
        mean_rank_diff = float(sub["rank_diff"].mean())
        median_rank_diff = float(sub["rank_diff"].median())
        n_improved = int((sub["rank_diff"] > 0).sum())
        n_worsened = int((sub["rank_diff"] < 0).sum())
        n_tied = int((sub["rank_diff"] == 0).sum())

        # Wilcoxon signed-rank on rank differences
        diffs = sub["rank_diff"].to_numpy()
        n_nonzero = int(np.count_nonzero(diffs))
        if n_nonzero >= 6:
            _, wil_p = stats.wilcoxon(diffs)
            wil_p = float(wil_p)
        else:
            wil_p = np.nan

        rows.append({
            "scope": scope,
            "n_positives": n,
            "mean_rank_A": mean_rank_a,
            "mean_rank_C": mean_rank_c,
            "mean_rank_improvement": mean_rank_diff,
            "median_rank_improvement": median_rank_diff,
            "n_improved": n_improved,
            "n_worsened": n_worsened,
            "n_tied": n_tied,
            "frac_improved": n_improved / n,
            "wilcoxon_p": wil_p,
        })

        logger.info(
            "  Rank %s: n=%d, mean_rank A=%.1f C=%.1f, improvement=%.2f, "
            "improved=%d worsened=%d tied=%d, Wilcoxon p=%.4f",
            scope, n, mean_rank_a, mean_rank_c, mean_rank_diff,
            n_improved, n_worsened, n_tied,
            wil_p if not np.isnan(wil_p) else -1,
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------

def plot_within_bin(
    bin_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Two-panel figure: bar charts of AUPRC per CFD bin and delta.

    Parameters
    ----------
    bin_df : pd.DataFrame
        Output of ``within_bin_analysis``, filtered to fixed bins only.
    out_path : Path
        Output file path.
    """
    fixed = bin_df[bin_df["bin_type"] == "fixed"].copy()
    fixed = fixed[fixed["n_positives"] >= MIN_POS_FOR_AUPRC].copy()

    if len(fixed) == 0:
        logger.warning("No fixed bins with >= %d positives; skipping figure", MIN_POS_FOR_AUPRC)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    labels = fixed["bin_label"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    auprc_a = fixed["auprc_A"].to_numpy()
    auprc_c = fixed["auprc_C"].to_numpy()
    n_pos = fixed["n_positives"].to_numpy()

    # Left panel: grouped bar chart
    bars_a = ax1.bar(x - width / 2, auprc_a, width, label="Model A (seq-only)",
                     color="#4472C4", edgecolor="black", linewidth=0.5)
    bars_c = ax1.bar(x + width / 2, auprc_c, width, label="Model C (seq+AG)",
                     color="#ED7D31", edgecolor="black", linewidth=0.5)

    # Annotate n_positives above each bin group
    for i, (xa, xc, n) in enumerate(zip(x - width / 2, x + width / 2, n_pos)):
        y_max = max(auprc_a[i], auprc_c[i])
        ax1.text(x[i], y_max + 0.02, f"n={n}", ha="center", va="bottom",
                 fontsize=8, fontweight="bold")

    ax1.set_xlabel("CFD Score Bin", fontsize=11)
    ax1.set_ylabel("AUPRC (within bin)", fontsize=11)
    ax1.set_title("Model Performance Within CFD Score Bins", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_ylim(0, min(1.0, max(max(auprc_a), max(auprc_c)) * 1.25))

    # Right panel: delta bar chart
    delta = fixed["delta_C_minus_A"].to_numpy()
    colors = ["#2E8B57" if d > 0 else "#CD5C5C" for d in delta]
    ax2.bar(x, delta, width=0.6, color=colors, edgecolor="black", linewidth=0.5)
    ax2.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax2.set_xlabel("CFD Score Bin", fontsize=11)
    ax2.set_ylabel("AUPRC Delta (C - A)", fontsize=11)
    ax2.set_title("Chromatin Value Added Per Bin", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)

    # Annotate delta values
    for i, d in enumerate(delta):
        va = "bottom" if d >= 0 else "top"
        offset = 0.005 if d >= 0 else -0.005
        ax2.text(x[i], d + offset, f"{d:+.3f}", ha="center", va=va, fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved within-bin figure -> %s", out_path)


def plot_rank_improvement(
    df: pd.DataFrame,
    pred_a: np.ndarray,
    pred_c: np.ndarray,
    out_path: Path,
) -> None:
    """Scatter plot of per-site rank under A vs C for true positives.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with ``guide_id``, ``label``, ``cell_type``.
    pred_a, pred_c : np.ndarray
        Site-level OOF predictions.
    out_path : Path
        Output file path.
    """
    work = df[["guide_id", "label", "cell_type"]].copy()
    work["pred_a"] = pred_a
    work["pred_c"] = pred_c

    # Within-guide percentile ranks (0-100, lower = higher risk)
    work["pctrank_a"] = work.groupby("guide_id")["pred_a"].rank(
        ascending=False, pct=True
    ) * 100
    work["pctrank_c"] = work.groupby("guide_id")["pred_c"].rank(
        ascending=False, pct=True
    ) * 100

    positives = work[work["label"] == 1].copy()

    # Also include hard/easy annotation
    guide_a_auprc: dict[str, float] = {}
    for gid, ggrp in work.groupby("guide_id"):
        n_pos = int(ggrp["label"].sum())
        if n_pos == 0 or n_pos == len(ggrp):
            continue
        g_y = ggrp["label"].to_numpy()
        ga = float(average_precision_score(g_y, ggrp["pred_a"].to_numpy()))
        guide_a_auprc[gid] = ga
    positives["guide_difficulty"] = positives["guide_id"].map(
        lambda g: "hard" if guide_a_auprc.get(g, 0) < 0.5 else "easy"
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    ct_colors = {"K562": "#4472C4", "HEK293": "#ED7D31", "CD4_Tcell": "#70AD47"}
    ct_labels_done: set[str] = set()

    for _, row in positives.iterrows():
        ct = row["cell_type"]
        color = ct_colors.get(ct, "gray")
        lbl = ct if ct not in ct_labels_done else None
        if lbl:
            ct_labels_done.add(ct)
        ax.scatter(
            row["pctrank_a"], row["pctrank_c"],
            c=color, alpha=0.5, s=20, edgecolors="none",
            label=lbl,
        )

    # Diagonal line
    ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, alpha=0.5)

    # Annotate regions
    ax.text(80, 20, "C improved\n(ranked higher)", fontsize=9, ha="center",
            color="#2E8B57", fontstyle="italic")
    ax.text(20, 80, "A better\n(C ranked lower)", fontsize=9, ha="center",
            color="#CD5C5C", fontstyle="italic")

    ax.set_xlabel("Percentile rank under Model A (lower = higher risk)", fontsize=11)
    ax.set_ylabel("Percentile rank under Model C (lower = higher risk)", fontsize=11)
    ax.set_title("Rank of True Positives: Model A vs Model C", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved rank improvement figure -> %s", out_path)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    _RESULTS.mkdir(parents=True, exist_ok=True)
    _FIGURES.mkdir(parents=True, exist_ok=True)

    # ----- Load feature matrix -----
    fm_path = _FEATURES / "feature_matrix_expanded.parquet"
    logger.info("Loading feature matrix: %s", fm_path)
    fm = pd.read_parquet(fm_path)
    logger.info("Feature matrix: %d rows x %d cols", len(fm), len(fm.columns))

    # ----- Filter for canonical LOGO CV -----
    df = fm.copy()
    n0 = len(df)

    # Exclude DISCOVER-seq
    if "dataset" in df.columns:
        mask = df["dataset"].isin(EXCLUDE_DATASETS)
        df = df[~mask]
        logger.info("Excluded %d DISCOVER-seq rows: %d -> %d", int(mask.sum()), n0, len(df))

    # Filter cell types
    mask = df["cell_type"].isin(MODEL_BC_CELL_TYPES)
    n1 = len(df)
    df = df[mask]
    logger.info("Cell type filter: %d -> %d", n1, len(df))

    # Verify no on-targets
    n_ontarget = int((df["num_mismatches"] == 0).sum())
    assert n_ontarget == 0, f"On-target sites found: {n_ontarget}"

    n_sites = len(df)
    n_pos = int(df["label"].sum())
    n_guides = df["guide_id"].nunique()
    logger.info("Canonical dataset: %d sites, %d positives (%.1f%%), %d guides",
                n_sites, n_pos, 100 * n_pos / n_sites, n_guides)

    # Reset index for alignment
    df = df.reset_index(drop=True)

    # ----- Get feature columns -----
    feat_a = get_model_features("A", include_missingness_indicators=False)
    feat_a = [c for c in feat_a if c in df.columns]

    feat_c = get_model_features("C", include_missingness_indicators=False)
    feat_c = [c for c in feat_c if c in df.columns]

    logger.info("Model A features: %d", len(feat_a))
    logger.info("Model C features (no _missing): %d", len(feat_c))

    # ----- Run LOGO CV for both models -----
    print("\n" + "=" * 70)
    print("Running LOGO CV for Model A (site-aligned OOF predictions)")
    print("=" * 70)
    pred_a = run_logo_cv_site_aligned(df, feat_a, "Model_A")

    print("\n" + "=" * 70)
    print("Running LOGO CV for Model C-no-missing (site-aligned OOF predictions)")
    print("=" * 70)
    pred_c = run_logo_cv_site_aligned(df, feat_c, "Model_C_nm")

    # Quick sanity: overall AUPRC
    overall_a = average_precision_score(df["label"].to_numpy(), pred_a)
    overall_c = average_precision_score(df["label"].to_numpy(), pred_c)
    print(f"\nOOF pooled AUPRC: Model A = {overall_a:.4f}, Model C = {overall_c:.4f}")

    # ===================================================================
    # ANALYSIS 1: Within-CFD-bin discrimination
    # ===================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Within-CFD-bin discrimination")
    print("=" * 70)

    bin_df = within_bin_analysis(df, pred_a, pred_c)
    bin_path = _RESULTS / "within_bin_analysis.csv"
    bin_df.to_csv(bin_path, index=False)
    logger.info("Saved within-bin analysis -> %s", bin_path)

    # Print fixed-bin results
    fixed_bins = bin_df[bin_df["bin_type"] == "fixed"].copy()
    print(f"\nFixed CFD bins (min {MIN_POS_FOR_AUPRC} positives required):")
    print(f"{'Bin':<16} {'N sites':<9} {'N pos':<7} {'AUPRC_A':<10} {'AUPRC_C':<10} "
          f"{'Delta':<10} {'Wilcoxon p':<12} {'N guides':<10}")
    print("-" * 84)
    for _, row in fixed_bins.iterrows():
        a_str = f"{row['auprc_A']:.4f}" if not np.isnan(row['auprc_A']) else "skip"
        c_str = f"{row['auprc_C']:.4f}" if not np.isnan(row['auprc_C']) else "skip"
        d_str = f"{row['delta_C_minus_A']:+.4f}" if not np.isnan(row['delta_C_minus_A']) else "skip"
        w_str = f"{row['wilcoxon_p']:.4f}" if not np.isnan(row['wilcoxon_p']) else "N/A"
        print(f"{row['bin_label']:<16} {row['n_sites']:<9} {row['n_positives']:<7} "
              f"{a_str:<10} {c_str:<10} {d_str:<10} {w_str:<12} {int(row['n_guides_tested']):<10}")

    # Print quintile results
    quint_bins = bin_df[bin_df["bin_type"] == "quintile"].copy()
    print(f"\nQuintile CFD bins:")
    print(f"{'Bin':<25} {'N sites':<9} {'N pos':<7} {'AUPRC_A':<10} {'AUPRC_C':<10} "
          f"{'Delta':<10}")
    print("-" * 71)
    for _, row in quint_bins.iterrows():
        a_str = f"{row['auprc_A']:.4f}" if not np.isnan(row['auprc_A']) else "skip"
        c_str = f"{row['auprc_C']:.4f}" if not np.isnan(row['auprc_C']) else "skip"
        d_str = f"{row['delta_C_minus_A']:+.4f}" if not np.isnan(row['delta_C_minus_A']) else "skip"
        print(f"{row['bin_label']:<25} {row['n_sites']:<9} {row['n_positives']:<7} "
              f"{a_str:<10} {c_str:<10} {d_str:<10}")

    # ===================================================================
    # ANALYSIS 2: Hard vs easy guide analysis
    # ===================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Hard vs Easy Guide Analysis")
    print("=" * 70)

    hard_easy_df, guide_df = hard_vs_easy_analysis(df, pred_a, pred_c)
    hard_easy_path = _RESULTS / "hard_vs_easy_guide_analysis.csv"
    hard_easy_df.to_csv(hard_easy_path, index=False)
    logger.info("Saved hard/easy analysis -> %s", hard_easy_path)

    print(f"\n{'Group':<20} {'N':<5} {'A':<8} {'C':<8} {'Delta':<10} "
          f"{'C>A':<8} {'Wilcoxon p':<12}")
    print("-" * 71)
    for _, row in hard_easy_df.iterrows():
        w_str = f"{row['wilcoxon_p']:.4f}" if not np.isnan(row['wilcoxon_p']) else "N/A"
        print(f"{row['group']:<20} {row['n_guides']:<5} "
              f"{row['mean_auprc_A']:<8.4f} {row['mean_auprc_C']:<8.4f} "
              f"{row['mean_delta_C_A']:+<10.4f} "
              f"{row['n_C_better']}/{row['n_guides']:<5} {w_str:<12}")

    # ===================================================================
    # ANALYSIS 3: Site-level rank improvement
    # ===================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Site-level Rank Improvement for Positives")
    print("=" * 70)

    rank_df = rank_improvement_analysis(df, pred_a, pred_c)
    rank_path = _RESULTS / "rank_improvement_positives.csv"
    rank_df.to_csv(rank_path, index=False)
    logger.info("Saved rank improvement analysis -> %s", rank_path)

    print(f"\n{'Scope':<12} {'N pos':<7} {'Rank A':<9} {'Rank C':<9} "
          f"{'Improve':<9} {'Better':<10} {'Worse':<10} {'Wilcoxon p':<12}")
    print("-" * 78)
    for _, row in rank_df.iterrows():
        w_str = f"{row['wilcoxon_p']:.4f}" if not np.isnan(row['wilcoxon_p']) else "N/A"
        print(f"{row['scope']:<12} {row['n_positives']:<7} "
              f"{row['mean_rank_A']:<9.1f} {row['mean_rank_C']:<9.1f} "
              f"{row['mean_rank_improvement']:+<9.2f} "
              f"{row['n_improved']:<10} {row['n_worsened']:<10} {w_str:<12}")

    # ===================================================================
    # FIGURES
    # ===================================================================
    print("\n" + "=" * 70)
    print("FIGURES")
    print("=" * 70)

    plot_within_bin(bin_df, _FIGURES / "fig_within_bin.png")
    plot_rank_improvement(df, pred_a, pred_c, _FIGURES / "fig_rank_improvement.png")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "=" * 70)
    print("CONDITIONAL VALUE ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  {bin_path}")
    print(f"  {hard_easy_path}")
    print(f"  {rank_path}")
    print(f"  {_FIGURES / 'fig_within_bin.png'}")
    print(f"  {_FIGURES / 'fig_rank_improvement.png'}")


if __name__ == "__main__":
    main()
