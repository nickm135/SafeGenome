#!/usr/bin/env python
"""Script 16: AG vs ENCODE concordance analysis on the full 60-guide dataset.

Computes Spearman correlations between AlphaGenome-predicted and ENCODE-experimental
chromatin features for K562 sites (the only cell type with both data sources).
Generates publication-quality scatter plots and bar charts.

Outputs:
    data/results/ag_vs_encode_concordance_expanded_v2.csv
    figures/fig_ag_vs_encode_scatter.png
    figures/fig_concordance_barplot.png
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_MATRIX = PROJECT_ROOT / "data" / "features" / "feature_matrix_expanded.parquet"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Original script 07 results for comparison (K562, n=3531)
# Values from data/results/ag_vs_encode_correlation.csv
SCRIPT_07_RESULTS: dict[str, float | None] = {
    "DNase mean": 0.4057,
    "DNase max": 0.4178,
    "DNase at site": 0.1771,
    "H3K4me3": 0.2704,
    "H3K27me3": 0.6734,
    "H3K9me3": None,  # NaN — AG constant at 0
    "H3K27ac": 0.6252,
    "Accessibility binary": 0.5073,
}

# Feature pairs: (ag_column, exp_column, display_label, category)
FEATURE_PAIRS: list[tuple[str, str, str, str]] = [
    ("ag_dnase_mean", "exp_dnase_mean", "DNase mean", "Accessibility"),
    ("ag_dnase_max", "exp_dnase_max", "DNase max", "Accessibility"),
    ("ag_dnase_at_site", "exp_dnase_at_site", "DNase at site", "Accessibility"),
    ("ag_h3k4me3_mean", "exp_h3k4me3_mean", "H3K4me3", "Histone"),
    ("ag_h3k27me3_mean", "exp_h3k27me3_mean", "H3K27me3", "Histone"),
    ("ag_h3k9me3_mean", "exp_h3k9me3_mean", "H3K9me3", "Histone"),
    ("ag_h3k27ac_mean", "exp_h3k27ac_mean", "H3K27ac", "Histone"),
    ("ag_accessibility_binary", "exp_accessibility_binary", "Accessibility binary", "Accessibility"),
]

CATEGORY_COLORS = {
    "Accessibility": "#2196F3",
    "Histone": "#FF9800",
}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_concordance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlation for each AG vs ENCODE feature pair.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix filtered to K562 sites.

    Returns
    -------
    pd.DataFrame
        One row per feature pair with spearman_r, p_value, n, category.
    """
    rows = []
    for ag_col, exp_col, label, category in FEATURE_PAIRS:
        valid = df[[ag_col, exp_col]].dropna()
        n = len(valid)
        if n < 3:
            logger.warning("%s: only %d valid pairs, skipping", label, n)
            rows.append({
                "label": label, "ag_feature": ag_col, "exp_feature": exp_col,
                "category": category, "spearman_r": np.nan,
                "p_value": np.nan, "n": n,
            })
            continue

        # Check for constant columns
        if valid[ag_col].nunique() <= 1 or valid[exp_col].nunique() <= 1:
            logger.warning("%s: constant column (AG nunique=%d, EXP nunique=%d)",
                           label, valid[ag_col].nunique(), valid[exp_col].nunique())
            rows.append({
                "label": label, "ag_feature": ag_col, "exp_feature": exp_col,
                "category": category, "spearman_r": np.nan,
                "p_value": np.nan, "n": n,
            })
            continue

        r, p = spearmanr(valid[ag_col], valid[exp_col])
        logger.info("  %s: r=%.4f, p=%.2e, n=%d", label, r, p, n)
        rows.append({
            "label": label, "ag_feature": ag_col, "exp_feature": exp_col,
            "category": category, "spearman_r": r,
            "p_value": p, "n": n,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_scatter_grid(df: pd.DataFrame, concordance: pd.DataFrame) -> None:
    """Multi-panel scatter plot: AG (y) vs ENCODE (x) for each feature pair.

    Subsamples to 1000 points for display; correlations computed on ALL points.
    """
    n_pairs = len(FEATURE_PAIRS)
    ncols = 4
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
    axes = axes.flatten()

    rng = np.random.RandomState(42)

    for i, (ag_col, exp_col, label, category) in enumerate(FEATURE_PAIRS):
        ax = axes[i]
        valid = df[[ag_col, exp_col]].dropna()

        row = concordance[concordance["label"] == label].iloc[0]
        r_val = row["spearman_r"]
        p_val = row["p_value"]
        n_val = int(row["n"])

        # Subsample for display
        if len(valid) > 1000:
            idx = rng.choice(len(valid), 1000, replace=False)
            plot_data = valid.iloc[idx]
        else:
            plot_data = valid

        color = CATEGORY_COLORS[category]
        ax.scatter(
            plot_data[exp_col], plot_data[ag_col],
            s=6, alpha=0.15, color=color, edgecolors="none", rasterized=True,
        )

        # Reference diagonal (identity line)
        lo = min(valid[ag_col].min(), valid[exp_col].min())
        hi = max(valid[ag_col].max(), valid[exp_col].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4, label="y=x")

        # Regression line (on all data)
        if not np.isnan(r_val):
            z = np.polyfit(valid[exp_col], valid[ag_col], 1)
            x_line = np.linspace(valid[exp_col].min(), valid[exp_col].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), "r-", lw=1.2, alpha=0.8)

        # Annotation
        if np.isnan(r_val):
            annot = f"r=NaN\nn={n_val}"
        else:
            annot = f"r={r_val:.3f}\np={p_val:.1e}\nn={n_val}"
        ax.text(
            0.05, 0.95, annot, transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

        ax.set_xlabel("ENCODE (experimental)", fontsize=9)
        ax.set_ylabel("AlphaGenome (predicted)", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for j in range(n_pairs, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "AlphaGenome vs ENCODE Chromatin — K562 (n=3,531)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out = FIGURES_DIR / "fig_ag_vs_encode_scatter.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scatter grid → %s", out)


def plot_concordance_barplot(concordance: pd.DataFrame) -> None:
    """Bar chart of Spearman r per feature, sorted by magnitude, colored by category."""
    plot_df = concordance.dropna(subset=["spearman_r"]).copy()
    plot_df = plot_df.sort_values("spearman_r", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = [CATEGORY_COLORS[cat] for cat in plot_df["category"]]
    bars = ax.barh(plot_df["label"], plot_df["spearman_r"], color=colors, edgecolor="white", height=0.6)

    # Value labels
    for bar, r_val in zip(bars, plot_df["spearman_r"]):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{r_val:.3f}", va="center", fontsize=9,
        )

    ax.set_xlabel("Spearman r", fontsize=11)
    ax.set_title("AG vs ENCODE Concordance — K562", fontsize=12, fontweight="bold")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(0, max(plot_df["spearman_r"]) + 0.1)

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS["Accessibility"], label="Accessibility / DNase"),
        Patch(facecolor=CATEGORY_COLORS["Histone"], label="Histone marks"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_concordance_barplot.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved bar chart → %s", out)


# ---------------------------------------------------------------------------
# Comparison with script 07
# ---------------------------------------------------------------------------

def compare_to_script_07(concordance: pd.DataFrame) -> pd.DataFrame:
    """Compare current concordance to original script 07 results."""
    rows = []
    for _, row in concordance.iterrows():
        label = row["label"]
        r_new = row["spearman_r"]
        r_old = SCRIPT_07_RESULTS.get(label)
        if r_old is None:
            delta = np.nan
        elif np.isnan(r_new):
            delta = np.nan
        else:
            delta = r_new - r_old
        rows.append({
            "label": label,
            "r_script07": r_old if r_old is not None else np.nan,
            "r_script16": r_new,
            "delta": delta,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("Script 16: AG vs ENCODE Concordance (Expanded v2)")
    logger.info("=" * 60)

    # Load data
    df = pd.read_parquet(FEATURE_MATRIX)
    logger.info("Loaded feature matrix: %d rows × %d cols", *df.shape)

    # Filter to K562
    k562 = df[df["cell_type"] == "K562"].copy()
    logger.info("K562 sites: %d", len(k562))

    # Compute concordance
    logger.info("\n--- Spearman Concordance (K562) ---")
    concordance = compute_concordance(k562)

    # Save results
    out_csv = RESULTS_DIR / "ag_vs_encode_concordance_expanded_v2.csv"
    concordance.to_csv(out_csv, index=False)
    logger.info("Saved concordance table → %s", out_csv)

    # Print concordance table
    print("\n" + "=" * 70)
    print("AG vs ENCODE Concordance — K562 (n=3,531)")
    print("=" * 70)
    print(f"{'Feature':<22} {'Category':<14} {'Spearman r':>10} {'p-value':>12} {'n':>6}")
    print("-" * 70)
    for _, row in concordance.iterrows():
        r_str = f"{row['spearman_r']:.4f}" if not np.isnan(row["spearman_r"]) else "NaN"
        p_str = f"{row['p_value']:.2e}" if not np.isnan(row["p_value"]) else "NaN"
        print(f"{row['label']:<22} {row['category']:<14} {r_str:>10} {p_str:>12} {int(row['n']):>6}")

    # Compare to script 07
    comparison = compare_to_script_07(concordance)

    print("\n" + "=" * 70)
    print("Comparison: Script 16 vs Script 07")
    print("=" * 70)
    print(f"{'Feature':<22} {'r (script07)':>12} {'r (script16)':>12} {'delta':>8}")
    print("-" * 56)
    for _, row in comparison.iterrows():
        r07_str = f"{row['r_script07']:.4f}" if not np.isnan(row["r_script07"]) else "NaN"
        r16_str = f"{row['r_script16']:.4f}" if not np.isnan(row["r_script16"]) else "NaN"
        d_str = f"{row['delta']:+.4f}" if not np.isnan(row["delta"]) else "N/A"
        print(f"{row['label']:<22} {r07_str:>12} {r16_str:>12} {d_str:>8}")

    max_delta = comparison["delta"].dropna().abs().max()
    all_match = max_delta < 1e-3
    if all_match:
        print("\n[OK] All correlations match script 07 (same K562 data).")
    else:
        print(f"\n[NOTE] Max delta: {max_delta:.4f} — see notes below.")

    # Generate figures
    logger.info("\n--- Generating figures ---")
    plot_scatter_grid(k562, concordance)
    plot_concordance_barplot(concordance)

    # Summary statistics
    valid_r = concordance["spearman_r"].dropna()
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"  Feature pairs:      {len(FEATURE_PAIRS)}")
    print(f"  Valid correlations:  {len(valid_r)} / {len(FEATURE_PAIRS)}")
    print(f"  Mean |r|:           {valid_r.abs().mean():.4f}")
    print(f"  Median |r|:         {valid_r.abs().median():.4f}")
    print(f"  Best:               {concordance.loc[valid_r.idxmax(), 'label']} (r={valid_r.max():.4f})")
    print(f"  Worst:              {concordance.loc[valid_r.idxmin(), 'label']} (r={valid_r.min():.4f})")
    print(f"  H3K9me3:            NaN (AG predicts constant 0 for K562)")

    logger.info("\nScript 16 complete.")


if __name__ == "__main__":
    main()
