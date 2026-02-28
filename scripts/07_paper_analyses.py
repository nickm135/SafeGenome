#!/usr/bin/env python
"""Paper analyses: AG vs ENCODE correlation (Validation 1) and SHAP importance.

Outputs:
    data/results/ag_vs_encode_correlation.csv
    figures/fig5_ag_vs_encode_correlation.png
    figures/fig4_feature_importance.png
"""

import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# ── project imports ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cell_type_transfer import compute_chromatin_correlation
from src.model import compute_shap_values, prepare_model_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("paper_analyses")

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FEATURE_MATRIX_PATH = ROOT / "data" / "features" / "feature_matrix_combined.parquet"
MODEL_C_PATH = ROOT / "data" / "models" / "model_c.pkl"
RESULTS_DIR = ROOT / "data" / "results"
FIGURES_DIR = ROOT / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── feature pairs for Validation 1 ──────────────────────────────────────
FEATURE_PAIRS: list[tuple[str, str, str]] = [
    ("ag_dnase_mean", "exp_dnase_mean", "DNase mean"),
    ("ag_dnase_max", "exp_dnase_max", "DNase max"),
    ("ag_dnase_at_site", "exp_dnase_at_site", "DNase at site"),
    ("ag_h3k4me3_mean", "exp_h3k4me3_mean", "H3K4me3"),
    ("ag_h3k27me3_mean", "exp_h3k27me3_mean", "H3K27me3"),
    ("ag_h3k9me3_mean", "exp_h3k9me3_mean", "H3K9me3"),
    ("ag_h3k27ac_mean", "exp_h3k27ac_mean", "H3K27ac"),
    ("ag_accessibility_binary", "exp_accessibility_binary", "Accessibility"),
]


def run_validation_1(fm: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlations between AG and ENCODE features at K562 sites."""
    logger.info("=== Validation 1: AlphaGenome vs ENCODE correlation (K562) ===")

    rows = []
    for ag_feat, exp_feat, label in FEATURE_PAIRS:
        result = compute_chromatin_correlation(
            fm, ag_feature=ag_feat, exp_feature=exp_feat, cell_type="K562"
        )
        rows.append(
            {
                "ag_feature": ag_feat,
                "exp_feature": exp_feat,
                "label": label,
                "spearman_r": result["spearman_r"],
                "p_value": result["p_value"],
                "n": result["n"],
            }
        )
        logger.info(
            "  %s: r=%.4f, p=%.2e, n=%d",
            label,
            result["spearman_r"],
            result["p_value"],
            result["n"],
        )

    corr_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "ag_vs_encode_correlation.csv"
    corr_df.to_csv(out_path, index=False)
    logger.info("Saved correlation summary → %s", out_path)
    return corr_df


def plot_scatter_grid(fm: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    """Scatter plot grid: one panel per AG vs ENCODE feature pair."""
    k562 = fm[fm["cell_type"] == "K562"].copy()

    n_pairs = len(FEATURE_PAIRS)
    ncols = 4
    nrows = (n_pairs + ncols - 1) // ncols  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for idx, (ag_feat, exp_feat, label) in enumerate(FEATURE_PAIRS):
        ax = axes[idx]
        valid = k562[[ag_feat, exp_feat]].dropna()
        x = valid[exp_feat].values
        y = valid[ag_feat].values

        ax.scatter(x, y, alpha=0.15, s=6, color="#2563eb", rasterized=True)

        # Spearman r from precomputed table
        row = corr_df[corr_df["label"] == label].iloc[0]
        r = row["spearman_r"]
        n = int(row["n"])

        ax.set_xlabel(f"ENCODE ({exp_feat.replace('exp_', '')})", fontsize=9)
        ax.set_ylabel(f"AlphaGenome ({ag_feat.replace('ag_', '')})", fontsize=9)
        ax.set_title(f"{label}\nr={r:.3f}, n={n:,}", fontsize=10)
        ax.tick_params(labelsize=8)

    # hide unused axes
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Validation 1: AlphaGenome vs ENCODE Chromatin Features (K562)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path = FIGURES_DIR / "fig5_ag_vs_encode_correlation.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scatter grid → %s", out_path)


def run_shap_analysis(fm: pd.DataFrame) -> None:
    """Compute SHAP values for Model C and generate summary plot."""
    logger.info("=== SHAP Feature Importance (Model C) ===")

    # Load Model C
    with open(MODEL_C_PATH, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    logger.info(
        "Loaded Model C: %d features, AUPRC=%.4f±%.4f",
        len(feature_cols),
        artifact["mean_auprc"],
        artifact["std_auprc"],
    )

    # Prepare K562 data with Model C features (including missingness indicators
    # to match training configuration)
    X, y, groups, feat_cols = prepare_model_data(
        fm, "C", cell_types=["K562"], include_missingness_indicators=True
    )
    logger.info("K562 Model C data: %d samples, %d features, %d positives", X.shape[0], X.shape[1], y.sum())

    # Compute SHAP values
    shap_values, importance_df = compute_shap_values(model, X, feat_cols)
    logger.info("SHAP values computed: shape=%s", shap_values.shape)

    # Save importance CSV
    importance_path = RESULTS_DIR / "shap_importance_model_C.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info("Saved SHAP importance → %s", importance_path)

    # Print top 15
    logger.info("Top 15 features by mean |SHAP|:")
    for _, row in importance_df.head(15).iterrows():
        logger.info("  %-30s  %.6f", row["feature"], row["mean_abs_shap"])

    # SHAP summary plot (beeswarm)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        features=X,
        feature_names=feat_cols,
        show=False,
        max_display=20,
        plot_size=None,
    )
    plt.title("SHAP Feature Importance — Model C (K562)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig4_feature_importance.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP summary plot → %s", out_path)


def main() -> None:
    logger.info("Loading feature matrix from %s", FEATURE_MATRIX_PATH)
    fm = pd.read_parquet(FEATURE_MATRIX_PATH)
    logger.info("Feature matrix: %d rows × %d cols", fm.shape[0], fm.shape[1])

    # Validation 1
    corr_df = run_validation_1(fm)
    plot_scatter_grid(fm, corr_df)

    # SHAP
    run_shap_analysis(fm)

    logger.info("All paper analyses complete.")


if __name__ == "__main__":
    main()
