"""Diagnostic 5: AlphaGenome sanity check + accessibility vs label analysis."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alphagenome_client import (
    INTERVAL_SIZE, BIN_SIZE, _extract_chromatin_features, TrackAvailability,
)
from src.utils import load_cache

# Hardcoded track availability (from previous AG queries — avoids needing live API)
_AVAIL = {
    "CL:0000624": TrackAvailability(
        ontology_term="CL:0000624", has_dnase=True, has_atac=True,
        histone_marks=["H3K27ac", "H3K27me3", "H3K4me3", "H3K9me3"],
        has_ctcf=True,
    ),
    "EFO:0002067": TrackAvailability(
        ontology_term="EFO:0002067", has_dnase=True, has_atac=True,
        histone_marks=["H3K27ac", "H3K27me3", "H3K4me3"],  # K562 has no H3K9me3
        has_ctcf=True,
    ),
}

_ROOT = Path(__file__).resolve().parents[1]
ag_cache_dir = _ROOT / "data" / "features" / "ag_cache"

# Load feature matrix for accessibility analysis
fm = pd.read_parquet(_ROOT / "data/features/feature_matrix_expanded.parquet")
fm_no_disc = fm[fm["dataset"] != "discover_seq_k562"].copy()
tcell = fm_no_disc[fm_no_disc["cell_type"] == "CD4_Tcell"]
k562 = fm_no_disc[fm_no_disc["cell_type"] == "K562"]

# ======================================================================
# DIAGNOSTIC 5a: Known T cell loci — query from AG cache
# ======================================================================
print("=" * 120)
print("DIAGNOSTIC 5a: AlphaGenome Predictions at Known T Cell Loci")
print("=" * 120)

known_loci = [
    ("CD4 promoter", "chr12", 6_789_500, "active in T cells"),
    ("TRAC (TCR alpha)", "chr14", 22_547_500, "highly active in T cells"),
    ("IL2 promoter", "chr4", 122_451_700, "active in T cells"),
    ("CD3E promoter", "chr11", 118_344_300, "active in T cells"),
    ("HBB (inactive in T)", "chr11", 5_227_000, "inactive in T cells"),
    ("GAPDH (housekeeping)", "chr12", 6_534_500, "active in all cells"),
]

def query_locus(chrom, pos, ontology):
    """Query AG cache for features at a specific locus."""
    tile_start = (pos // INTERVAL_SIZE) * INTERVAL_SIZE
    tile_end = tile_start + INTERVAL_SIZE
    ont_safe = ontology.replace(":", "_")

    cache_key = f"{chrom}__{tile_start}__{tile_end}__{ont_safe}"
    matches = list(ag_cache_dir.glob(f"{cache_key}__*.pkl"))

    if not matches:
        return None

    with open(matches[0], "rb") as f:
        output = pickle.load(f)

    site_offset = pos - tile_start
    avail = _AVAIL[ontology]
    features = _extract_chromatin_features(output, site_offset, avail)
    return features

for ct_label, ct_ontology in [("CD4_Tcell (CL:0000624)", "CL:0000624"),
                               ("K562 (EFO:0002067)", "EFO:0002067")]:
    print(f"\n  {ct_label}:")
    print(f"  {'Locus':<25s} {'Expected':<25s} {'DNase_mean':>11s} {'DNase_max':>11s} {'DNase@site':>11s} {'ATAC_mean':>11s} {'H3K27ac':>9s} {'Access':>7s}")
    print(f"  {'-'*110}")

    for name, chrom, pos, expected in known_loci:
        feats = query_locus(chrom, pos, ct_ontology)
        if feats is None:
            print(f"  {name:<25s} {expected:<25s} {'(no cache)':>11s}")
            continue

        dnase_m = feats.get("ag_dnase_mean", float("nan"))
        dnase_x = feats.get("ag_dnase_max", float("nan"))
        dnase_s = feats.get("ag_dnase_at_site", float("nan"))
        atac_m = feats.get("ag_atac_mean", float("nan"))
        h3k27ac = feats.get("ag_h3k27ac_mean", float("nan"))
        access = feats.get("ag_accessibility_binary", float("nan"))

        print(f"  {name:<25s} {expected:<25s} {dnase_m:>11.4f} {dnase_x:>11.4f} {dnase_s:>11.4f} {atac_m:>11.4f} {h3k27ac:>9.1f} {access:>7.0f}")

# ======================================================================
# DIAGNOSTIC 5b: Accessibility distribution by label
# ======================================================================
print("\n" + "=" * 120)
print("DIAGNOSTIC 5b: Chromatin Accessibility vs Off-Target Label")
print("=" * 120)

# For each cell type, compare positive vs negative sites
for ct_name, ct_df in [("K562", k562), ("CD4_Tcell", tcell)]:
    pos = ct_df[ct_df["label"] == 1]
    neg = ct_df[ct_df["label"] == 0]

    print(f"\n  {ct_name} (n_pos={len(pos)}, n_neg={len(neg)}):")
    print(f"  {'Feature':<30s} {'Pos mean':>10s} {'Neg mean':>10s} {'Diff':>10s} {'Effect(d)':>10s} {'Mann-W p':>12s} {'Direction':>10s}")
    print(f"  {'-'*95}")

    for col in ["ag_dnase_mean", "ag_dnase_max", "ag_dnase_at_site", "ag_atac_mean",
                 "ag_atac_at_site", "ag_h3k27ac_mean", "ag_h3k27me3_mean", "ag_h3k4me3_mean",
                 "ag_h3k9me3_mean", "ag_ctcf_mean", "ag_accessibility_binary"]:
        p_vals = pos[col].dropna().values
        n_vals = neg[col].dropna().values

        if len(p_vals) < 5 or len(n_vals) < 5:
            print(f"  {col:<30s} insufficient data")
            continue

        p_mean = p_vals.mean()
        n_mean = n_vals.mean()
        diff = p_mean - n_mean

        # Cohen's d
        pooled_std = np.sqrt((p_vals.std()**2 * len(p_vals) + n_vals.std()**2 * len(n_vals)) / (len(p_vals) + len(n_vals)))
        cohen_d = diff / pooled_std if pooled_std > 0 else 0

        # Mann-Whitney U
        u_stat, u_p = stats.mannwhitneyu(p_vals, n_vals, alternative='two-sided')

        direction = "+" if diff > 0 else "-"
        sig = "*" if u_p < 0.05 else ""
        print(f"  {col:<30s} {p_mean:>10.4f} {n_mean:>10.4f} {diff:>+10.4f} {cohen_d:>+10.4f} {u_p:>12.2e} {direction:>8s}{sig}")

# ======================================================================
# DIAGNOSTIC 5c: Percentile analysis
# ======================================================================
print("\n" + "=" * 120)
print("DIAGNOSTIC 5c: Are Positives in More Accessible Regions?")
print("=" * 120)

for ct_name, ct_df in [("K562", k562), ("CD4_Tcell", tcell)]:
    print(f"\n  {ct_name}:")
    # What percentile of DNase accessibility are the positives at?
    for col in ["ag_dnase_mean", "ag_atac_mean", "ag_h3k27ac_mean"]:
        all_vals = ct_df[col].values
        pos_vals = ct_df[ct_df["label"] == 1][col].values
        neg_vals = ct_df[ct_df["label"] == 0][col].values

        # Fraction of positives in top quartile vs bottom quartile
        q75 = np.percentile(all_vals, 75)
        q25 = np.percentile(all_vals, 25)

        pos_in_top = (pos_vals > q75).sum()
        pos_in_bot = (pos_vals < q25).sum()
        neg_in_top = (neg_vals > q75).sum()
        neg_in_bot = (neg_vals < q25).sum()

        pos_top_pct = 100 * pos_in_top / len(pos_vals) if len(pos_vals) > 0 else 0
        pos_bot_pct = 100 * pos_in_bot / len(pos_vals) if len(pos_vals) > 0 else 0
        neg_top_pct = 100 * neg_in_top / len(neg_vals) if len(neg_vals) > 0 else 0
        neg_bot_pct = 100 * neg_in_bot / len(neg_vals) if len(neg_vals) > 0 else 0

        enrichment_top = pos_top_pct / neg_top_pct if neg_top_pct > 0 else float("inf")

        print(f"    {col:<25s}: Pos in top Q: {pos_top_pct:.1f}%, Neg in top Q: {neg_top_pct:.1f}%, Enrichment: {enrichment_top:.2f}x")

# ======================================================================
# DIAGNOSTIC 5d: Compare AG feature distributions
# ======================================================================
print("\n" + "=" * 120)
print("DIAGNOSTIC 5d: AG Feature Distribution Comparison (K562 vs T cell)")
print("=" * 120)

print(f"\n  Key question: Are T cell sites more homogeneous in chromatin accessibility?")
print(f"\n  {'Feature':<30s} {'K562 IQR':>12s} {'Tcell IQR':>12s} {'K562 Q25-Q75':>25s} {'Tcell Q25-Q75':>25s}")
print(f"  {'-'*110}")

for col in ["ag_dnase_mean", "ag_atac_mean", "ag_h3k27ac_mean", "ag_ctcf_mean", "ag_h3k4me3_mean"]:
    k_q25, k_q75 = np.percentile(k562[col], [25, 75])
    t_q25, t_q75 = np.percentile(tcell[col], [25, 75])
    k_iqr = k_q75 - k_q25
    t_iqr = t_q75 - t_q25
    print(f"  {col:<30s} {k_iqr:>12.4f} {t_iqr:>12.4f} {f'{k_q25:.4f} - {k_q75:.4f}':>25s} {f'{t_q25:.4f} - {t_q75:.4f}':>25s}")

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 120)
print("DIAGNOSTIC SUMMARY")
print("=" * 120)
print("""
Key findings will be printed here after all diagnostics complete.
""")
