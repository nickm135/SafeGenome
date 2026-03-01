"""Diagnostic script: Why T cell AlphaGenome features show zero benefit."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import prepare_model_data, compute_scale_pos_weight, make_xgb_classifier
from src.feature_matrix import get_model_features

_ROOT = Path(__file__).resolve().parents[1]

# Load data
fm = pd.read_parquet(_ROOT / "data/features/feature_matrix_expanded.parquet")
fm_no_disc = fm[fm["dataset"] != "discover_seq_k562"].copy()

ag_cols = [c for c in fm.columns if c.startswith("ag_") and not c.endswith("_missing")]

k562 = fm_no_disc[fm_no_disc["cell_type"] == "K562"]
tcell = fm_no_disc[fm_no_disc["cell_type"] == "CD4_Tcell"]

MODEL_BC_CELL_TYPES = ["K562", "HEK293", "CD4_Tcell"]

# ======================================================================
# DIAGNOSTIC 3: SHAP by cell type
# ======================================================================
print("=" * 120)
print("DIAGNOSTIC 3: Mean |SHAP| by Cell Type for AlphaGenome Features")
print("=" * 120)

X, y, groups, feature_cols = prepare_model_data(
    fm_no_disc, "C",
    cell_types=MODEL_BC_CELL_TYPES,
    exclude_datasets=["discover_seq_k562"],
    include_missingness_indicators=True,
)
print(f"Model C data: {X.shape[0]} rows, {X.shape[1]} features, {int(y.sum())} pos")

spw = compute_scale_pos_weight(y)
clf = make_xgb_classifier(spw, random_state=42)
clf.fit(X, y)

import shap
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)

# Get cell types aligned with the subset
mask = fm_no_disc["cell_type"].isin(MODEL_BC_CELL_TYPES)
fm_subset = fm_no_disc[mask].copy()
cell_types_arr = fm_subset["cell_type"].values

shap_df = pd.DataFrame(shap_values, columns=feature_cols)
shap_df["cell_type"] = cell_types_arr

ag_missing_cols = [c for c in feature_cols if c.startswith("ag_") and c.endswith("_missing")]
ag_feat_cols = [c for c in feature_cols if c.startswith("ag_") and not c.endswith("_missing")]

all_features_shap = {}
for ct in ["K562", "CD4_Tcell", "HEK293"]:
    ct_mask = shap_df["cell_type"] == ct
    ct_shap = shap_df[ct_mask][feature_cols].abs().mean()
    all_features_shap[ct] = ct_shap

print(f"\n{'Feature':<35s} {'K562 |SHAP|':>14s} {'Tcell |SHAP|':>14s} {'HEK |SHAP|':>14s} {'K562 rank':>10s} {'Tcell rank':>10s}")
print("-" * 100)

for col in sorted(ag_feat_cols):
    k_shap = all_features_shap["K562"][col]
    t_shap = all_features_shap["CD4_Tcell"][col]
    h_shap = all_features_shap["HEK293"][col]
    k_rank = int((all_features_shap["K562"] > k_shap).sum() + 1)
    t_rank = int((all_features_shap["CD4_Tcell"] > t_shap).sum() + 1)
    print(f"{col:<35s} {k_shap:>14.6f} {t_shap:>14.6f} {h_shap:>14.6f} {k_rank:>10d} {t_rank:>10d}")

print(f"\nMissingness indicators:")
for col in sorted(ag_missing_cols):
    k_shap = all_features_shap["K562"][col]
    t_shap = all_features_shap["CD4_Tcell"][col]
    h_shap = all_features_shap["HEK293"][col]
    print(f"{col:<35s} {k_shap:>14.6f} {t_shap:>14.6f} {h_shap:>14.6f}")

print(f"\nTop 10 features by mean |SHAP| per cell type:")
for ct in ["K562", "CD4_Tcell"]:
    ranked = all_features_shap[ct].sort_values(ascending=False)
    print(f"\n  {ct}:")
    for i, (feat, val) in enumerate(ranked.head(15).items()):
        is_ag = "AG" if feat.startswith("ag_") and not feat.endswith("_missing") else "miss" if feat.endswith("_missing") else "seq"
        is_ag = "EXP" if feat.startswith("exp_") and not feat.endswith("_missing") else is_ag
        print(f"    {i+1:2d}. {feat:<35s} {val:.6f}  [{is_ag}]")

print(f"\nTotal AG feature SHAP budget:")
for ct in ["K562", "CD4_Tcell", "HEK293"]:
    ag_total = sum(all_features_shap[ct][c] for c in ag_feat_cols)
    all_total = all_features_shap[ct].sum()
    pct = 100 * ag_total / all_total if all_total > 0 else 0
    miss_total = sum(all_features_shap[ct][c] for c in ag_missing_cols if c in all_features_shap[ct])
    miss_pct = 100 * miss_total / all_total if all_total > 0 else 0
    print(f"  {ct}: AG |SHAP| = {ag_total:.4f} ({pct:.1f}%), Missingness |SHAP| = {miss_total:.4f} ({miss_pct:.1f}%)")

# ======================================================================
# DIAGNOSTIC 4: T cell guide difficulty
# ======================================================================
print("\n" + "=" * 120)
print("DIAGNOSTIC 4: Guide Difficulty (Model A AUPRC per guide, by cell type)")
print("=" * 120)

per_guide = pd.read_csv(_ROOT / "data/results/per_guide_auprc_expanded_v2.csv")
summary_labels = {"median", "IQR_25", "IQR_75", "mean", "std"}
guides = per_guide[~per_guide["guide_id"].isin(summary_labels)].copy()

guide_celltype = fm_no_disc.groupby("guide_id")["cell_type"].first().to_dict()
guides["cell_type"] = guides["guide_id"].map(guide_celltype)

print(f"\n{'Cell type':<15s} {'N guides':>10s} {'N with pos':>12s} {'A mean':>10s} {'A median':>10s} {'A std':>10s} {'C mean':>10s} {'C-A gap':>10s}")
print("-" * 95)

for ct in ["K562", "HEK293", "CD4_Tcell"]:
    ct_guides = guides[guides["cell_type"] == ct]
    ct_with_pos = ct_guides[ct_guides["n_positives"] > 0]
    a_mean = ct_with_pos["auprc_A"].mean()
    a_median = ct_with_pos["auprc_A"].median()
    a_std = ct_with_pos["auprc_A"].std()
    c_mean = ct_with_pos["auprc_C"].mean()
    gap = c_mean - a_mean
    print(f"{ct:<15s} {len(ct_guides):>10d} {len(ct_with_pos):>12d} {a_mean:>10.4f} {a_median:>10.4f} {a_std:>10.4f} {c_mean:>10.4f} {gap:>+10.4f}")

# Distribution of Model A AUPRC
print(f"\nModel A AUPRC distribution by cell type:")
for ct in ["K562", "CD4_Tcell"]:
    ct_vals = guides[(guides["cell_type"] == ct) & (guides["auprc_A"].notna())]["auprc_A"]
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    counts = pd.cut(ct_vals, bins=bins).value_counts().sort_index()
    print(f"\n  {ct} (n={len(ct_vals)}):")
    for interval, count in counts.items():
        pct = 100 * count / len(ct_vals) if len(ct_vals) > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"    {str(interval):<15s}: {count:3d} ({pct:5.1f}%) {bar}")

# Detailed per-guide comparison: C-A gap by difficulty bucket
print(f"\nC-A gap by Model A difficulty bucket:")
for ct in ["K562", "CD4_Tcell"]:
    ct_g = guides[(guides["cell_type"] == ct) & (guides["auprc_A"].notna()) & (guides["auprc_C"].notna())]
    ct_g = ct_g.copy()
    ct_g["difficulty"] = pd.cut(ct_g["auprc_A"], bins=[0, 0.3, 0.6, 0.8, 1.01],
                                labels=["hard(<0.3)", "medium(0.3-0.6)", "easy(0.6-0.8)", "trivial(>0.8)"])
    ct_g["c_a_gap"] = ct_g["auprc_C"] - ct_g["auprc_A"]
    print(f"\n  {ct}:")
    for bucket in ["hard(<0.3)", "medium(0.3-0.6)", "easy(0.6-0.8)", "trivial(>0.8)"]:
        bucket_data = ct_g[ct_g["difficulty"] == bucket]
        if len(bucket_data) > 0:
            gap_mean = bucket_data["c_a_gap"].mean()
            gap_std = bucket_data["c_a_gap"].std()
            print(f"    {bucket:<20s}: n={len(bucket_data):2d}, C-A gap = {gap_mean:+.4f} +/- {gap_std:.4f}")
        else:
            print(f"    {bucket:<20s}: n= 0")

# Number of positives per guide
print(f"\nPositives per guide distribution:")
for ct in ["K562", "CD4_Tcell"]:
    ct_g = guides[guides["cell_type"] == ct]
    print(f"\n  {ct}:")
    print(f"    Mean positives:   {ct_g['n_positives'].mean():.1f}")
    print(f"    Median positives: {ct_g['n_positives'].median():.1f}")
    print(f"    Guides with 0:    {(ct_g['n_positives'] == 0).sum()}")
    print(f"    Guides with 1-2:  {((ct_g['n_positives'] >= 1) & (ct_g['n_positives'] <= 2)).sum()}")
    print(f"    Guides with 3-5:  {((ct_g['n_positives'] >= 3) & (ct_g['n_positives'] <= 5)).sum()}")
    print(f"    Guides with >5:   {(ct_g['n_positives'] > 5).sum()}")

print("\n" + "=" * 120)
print("DIAGNOSTIC 5: AlphaGenome Prediction Sanity Check at Known T Cell Loci")
print("=" * 120)

# Check AG predictions at known T cell active loci
# CD4 promoter (chr12:6,789,500-6,789,600 hg38)
# TRAC locus (chr14:22,547,500-22,547,600 hg38 - T cell receptor alpha constant)
# IL2 gene promoter (chr4:122,451,700-122,451,800 hg38)
# CD3E promoter (chr11:118,344,300-118,344,400 hg38)
# Also check a known inaccessible locus for contrast:
# HBB promoter (chr11:5,227,000-5,227,100 hg38 - should be inactive in T cells)

# Look at the AG cache for these loci
import pickle
from pathlib import Path

ag_cache_dir = _ROOT / "data" / "features" / "ag_cache"

known_loci = [
    ("CD4 promoter", "chr12", 6_789_500),
    ("TRAC (TCR alpha)", "chr14", 22_547_500),
    ("IL2 promoter", "chr4", 122_451_700),
    ("CD3E promoter", "chr11", 118_344_300),
    ("HBB (inactive in T)", "chr11", 5_227_000),
    ("GAPDH (housekeeping)", "chr12", 6_534_500),
]

# Helper to find the right cache tile
def find_cache_tile(chrom: str, pos: int, ontology: str) -> dict | None:
    tile_size = 131072
    tile_start = (pos // tile_size) * tile_size
    tile_end = tile_start + tile_size
    ont_safe = ontology.replace(":", "_")

    # Search for matching cache file
    pattern = f"{chrom}__{tile_start}__{tile_end}__{ont_safe}__*.pkl"
    matches = list(ag_cache_dir.glob(pattern))
    if matches:
        with open(matches[0], "rb") as f:
            return pickle.load(f)
    return None

# Feature extraction from cache tile
def extract_features_from_tile(tile_data: dict, pos: int, tile_start: int) -> dict:
    """Extract AG features at a specific position from cached tile data."""
    offset = pos - tile_start
    features = {}
    for key, arr in tile_data.items():
        if isinstance(arr, np.ndarray):
            if len(arr) >= 131072:
                # 1bp resolution - take value at offset
                features[key] = float(arr[offset])
            else:
                # binned resolution (128bp)
                bin_idx = offset // 128
                if bin_idx < len(arr):
                    features[key] = float(arr[bin_idx])
                else:
                    features[key] = float("nan")
        else:
            features[key] = arr
    return features

# Query each locus
for ct_label, ct_ontology in [("CD4_Tcell (CL:0000624)", "CL:0000624"), ("K562 (EFO:0002067)", "EFO:0002067")]:
    print(f"\n  {ct_label}:")
    print(f"  {'Locus':<25s} {'DNase@site':>12s} {'DNase_mean':>12s} {'ATAC@site':>12s} {'ATAC_mean':>12s} {'H3K27ac':>12s} {'Accessible':>12s}")
    print(f"  {'-'*100}")

    for name, chrom, pos in known_loci:
        tile_data = find_cache_tile(chrom, pos, ct_ontology)
        if tile_data is None:
            print(f"  {name:<25s}  (no cache tile found)")
            continue

        tile_start = (pos // 131072) * 131072
        feats = extract_features_from_tile(tile_data, pos, tile_start)

        # Print key accessibility features
        dnase_site = feats.get("dnase", float("nan"))
        atac_site = feats.get("atac", float("nan"))

        # For mean values, we need to look at a window around the site
        # The cache stores raw tracks, not summarized features
        # Use the at-site value as a proxy
        print(f"  {name:<25s} {dnase_site:>12.4f} {'--':>12s} {atac_site:>12.4f} {'--':>12s} {'--':>12s} {'--':>12s}")

# Actually, let's look at the feature matrix values for sites near known loci
# Find sites closest to known active regions
print(f"\n  Feature matrix values at off-target sites in high/low accessibility regions:")
print(f"  Comparing sites with highest vs lowest DNase signal in T cells:")

tcell_sites = fm_no_disc[fm_no_disc["cell_type"] == "CD4_Tcell"].copy()
tcell_sites_sorted = tcell_sites.sort_values("ag_dnase_mean", ascending=False)

print(f"\n  Top 10 T cell sites by ag_dnase_mean (most accessible):")
print(f"  {'guide_id':<12s} {'chrom':<6s} {'start':>12s} {'label':>6s} {'dnase_mean':>12s} {'atac_mean':>12s} {'h3k27ac':>10s} {'access_bin':>10s}")
for _, row in tcell_sites_sorted.head(10).iterrows():
    print(f"  {row['guide_id']:<12s} {row['chrom']:<6s} {int(row['start']):>12d} {int(row['label']):>6d} {row['ag_dnase_mean']:>12.4f} {row['ag_atac_mean']:>12.4f} {row['ag_h3k27ac_mean']:>10.1f} {row['ag_accessibility_binary']:>10.0f}")

print(f"\n  Bottom 10 T cell sites by ag_dnase_mean (least accessible):")
for _, row in tcell_sites_sorted.tail(10).iterrows():
    print(f"  {row['guide_id']:<12s} {row['chrom']:<6s} {int(row['start']):>12d} {int(row['label']):>6d} {row['ag_dnase_mean']:>12.4f} {row['ag_atac_mean']:>12.4f} {row['ag_h3k27ac_mean']:>10.1f} {row['ag_accessibility_binary']:>10.0f}")

# Key question: are positive T cell off-targets in more accessible regions?
print(f"\n  T cell accessibility by label:")
for label_val, label_name in [(1, "Positive (true OT)"), (0, "Negative")]:
    subset = tcell_sites[tcell_sites["label"] == label_val]
    print(f"\n    {label_name} (n={len(subset)}):")
    for col in ["ag_dnase_mean", "ag_atac_mean", "ag_h3k27ac_mean", "ag_accessibility_binary"]:
        m = subset[col].mean()
        s = subset[col].std()
        med = subset[col].median()
        print(f"      {col:<30s}: mean={m:.4f}, std={s:.4f}, median={med:.4f}")

# Same for K562
print(f"\n  K562 accessibility by label:")
for label_val, label_name in [(1, "Positive (true OT)"), (0, "Negative")]:
    subset = k562[k562["label"] == label_val]
    print(f"\n    {label_name} (n={len(subset)}):")
    for col in ["ag_dnase_mean", "ag_atac_mean", "ag_h3k27ac_mean", "ag_accessibility_binary"]:
        m = subset[col].mean()
        s = subset[col].std()
        med = subset[col].median()
        print(f"      {col:<30s}: mean={m:.4f}, std={s:.4f}, median={med:.4f}")
