"""
scripts/04b_extract_features_lazzarotto.py — Extract features for T cell
candidates and merge with existing feature matrix.

Takes the Lazzarotto 2020 candidates (from script 03b) and:
1. Labels them against the parsed Lazzarotto GUIDE-seq positives
2. Extracts sequence features (CFD, MIT, mismatch encoding, GC, etc.)
3. Extracts AlphaGenome features using CD4+ T cell ontology (CL:0000624)
4. Sets ENCODE features to NaN (no BigWig data for primary T cells)
5. Merges with existing feature_matrix_combined.parquet → expanded matrix

Usage
-----
    source venv/Scripts/activate
    python scripts/04b_extract_features_lazzarotto.py [options]

    --candidates PATH     T cell candidates parquet (default: data/processed/candidates_lazzarotto.parquet)
    --existing PATH       Existing feature matrix to merge with (default: data/features/feature_matrix_combined.parquet)
    --output PATH         Output expanded parquet (default: data/features/feature_matrix_expanded.parquet)
    --ag-cache-dir PATH   AlphaGenome cache directory (default: data/features/ag_cache)
    --no-ag               Skip AlphaGenome features

Cell-type notes
---------------
* CD4+ T cell (CL:0000624): full AlphaGenome coverage (12/12 features).
  DNase, ATAC, 6 histone marks (H3K27ac, H3K27me3, H3K36me3, H3K4me1,
  H3K4me3, H3K9me3), and CTCF.  Zero imputation needed.
* No ENCODE BigWig data exists for primary T cells. All ENCODE features
  will be NaN, imputed to 0 with _missing=1 indicators. Model B is
  effectively sequence-only for T cell sites (same as HEK293).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# --- path setup so src imports work when run as a script ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_matrix import build_feature_matrix
from src.utils import get_logger

logger = get_logger("extract_features_lazzarotto")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]
_PROCESSED = _ROOT / "data" / "processed"
_FEATURES = _ROOT / "data" / "features"

_CANDIDATES_DEFAULT = _PROCESSED / "candidates_lazzarotto.parquet"
_LAZZAROTTO_PARQUET = _PROCESSED / "lazzarotto2020_guideseq.parquet"
_EXISTING_FM_DEFAULT = _FEATURES / "feature_matrix_combined.parquet"
_OUTPUT_DEFAULT = _FEATURES / "feature_matrix_expanded.parquet"
_AG_CACHE_DEFAULT = _FEATURES / "ag_cache"

# AlphaGenome ontology term for CD4+ T cells
_CD4_TCELL_ONTOLOGY = "CL:0000624"
_CELL_TYPE = "CD4_Tcell"
_DATASET_NAME = "guide_seq_lazzarotto2020"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_lazzarotto_positives(path: Path) -> pd.DataFrame:
    """Load T cell positives for label assignment.

    Returns DataFrame with columns: guide_id, chrom, start.
    """
    if not path.exists():
        logger.error(
            "Lazzarotto parquet not found: %s\n"
            "Run scripts/02b_parse_lazzarotto2020.py first.",
            path,
        )
        sys.exit(1)

    df = pd.read_parquet(path, columns=["guide_id", "chrom", "start"])
    logger.info("T cell positives: %d sites, %d guides", len(df), df["guide_id"].nunique())
    return df


def _load_ag_model():
    """Load the AlphaGenome DnaClient from config.ALPHAGENOME_API_KEY."""
    try:
        import config as cfg  # type: ignore
    except ImportError:
        logger.error(
            "config.py not found.  Create config.py with ALPHAGENOME_API_KEY = '...'."
        )
        return None

    api_key = getattr(cfg, "ALPHAGENOME_API_KEY", None)
    if not api_key:
        logger.error("ALPHAGENOME_API_KEY not set in config.py.")
        return None

    try:
        import src.alphagenome_client as ag_mod
        model = ag_mod.create_client(api_key)
        logger.info("AlphaGenome client created.")
        return model
    except Exception as exc:
        logger.error("Failed to create AlphaGenome client: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract features for T cell candidates and merge with existing matrix"
    )
    parser.add_argument(
        "--candidates", type=Path, default=_CANDIDATES_DEFAULT,
        help="T cell candidates parquet from script 03b",
    )
    parser.add_argument(
        "--existing", type=Path, default=_EXISTING_FM_DEFAULT,
        help="Existing feature matrix to merge with",
    )
    parser.add_argument(
        "--output", type=Path, default=_OUTPUT_DEFAULT,
        help="Output expanded feature matrix parquet",
    )
    parser.add_argument(
        "--ag-cache-dir", type=Path, default=_AG_CACHE_DEFAULT,
        help="AlphaGenome cache directory",
    )
    parser.add_argument(
        "--no-ag", action="store_true",
        help="Skip AlphaGenome features",
    )
    args = parser.parse_args()

    # ---- Validate inputs ---------------------------------------------------
    if not args.candidates.exists():
        logger.error(
            "T cell candidates not found: %s\n"
            "Run script 03b first: python scripts/03b_generate_candidates_lazzarotto.py",
            args.candidates,
        )
        sys.exit(1)

    if not args.existing.exists():
        logger.error(
            "Existing feature matrix not found: %s\n"
            "Run script 04 first: python scripts/04_extract_features.py",
            args.existing,
        )
        sys.exit(1)

    # ---- Load candidates and positives -------------------------------------
    candidates = pd.read_parquet(args.candidates)
    logger.info(
        "Loaded %d T cell candidate sites for %d guides",
        len(candidates), candidates["guide_id"].nunique(),
    )

    positives = _load_lazzarotto_positives(_LAZZAROTTO_PARQUET)

    # Assign dataset column to all candidates
    candidates["dataset"] = _DATASET_NAME

    # ---- Load AlphaGenome client -------------------------------------------
    ag_model = None
    if not args.no_ag:
        ag_model = _load_ag_model()
        if ag_model is None:
            logger.warning(
                "AlphaGenome client unavailable — T cell features will lack "
                "chromatin predictions. Pass --no-ag to suppress."
            )
    else:
        logger.info("Skipping AlphaGenome features (--no-ag)")

    args.ag_cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ---- Build T cell feature matrix (no ENCODE — pass None for manifest) --
    tcell_out = args.output.parent / f"feature_matrix_{_CELL_TYPE}.parquet"

    logger.info("=" * 60)
    logger.info("Building feature matrix for cell_type=%s", _CELL_TYPE)
    logger.info("  AlphaGenome ontology: %s", _CD4_TCELL_ONTOLOGY)
    logger.info("  ENCODE: None (no BigWig for primary T cells)")
    logger.info("=" * 60)

    fm_tcell = build_feature_matrix(
        candidates,
        positives,
        cell_type=_CELL_TYPE,
        encode_manifest=None,       # No ENCODE data for T cells
        ag_model=ag_model,
        ag_ontology_term=_CD4_TCELL_ONTOLOGY if ag_model else None,
        ag_cache_dir=args.ag_cache_dir if ag_model else None,
        output_path=tcell_out,
    )

    n_pos = int(fm_tcell["label"].sum()) if "label" in fm_tcell.columns else 0
    logger.info(
        "T cell feature matrix: %d rows, %d positive (%.1f%%), %d columns",
        len(fm_tcell), n_pos,
        100.0 * n_pos / len(fm_tcell) if len(fm_tcell) else 0.0,
        len(fm_tcell.columns),
    )

    # ---- Load existing feature matrix and merge ----------------------------
    logger.info("Loading existing feature matrix: %s", args.existing)
    fm_existing = pd.read_parquet(args.existing)
    logger.info(
        "Existing: %d rows × %d columns | %d positive",
        len(fm_existing), len(fm_existing.columns),
        int(fm_existing["label"].sum()) if "label" in fm_existing.columns else 0,
    )

    # Ensure T cell matrix has all the same columns as existing
    # Add missing ENCODE columns as NaN (they'll be imputed downstream)
    for col in fm_existing.columns:
        if col not in fm_tcell.columns:
            if col.endswith("_missing"):
                # Missing indicators for ENCODE features: 1 = missing
                fm_tcell[col] = 1
            else:
                fm_tcell[col] = float("nan")

    # Ensure both have same column set (in case T cell has extra cols)
    common_cols = list(fm_existing.columns)
    extra_tcell = [c for c in fm_tcell.columns if c not in common_cols]
    if extra_tcell:
        logger.info(
            "T cell has %d extra columns not in existing matrix (adding to existing as NaN): %s",
            len(extra_tcell), extra_tcell[:10],
        )
        for col in extra_tcell:
            fm_existing[col] = float("nan")
        common_cols.extend(extra_tcell)

    # ---- Impute ENCODE NaN columns for T cell data -------------------------
    # ENCODE features need to be imputed since T cells have no BigWig data.
    # The imputation was already done in build_feature_matrix for AG features,
    # but ENCODE features were skipped (encode_manifest=None).
    # We need to ensure ENCODE columns exist with NaN, then impute.
    from src.feature_matrix import EXP_FEATURE_NAMES, MISSING_INDICATOR_SUFFIX, impute_chromatin_features

    for col in EXP_FEATURE_NAMES:
        if col not in fm_tcell.columns:
            fm_tcell[col] = float("nan")
        miss_col = col + MISSING_INDICATOR_SUFFIX
        if miss_col not in fm_tcell.columns:
            fm_tcell[miss_col] = 1  # all missing for T cells

    # Impute ENCODE features (will fill NaN with 0.0 since entire column is NaN)
    fm_tcell = impute_chromatin_features(fm_tcell, EXP_FEATURE_NAMES)

    # ---- Concatenate -------------------------------------------------------
    combined = pd.concat([fm_existing, fm_tcell], ignore_index=True)

    # Sort columns for consistency
    meta_cols = ["guide_id", "chrom", "start", "strand", "guide_seq",
                 "off_target_seq", "pam_seq", "cell_type", "dataset", "label"]
    meta_present = [c for c in meta_cols if c in combined.columns]
    other_cols = sorted([c for c in combined.columns if c not in meta_present])
    combined = combined[meta_present + other_cols]

    combined.to_parquet(args.output, index=False)

    n_pos_total = int(combined["label"].sum()) if "label" in combined.columns else 0
    logger.info("=" * 60)
    logger.info(
        "EXPANDED feature matrix: %d rows × %d columns",
        len(combined), len(combined.columns),
    )
    logger.info(
        "  %d positive (%.1f%%), %d negative",
        n_pos_total,
        100.0 * n_pos_total / len(combined) if len(combined) else 0.0,
        len(combined) - n_pos_total,
    )

    # Per-cell-type breakdown
    if "cell_type" in combined.columns:
        for ct in sorted(combined["cell_type"].unique()):
            ct_data = combined[combined["cell_type"] == ct]
            ct_pos = int(ct_data["label"].sum()) if "label" in ct_data.columns else 0
            ct_guides = ct_data["guide_id"].nunique()
            logger.info(
                "  %s: %d rows, %d positive, %d guides",
                ct, len(ct_data), ct_pos, ct_guides,
            )

    logger.info("Saved → %s", args.output)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
