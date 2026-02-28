"""
scripts/04_extract_features.py — Extract sequence, AlphaGenome, and ENCODE features.

Builds per-cell-type feature matrices for K562 and HEK293, then concatenates
them into a single combined feature matrix for model training.

Requires
--------
* data/processed/candidates_all.parquet  (from script 03)
* data/processed/guide_seq_tsai2015.parquet
* data/processed/deepcrispr_k562.parquet
* data/processed/discover_seq_k562.parquet
* config.py with ALPHAGENOME_API_KEY (for Model C features)
* data/experimental_chromatin/*.bigWig   (for Model B features)

Usage
-----
    source venv/Scripts/activate
    python scripts/04_extract_features.py [options]

    --candidates PATH   Candidates parquet (default: data/processed/candidates_all.parquet)
    --bigwig-dir PATH   ENCODE BigWig directory (default: data/experimental_chromatin)
    --ag-cache-dir PATH AlphaGenome cache directory (default: data/features/ag_cache)
    --output-dir PATH   Output directory (default: data/features)
    --no-ag             Skip AlphaGenome features (produce sequence-only + ENCODE)
    --no-encode         Skip ENCODE features (produce sequence-only + AlphaGenome)
    --cell-types LIST   Comma-separated cell types to process (default: K562,HEK293)

Outputs
-------
    data/features/feature_matrix_K562.parquet
    data/features/feature_matrix_HEK293.parquet
    data/features/feature_matrix_combined.parquet   ← used by script 05

Cell-type notes
---------------
* K562 (EFO:0002067): full AlphaGenome tracks + all 5 ENCODE assays
* HEK293 (EFO:0001182): AlphaGenome has no DNase/ATAC → those features = NaN
  HEK293 ENCODE: missing DNase + H3K27me3 BigWig → those features = NaN
* U2OS: no AlphaGenome tracks; no ENCODE BigWig data downloaded.
  U2OS rows (from GUIDE-seq) are NOT included by default.  They are useful
  only for Model A (sequence-only).  To include them, add U2OS to --cell-types
  and adjust positives loading logic.
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

logger = get_logger("extract_features")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]
_PROCESSED = _ROOT / "data" / "processed"
_FEATURES = _ROOT / "data" / "features"

_CANDIDATES_DEFAULT = _PROCESSED / "candidates_all.parquet"
_GUIDE_SEQ_PATH = _PROCESSED / "guide_seq_tsai2015.parquet"
_DEEPCRISPR_PATH = _PROCESSED / "deepcrispr_k562.parquet"
_DISCOVER_SEQ_PATH = _PROCESSED / "discover_seq_k562.parquet"
_BIGWIG_DIR_DEFAULT = _ROOT / "data" / "experimental_chromatin"
_AG_CACHE_DEFAULT = _FEATURES / "ag_cache"

# AlphaGenome ontology terms
_ONTOLOGY: dict[str, str] = {
    "K562": "EFO:0002067",
    "HEK293": "EFO:0001182",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_positives(
    guide_seq_path: Path,
    deepcrispr_path: Path,
    discover_seq_path: Path,
) -> dict[str, pd.DataFrame]:
    """Load and split positives by cell type.

    Returns a dict: cell_type → positives DataFrame with columns
    (guide_id, chrom, start) — only the columns required by label_candidates.

    K562 positives: DeepCRISPR K562 + DISCOVER-seq K562.
    HEK293 positives: GUIDE-seq rows where cell_type == "HEK293".
    """
    positives_by_ct: dict[str, pd.DataFrame] = {}

    # --- GUIDE-seq → split by cell_type ---
    if guide_seq_path.exists():
        gs = pd.read_parquet(guide_seq_path)
        hek_pos = gs[gs["cell_type"] == "HEK293"][["guide_id", "chrom", "start"]].copy()
        if len(hek_pos) > 0:
            positives_by_ct["HEK293"] = hek_pos
            logger.info("HEK293 positives from GUIDE-seq: %d sites", len(hek_pos))
        else:
            logger.warning("No HEK293 rows found in GUIDE-seq parquet.")
    else:
        logger.warning("GUIDE-seq parquet not found: %s", guide_seq_path)

    # --- DeepCRISPR K562 ---
    k562_frames: list[pd.DataFrame] = []
    if deepcrispr_path.exists():
        dc = pd.read_parquet(deepcrispr_path)[["guide_id", "chrom", "start"]]
        k562_frames.append(dc)
        logger.info("K562 positives from DeepCRISPR: %d sites", len(dc))
    else:
        logger.warning("DeepCRISPR parquet not found: %s", deepcrispr_path)

    # --- DISCOVER-seq K562 ---
    if discover_seq_path.exists():
        ds = pd.read_parquet(discover_seq_path)[["guide_id", "chrom", "start"]]
        k562_frames.append(ds)
        logger.info("K562 positives from DISCOVER-seq: %d sites", len(ds))
    else:
        logger.warning("DISCOVER-seq parquet not found: %s", discover_seq_path)

    if k562_frames:
        k562_pos = pd.concat(k562_frames, ignore_index=True).drop_duplicates()
        positives_by_ct["K562"] = k562_pos
        logger.info("K562 positives total: %d sites", len(k562_pos))
    else:
        logger.warning("No K562 positives loaded.")

    return positives_by_ct


def _build_guide_dataset_map(
    guide_seq_path: Path,
    deepcrispr_path: Path,
    discover_seq_path: Path,
) -> dict[str, dict[str, str]]:
    """Build per-cell-type guide_id → dataset_name mappings.

    Returns a nested dict: cell_type → {guide_id: dataset_name}.
    Used to set the 'dataset' column on candidates before feature extraction
    so that model.py's exclude_datasets parameter works correctly.
    """
    mapping: dict[str, dict[str, str]] = {"K562": {}, "HEK293": {}}

    if guide_seq_path.exists():
        gs = pd.read_parquet(guide_seq_path, columns=["guide_id", "cell_type"])
        for gid in gs[gs["cell_type"] == "HEK293"]["guide_id"].unique():
            mapping["HEK293"][gid] = "guide_seq_tsai2015"

    if deepcrispr_path.exists():
        dc = pd.read_parquet(deepcrispr_path, columns=["guide_id"])
        for gid in dc["guide_id"].unique():
            mapping["K562"][gid] = "deepcrispr_k562"

    if discover_seq_path.exists():
        ds = pd.read_parquet(discover_seq_path, columns=["guide_id"])
        for gid in ds["guide_id"].unique():
            # DISCOVER-seq takes priority over DeepCRISPR if guide appears in both
            mapping["K562"][gid] = "discover_seq_k562"

    return mapping


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
        import src.alphagenome_client as ag_mod  # type: ignore
        model = ag_mod.create_client(api_key)
        logger.info("AlphaGenome client created.")
        return model
    except Exception as exc:
        logger.error("Failed to create AlphaGenome client: %s", exc)
        return None


def _load_encode_manifest(bigwig_dir: Path):
    """Build ENCODE BigWig manifest from the given directory."""
    import src.encode_client as enc_mod  # type: ignore

    if not bigwig_dir.exists():
        logger.warning("BigWig directory not found: %s", bigwig_dir)
        return None

    manifest = enc_mod.build_manifest(bigwig_dir)
    available = list(manifest.files.keys()) if hasattr(manifest, "files") else []
    logger.info("ENCODE manifest built.  Available cell-type+assay keys: %s", available)
    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract sequence + chromatin features → per-cell-type feature matrices"
    )
    parser.add_argument(
        "--candidates", type=Path, default=_CANDIDATES_DEFAULT,
        help="Candidates parquet from script 03 (default: data/processed/candidates_all.parquet)",
    )
    parser.add_argument(
        "--bigwig-dir", type=Path, default=_BIGWIG_DIR_DEFAULT,
        help="ENCODE BigWig directory (default: data/experimental_chromatin)",
    )
    parser.add_argument(
        "--ag-cache-dir", type=Path, default=_AG_CACHE_DEFAULT,
        help="AlphaGenome cache directory (default: data/features/ag_cache)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_FEATURES,
        help="Output directory for feature matrix parquets (default: data/features)",
    )
    parser.add_argument(
        "--no-ag", action="store_true",
        help="Skip AlphaGenome features (sequence + ENCODE only)",
    )
    parser.add_argument(
        "--no-encode", action="store_true",
        help="Skip ENCODE features (sequence + AlphaGenome only)",
    )
    parser.add_argument(
        "--cell-types", type=str, default="K562,HEK293",
        help="Comma-separated cell types to process (default: K562,HEK293)",
    )
    args = parser.parse_args()

    cell_types = [ct.strip() for ct in args.cell_types.split(",")]
    logger.info("Processing cell types: %s", cell_types)

    # ---- Validate candidates -----------------------------------------------
    if not args.candidates.exists():
        logger.error(
            "Candidates parquet not found: %s\n"
            "Run script 03 first: python scripts/03_generate_candidates.py",
            args.candidates,
        )
        sys.exit(1)

    candidates = pd.read_parquet(args.candidates)
    logger.info("Loaded %d candidate sites for %d guides", len(candidates),
                candidates["guide_id"].nunique())

    # ---- Load positives and dataset mappings --------------------------------
    positives_by_ct = _load_positives(
        _GUIDE_SEQ_PATH, _DEEPCRISPR_PATH, _DISCOVER_SEQ_PATH
    )
    guide_dataset_map = _build_guide_dataset_map(
        _GUIDE_SEQ_PATH, _DEEPCRISPR_PATH, _DISCOVER_SEQ_PATH
    )

    # ---- Load chromatin data sources ----------------------------------------
    ag_model = None
    if not args.no_ag:
        ag_model = _load_ag_model()
        if ag_model is None:
            logger.warning(
                "AlphaGenome client unavailable.  "
                "Feature matrices will lack Model C (AG) features.  "
                "Pass --no-ag to suppress this warning."
            )
    else:
        logger.info("Skipping AlphaGenome features (--no-ag)")

    encode_manifest = None
    if not args.no_encode:
        encode_manifest = _load_encode_manifest(args.bigwig_dir)
        if encode_manifest is None:
            logger.warning(
                "ENCODE manifest unavailable.  "
                "Feature matrices will lack Model B (ENCODE) features."
            )
    else:
        logger.info("Skipping ENCODE features (--no-encode)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.ag_cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build per-cell-type feature matrices --------------------------------
    feature_matrix_paths: list[Path] = []

    for cell_type in cell_types:
        logger.info("=" * 60)
        logger.info("Building feature matrix for cell_type=%s", cell_type)
        logger.info("=" * 60)

        if cell_type not in positives_by_ct:
            logger.error(
                "No positives loaded for cell_type=%s — skipping.  "
                "Check that processed parquets exist.",
                cell_type,
            )
            continue

        positives = positives_by_ct[cell_type]

        # Filter candidates to guides that have positives in this cell type
        valid_guide_ids = set(positives["guide_id"].unique())
        ct_candidates = candidates[candidates["guide_id"].isin(valid_guide_ids)].copy()
        logger.info(
            "Filtered to %d guides with %s positives: %d candidates",
            len(valid_guide_ids), cell_type, len(ct_candidates),
        )

        if ct_candidates.empty:
            logger.error("No candidates for %s after guide_id filter.", cell_type)
            continue

        # Assign dataset column based on which source dataset each guide is from
        ct_guide_dataset = guide_dataset_map.get(cell_type, {})
        ct_candidates["dataset"] = (
            ct_candidates["guide_id"]
            .map(ct_guide_dataset)
            .fillna("other")
        )

        # Select AlphaGenome ontology term for this cell type
        ag_ontology = _ONTOLOGY.get(cell_type) if ag_model is not None else None

        out_path = args.output_dir / f"feature_matrix_{cell_type}.parquet"

        fm = build_feature_matrix(
            ct_candidates,
            positives,
            cell_type=cell_type,
            encode_manifest=encode_manifest,
            ag_model=ag_model,
            ag_ontology_term=ag_ontology,
            ag_cache_dir=args.ag_cache_dir,
            output_path=out_path,
        )

        n_pos = int(fm["label"].sum()) if "label" in fm.columns else 0
        logger.info(
            "Cell type %s complete: %d rows, %d positive (%.1f%%)",
            cell_type, len(fm), n_pos,
            100.0 * n_pos / len(fm) if len(fm) else 0.0,
        )
        feature_matrix_paths.append(out_path)

    # ---- Combine all cell-type matrices into one file -----------------------
    if not feature_matrix_paths:
        logger.error("No feature matrices were built — check errors above.")
        sys.exit(1)

    logger.info("Concatenating %d cell-type matrices → combined", len(feature_matrix_paths))
    combined_frames = [pd.read_parquet(p) for p in feature_matrix_paths]
    combined = pd.concat(combined_frames, ignore_index=True)

    combined_path = args.output_dir / "feature_matrix_combined.parquet"
    combined.to_parquet(combined_path, index=False)

    n_pos_total = int(combined["label"].sum()) if "label" in combined.columns else 0
    logger.info(
        "Combined feature matrix: %d rows × %d columns | "
        "%d positive (%.1f%%) | %d negative | saved → %s",
        len(combined), len(combined.columns),
        n_pos_total,
        100.0 * n_pos_total / len(combined) if len(combined) else 0.0,
        len(combined) - n_pos_total,
        combined_path,
    )


if __name__ == "__main__":
    main()
