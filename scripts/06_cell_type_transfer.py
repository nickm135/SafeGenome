"""
scripts/06_cell_type_transfer.py — Cross-cell-type CRISPR off-target risk scoring.

Loads the trained Model C from script 05 and applies it to score candidate
off-target sites across multiple cell types using AlphaGenome predicted
chromatin.  This is the core demonstration of SafeGenome's key capability:
predicting cell-type-specific off-target risk WITHOUT experimental chromatin data.

Requires
--------
* data/models/model_c.pkl              (from script 05)
* data/processed/candidates_all.parquet (from script 03)
* config.py with ALPHAGENOME_API_KEY

Usage
-----
    source venv/Scripts/activate
    python scripts/06_cell_type_transfer.py [options]

    --model PATH            Model C pickle (default: data/models/model_c.pkl)
    --candidates PATH       Candidates parquet (default: data/processed/candidates_all.parquet)
    --ag-cache-dir PATH     AlphaGenome cache (default: data/features/ag_cache)
    --output-dir PATH       Output directory (default: data/results/cell_type_transfer)
    --cell-types LIST       Comma-separated "Name:OntologyTerm" pairs
                            (default: K562:EFO:0002067,HEK293:EFO:0001182,
                                      CD4_T_cell:CL:0000624,Hepatocyte:CL:0000182)
    --guide-ids LIST        Comma-separated guide_ids to score (default: all)
    --top-n-heatmap N       Number of sites to show in heatmap (default: 50)
    --validate              Also compute AUPRC against K562/HEK293 positives

Outputs (written to --output-dir)
----------------------------------
    cell_type_transfer_scores.csv       Risk scores for all sites × cell types
    cell_type_score_matrix.csv          Wide matrix: sites × cell types
    cell_type_rank_correlation.csv      Spearman rank correlation matrix
    transfer_validation.csv             AUPRC per cell type (if --validate)
    figures/fig3_cell_type_heatmap.png
    figures/cell_type_rank_correlation_heatmap.png

Biological interpretation
--------------------------
The score matrix and heatmap show how off-target cleavage risk varies by
cell type.  Sites with uniformly high risk across all cell types should be
flagged regardless of clinical application.  Sites with high risk only in
specific cell types (e.g. K562 but not T cells) warrant cell-type-specific
safety evaluation.

The rank correlation heatmap shows how similar the risk rankings are across
cell types.  High correlation indicates the guide's off-target landscape is
similar across cell types (likely driven by sequence accessibility shared by
most cell types).  Low correlation indicates cell-type-specific chromatin
drives differential risk.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

# --- path setup so src imports work when run as a script ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cell_type_transfer import RECOMMENDED_CELL_TYPES, run_cell_type_transfer
from src.utils import get_logger

logger = get_logger("cell_type_transfer")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]
_MODELS = _ROOT / "data" / "models"
_PROCESSED = _ROOT / "data" / "processed"
_FEATURES = _ROOT / "data" / "features"
_RESULTS = _ROOT / "data" / "results" / "cell_type_transfer"

_MODEL_C_DEFAULT = _MODELS / "model_c.pkl"
_CANDIDATES_DEFAULT = _PROCESSED / "candidates_all.parquet"
_AG_CACHE_DEFAULT = _FEATURES / "ag_cache"

# Default cell types for transfer demo (all have AlphaGenome tracks)
_DEFAULT_CELL_TYPE_MAP: dict[str, str] = {
    "K562": "EFO:0002067",         # Chronic myelogenous leukemia (full tracks)
    "HEK293": "EFO:0001182",       # Embryonic kidney (no DNase/ATAC tracks)
    "CD4_T_cell": "CL:0000624",    # CD4+ T cell (relevant for therapeutic CRISPR)
    "Hepatocyte": "CL:0000182",    # Hepatocyte (liver; relevant for in vivo delivery)
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_cell_type_arg(cell_type_str: str) -> dict[str, str]:
    """Parse 'Name:OntologyTerm,...' argument into a dict.

    Handles both formats:
        K562:EFO:0002067          (ontology with colon in term)
        K562:EFO_0002067          (underscore separator)
    Strategy: split on first colon → name, then rest → ontology term.
    """
    result: dict[str, str] = {}
    for entry in cell_type_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        # Split on first colon only: "K562:EFO:0002067" → ("K562", "EFO:0002067")
        parts = entry.split(":", maxsplit=1)
        if len(parts) != 2:
            logger.warning("Could not parse cell type entry: %r (expected Name:Term)", entry)
            continue
        name, term = parts[0].strip(), parts[1].strip()
        result[name] = term
    return result


def _load_model_c(path: Path) -> tuple:
    """Load Model C artifact from pickle.

    Returns (model, feature_cols, metadata_dict).
    """
    if not path.exists():
        logger.error(
            "Model C not found: %s\n"
            "Run script 05 first: python scripts/05_train_models.py",
            path,
        )
        sys.exit(1)

    with open(path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    logger.info(
        "Loaded Model C: %d features, AUPRC=%.4f±%.4f",
        len(feature_cols),
        artifact.get("mean_auprc", float("nan")),
        artifact.get("std_auprc", float("nan")),
    )
    return model, feature_cols, artifact


def _load_positives_for_validation() -> dict[str, pd.DataFrame] | None:
    """Load K562 and HEK293 positives for AUPRC validation.

    Returns None if parquets are not found.
    """
    pos_by_ct: dict[str, pd.DataFrame] = {}

    guide_seq_path = _PROCESSED / "guide_seq_tsai2015.parquet"
    deepcrispr_path = _PROCESSED / "deepcrispr_k562.parquet"
    discover_seq_path = _PROCESSED / "discover_seq_k562.parquet"

    if guide_seq_path.exists():
        gs = pd.read_parquet(guide_seq_path)
        hek_pos = gs[gs["cell_type"] == "HEK293"][["guide_id", "chrom", "start"]].copy()
        if len(hek_pos) > 0:
            pos_by_ct["HEK293"] = hek_pos

    k562_frames = []
    for path in [deepcrispr_path, discover_seq_path]:
        if path.exists():
            k562_frames.append(pd.read_parquet(path)[["guide_id", "chrom", "start"]])
    if k562_frames:
        pos_by_ct["K562"] = pd.concat(k562_frames, ignore_index=True).drop_duplicates()

    return pos_by_ct if pos_by_ct else None


def _load_ag_client():
    """Load AlphaGenome DnaClient from config.py."""
    try:
        import config as cfg  # type: ignore
    except ImportError:
        logger.error(
            "config.py not found.  Create config.py with ALPHAGENOME_API_KEY = '...'."
        )
        sys.exit(1)

    api_key = getattr(cfg, "ALPHAGENOME_API_KEY", None)
    if not api_key:
        logger.error("ALPHAGENOME_API_KEY not set in config.py.")
        sys.exit(1)

    import src.alphagenome_client as ag_mod  # type: ignore
    client = ag_mod.create_client(api_key)
    logger.info("AlphaGenome client created.")
    return client


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-cell-type off-target risk scoring with trained Model C"
    )
    parser.add_argument(
        "--model", type=Path, default=_MODEL_C_DEFAULT,
        help="Model C pickle from script 05 (default: data/models/model_c.pkl)",
    )
    parser.add_argument(
        "--candidates", type=Path, default=_CANDIDATES_DEFAULT,
        help="Candidates parquet from script 03 (default: data/processed/candidates_all.parquet)",
    )
    parser.add_argument(
        "--ag-cache-dir", type=Path, default=_AG_CACHE_DEFAULT,
        help="AlphaGenome cache directory (default: data/features/ag_cache)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_RESULTS,
        help="Output directory (default: data/results/cell_type_transfer)",
    )
    parser.add_argument(
        "--cell-types", type=str, default=None,
        help=(
            "Comma-separated 'Name:OntologyTerm' pairs.  "
            "Default: K562:EFO:0002067,HEK293:EFO:0001182,"
            "CD4_T_cell:CL:0000624,Hepatocyte:CL:0000182"
        ),
    )
    parser.add_argument(
        "--guide-ids", type=str, default=None,
        help="Comma-separated guide_ids to score (default: all guides in candidates)",
    )
    parser.add_argument(
        "--top-n-heatmap", type=int, default=50,
        help="Number of top sites to display in the heatmap figure (default: 50)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Compute AUPRC against K562/HEK293 experimental positives",
    )
    args = parser.parse_args()

    # ---- Parse cell-type map ------------------------------------------------
    if args.cell_types:
        cell_type_map = _parse_cell_type_arg(args.cell_types)
        if not cell_type_map:
            logger.error("No valid cell types parsed from: %r", args.cell_types)
            sys.exit(1)
    else:
        cell_type_map = _DEFAULT_CELL_TYPE_MAP

    logger.info("Cell types for transfer: %s", list(cell_type_map.keys()))

    # ---- Load model ---------------------------------------------------------
    model, feature_cols, _ = _load_model_c(args.model)

    # ---- Load candidates ----------------------------------------------------
    if not args.candidates.exists():
        logger.error(
            "Candidates parquet not found: %s\n"
            "Run script 03 first: python scripts/03_generate_candidates.py",
            args.candidates,
        )
        sys.exit(1)

    candidates = pd.read_parquet(args.candidates)
    logger.info(
        "Loaded %d candidates for %d guides",
        len(candidates), candidates["guide_id"].nunique(),
    )

    # Filter to specified guide_ids if requested
    if args.guide_ids:
        requested_ids = {g.strip() for g in args.guide_ids.split(",")}
        before = len(candidates)
        candidates = candidates[candidates["guide_id"].isin(requested_ids)].copy()
        logger.info(
            "Filtered to guide_ids=%s: %d → %d candidates",
            requested_ids, before, len(candidates),
        )
        if candidates.empty:
            logger.error(
                "No candidates remain after guide_id filter.  "
                "Available guide_ids: %s",
                sorted(pd.read_parquet(args.candidates)["guide_id"].unique()),
            )
            sys.exit(1)

    # ---- Load AlphaGenome client --------------------------------------------
    ag_client = _load_ag_client()

    # ---- Load positives for validation (optional) ---------------------------
    positives_by_ct: dict[str, pd.DataFrame] | None = None
    if args.validate:
        positives_by_ct = _load_positives_for_validation()
        if positives_by_ct:
            logger.info(
                "Validation positives loaded for: %s", list(positives_by_ct.keys())
            )
        else:
            logger.warning("No positives found for validation — skipping AUPRC computation.")

    # ---- Create output directories ------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.ag_cache_dir.mkdir(parents=True, exist_ok=True)

    # Create figures subdirectory within output_dir
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run cell-type transfer ---------------------------------------------
    logger.info("Starting cell-type transfer analysis ...")
    logger.info(
        "Scoring %d candidates across %d cell types",
        len(candidates), len(cell_type_map),
    )

    results = run_cell_type_transfer(
        candidates=candidates,
        cell_type_map=cell_type_map,
        model=model,
        feature_cols=feature_cols,
        ag_client=ag_client,
        ag_cache_dir=args.ag_cache_dir,
        output_dir=args.output_dir,
        top_n_heatmap=args.top_n_heatmap,
        sort_heatmap_by="cfd_score",
        positives_by_cell_type=positives_by_ct,
    )

    # ---- Summary -----------------------------------------------------------
    transfer_df = results.get("transfer_df")
    score_matrix = results.get("score_matrix")
    rank_corr = results.get("rank_corr")
    validation = results.get("validation", {})

    if score_matrix is not None:
        logger.info(
            "Score matrix: %d sites × %d cell types",
            score_matrix.shape[0], score_matrix.shape[1],
        )

    if rank_corr is not None:
        logger.info("Rank correlation matrix:\n%s", rank_corr.round(3).to_string())

    if validation:
        print("\n" + "=" * 55)
        print("Validation AUPRC (Model C applied to training cell types)")
        print("=" * 55)
        for ct, metrics in validation.items():
            print(
                f"  {ct:<15} AUPRC={metrics.get('auprc', float('nan')):.4f}  "
                f"n_pos={metrics.get('n_positives', 'N/A')}"
            )
        print()
        print("Note: Validation uses sites/positives seen during training.")
        print("This is an in-sample check, not out-of-sample generalisation.")
        print("=" * 55 + "\n")

    logger.info(
        "Cell-type transfer complete.  Results saved → %s", args.output_dir
    )
    logger.info(
        "Key outputs:\n"
        "  %s/cell_type_transfer_scores.csv\n"
        "  %s/cell_type_score_matrix.csv\n"
        "  %s/figures/fig3_cell_type_heatmap.png",
        args.output_dir, args.output_dir, args.output_dir,
    )


if __name__ == "__main__":
    main()
