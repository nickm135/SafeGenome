"""
scripts/03_generate_candidates.py — Enumerate candidate off-target sites with Cas-OFFinder.

For each unique guide RNA in the training datasets (GUIDE-seq, DeepCRISPR K562,
DISCOVER-seq K562), run Cas-OFFinder against hg38 to find all candidate off-target
sites within the mismatch threshold.

CHANGE-seq guides are excluded by default: that assay is performed on naked DNA
and cannot be used as positive labels for in-cell cleavage (naked DNA, no chromatin context).
Pass --include-change-seq only if you need the extra candidate sites.

Usage
-----
    source venv/Scripts/activate
    python scripts/03_generate_candidates.py [options]

    --genome PATH           Path to per-chrom FASTA directory (default: data/raw/genome/chroms)
    --max-mismatches N      Cas-OFFinder mismatch threshold (default: 4)
    --output PATH           Output parquet (default: data/processed/candidates_all.parquet)
    --include-change-seq    Also run Cas-OFFinder for CHANGE-seq guides (110 extra guides)
    --guide-seq-only        Only run for GUIDE-seq guides (9 guides; faster for testing)

Output schema
-------------
    guide_id            str         e.g. "VEGFA_site2"
    guide_seq           str         20-nt protospacer (no PAM, uppercase)
    chrom               str         normalized chromosome, e.g. "chr6"
    start               int64       0-based start (BED convention)
    strand              str         "+" or "-"
    off_target_seq      str         20-nt off-target spacer (no PAM)
    pam_seq             str         3-nt PAM observed at the site
    num_mismatches      int64       spacer mismatches (0 = on-target; removed)
    mismatch_positions  object      list[int], 1-based positions of mismatches
    source_datasets     str         comma-separated datasets this guide appears in
    primary_dataset     str         primary source dataset (for exclude_datasets filtering)

Design notes
------------
* Cas-OFFinder is run ONCE per unique guide_seq.  If the same guide_seq appears
  under multiple guide_ids (e.g. "VEGFA_site2" in both GUIDE-seq and DISCOVER-seq),
  the candidate rows are duplicated so each guide_id gets its own rows.  This
  is necessary because positive labels are assigned per-guide_id in script 04.
* On-target sites (num_mismatches == 0) are removed by find_offtargets
  (remove_ontarget=True) per project protocol.
* All data drops are logged for the paper Methods section.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# --- path setup so src imports work when run as a script ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.offtarget_finder import find_offtargets
from src.utils import get_logger

logger = get_logger("generate_candidates")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]
_RAW = _ROOT / "data" / "raw"
_PROCESSED = _ROOT / "data" / "processed"

_GENOME_DEFAULT = _RAW / "genome" / "chroms"  # directory of per-chrom FASTAs (Cas-OFFinder v2 requires this)
_GUIDE_SEQ_PATH = _PROCESSED / "guide_seq_tsai2015.parquet"
_CHANGE_SEQ_PATH = _PROCESSED / "change_seq_lazzarotto2020.parquet"
_DEEPCRISPR_PATH = _PROCESSED / "deepcrispr_k562.parquet"
_DISCOVER_SEQ_PATH = _PROCESSED / "discover_seq_k562.parquet"
_CANDIDATES_OUT = _PROCESSED / "candidates_all.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_guides_from_parquet(
    path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """Load (guide_id, guide_seq, dataset) from a processed parquet.

    Returns a DataFrame with columns: guide_id, guide_seq, dataset.
    Drops guide_ids with missing guide_seq.
    """
    if not path.exists():
        logger.warning("Parquet not found, skipping: %s", path)
        return pd.DataFrame(columns=["guide_id", "guide_seq", "dataset"])

    df = pd.read_parquet(path, columns=["guide_id", "guide_seq"])
    before = len(df)
    df = df.dropna(subset=["guide_seq"])
    if len(df) < before:
        logger.warning(
            "[%s] Dropped %d rows with missing guide_seq",
            dataset_name, before - len(df),
        )

    # Unique (guide_id, guide_seq) pairs
    df = df.drop_duplicates(subset=["guide_id", "guide_seq"])
    df["dataset"] = dataset_name
    logger.info("[%s] %d unique guide_ids loaded", dataset_name, len(df))
    return df[["guide_id", "guide_seq", "dataset"]]


def _run_casoffinder_for_guide(
    guide_id: str,
    guide_seq: str,
    genome_path: Path,
    max_mismatches: int,
) -> pd.DataFrame | None:
    """Run Cas-OFFinder for a single guide and return a tidy DataFrame.

    Returns None if Cas-OFFinder fails for this guide.
    Renames output column 'guide' → 'guide_seq' and adds 'guide_id'.
    """
    try:
        df = find_offtargets(
            guide_seq,
            genome_path,
            max_mismatches=max_mismatches,
            remove_ontarget=True,
        )
    except Exception as exc:
        logger.error(
            "Cas-OFFinder failed for guide_id=%s guide_seq=%s: %s",
            guide_id, guide_seq, exc,
        )
        return None

    if df.empty:
        logger.warning("No candidates found for guide_id=%s", guide_id)
        return df

    # find_offtargets returns column 'guide' (the input sequence); rename to guide_seq
    df = df.rename(columns={"guide": "guide_seq"})
    df["guide_id"] = guide_id
    logger.info(
        "  guide_id=%-25s  candidates=%d  max_mm=%d",
        guide_id, len(df), df["num_mismatches"].max() if len(df) else 0,
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Cas-OFFinder for all training guides → candidates_all.parquet"
    )
    parser.add_argument(
        "--genome", type=Path, default=_GENOME_DEFAULT,
        help="Path to directory of per-chrom FASTA files (default: data/raw/genome/chroms)",
    )
    parser.add_argument(
        "--max-mismatches", type=int, default=4,
        help="Cas-OFFinder mismatch threshold (default: 4)",
    )
    parser.add_argument(
        "--output", type=Path, default=_CANDIDATES_OUT,
        help="Output parquet path (default: data/processed/candidates_all.parquet)",
    )
    parser.add_argument(
        "--include-change-seq", action="store_true",
        help="Also run Cas-OFFinder for CHANGE-seq guides (110 guides, slow)",
    )
    parser.add_argument(
        "--guide-seq-only", action="store_true",
        help="Only run for GUIDE-seq guides (9 guides; useful for quick testing)",
    )
    args = parser.parse_args()

    genome_path: Path = args.genome
    if not genome_path.exists():
        logger.error("Genome directory not found: %s", genome_path)
        logger.error(
            "Expected a directory of per-chromosome FASTA files (chr1.fa, chr2.fa, ...).\n"
            "Split hg38.fa first — see the genome split script or README."
        )
        sys.exit(1)

    # ---- Collect guides from each dataset -----------------------------------
    guide_frames: list[pd.DataFrame] = []

    guide_seq_guides = _load_guides_from_parquet(_GUIDE_SEQ_PATH, "guide_seq_tsai2015")
    guide_frames.append(guide_seq_guides)

    if not args.guide_seq_only:
        deepcrispr_guides = _load_guides_from_parquet(_DEEPCRISPR_PATH, "deepcrispr_k562")
        discover_guides = _load_guides_from_parquet(_DISCOVER_SEQ_PATH, "discover_seq_k562")
        guide_frames.extend([deepcrispr_guides, discover_guides])

        if args.include_change_seq:
            change_guides = _load_guides_from_parquet(
                _CHANGE_SEQ_PATH, "change_seq_lazzarotto2020"
            )
            guide_frames.append(change_guides)
            logger.warning(
                "Including CHANGE-seq: %d guides. "
                "Remember: CHANGE-seq sites are NOT valid in-cell positives (naked DNA, no chromatin context).",
                len(change_guides),
            )

    all_guides = pd.concat(guide_frames, ignore_index=True)

    # ---- Build guide_seq → set of (guide_id, datasets) mapping -------------
    # Group by guide_seq so we run Cas-OFFinder ONCE per unique sequence.
    # Multiple guide_ids sharing the same guide_seq get separate output rows.
    guide_seq_to_rows: dict[str, list[dict]] = {}
    for _, row in all_guides.iterrows():
        seq = row["guide_seq"]
        if seq not in guide_seq_to_rows:
            guide_seq_to_rows[seq] = []
        guide_seq_to_rows[seq].append(
            {"guide_id": row["guide_id"], "dataset": row["dataset"]}
        )

    # For guides with the same guide_id but different datasets, merge datasets
    # into source_datasets and keep the first (primary) dataset
    guide_id_to_info: dict[str, dict] = {}
    guide_id_to_seq: dict[str, str] = {}
    for seq, rows in guide_seq_to_rows.items():
        for r in rows:
            gid = r["guide_id"]
            if gid not in guide_id_to_info:
                guide_id_to_info[gid] = {"datasets": [r["dataset"]], "primary": r["dataset"]}
                guide_id_to_seq[gid] = seq
            else:
                if r["dataset"] not in guide_id_to_info[gid]["datasets"]:
                    guide_id_to_info[gid]["datasets"].append(r["dataset"])

    total_unique_seqs = len(set(guide_id_to_seq.values()))
    total_guide_ids = len(guide_id_to_seq)
    logger.info(
        "Total: %d unique guide_ids across %d unique sequences — "
        "will run Cas-OFFinder %d times",
        total_guide_ids, total_unique_seqs, total_unique_seqs,
    )

    # ---- Run Cas-OFFinder per unique guide_seq, duplicate for each guide_id -
    # We group guide_ids by their guide_seq to run the binary only once per seq.
    seq_to_guide_ids: dict[str, list[str]] = {}
    for gid, seq in guide_id_to_seq.items():
        seq_to_guide_ids.setdefault(seq, []).append(gid)

    all_candidates: list[pd.DataFrame] = []
    failed: list[str] = []

    for i, (guide_seq, guide_ids) in enumerate(seq_to_guide_ids.items(), start=1):
        logger.info(
            "Running Cas-OFFinder %d/%d  guide_seq=%s  guide_ids=%s",
            i, len(seq_to_guide_ids), guide_seq, guide_ids,
        )
        # Run once for this sequence (use first guide_id as label for the run)
        primary_gid = guide_ids[0]
        df = _run_casoffinder_for_guide(
            primary_gid, guide_seq, genome_path, args.max_mismatches
        )
        if df is None:
            failed.extend(guide_ids)
            continue
        if df.empty:
            continue

        # Duplicate for each guide_id sharing this sequence
        for gid in guide_ids:
            gdf = df.copy()
            gdf["guide_id"] = gid
            info = guide_id_to_info[gid]
            gdf["source_datasets"] = ",".join(info["datasets"])
            gdf["primary_dataset"] = info["primary"]
            all_candidates.append(gdf)

    if failed:
        logger.error("Cas-OFFinder failed for %d guide_ids: %s", len(failed), failed)

    if not all_candidates:
        logger.error("No candidates generated — check genome path and Cas-OFFinder installation.")
        sys.exit(1)

    # ---- Assemble and save --------------------------------------------------
    combined = pd.concat(all_candidates, ignore_index=True)

    # Enforce correct dtypes
    combined["start"] = combined["start"].astype("int64")
    combined["num_mismatches"] = combined["num_mismatches"].astype("int64")

    # Column ordering
    ordered_cols = [
        "guide_id", "guide_seq", "chrom", "start", "strand",
        "off_target_seq", "pam_seq", "num_mismatches", "mismatch_positions",
        "source_datasets", "primary_dataset",
    ]
    extra_cols = [c for c in combined.columns if c not in ordered_cols]
    combined = combined[ordered_cols + extra_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)

    logger.info(
        "Saved %d candidate sites for %d guide_ids → %s",
        len(combined), combined["guide_id"].nunique(), args.output,
    )
    logger.info(
        "Mismatch distribution:\n%s",
        combined["num_mismatches"].value_counts().sort_index().to_string(),
    )


if __name__ == "__main__":
    main()
