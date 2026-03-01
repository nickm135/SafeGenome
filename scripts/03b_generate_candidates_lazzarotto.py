"""
scripts/03b_generate_candidates_lazzarotto.py — Enumerate candidate off-target
sites for Lazzarotto 2020 T cell GUIDE-seq guides using Cas-OFFinder.

Loads the 44 T cell guide sequences from the parsed Lazzarotto 2020 parquet
and runs Cas-OFFinder against hg38 (max_mismatches=4, GPU mode) to generate
candidate off-target sites.  Output schema matches candidates_all.parquet
exactly so it can be used directly by script 04b.

Usage
-----
    source venv/Scripts/activate
    python scripts/03b_generate_candidates_lazzarotto.py [options]

    --genome PATH           Path to per-chrom FASTA directory (default: data/raw/genome/chroms)
    --max-mismatches N      Cas-OFFinder mismatch threshold (default: 4)
    --output PATH           Output parquet (default: data/processed/candidates_lazzarotto.parquet)

Output schema (identical to candidates_all.parquet)
---------------------------------------------------
    guide_id            str         e.g. "Laz_01"
    guide_seq           str         20-nt protospacer (no PAM, uppercase)
    chrom               str         normalized chromosome, e.g. "chr6"
    start               int64       0-based start (BED convention)
    strand              str         "+" or "-"
    off_target_seq      str         20-nt off-target spacer (no PAM)
    pam_seq             str         3-nt PAM observed at the site
    num_mismatches      int64       spacer mismatches (0 = on-target; removed)
    mismatch_positions  object      list[int], 1-based positions of mismatches
    source_datasets     str         "guide_seq_lazzarotto2020"
    primary_dataset     str         "guide_seq_lazzarotto2020"
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

logger = get_logger("generate_candidates_lazzarotto")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]
_RAW = _ROOT / "data" / "raw"
_PROCESSED = _ROOT / "data" / "processed"

_GENOME_DEFAULT = _RAW / "genome" / "chroms"
_LAZZAROTTO_PATH = _PROCESSED / "lazzarotto2020_guideseq.parquet"
_CANDIDATES_OUT = _PROCESSED / "candidates_lazzarotto.parquet"

_DATASET_NAME = "guide_seq_lazzarotto2020"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_lazzarotto_guides(path: Path) -> pd.DataFrame:
    """Load unique (guide_id, guide_seq) pairs from parsed Lazzarotto parquet.

    Returns
    -------
    pd.DataFrame
        Columns: guide_id, guide_seq (20-nt, no PAM).
    """
    if not path.exists():
        logger.error(
            "Lazzarotto parquet not found: %s\n"
            "Run scripts/02b_parse_lazzarotto2020.py first.",
            path,
        )
        sys.exit(1)

    df = pd.read_parquet(path, columns=["guide_id", "guide_seq"])
    df = df.drop_duplicates(subset=["guide_id", "guide_seq"])
    df = df.dropna(subset=["guide_seq"])
    logger.info(
        "Loaded %d unique T cell guides from %s",
        len(df), path,
    )
    return df


def _run_casoffinder_for_guide(
    guide_id: str,
    guide_seq: str,
    genome_path: Path,
    max_mismatches: int,
) -> pd.DataFrame | None:
    """Run Cas-OFFinder for a single guide and return a tidy DataFrame.

    Returns None if Cas-OFFinder fails for this guide.
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

    # find_offtargets returns column 'guide' (the input sequence); rename
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
        description="Run Cas-OFFinder for Lazzarotto 2020 T cell guides"
    )
    parser.add_argument(
        "--genome", type=Path, default=_GENOME_DEFAULT,
        help="Path to directory of per-chrom FASTA files",
    )
    parser.add_argument(
        "--max-mismatches", type=int, default=4,
        help="Cas-OFFinder mismatch threshold (default: 4)",
    )
    parser.add_argument(
        "--output", type=Path, default=_CANDIDATES_OUT,
        help="Output parquet path",
    )
    args = parser.parse_args()

    genome_path: Path = args.genome
    if not genome_path.exists():
        logger.error("Genome directory not found: %s", genome_path)
        sys.exit(1)

    # ---- Load guides --------------------------------------------------------
    guides_df = _load_lazzarotto_guides(_LAZZAROTTO_PATH)

    # Deduplicate by guide_seq (run Cas-OFFinder once per unique sequence)
    seq_to_guide_ids: dict[str, list[str]] = {}
    for _, row in guides_df.iterrows():
        seq = row["guide_seq"]
        gid = row["guide_id"]
        seq_to_guide_ids.setdefault(seq, []).append(gid)

    total_unique_seqs = len(seq_to_guide_ids)
    total_guide_ids = sum(len(v) for v in seq_to_guide_ids.values())
    logger.info(
        "Total: %d guide_ids across %d unique sequences — "
        "will run Cas-OFFinder %d times",
        total_guide_ids, total_unique_seqs, total_unique_seqs,
    )

    # ---- Run Cas-OFFinder per unique guide_seq ----
    all_candidates: list[pd.DataFrame] = []
    failed: list[str] = []

    for i, (guide_seq, guide_ids) in enumerate(
        sorted(seq_to_guide_ids.items()), start=1
    ):
        logger.info(
            "Running Cas-OFFinder %d/%d  guide_seq=%s  guide_ids=%s",
            i, total_unique_seqs, guide_seq, guide_ids,
        )
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
            gdf["source_datasets"] = _DATASET_NAME
            gdf["primary_dataset"] = _DATASET_NAME
            all_candidates.append(gdf)

    if failed:
        logger.error(
            "Cas-OFFinder failed for %d guide_ids: %s",
            len(failed), failed,
        )

    if not all_candidates:
        logger.error(
            "No candidates generated — check genome path and Cas-OFFinder."
        )
        sys.exit(1)

    # ---- Assemble and save --------------------------------------------------
    combined = pd.concat(all_candidates, ignore_index=True)

    # Enforce correct dtypes
    combined["start"] = combined["start"].astype("int64")
    combined["num_mismatches"] = combined["num_mismatches"].astype("int64")

    # Column ordering (match candidates_all.parquet exactly)
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
