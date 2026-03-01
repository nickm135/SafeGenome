"""
scripts/02b_parse_lazzarotto2020.py — Parse Lazzarotto 2020 GUIDE-seq data
(reprocessed by Yaish & Orenstein 2024) into the standardised SafeGenome
off-target table.

Usage
-----
    python scripts/02b_parse_lazzarotto2020.py [options]

    --input PATH   Path to GUIDEseq_CR_Lazzarotto_2020_dataset.csv
                   (default: data/raw/lazzarotto2020/GUIDEseq_CR_Lazzarotto_2020_dataset.csv)
    --output PATH  Output parquet path
                   (default: data/processed/lazzarotto2020_guideseq.parquet)

Data provenance
---------------
Original study:
    Lazzarotto et al. (2020) Nat Biotechnol 38:1317-1327.
    PMID: 32541958.  GUIDE-seq on 58 sgRNAs in primary CD4+ T cells.

Reprocessed by:
    Yaish & Orenstein (2024) Nucleic Acids Res 52(10):e50.
    PMID: 38813823.  github.com/OrensteinLab/CRISPR-Bulge.

The CSV contains ALL candidate off-target sites (active + inactive) for 58
guides in hg38 coordinates.  Active sites have reads > 0.  Bulge sites
(Align.#Bulges > 0) are filtered out for compatibility with 20-nt CFD
scoring.

Output schema matches 02_parse_datasets.py (see that file's docstring).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- path setup so src imports work when run as a script ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import get_logger, log_filter, normalize_chrom

logger = get_logger("parse_lazzarotto2020")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]
_RAW_DIR = _ROOT / "data" / "raw" / "lazzarotto2020"
_DEFAULT_INPUT = _RAW_DIR / "GUIDEseq_CR_Lazzarotto_2020_dataset.csv"
_PROCESSED = _ROOT / "data" / "processed"
_DEFAULT_OUTPUT = _PROCESSED / "lazzarotto2020_guideseq.parquet"

# ---------------------------------------------------------------------------
# Standard output schema (must match 02_parse_datasets.py)
# ---------------------------------------------------------------------------

_OUTPUT_COLUMNS: list[str] = [
    "guide_id",
    "chrom",
    "start",
    "end",
    "strand",
    "guide_seq",
    "offtarget_seq",
    "pam_seq",
    "num_mismatches",
    "read_count",
    "cell_type",
    "dataset",
    "genome_build",
]

_OUTPUT_DTYPES: dict[str, str] = {
    "start": "int64",
    "end": "int64",
    "num_mismatches": "int64",
    "read_count": "float64",
}


# ---------------------------------------------------------------------------
# Guide ID assignment
# ---------------------------------------------------------------------------

def _assign_guide_ids(guide_seqs: list[str]) -> dict[str, str]:
    """Assign human-readable guide IDs like ``Laz_01`` to ``Laz_58``.

    Guides are sorted alphabetically by their 20-nt spacer sequence
    (without PAM) to ensure deterministic, reproducible IDs.

    Parameters
    ----------
    guide_seqs : list[str]
        Unique 23-nt guide sequences (spacer + NNN PAM).

    Returns
    -------
    dict[str, str]
        Mapping from 23-nt guide sequence to human-readable guide_id.
    """
    # Sort by 20-nt spacer (ignore PAM) for reproducibility
    sorted_guides = sorted(set(guide_seqs), key=lambda s: s[:20])
    return {seq: f"Laz_{i+1:02d}" for i, seq in enumerate(sorted_guides)}


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_lazzarotto_guideseq(path: Path = _DEFAULT_INPUT) -> pd.DataFrame:
    """Parse the Yaish-reprocessed Lazzarotto 2020 GUIDE-seq dataset.

    Filters to substitution-only active sites, maps to SafeGenome schema.

    Parameters
    ----------
    path : Path
        Path to GUIDEseq_CR_Lazzarotto_2020_dataset.csv.

    Returns
    -------
    pd.DataFrame
        Standardised off-target table with columns matching _OUTPUT_COLUMNS.

    Raises
    ------
    FileNotFoundError
        If the input CSV is not found.
    """
    logger.info("=== Parsing Lazzarotto 2020 GUIDE-seq (Yaish reprocessed) from %s ===", path)

    if not path.exists():
        raise FileNotFoundError(
            f"Lazzarotto GUIDE-seq file not found: {path}\n"
            "Download instructions:\n"
            "  1. git clone https://github.com/OrensteinLab/CRISPR-Bulge.git\n"
            "  2. unzip files/datasets.zip\n"
            "  3. Copy datasets/GUIDEseq/include_on_targets/"
            "GUIDEseq_CR_Lazzarotto_2020_dataset.csv\n"
            "     to data/raw/lazzarotto2020/\n"
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    df = pd.read_csv(path)
    n_raw = len(df)
    n_guides_raw = df["sgRNA"].nunique()
    logger.info("Loaded %d rows, %d unique guides", n_raw, n_guides_raw)

    # Verify expected columns
    expected_cols = {
        "sgRNA", "chrom", "Align.strand", "Align.chromStart",
        "Align.chromEnd", "Align.off-target", "Align.sgRNA",
        "Align.#Mismatches", "Align.#Bulges", "reads",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}\n"
            f"Actual columns: {sorted(df.columns)}"
        )

    # ------------------------------------------------------------------
    # Filter 1: keep only active sites (reads > 0)
    # ------------------------------------------------------------------
    # reads == 0 means the off-target was not detected by GUIDE-seq.
    # These are candidate negatives — SafeGenome uses Cas-OFFinder for
    # negative enumeration instead.
    n_before = len(df)
    df = df[df["reads"] > 0].copy()
    log_filter(logger, "inactive sites (reads == 0)", n_before, len(df))

    # ------------------------------------------------------------------
    # Filter 2: remove bulge sites (Align.#Bulges > 0)
    # ------------------------------------------------------------------
    # Bulge sites have insertions/deletions in the alignment, making
    # sequences != 20 nt.  Incompatible with CFD scoring and the
    # SafeGenome feature pipeline.
    n_before = len(df)
    n_bulge = (df["Align.#Bulges"] > 0).sum()
    df = df[df["Align.#Bulges"] == 0].copy()
    log_filter(logger, f"bulge sites (Align.#Bulges > 0, n={n_bulge})", n_before, len(df))

    # ------------------------------------------------------------------
    # Filter 3: remove on-target sites (Align.#Mismatches == 0)
    # ------------------------------------------------------------------
    # On-target sites trivially have high reads and 0 mismatches.
    # Including them teaches the model "0 mismatches = positive" which
    # is useless.  Critical rule from CLAUDE.md.
    n_before = len(df)
    df = df[df["Align.#Mismatches"] > 0].copy()
    log_filter(logger, "on-target sites (0 mismatches)", n_before, len(df))

    # ------------------------------------------------------------------
    # Filter 4: require 23-nt sequences (spacer + PAM)
    # ------------------------------------------------------------------
    # Substitution-only sites should always be 23 nt, but verify.
    n_before = len(df)
    ot_lens = df["Align.off-target"].str.len()
    guide_lens = df["Align.sgRNA"].str.len()
    mask_len = (ot_lens == 23) & (guide_lens == 23)
    df = df[mask_len].copy()
    log_filter(logger, "non-23-nt sequences", n_before, len(df))

    # ------------------------------------------------------------------
    # Verify no gap characters remain (sanity check after bulge filter)
    # ------------------------------------------------------------------
    has_gap_ot = df["Align.off-target"].str.contains("-", na=False)
    has_gap_guide = df["Align.sgRNA"].str.contains("-", na=False)
    if has_gap_ot.any() or has_gap_guide.any():
        n_gap = (has_gap_ot | has_gap_guide).sum()
        logger.warning(
            "UNEXPECTED: %d rows still have gap chars after bulge filter. Removing.", n_gap
        )
        n_before = len(df)
        df = df[~(has_gap_ot | has_gap_guide)].copy()
        log_filter(logger, "residual gap characters", n_before, len(df))

    # ------------------------------------------------------------------
    # Extract guide/off-target sequences and PAM
    # ------------------------------------------------------------------
    # sgRNA column: 23-nt (20-nt spacer + NNN PAM placeholder)
    # Align.off-target: 23-nt (20-nt off-target spacer + 3-nt observed PAM)
    # Align.sgRNA: 23-nt (should match sgRNA for substitution-only sites)
    df["guide_seq"] = df["sgRNA"].str[:20].str.upper()
    df["offtarget_seq"] = df["Align.off-target"].str[:20].str.upper()
    df["pam_seq"] = df["Align.off-target"].str[20:].str.upper()

    # ------------------------------------------------------------------
    # Assign human-readable guide IDs (Laz_01 through Laz_58)
    # ------------------------------------------------------------------
    guide_id_map = _assign_guide_ids(df["sgRNA"].unique().tolist())
    df["guide_id"] = df["sgRNA"].map(guide_id_map)
    logger.info(
        "Assigned guide IDs: %d guides (Laz_01 to Laz_%02d)",
        len(guide_id_map), len(guide_id_map),
    )

    # ------------------------------------------------------------------
    # Normalize chromosome names (already hg38, no liftover needed)
    # ------------------------------------------------------------------
    n_before = len(df)
    norm_chroms: list[str] = []
    valid_mask: list[bool] = []
    for c in df["chrom"].astype(str):
        try:
            norm_chroms.append(normalize_chrom(c))
            valid_mask.append(True)
        except ValueError:
            norm_chroms.append("")
            valid_mask.append(False)

    df["chrom"] = norm_chroms
    df = df[valid_mask].copy()
    log_filter(logger, "non-standard contigs", n_before, len(df))

    # ------------------------------------------------------------------
    # Coordinates: already hg38 BED (0-based, half-open, length 23)
    # ------------------------------------------------------------------
    df["start"] = df["Align.chromStart"].astype("int64")
    df["end"] = df["Align.chromEnd"].astype("int64")

    # Verify consistent interval length
    site_lens = df["end"] - df["start"]
    if not (site_lens == 23).all():
        bad = (site_lens != 23).sum()
        logger.warning("UNEXPECTED: %d sites have length != 23. Keeping anyway.", bad)

    # ------------------------------------------------------------------
    # Strand
    # ------------------------------------------------------------------
    df["strand"] = df["Align.strand"].astype(str)

    # ------------------------------------------------------------------
    # Remaining fields
    # ------------------------------------------------------------------
    df["num_mismatches"] = df["Align.#Mismatches"].astype("int64")
    df["read_count"] = df["reads"].astype("float64")
    df["cell_type"] = "CD4_Tcell"
    df["dataset"] = "guide_seq_lazzarotto2020"
    df["genome_build"] = "hg38"

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    out = df[_OUTPUT_COLUMNS].copy()
    for col, dtype in _OUTPUT_DTYPES.items():
        out[col] = out[col].astype(dtype)

    logger.info(
        "Lazzarotto GUIDE-seq final: %d rows, %d guides, cell_type=%s",
        len(out),
        out["guide_id"].nunique(),
        sorted(out["cell_type"].unique()),
    )
    return out


# ---------------------------------------------------------------------------
# Validation and summary
# ---------------------------------------------------------------------------

def validate_parsed_data(df: pd.DataFrame) -> None:
    """Run sanity checks on parsed data and print summary.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed output from parse_lazzarotto_guideseq().
    """
    logger.info("=== Validation checks ===")

    # 1. Check for duplicate sites (same guide + chrom + position)
    dup_cols = ["guide_id", "chrom", "start"]
    n_dup = df.duplicated(subset=dup_cols, keep=False).sum()
    if n_dup > 0:
        logger.warning("DUPLICATES FOUND: %d rows share (guide_id, chrom, start)", n_dup)
        # Show details
        dups = df[df.duplicated(subset=dup_cols, keep=False)]
        logger.warning("Duplicate details:\n%s", dups[dup_cols].head(10).to_string())
    else:
        logger.info("No duplicate sites (guide_id + chrom + start). PASS")

    # 2. Check chromosomes are standard
    standard_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
    actual_chroms = set(df["chrom"].unique())
    non_standard = actual_chroms - standard_chroms
    if non_standard:
        n_ns = df[df["chrom"].isin(non_standard)].shape[0]
        logger.warning(
            "Non-standard chromosomes found: %s (%d rows)", non_standard, n_ns
        )
    else:
        logger.info("All chromosomes are standard (chr1-22, chrX, chrY). PASS")

    # 3. Count guides with >= 5 off-target sites
    guide_counts = df.groupby("guide_id").size()
    n_gte5 = (guide_counts >= 5).sum()
    n_gte1 = (guide_counts >= 1).sum()
    logger.info(
        "Guides with >= 5 sites (usable for LOGO CV): %d / %d",
        n_gte5, n_gte1,
    )

    # 4. Per-guide site distribution
    logger.info(
        "Sites per guide: min=%d, median=%.1f, mean=%.1f, max=%d",
        guide_counts.min(),
        guide_counts.median(),
        guide_counts.mean(),
        guide_counts.max(),
    )

    # 5. Mismatch distribution
    logger.info("Mismatch distribution:\n%s",
                df["num_mismatches"].value_counts().sort_index().to_string())

    # 6. Read count distribution
    logger.info(
        "Read count: min=%.0f, median=%.0f, mean=%.0f, max=%.0f",
        df["read_count"].min(),
        df["read_count"].median(),
        df["read_count"].mean(),
        df["read_count"].max(),
    )

    # 7. Final summary
    print(f"\n{'='*60}")
    print(f"PARSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total guides:   {df['guide_id'].nunique()}")
    print(f"Total sites:    {len(df)}")
    print(f"Guides >= 5 sites (LOGO CV usable): {n_gte5}")
    print(f"Guides >= 1 site:                   {n_gte1}")
    print(f"Cell type:      {df['cell_type'].unique()[0]}")
    print(f"Genome build:   {df['genome_build'].unique()[0]}")
    print(f"{'='*60}")
    print(f"\nPer-guide site counts:")
    for gid, count in guide_counts.sort_values(ascending=False).items():
        usable = " [LOGO CV]" if count >= 5 else ""
        print(f"  {gid}: {count:4d} sites{usable}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Parse Lazzarotto 2020 GUIDE-seq (Yaish reprocessed)"
    )
    parser.add_argument(
        "--input", type=Path, default=_DEFAULT_INPUT,
        help="Path to GUIDEseq_CR_Lazzarotto_2020_dataset.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT,
        help="Output parquet path",
    )
    args = parser.parse_args()

    # Parse
    df = parse_lazzarotto_guideseq(args.input)

    # Validate
    validate_parsed_data(df)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    logger.info("Saved %d rows to %s", len(df), args.output)
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
