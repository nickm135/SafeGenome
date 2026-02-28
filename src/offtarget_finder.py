"""
offtarget_finder.py — Cas-OFFinder wrapper for genome-wide CRISPR off-target discovery.

Wraps the Cas-OFFinder binary (Bae et al. 2014, Bioinformatics) to enumerate
candidate off-target sites for a guide RNA across the full genome.

Coordinate convention
---------------------
All positions in the returned DataFrame use 0-based half-open (BED) coordinates,
matching Cas-OFFinder's native output and SafeGenome's internal convention.

Off-target sequence orientation
--------------------------------
Cas-OFFinder returns the off-target sequence in the same orientation as the guide
(protospacer strand, 5'→3'), for both + and − strand hits.  This means guide and
off_target_seq can be compared character-by-character to find mismatches without
any reverse-complementing.

Cas-OFFinder input format (written by _write_input)
----------------------------------------------------
Line 1: absolute path to genome FASTA file
Line 2: CRISPR type pattern — "N" × spacer_length + PAM  (e.g. NNNNNNNNNNNNNNNNNNNNNGG)
Line 3: guide_sequence + PAM + " " + max_mismatches       (e.g. GAGTCCGAGCAGAAGAAGAANGG 4)

Cas-OFFinder output format (parsed by _parse_output)
-----------------------------------------------------
Tab-separated, no header, 6 columns per line:
  1  guide_seq_with_pam  — the input guide+PAM (as written in the input file)
  2  chrom               — chromosome name (may or may not have "chr" prefix)
  3  position            — 0-based start of the off-target site
  4  off_target_seq      — found sequence including PAM, guide-length + pam-length chars
  5  strand              — "+" or "−"
  6  mismatch_count      — mismatches in the spacer region only (PAM not counted)

Binary / GPU configuration
---------------------------
Default binary:  C:/cas-offinder/cas-offinder.exe  (from config.py)
Default mode:    "G"  (GPU — matches RTX 3070)
Override either via config.py keys CASOFFINDER_BIN and CASOFFINDER_MODE.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import get_logger, log_filter, normalize_chrom, normalize_guide

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration — read from config.py if present, else use defaults
# ---------------------------------------------------------------------------

try:
    import config as _cfg  # type: ignore

    _DEFAULT_BINARY: Path = Path(
        getattr(_cfg, "CASOFFINDER_BIN", r"C:\cas-offinder\cas-offinder.exe")
    )
    _DEFAULT_MODE: str = getattr(_cfg, "CASOFFINDER_MODE", "G")
except ImportError:
    _DEFAULT_BINARY = Path(r"C:\cas-offinder\cas-offinder.exe")
    _DEFAULT_MODE = "G"

_SPACER_LENGTH: int = 20


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_offtargets(
    guide: str,
    genome_path: Path,
    *,
    pam: str = "NGG",
    max_mismatches: int = 4,
    binary_path: Path = _DEFAULT_BINARY,
    mode: str = _DEFAULT_MODE,
    work_dir: Path | None = None,
    remove_ontarget: bool = True,
) -> pd.DataFrame:
    """Find all genome-wide off-target candidates for a guide RNA.

    Runs Cas-OFFinder and returns a tidy DataFrame.  Coordinates are 0-based
    half-open (BED), genome build hg38.

    Parameters
    ----------
    guide : str
        20-nt protospacer sequence (DNA, uppercase, 5'→3').  Accepts 23-nt
        with PAM suffix — the PAM is stripped automatically by normalize_guide.
    genome_path : Path
        Path to the reference genome FASTA (hg38.fa or a directory of per-chrom
        FASTA files — Cas-OFFinder accepts both).
    pam : str
        PAM sequence to search for (IUPAC wildcards allowed).  Default "NGG".
    max_mismatches : int
        Maximum mismatches in the spacer region.  Default 4.
    binary_path : Path
        Path to the cas-offinder executable.
    mode : str
        Cas-OFFinder hardware mode: "G" (GPU), "C" (CPU), "A" (all OpenCL).
    work_dir : Path | None
        Parent directory for the temporary working directory.  Uses the system
        temp directory if None.
    remove_ontarget : bool
        If True (default), remove sites with 0 mismatches (the on-target site).
        The on-target must be excluded from training/evaluation.

    Returns
    -------
    pd.DataFrame
        One row per candidate off-target site.  Columns:

        - ``guide``              : str — the normalized 20-nt guide sequence
        - ``chrom``              : str — normalized chromosome (e.g. "chr7")
        - ``start``              : int — 0-based start of the protospacer
        - ``strand``             : str — "+" or "−"
        - ``off_target_seq``     : str — 20-nt off-target spacer (no PAM)
        - ``pam_seq``            : str — PAM sequence observed at this site
        - ``num_mismatches``     : int — mismatches in spacer (0 = on-target)
        - ``mismatch_positions`` : list[int] — 1-based positions of mismatches
                                   (index convention matches sequence_features.py)

    Raises
    ------
    FileNotFoundError
        If genome_path or binary_path does not exist.
    subprocess.CalledProcessError
        If the Cas-OFFinder binary exits with a non-zero return code.
    """
    guide = normalize_guide(guide)
    genome_path = Path(genome_path)
    binary_path = Path(binary_path)

    if not genome_path.exists():
        raise FileNotFoundError(f"Genome not found: {genome_path}")
    if not binary_path.exists():
        raise FileNotFoundError(
            f"Cas-OFFinder binary not found: {binary_path}\n"
            "Install from https://github.com/snugel/cas-offinder and configure "
            "CASOFFINDER_BIN in config.py."
        )

    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        tmp_path = Path(tmp)
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.txt"

        _write_input(input_file, guide, genome_path, pam, max_mismatches)
        _run_binary(input_file, output_file, binary_path, mode)
        df = _parse_output(output_file, guide, len(pam))

    n_before = len(df)
    if remove_ontarget and n_before > 0:
        df = df[df["num_mismatches"] > 0].reset_index(drop=True)
        log_filter(logger, "on-target sites (0 mismatches)", n_before, len(df))

    logger.info(
        "find_offtargets: guide=%s  candidates=%d  max_mm=%d  genome=%s",
        guide,
        len(df),
        max_mismatches,
        genome_path.name,
    )
    return df


# ---------------------------------------------------------------------------
# Input file
# ---------------------------------------------------------------------------


def _write_input(
    path: Path,
    guide: str,
    genome_path: Path,
    pam: str,
    max_mismatches: int,
) -> None:
    """Write a Cas-OFFinder input file for a single guide.

    Parameters
    ----------
    path : Path
        Destination file path.
    guide : str
        Normalized 20-nt guide sequence (no PAM).
    genome_path : Path
        Absolute path to genome FASTA or directory.
    pam : str
        PAM pattern (IUPAC), e.g. "NGG".
    max_mismatches : int
        Maximum mismatches in the spacer region.
    """
    pam_upper = pam.upper()
    crispr_pattern = "N" * _SPACER_LENGTH + pam_upper
    guide_entry = f"{guide}{pam_upper} {max_mismatches}"

    with path.open("w", newline="\n") as fh:
        fh.write(str(genome_path.resolve()) + "\n")
        fh.write(crispr_pattern + "\n")
        fh.write(guide_entry + "\n")

    logger.debug("Wrote Cas-OFFinder input: %s", path)


# ---------------------------------------------------------------------------
# Binary invocation
# ---------------------------------------------------------------------------


def _run_binary(
    input_path: Path,
    output_path: Path,
    binary_path: Path,
    mode: str,
) -> None:
    """Execute the Cas-OFFinder binary and wait for it to finish.

    Parameters
    ----------
    input_path : Path
        Path to the prepared Cas-OFFinder input file.
    output_path : Path
        Path where Cas-OFFinder should write its output.
    binary_path : Path
        Path to the cas-offinder executable.
    mode : str
        Hardware mode: "G" (GPU), "C" (CPU), "A" (all).

    Raises
    ------
    subprocess.CalledProcessError
        If Cas-OFFinder returns a non-zero exit code.
    """
    cmd = [str(binary_path), str(input_path), mode, str(output_path)]
    logger.info("Running Cas-OFFinder: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error("Cas-OFFinder stderr:\n%s", result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    if result.stdout:
        logger.debug("Cas-OFFinder stdout:\n%s", result.stdout)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _parse_output(
    path: Path,
    guide: str,
    pam_len: int = 3,
) -> pd.DataFrame:
    """Parse a Cas-OFFinder tab-separated output file into a DataFrame.

    Handles empty output files and malformed lines gracefully.
    Chromosome names are normalized via normalize_chrom; sites on non-standard
    contigs (e.g. scaffolds) are skipped with a warning.

    Parameters
    ----------
    path : Path
        Path to the Cas-OFFinder output file.
    guide : str
        Normalized 20-nt guide (used to compute mismatch_positions).
    pam_len : int
        Number of PAM characters at the end of the off-target sequence.

    Returns
    -------
    pd.DataFrame
        See find_offtargets docstring for column descriptions.
    """
    if not path.exists() or path.stat().st_size == 0:
        logger.warning("Cas-OFFinder output is empty or missing: %s", path)
        return _empty_dataframe()

    rows: list[dict[str, Any]] = []
    n_skipped_format = 0
    n_skipped_chrom = 0

    with path.open() as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 6:
                logger.warning(
                    "Skipping malformed line %d (%d fields): %r",
                    lineno,
                    len(parts),
                    line,
                )
                n_skipped_format += 1
                continue

            _guide_col, chrom_raw, pos_str, ot_seq_full, strand, mm_str = parts

            try:
                start = int(pos_str)
                num_mm = int(mm_str)
            except ValueError:
                logger.warning(
                    "Non-integer position or mismatch count on line %d: %r",
                    lineno,
                    line,
                )
                n_skipped_format += 1
                continue

            try:
                chrom = normalize_chrom(chrom_raw)
            except ValueError:
                # scaffold / alt contig — skip silently at debug level
                logger.debug("Skipping non-standard contig: %r", chrom_raw)
                n_skipped_chrom += 1
                continue

            spacer = ot_seq_full[:_SPACER_LENGTH]
            pam_seq = ot_seq_full[_SPACER_LENGTH : _SPACER_LENGTH + pam_len]
            mm_positions = _compute_mismatch_positions(guide, spacer)

            rows.append(
                {
                    "guide": guide,
                    "chrom": chrom,
                    "start": start,
                    "strand": strand,
                    "off_target_seq": spacer,
                    "pam_seq": pam_seq,
                    "num_mismatches": num_mm,
                    "mismatch_positions": mm_positions,
                }
            )

    if n_skipped_format:
        logger.warning(
            "Skipped %d malformed lines in Cas-OFFinder output", n_skipped_format
        )
    if n_skipped_chrom:
        logger.info(
            "Skipped %d hits on non-standard contigs", n_skipped_chrom
        )

    if not rows:
        return _empty_dataframe()

    df = pd.DataFrame(rows)
    df["start"] = df["start"].astype("int64")
    df["num_mismatches"] = df["num_mismatches"].astype("int64")
    return df


def _empty_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical output schema."""
    return pd.DataFrame(
        columns=[
            "guide",
            "chrom",
            "start",
            "strand",
            "off_target_seq",
            "pam_seq",
            "num_mismatches",
            "mismatch_positions",
        ]
    )


# ---------------------------------------------------------------------------
# Mismatch position computation
# ---------------------------------------------------------------------------


def _compute_mismatch_positions(guide: str, off_target_spacer: str) -> list[int]:
    """Return 1-based positions where guide and off-target spacer differ.

    Position numbering is identical to the convention in sequence_features.py:
    index 0 in the sequence string → position 1 (first character, 5' end).

    Parameters
    ----------
    guide : str
        20-nt guide sequence (uppercase DNA).
    off_target_spacer : str
        20-nt off-target spacer (without PAM), same strand orientation as guide.

    Returns
    -------
    list[int]
        Sorted 1-based mismatch positions.  Empty list for a perfect match.
    """
    return [
        i + 1
        for i, (g, o) in enumerate(zip(guide, off_target_spacer))
        if g.upper() != o.upper()
    ]
