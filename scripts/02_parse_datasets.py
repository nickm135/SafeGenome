"""
scripts/02_parse_datasets.py — Parse raw supplementary tables into the
standardised SafeGenome off-target table.

Usage
-----
    python scripts/02_parse_datasets.py [options]

    --guide-seq PATH        Tsai 2015 GUIDE-seq TSV
    --change-seq PATH       Lazzarotto 2020 CHANGE-seq XLSX
    --deepcrispr PATH       Chuai 2018 Additional File 7 (CSV/XLSX)
    --discover-seq-s1 PATH  Wienert 2019 Supplementary Table S1 XLSX
    --discover-seq-s3 PATH  Wienert 2019 Supplementary Table S3 XLSX
    --deepcrispr-genome-build {hg19,hg38}  (default: hg19)

Outputs
-------
    data/processed/guide_seq_tsai2015.parquet
    data/processed/change_seq_lazzarotto2020.parquet
    data/processed/deepcrispr_k562.parquet
    data/processed/discover_seq_k562.parquet

Output schema (all files)
--------------------------
  guide_id        str     guide name, e.g. "VEGFA_site1"
  chrom           str     normalised chromosome, e.g. "chr6"
  start           int64   0-based start (BED convention)
  end             int64   0-based half-open end  (= start + 23 for NGG sites)
  strand          str     "+" or "−"
  guide_seq       str     20-nt protospacer sequence (DNA, uppercase, no PAM)
  offtarget_seq   str     20-nt off-target spacer (DNA, uppercase, no PAM)
  pam_seq         str     3-nt PAM observed at the off-target site
                          ("NNN" when not available in source — must be
                          extracted from genome in feature_matrix.py)
  num_mismatches  int64   mismatches in spacer only (PAM mismatches not counted)
  read_count      float64 raw experimental read/score value; interpretation
                          depends on dataset:
                          - guide_seq / discover_seq: GUIDE-seq / MRE11 reads
                          - change_seq: CHANGE-seq reads
                          - deepcrispr: cleavage frequency or binary label
  cell_type       str     cell line, e.g. "HEK293", "U2OS", "K562", "T cells"
  dataset         str     dataset tag
  genome_build    str     always "hg38" (liftover applied where needed)

Design decisions
-----------------
* Truncated guides (tru_*) are excluded — their spacers are <20 nt and are
  incompatible with 20-nt CFD / MIT scoring.
* On-target sites (num_mismatches == 0) are removed per project protocol.
  The model must NOT learn "0 mismatches = positive".
* GUIDE-seq and DeepCRISPR coordinates are originally hg19; lifted to hg38.
* CHANGE-seq and DISCOVER-seq coordinates are already hg38.  No liftover.
* Every filtering step is logged with log_filter for the paper methods section.
* DeepCRISPR and DISCOVER-seq are POSITIVES-ONLY datasets.  Negatives must
  be generated separately via Cas-OFFinder.
* Column detection uses alias lists; if a required column is missing the
  parser raises ValueError listing the actual columns found for debugging.

Data sources
------------
GUIDE-seq:
  Tsai et al. (2015) Nat Biotechnol 33:187-197.  doi:10.1038/nbt.3117
  Source: https://raw.githubusercontent.com/maximilianh/crisporpaper/
          master/origData/Tsai2015/nbt.3117-S2.txt
  Save as: data/raw/tsai2015_guide_seq.tsv
  Genome build: hg19 → lifted to hg38 here.

CHANGE-seq:
  Lazzarotto et al. (2020) Nat Biotechnol 38:1317-1327.
  doi:10.1038/s41587-020-0555-7
  Source: Supplementary Tables XLSX from PMC7652380.
  Save as: data/raw/lazzarotto2020_change_seq_supp_tables.xlsx
  Instructions: https://pmc.ncbi.nlm.nih.gov/articles/PMC7652380/
                → Supplementary Material → SuppTables.xlsx
  Genome build: hg38 (no liftover needed).

DeepCRISPR K562:
  Chuai et al. (2018) Genome Biol 19:80.  doi:10.1186/s13059-018-1459-4
  Source: Additional File 7 (PMC6020378 supplementary).
  Instructions: https://genomebiology.biomedcentral.com/articles/
                10.1186/s13059-018-1459-4  → Additional files → File 7
  Save as: data/raw/chuai2018_deepcrispr_k562.xlsx  (or .csv)
  Genome build: assumed hg19 (override with --deepcrispr-genome-build hg38).
  Content: 12 sgRNAs, ~120 positive off-target sites in K562 (file also
           contains HEK293 rows; K562 rows are extracted by cell-type filter).

DISCOVER-seq K562:
  Wienert et al. (2019) Science 364:286-289.  doi:10.1126/science.aav9023
  Source: Supplementary Tables S1 and S3 from PMC6589096.
  Instructions: https://pmc.ncbi.nlm.nih.gov/articles/PMC6589096/
                → Supplementary Materials
                → Download Table_S1.xlsx  (guide info and on-target sites)
                → Download Table_S3.xlsx  (DISCOVER-seq off-target sites)
  Save as: data/raw/wienert2019_discover_seq_S1.xlsx
           data/raw/wienert2019_discover_seq_S3.xlsx
  Genome build: hg38 (no liftover needed).
  Content: 3 sgRNAs in K562 — VEGFA_site2 (57 OTs), RNF2 (0), HBB (2).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# --- path setup so src imports work when run as a script ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import get_logger, log_filter, liftover_hg19_to_hg38, normalize_chrom

logger = get_logger("parse_datasets")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]
_RAW = _ROOT / "data" / "raw"
_PROCESSED = _ROOT / "data" / "processed"

_GUIDE_SEQ_RAW = _RAW / "tsai2015_guide_seq.tsv"
_CHANGE_SEQ_RAW = _RAW / "lazzarotto2020_change_seq_supp_tables.xlsx"
_DEEPCRISPR_RAW = _RAW / "chuai2018_deepcrispr_k562.xlsx"
_DISCOVER_SEQ_S1_RAW = _RAW / "wienert2019_discover_seq_S1.xlsx"
_DISCOVER_SEQ_S3_RAW = _RAW / "wienert2019_discover_seq_S3.xlsx"

_GUIDE_SEQ_OUT = _PROCESSED / "guide_seq_tsai2015.parquet"
_CHANGE_SEQ_OUT = _PROCESSED / "change_seq_lazzarotto2020.parquet"
_DEEPCRISPR_OUT = _PROCESSED / "deepcrispr_k562.parquet"
_DISCOVER_SEQ_OUT = _PROCESSED / "discover_seq_k562.parquet"

# ---------------------------------------------------------------------------
# Standard output schema
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
# Shared helpers for flexible column detection
# ---------------------------------------------------------------------------


def _pick_col(df: pd.DataFrame, aliases: list[str], label: str) -> str:
    """Return the first alias that exists in *df*; raise ``ValueError`` if none.

    Parameters
    ----------
    df : pd.DataFrame
    aliases : list[str]
        Column name candidates tried in order of preference.
    label : str
        Human-readable field name used in the error message.

    Returns
    -------
    str
        The matched column name.

    Raises
    ------
    ValueError
        With a message that lists the actual columns found, to help users
        diagnose format mismatches without reading the source code.
    """
    for a in aliases:
        if a in df.columns:
            return a
    raise ValueError(
        f"Cannot find '{label}' column.\n"
        f"  Tried aliases : {aliases}\n"
        f"  Actual columns: {sorted(df.columns)}"
    )


def _load_tabular(path: Path) -> pd.DataFrame:
    """Load a CSV, TSV, or Excel file into a DataFrame.

    Tries tab-separated first; falls back to comma-separated only when the
    tab-split produced exactly one column (i.e. no tabs found in the file).
    """
    sfx = path.suffix.lower()
    if sfx in {".xlsx", ".xls"}:
        return pd.read_excel(path, engine="openpyxl")
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")
    return df


def _apply_normalise_chrom(
    df: pd.DataFrame, col: str
) -> tuple[pd.DataFrame, "pd.Series[bool]"]:
    """Normalise *col* via ``normalize_chrom()`` in-place.

    Returns the modified DataFrame and a boolean ``pd.Series`` keep-mask
    (``True`` = valid chrom).  Using a Series (rather than a plain Python
    list) ensures ``df[mask]`` is always interpreted as row-selection, even
    for empty DataFrames where ``df[[]]`` would be column-selection.
    """
    norm: list[str] = []
    mask: list[bool] = []
    for c in df[col].astype(str):
        try:
            norm.append(normalize_chrom(c))
            mask.append(True)
        except ValueError:
            norm.append("")
            mask.append(False)
    df = df.copy()
    df[col] = norm
    return df, pd.Series(mask, index=df.index, dtype=bool)


# ---------------------------------------------------------------------------
# GUIDE-seq parser  (Tsai 2015, hg19 → hg38)
# ---------------------------------------------------------------------------


def parse_guide_seq(path: Path = _GUIDE_SEQ_RAW) -> pd.DataFrame:
    """Parse the Tsai 2015 GUIDE-seq supplementary table (crisporpaper TSV).

    Parameters
    ----------
    path : Path
        Path to tsai2015_guide_seq.tsv.

    Returns
    -------
    pd.DataFrame
        Standardised off-target table (see module docstring for schema).
    """
    logger.info("=== Parsing GUIDE-seq (Tsai 2015) from %s ===", path)

    if not path.exists():
        raise FileNotFoundError(
            f"GUIDE-seq file not found: {path}\n"
            "Download from:\n"
            "  https://raw.githubusercontent.com/maximilianh/crisporpaper/"
            "master/origData/Tsai2015/nbt.3117-S2.txt\n"
            "Save as: data/raw/tsai2015_guide_seq.tsv"
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    # The file has a '#' prefix on the header; skip it with comment='#'
    # and supply column names manually.
    col_names = [
        "Chromosome",
        "Start",
        "End",
        "Name",
        "GUIDE_Seq_Reads",
        "Strand",
        "Cells",
        "Targetsite",
        "Target_Sequence",
        "Offtarget_Sequence",
        "Mismatches_20bp",
        "Mismatches_3bp_PAM",
        "Mismatch_Total",
    ]
    df = pd.read_csv(path, sep="\t", comment="#", names=col_names)
    n_raw = len(df)
    logger.info("Loaded %d rows", n_raw)

    # ------------------------------------------------------------------
    # Filter 1: remove truncated guides (tru_*)
    # ------------------------------------------------------------------
    # tru_* guides have spacers <20 nt — incompatible with 20-nt CFD scoring.
    mask_tru = df["Targetsite"].str.startswith("tru_")
    df = df[~mask_tru].copy()
    log_filter(logger, "truncated guides (tru_*)", n_raw, len(df))

    # ------------------------------------------------------------------
    # Filter 2: require 23-nt sequences (guide + 3-nt PAM)
    # ------------------------------------------------------------------
    n_before = len(df)
    mask_len = (df["Target_Sequence"].str.len() == 23) & (
        df["Offtarget_Sequence"].str.len() == 23
    )
    df = df[mask_len].copy()
    log_filter(logger, "non-23-nt sequences", n_before, len(df))

    # ------------------------------------------------------------------
    # Extract guide/off-target sequences and PAM
    # ------------------------------------------------------------------
    # Target_Sequence is the on-target genomic site (guide + PAM), 23 nt.
    # The first 20 nt are the protospacer; the last 3 nt are the PAM.
    # For the off-target PAM we use Offtarget_Sequence[20:] (actual observed).
    df["guide_seq"] = df["Target_Sequence"].str[:20].str.upper()
    df["offtarget_seq"] = df["Offtarget_Sequence"].str[:20].str.upper()
    df["pam_seq"] = df["Offtarget_Sequence"].str[20:].str.upper()

    # ------------------------------------------------------------------
    # Filter 3: remove on-target sites (Mismatch_Total == 0)
    # ------------------------------------------------------------------
    n_before = len(df)
    df = df[df["Mismatch_Total"] > 0].copy()
    log_filter(logger, "on-target sites (Mismatch_Total == 0)", n_before, len(df))

    # ------------------------------------------------------------------
    # Normalize cell-line names
    # ------------------------------------------------------------------
    df["cell_type"] = df["Cells"].replace({"293": "HEK293"})

    # ------------------------------------------------------------------
    # Coordinate normalization: hg19 BED (0-based) → hg38 (0-based)
    # ------------------------------------------------------------------
    n_before = len(df)
    interval_len = (df["End"] - df["Start"]).values  # usually 23

    lifted_chroms: list[str] = []
    lifted_starts: list[int] = []
    keep_mask: list[bool] = []

    for _, row in df.iterrows():
        result = liftover_hg19_to_hg38(str(row["Chromosome"]), int(row["Start"]))
        if result is None:
            keep_mask.append(False)
            lifted_chroms.append("")
            lifted_starts.append(0)
        else:
            hg38_chrom, hg38_start = result
            keep_mask.append(True)
            lifted_chroms.append(hg38_chrom)
            lifted_starts.append(hg38_start)

    df["_keep"] = keep_mask
    df["_hg38_chrom"] = lifted_chroms
    df["_hg38_start"] = lifted_starts

    df = df[df["_keep"]].copy()
    log_filter(logger, "liftover failures (unmappable hg19 positions)", n_before, len(df))

    # Recompute end: hg38_start + original interval length (End - Start unchanged)
    df["start"] = df["_hg38_start"].astype("int64")
    df["end"] = (df["_hg38_start"] + (df["End"] - df["Start"])).astype("int64")

    # ------------------------------------------------------------------
    # Normalize chromosome names
    # ------------------------------------------------------------------
    n_before = len(df)
    valid_chroms: list[bool] = []
    norm_chroms: list[str] = []
    for c in df["_hg38_chrom"]:
        try:
            norm_chroms.append(normalize_chrom(c))
            valid_chroms.append(True)
        except ValueError:
            norm_chroms.append("")
            valid_chroms.append(False)

    df["chrom"] = norm_chroms
    df = df[valid_chroms].copy()
    log_filter(logger, "non-standard contigs after liftover", n_before, len(df))

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    df["guide_id"] = df["Targetsite"]
    df["strand"] = df["Strand"]
    df["num_mismatches"] = df["Mismatches_20bp"].astype("int64")
    df["read_count"] = df["GUIDE_Seq_Reads"].astype("float64")
    df["dataset"] = "guide_seq_tsai2015"
    df["genome_build"] = "hg38"

    out = df[_OUTPUT_COLUMNS].copy()
    for col, dtype in _OUTPUT_DTYPES.items():
        out[col] = out[col].astype(dtype)

    logger.info(
        "GUIDE-seq final: %d rows, %d guides, cells: %s",
        len(out),
        out["guide_id"].nunique(),
        sorted(out["cell_type"].unique()),
    )
    return out


# ---------------------------------------------------------------------------
# CHANGE-seq parser  (Lazzarotto 2020, hg38)
# ---------------------------------------------------------------------------


def parse_change_seq(path: Path = _CHANGE_SEQ_RAW) -> pd.DataFrame:
    """Parse the Lazzarotto 2020 CHANGE-seq supplementary tables Excel file.

    The full dataset is in Supplementary Table 3 (~202,000 rows, hg38).

    Parameters
    ----------
    path : Path
        Path to lazzarotto2020_change_seq_supp_tables.xlsx.

    Returns
    -------
    pd.DataFrame
        Standardised off-target table.

    Raises
    ------
    FileNotFoundError
        With download instructions if the file is missing.
    ValueError
        If the expected sheet / columns cannot be found in the workbook.
    """
    logger.info("=== Parsing CHANGE-seq (Lazzarotto 2020) from %s ===", path)

    if not path.exists():
        raise FileNotFoundError(
            f"CHANGE-seq supplementary file not found: {path}\n\n"
            "Download instructions:\n"
            "  1. Open https://pmc.ncbi.nlm.nih.gov/articles/PMC7652380/ in a browser.\n"
            "  2. Scroll to the bottom → 'Supplementary Material' section.\n"
            "  3. Click 'SuppTables.xlsx' (≈15 MB).\n"
            "  4. Save as:  data/raw/lazzarotto2020_change_seq_supp_tables.xlsx\n"
        )

    # ------------------------------------------------------------------
    # Load the CHANGE-seq off-target sheet
    # ------------------------------------------------------------------
    # Supp Table 3 contains the reformatted BED-style off-target table.
    # Columns: chrom, chromStart, chromEnd, name, CHANGEseq_reads, strand,
    #          offtarget_sequence, (Unnamed:7), chromStart:chromEnd,
    #          distance, target
    # "name" = guide label (e.g. "AAVS1_site_1"), shared across all hits
    #   for that guide.
    # "target" = 23-nt guide+PAM sequence.
    # "distance" = mismatches in the 20-nt spacer only (PAM not counted).
    # All coordinates are hg38 BED (0-based, half-open).
    # Cell type: all experiments used primary human T-cell genomic DNA as
    # the naked-DNA substrate (no per-row cell annotation in the table).
    _SHEET = "CHANGE-seq_Supp_Table_3"
    _REQUIRED_COLS = {"chrom", "chromStart", "chromEnd", "name",
                      "CHANGEseq_reads", "strand", "offtarget_sequence",
                      "distance", "target"}

    logger.info("Reading Excel workbook sheet list …")
    xl = pd.ExcelFile(path, engine="openpyxl")

    if _SHEET in xl.sheet_names:
        sheet_name: str = _SHEET
    else:
        # Fallback: find any sheet with the required columns
        sheet_name = ""
        for sname in xl.sheet_names:
            try:
                peek = xl.parse(sname, nrows=3)
                if _REQUIRED_COLS.issubset(set(peek.columns)):
                    sheet_name = sname
                    break
            except Exception:
                continue
        if not sheet_name:
            raise ValueError(
                f"Could not find CHANGE-seq sheet in {path}.\n"
                f"Available sheets: {xl.sheet_names}\n"
                f"Expected sheet: {_SHEET!r} with columns: {_REQUIRED_COLS}"
            )

    logger.info("Found CHANGE-seq data on sheet %r", sheet_name)

    df = xl.parse(sheet_name, engine="openpyxl")
    n_raw = len(df)
    logger.info("Loaded %d rows", n_raw)

    # ------------------------------------------------------------------
    # Filter 1: drop rows with missing offtarget_sequence
    # ------------------------------------------------------------------
    n_before = len(df)
    df = df[df["offtarget_sequence"].notna()].copy()
    log_filter(logger, "rows with missing offtarget_sequence", n_before, len(df))

    # ------------------------------------------------------------------
    # Filter 2: remove on-target sites (distance == 0)
    # ------------------------------------------------------------------
    n_before = len(df)
    df = df[df["distance"] > 0].copy()
    log_filter(logger, "on-target sites (distance == 0)", n_before, len(df))

    # ------------------------------------------------------------------
    # Filter 3: require 23-nt sequences
    # ------------------------------------------------------------------
    n_before = len(df)
    mask_len = (df["offtarget_sequence"].astype(str).str.len() == 23) & (
        df["target"].astype(str).str.len() == 23
    )
    df = df[mask_len].copy()
    log_filter(logger, "non-23-nt sequences", n_before, len(df))

    # ------------------------------------------------------------------
    # Extract guide/off-target sequences and PAM
    # ------------------------------------------------------------------
    # offtarget_sequence: 23-nt (spacer + PAM observed at off-target site).
    # target:             23-nt guide+PAM (N in PAM position).
    df["guide_seq"] = df["target"].astype(str).str[:20].str.upper()
    df["offtarget_seq"] = df["offtarget_sequence"].astype(str).str[:20].str.upper()
    df["pam_seq"] = df["offtarget_sequence"].astype(str).str[20:].str.upper()

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
    # Coordinates: already hg38 BED (0-based)
    # ------------------------------------------------------------------
    df["start"] = pd.to_numeric(df["chromStart"], errors="coerce").astype("int64")
    df["end"] = pd.to_numeric(df["chromEnd"], errors="coerce").astype("int64")

    # ------------------------------------------------------------------
    # Cell type: naked DNA from primary human T cells (no per-row annotation)
    # ------------------------------------------------------------------
    df["cell_type"] = "T cells"

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    df["guide_id"] = df["name"].astype(str)
    df["strand"] = df["strand"].astype(str)
    df["num_mismatches"] = df["distance"].astype("int64")
    df["read_count"] = pd.to_numeric(
        df["CHANGEseq_reads"], errors="coerce"
    ).astype("float64")
    df["dataset"] = "change_seq_lazzarotto2020"
    df["genome_build"] = "hg38"

    out = df[_OUTPUT_COLUMNS].copy()
    for col, dtype in _OUTPUT_DTYPES.items():
        out[col] = out[col].astype(dtype)

    logger.info(
        "CHANGE-seq final: %d rows, %d guides, cells: %s",
        len(out),
        out["guide_id"].nunique(),
        sorted(out["cell_type"].unique()),
    )
    return out


# ---------------------------------------------------------------------------
# DeepCRISPR K562 parser  (Chuai 2018, Additional File 7)
# ---------------------------------------------------------------------------


def _parse_deepcrispr_twosheets(
    xl: "pd.ExcelFile",
    path: Path,
    src_genome_build: str,
) -> pd.DataFrame:
    """Parse the actual Chuai 2018 two-sheet Excel format.

    Reads 'target' sheet (guide sequences) and 'off-target' sheet, joins
    them on 'Target ID', then applies sequence normalisation, mismatch
    computation, on-target filtering, and hg19→hg38 liftover.

    Parameters
    ----------
    xl : pd.ExcelFile
        Already-opened ExcelFile handle.
    path : Path
        Used only for logging messages.
    src_genome_build : str
        ``"hg19"`` triggers liftover; ``"hg38"`` passes coordinates through.

    Returns
    -------
    pd.DataFrame
        Standardised off-target table (see module docstring for schema).
    """
    # ------------------------------------------------------------------
    # Read target sheet → build Target ID → 20-nt guide_seq map
    # 'Target sgRNA' is 23-nt (spacer + PAM); take first 20-nt.
    # ------------------------------------------------------------------
    df_target = xl.parse("target", engine="openpyxl")
    logger.info(
        "target sheet: %d rows, columns: %s", len(df_target), list(df_target.columns)
    )
    guide_map: dict[str, str] = {
        str(tid): str(seq)[:20].upper()
        for tid, seq in zip(
            df_target["Target ID"].astype(str),
            df_target["Target sgRNA"].astype(str),
        )
    }

    # ------------------------------------------------------------------
    # Read off-target sheet
    # Columns: OT-ID, Target ID, Chromosome, Start, End, Strand, OT,
    #          Cleavage Frequency
    # ------------------------------------------------------------------
    df = xl.parse("off-target", engine="openpyxl")
    n_raw = len(df)
    logger.info(
        "off-target sheet: %d rows, columns: %s", n_raw, list(df.columns)
    )

    # Map guide sequence from target sheet
    df["_guide_str"] = df["Target ID"].astype(str).map(guide_map)

    # Drop rows whose Target ID is not in the target sheet
    n_before = n_raw
    df = df[df["_guide_str"].notna()].copy()
    log_filter(logger, "rows with unknown Target ID", n_before, len(df))

    # ------------------------------------------------------------------
    # OT sequence: 23-nt → spacer (20-nt) + PAM (3-nt or "NNN")
    # ------------------------------------------------------------------
    df["_ot_str"] = df["OT"].astype(str).str.strip().str.upper()
    n_before = len(df)
    mask_len = df["_ot_str"].str.len().between(20, 23)
    df = df[mask_len].copy()
    log_filter(logger, "OT sequences outside 20-23 nt range", n_before, len(df))

    df["offtarget_seq"] = df["_ot_str"].str[:20]
    df["guide_seq"] = df["_guide_str"].str[:20]
    df["pam_seq"] = df["_ot_str"].str[20:].where(
        df["_ot_str"].str.len() == 23, other="NNN"
    )

    pam_missing = (df["pam_seq"] == "NNN").sum()
    if pam_missing:
        logger.warning(
            "%d rows have no PAM in OT sequence (pam_seq='NNN'); "
            "feature_matrix.py must fill from genome.",
            pam_missing,
        )

    # ------------------------------------------------------------------
    # Mismatch count (no column in source — compute from sequences)
    # ------------------------------------------------------------------
    df["num_mismatches"] = [
        sum(g != o for g, o in zip(gs, os))
        for gs, os in zip(df["guide_seq"], df["offtarget_seq"])
    ]

    # ------------------------------------------------------------------
    # Filter: remove on-target sites (0 mismatches)
    # ------------------------------------------------------------------
    n_before = len(df)
    df = df[df["num_mismatches"] > 0].copy()
    log_filter(logger, "on-target sites (num_mismatches == 0)", n_before, len(df))

    # ------------------------------------------------------------------
    # Coordinates: liftover hg19 → hg38 or pass through
    # ------------------------------------------------------------------
    if src_genome_build.lower() == "hg19":
        logger.info("Lifting over DeepCRISPR coordinates hg19 → hg38 …")
        n_before = len(df)
        lifted_chroms: list[str] = []
        lifted_starts: list[int] = []
        keep_mask: list[bool] = []

        for _, row in df.iterrows():
            result = liftover_hg19_to_hg38(str(row["Chromosome"]), int(row["Start"]))
            if result is None:
                keep_mask.append(False)
                lifted_chroms.append("")
                lifted_starts.append(0)
            else:
                keep_mask.append(True)
                lifted_chroms.append(result[0])
                lifted_starts.append(result[1])

        df["_keep"] = keep_mask
        df["_h38_chrom"] = lifted_chroms
        df["_h38_start"] = lifted_starts
        df = df[df["_keep"]].copy()
        log_filter(logger, "liftover failures", n_before, len(df))

        interval_len = (
            pd.to_numeric(df["End"], errors="coerce")
            - pd.to_numeric(df["Start"], errors="coerce")
        ).astype("int64")
        df["start"] = df["_h38_start"].astype("int64")
        df["end"] = (df["_h38_start"] + interval_len).astype("int64")
        df["chrom"] = df["_h38_chrom"]
    else:
        logger.info("Coordinates already hg38; skipping liftover.")
        df["start"] = pd.to_numeric(df["Start"], errors="coerce").astype("int64")
        df["end"] = pd.to_numeric(df["End"], errors="coerce").astype("int64")
        df["chrom"] = df["Chromosome"].astype(str)

    # ------------------------------------------------------------------
    # Normalise chromosome names
    # ------------------------------------------------------------------
    n_before = len(df)
    df, valid_mask = _apply_normalise_chrom(df, "chrom")
    df = df[valid_mask].copy()
    log_filter(logger, "non-standard contigs", n_before, len(df))

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    df["guide_id"] = df["Target ID"].astype(str)
    df["strand"] = df["Strand"].astype(str)
    df["read_count"] = (
        pd.to_numeric(df["Cleavage Frequency"], errors="coerce").astype("float64")
    )
    df["cell_type"] = "K562"
    df["dataset"] = "deepcrispr_k562"
    df["genome_build"] = "hg38"

    out = df[_OUTPUT_COLUMNS].copy()
    for col, dtype in _OUTPUT_DTYPES.items():
        out[col] = out[col].astype(dtype)

    logger.info(
        "DeepCRISPR K562 final: %d rows, %d guides",
        len(out),
        out["guide_id"].nunique(),
    )
    return out


def parse_deepcrispr_k562(
    path: Path = _DEEPCRISPR_RAW,
    *,
    src_genome_build: str = "hg19",
) -> pd.DataFrame:
    """Parse Additional File 7 from Chuai et al. 2018 (DeepCRISPR, Genome Biology).

    The file aggregates experimentally verified off-target sites for ~30 sgRNAs
    (12 in K562, 18 in HEK293) detected by Digenome-seq, GUIDE-seq, BLESS,
    HTGTS, and IDLV.  This parser extracts **K562 rows only**.

    **POSITIVES ONLY.**  Negatives must be generated via Cas-OFFinder.

    Parameters
    ----------
    path : Path
        Path to the downloaded file.  Accepts ``.xlsx``, ``.xls``, ``.tsv``,
        or ``.csv``.  Default: ``data/raw/chuai2018_deepcrispr_k562.xlsx``.
    src_genome_build : str
        Coordinate system of the source file.  ``"hg19"`` triggers liftover to
        hg38; ``"hg38"`` skips liftover.  Default ``"hg19"`` (data originates
        from Kim 2015 Digenome-seq).  Override with ``"hg38"`` if the file was
        already lifted.

    Returns
    -------
    pd.DataFrame
        Standardised off-target table (see module docstring for schema).

    Raises
    ------
    FileNotFoundError
        With exact download instructions if the file is missing.
    ValueError
        If a required column cannot be found; message lists actual columns.
    """
    logger.info("=== Parsing DeepCRISPR K562 (Chuai 2018) from %s ===", path)

    if not path.exists():
        raise FileNotFoundError(
            f"DeepCRISPR file not found: {path}\n\n"
            "Download instructions:\n"
            "  1. Open https://genomebiology.biomedcentral.com/articles/"
            "10.1186/s13059-018-1459-4\n"
            "  2. Scroll to 'Additional files' → click 'Additional file 7'.\n"
            "  3. Download the file (CSV or XLSX).\n"
            f"  4. Save as: {path}\n"
            "  Alternate PMC link: https://www.ncbi.nlm.nih.gov/pmc/articles/"
            "PMC6020378/ (Supplementary Material → Additional file 7)\n"
            "\n"
            f"  If coordinates are in hg38, re-run with --deepcrispr-genome-build hg38"
        )

    # ------------------------------------------------------------------
    # Load — detect actual Chuai 2018 two-sheet Excel format
    # vs. flat-file / alternate format (backward-compatible fallback)
    # ------------------------------------------------------------------
    sfx = path.suffix.lower()
    if sfx in {".xlsx", ".xls"}:
        _xl = pd.ExcelFile(path, engine="openpyxl")
        if "off-target" in _xl.sheet_names and "target" in _xl.sheet_names:
            logger.info(
                "Detected two-sheet Excel format ('target' + 'off-target'); "
                "using _parse_deepcrispr_twosheets()."
            )
            return _parse_deepcrispr_twosheets(_xl, path, src_genome_build)

    df = _load_tabular(path)
    n_raw = len(df)
    logger.info("Loaded %d rows. Columns: %s", n_raw, list(df.columns))

    # ------------------------------------------------------------------
    # Filter 1: K562 only (if a cell-type column exists)
    # ------------------------------------------------------------------
    _CT_ALIASES = [
        "Cell_type", "cell_type", "Cell_Type", "CellType",
        "cell_line", "Cell_Line", "Cell", "cell",
    ]
    ct_col = next((c for c in _CT_ALIASES if c in df.columns), None)
    if ct_col is not None:
        n_before = len(df)
        df = df[df[ct_col].astype(str).str.upper().str.contains("K562")].copy()
        log_filter(logger, "non-K562 rows", n_before, len(df))
    else:
        logger.info(
            "No cell-type column found; treating all %d rows as K562.", len(df)
        )

    # ------------------------------------------------------------------
    # Column detection
    # ------------------------------------------------------------------
    col_guide_id = _pick_col(df, [
        "Target", "target", "Guide_Name", "guide_name", "sgRNA", "Name",
        "name", "Target_Name", "Guide", "guide", "ID",
    ], "guide_id")

    col_guide_seq = _pick_col(df, [
        "Target_Sequence", "target_sequence", "guide_seq", "Guide_Sequence",
        "GuideSeq", "protospacer", "sgRNA_seq",
    ], "guide_seq")

    col_ot_seq = _pick_col(df, [
        "Offtarget_Sequence", "Off_Target_Sequence", "offtarget_seq",
        "OT_Sequence", "OT_seq", "off_target_seq", "off_target",
        "OffTarget_Sequence",
    ], "offtarget_seq")

    col_chrom = _pick_col(df, [
        "Chromosome", "#Chromosome", "chrom", "chr", "Chr", "CHROM",
        "chromosome",
    ], "chrom")

    col_start = _pick_col(df, [
        "Start", "start", "chromStart", "POS", "pos",
    ], "start")

    col_end = _pick_col(df, ["End", "end", "chromEnd"], "end")

    col_strand = _pick_col(df, ["Strand", "strand", "STRAND"], "strand")

    col_freq = _pick_col(df, [
        "Cleavage_Frequency", "cleavage_frequency", "CleaveFreq",
        "cleavage_freq", "Frequency", "frequency", "Score", "score",
        "Label", "label", "Cleave_Freq", "cleavage",
    ], "cleavage_frequency / score")

    # Mismatch count is optional; computed from sequences if absent
    _MM_ALIASES = [
        "Mismatches", "mismatches", "num_mismatches", "n_mismatches",
        "Mismatch_count", "distance", "#Mismatches",
    ]
    col_mm = next((c for c in _MM_ALIASES if c in df.columns), None)
    if col_mm is None:
        logger.info(
            "No mismatch column found; will compute from guide/offtarget sequences."
        )

    # ------------------------------------------------------------------
    # Filter 2: drop rows with missing off-target sequence
    # ------------------------------------------------------------------
    n_before = len(df)
    df = df[df[col_ot_seq].notna()].copy()
    log_filter(logger, "rows with missing offtarget_seq", n_before, len(df))

    # ------------------------------------------------------------------
    # Sequence normalisation: 20-nt spacer + 3-nt PAM (or "NNN" placeholder)
    # Sequences may be 20-nt (spacer only) or 23-nt (spacer + PAM).
    # ------------------------------------------------------------------
    df["_ot_str"] = df[col_ot_seq].astype(str).str.strip().str.upper()
    df["_guide_str"] = df[col_guide_seq].astype(str).str.strip().str.upper()

    n_before = len(df)
    mask_len = (
        df["_ot_str"].str.len().between(20, 23)
        & df["_guide_str"].str.len().between(20, 23)
    )
    df = df[mask_len].copy()
    log_filter(logger, "sequences outside 20-23 nt range", n_before, len(df))

    df["offtarget_seq"] = df["_ot_str"].str[:20]
    df["guide_seq"] = df["_guide_str"].str[:20]
    # "NNN" when PAM not in source; feature_matrix.py must fill from genome
    df["pam_seq"] = df["_ot_str"].str[20:].where(
        df["_ot_str"].str.len() == 23, other="NNN"
    )

    pam_missing = (df["pam_seq"] == "NNN").sum()
    if pam_missing:
        logger.warning(
            "%d rows have no PAM in source sequence (pam_seq='NNN'); "
            "CFD scoring will require genome lookup in feature_matrix.py.",
            pam_missing,
        )

    # ------------------------------------------------------------------
    # Mismatch count
    # ------------------------------------------------------------------
    if col_mm is not None:
        df["num_mismatches"] = (
            pd.to_numeric(df[col_mm], errors="coerce").fillna(-1).astype("int64")
        )
    else:
        df["num_mismatches"] = [
            sum(g != o for g, o in zip(gs, os))
            for gs, os in zip(df["guide_seq"], df["offtarget_seq"])
        ]

    # ------------------------------------------------------------------
    # Filter 3: remove on-target sites (0 mismatches)
    # ------------------------------------------------------------------
    n_before = len(df)
    df = df[df["num_mismatches"] > 0].copy()
    log_filter(logger, "on-target sites (num_mismatches == 0)", n_before, len(df))

    # ------------------------------------------------------------------
    # Coordinates: liftover hg19 → hg38 or pass through hg38 as-is
    # ------------------------------------------------------------------
    if src_genome_build.lower() == "hg19":
        logger.info("Lifting over DeepCRISPR coordinates hg19 → hg38 …")
        n_before = len(df)
        lifted_chroms: list[str] = []
        lifted_starts: list[int] = []
        keep_mask: list[bool] = []

        for _, row in df.iterrows():
            result = liftover_hg19_to_hg38(str(row[col_chrom]), int(row[col_start]))
            if result is None:
                keep_mask.append(False)
                lifted_chroms.append("")
                lifted_starts.append(0)
            else:
                keep_mask.append(True)
                lifted_chroms.append(result[0])
                lifted_starts.append(result[1])

        df["_keep"] = keep_mask
        df["_h38_chrom"] = lifted_chroms
        df["_h38_start"] = lifted_starts
        df = df[df["_keep"]].copy()
        log_filter(logger, "liftover failures", n_before, len(df))

        interval_len = (
            pd.to_numeric(df[col_end], errors="coerce")
            - pd.to_numeric(df[col_start], errors="coerce")
        ).astype("int64")
        df["start"] = df["_h38_start"].astype("int64")
        df["end"] = (df["_h38_start"] + interval_len).astype("int64")
        df["chrom"] = df["_h38_chrom"]
    else:
        logger.info("Coordinates already hg38; skipping liftover.")
        df["start"] = pd.to_numeric(df[col_start], errors="coerce").astype("int64")
        df["end"] = pd.to_numeric(df[col_end], errors="coerce").astype("int64")
        df["chrom"] = df[col_chrom].astype(str)

    # ------------------------------------------------------------------
    # Normalise chromosome names
    # ------------------------------------------------------------------
    n_before = len(df)
    df, valid_mask = _apply_normalise_chrom(df, "chrom")
    df = df[valid_mask].copy()
    log_filter(logger, "non-standard contigs", n_before, len(df))

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    df["guide_id"] = df[col_guide_id].astype(str)
    df["strand"] = df[col_strand].astype(str)
    df["read_count"] = pd.to_numeric(df[col_freq], errors="coerce").astype("float64")
    df["cell_type"] = "K562"
    df["dataset"] = "deepcrispr_k562"
    df["genome_build"] = "hg38"

    out = df[_OUTPUT_COLUMNS].copy()
    for col, dtype in _OUTPUT_DTYPES.items():
        out[col] = out[col].astype(dtype)

    logger.info(
        "DeepCRISPR K562 final: %d rows, %d guides",
        len(out),
        out["guide_id"].nunique(),
    )
    return out


# ---------------------------------------------------------------------------
# DISCOVER-seq K562 parser  (Wienert 2019, hg38)
# ---------------------------------------------------------------------------


def _parse_coord_string(coord: str) -> tuple[str, int, int]:
    """Parse 'chr6:43770822-43770841' → ('chr6', 43770822, 43770841).

    Parameters
    ----------
    coord : str
        Coordinate string in the format ``<chrom>:<start>-<end>``.

    Returns
    -------
    tuple[str, int, int]
        (chrom, start, end) as parsed from the string.

    Raises
    ------
    ValueError
        If the string cannot be parsed.
    """
    try:
        chrom, rest = coord.split(":", 1)
        start_s, end_s = rest.split("-", 1)
        return chrom.strip(), int(start_s.strip()), int(end_s.strip())
    except Exception as exc:
        raise ValueError(f"Cannot parse coordinate string {coord!r}: {exc}") from exc


def _parse_discover_sheet(
    df: pd.DataFrame,
    guide_id: str,
    coord_col: str,
    strand_col: str,
    pam_col: str,
    guide_seq_col: str,
    mismatches_col: str,
) -> pd.DataFrame:
    """Extract off-target rows from a single DISCOVER-seq sheet.

    Common processing applied to VEGFA, RNF2, and HBB sheets after they
    are read with the correct ``header`` argument.  Handles:

    * NaN separator rows (filter by ``coord_col.notna()``)
    * Coordinate-string parsing (``'chrN:start-end'``)
    * On-target filtering (``mismatches > 0``)
    * Strand normalisation (sense/antisense → +/−)
    * read_count = sum of all numeric columns after the mismatch column

    Parameters
    ----------
    df : pd.DataFrame
        Raw sheet as read by ``pd.read_excel``.
    guide_id : str
        Constant guide identifier assigned to all rows (e.g. ``'VEGFA'``).
    coord_col, strand_col, pam_col, guide_seq_col, mismatches_col : str
        Column names in *df*.

    Returns
    -------
    pd.DataFrame
        Rows with the standardised output columns populated.
    """
    # Validate required columns
    for col in [coord_col, strand_col, pam_col, guide_seq_col, mismatches_col]:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found in DISCOVER-seq sheet "
                f"(guide_id='{guide_id}').\n"
                f"  Actual columns: {sorted(df.columns)}"
            )

    # Capture rep column names NOW before any new columns are added.
    # rep cols = all columns after the mismatches column in the original sheet.
    mm_idx = df.columns.tolist().index(mismatches_col)
    rep_cols = list(df.columns[mm_idx + 1:])

    # Drop NaN / separator rows (coord_col is NaN in separator rows)
    n_before = len(df)
    df = df[df[coord_col].notna()].copy()
    df = df[df[coord_col].astype(str).str.contains(":", na=False)].copy()
    log_filter(logger, f"{guide_id} separator / non-coordinate rows", n_before, len(df))

    if df.empty:
        # Return empty DataFrame with correct columns populated
        for col in ["chrom", "start", "end", "guide_seq", "num_mismatches",
                    "offtarget_seq", "pam_seq", "strand", "read_count",
                    "guide_id", "cell_type", "dataset", "genome_build"]:
            df[col] = pd.Series(dtype=object)
        return df

    # Parse coordinate string → chrom, start, end
    parsed = df[coord_col].astype(str).apply(
        lambda s: _parse_coord_string(s) if ":" in s else (None, None, None)
    )
    df["chrom"] = [t[0] for t in parsed]
    df["start"] = pd.to_numeric([t[1] for t in parsed], errors="coerce")
    df["end"] = pd.to_numeric([t[2] for t in parsed], errors="coerce")

    # Drop rows with unparseable coordinates
    n_before = len(df)
    df = df[df["chrom"].notna() & df["start"].notna() & df["end"].notna()].copy()
    log_filter(logger, f"{guide_id} unparseable coordinates", n_before, len(df))

    # Mismatch count (needed before guide_seq extraction below)
    df["num_mismatches"] = (
        pd.to_numeric(df[mismatches_col], errors="coerce").fillna(-1).astype("int64")
    )

    # Guide sequence: extract from the on-target row (mismatches == 0) BEFORE
    # filtering it.  The "Guide sequence" column in DISCOVER-seq S1 contains the
    # GENOMIC sequence at each locus (i.e. the off-target sequence with mismatches),
    # NOT the guide RNA itself.  The on-target row has 0 mismatches, so its
    # "Guide sequence" value IS the true 20-nt protospacer.
    on_target_rows = df[df["num_mismatches"] == 0]
    if len(on_target_rows) > 0:
        guide_seq_value: str = (
            on_target_rows[guide_seq_col]
            .astype(str).str.strip().str.upper().str[:20]
            .iloc[0]
        )
        logger.info(
            "%s: extracted guide_seq '%s' from on-target row (mismatches==0)",
            guide_id, guide_seq_value,
        )
    else:
        guide_seq_value = "N" * 20
        logger.warning(
            "%s: no on-target row found; guide_seq set to placeholder N×20", guide_id
        )

    # Off-target sequence: the "Guide sequence" column holds the genomic sequence
    # at each off-target locus (20 nt, same strand as the guide).
    df["offtarget_seq"] = (
        df[guide_seq_col].astype(str).str.strip().str.upper().str[:20]
    )

    # Filter on-target rows (mismatches == 0) — MUST come AFTER guide_seq extraction
    n_before = len(df)
    df = df[df["num_mismatches"] > 0].copy()
    log_filter(logger, f"{guide_id} on-target sites (mismatches == 0)", n_before, len(df))

    # Assign constant guide_seq to all remaining (off-target) rows
    df["guide_seq"] = guide_seq_value

    # PAM
    df["pam_seq"] = df[pam_col].astype(str).str.strip().str.upper().str[:3]

    # Strand: sense/antisense → +/−
    _STRAND_MAP = {"sense": "+", "antisense": "-", "+": "+", "-": "-"}
    df["strand"] = (
        df[strand_col].astype(str).str.strip().str.lower().map(_STRAND_MAP)
        .fillna(df[strand_col].astype(str).str.strip())
    )

    # read_count = sum of all replicate columns (captured before any new cols)
    df["read_count"] = (
        df[rep_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    )

    # Constant metadata
    df["guide_id"] = guide_id
    df["cell_type"] = "K562"
    df["dataset"] = "discover_seq_k562"
    df["genome_build"] = "hg38"

    return df


def parse_discover_seq_k562(
    s1_path: Path = _DISCOVER_SEQ_S1_RAW,
) -> pd.DataFrame:
    """Parse DISCOVER-seq K562 data from Wienert et al. 2019 (Science).

    Reads three DISCOVER sheets from Supplementary Table S1:

    * ``DISCOVER VEGFA`` — header row 2 (0-indexed); 57 total sites
    * ``DISCOVER RNF2``  — header row 1; 0 off-targets after filtering
    * ``DISCOVER HBB``   — header row 1; NaN separator rows dropped

    DISCOVER-seq identifies in-cell DSBs via MRE11 ChIP-seq.  Coordinates are
    already hg38; no liftover is applied.

    Expected content (3 sgRNAs in K562):
    - VEGFA : 56 off-target sites  (57 total; 1 on-target removed)
    - RNF2  :  0 off-target sites
    - HBB   :  4 off-target sites  (WT Cas9 + HiFi Cas9 rows kept separately)

    **POSITIVES ONLY.**  Negatives must be generated via Cas-OFFinder.

    Parameters
    ----------
    s1_path : Path
        Path to Supplementary Table S1 (``wienert2019_discover_seq_S1.xlsx``).
        The file must contain sheets named ``'DISCOVER VEGFA'``,
        ``'DISCOVER RNF2'``, and ``'DISCOVER HBB'``.
        Default: ``data/raw/wienert2019_discover_seq_S1.xlsx``.

    Returns
    -------
    pd.DataFrame
        Standardised off-target table (see module docstring for schema).

    Raises
    ------
    FileNotFoundError
        With download instructions if the file is missing.
    ValueError
        If a required column cannot be found; message lists actual columns.
    """
    logger.info(
        "=== Parsing DISCOVER-seq K562 (Wienert 2019) from %s ===", s1_path
    )

    if not s1_path.exists():
        raise FileNotFoundError(
            f"DISCOVER-seq S1 file not found: {s1_path}\n\n"
            "Download instructions:\n"
            "  1. Open https://pmc.ncbi.nlm.nih.gov/articles/PMC6589096/\n"
            "  2. Scroll to 'Supplementary Materials'.\n"
            f"  3. Download Table S1 → save as: {_DISCOVER_SEQ_S1_RAW}\n"
        )

    xl = pd.ExcelFile(s1_path, engine="openpyxl")
    logger.info("Sheets in S1: %s", xl.sheet_names)

    frames: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # DISCOVER VEGFA  (header=2: rows 0-1 are title/span, row 2 is header)
    # Columns: Unnamed:0 (off-target gene), Chr:Start-End, Cutsite, Strand,
    #          PAM, Guide sequence, Mismatches, Rep1 (BW43), Rep2 (BW42),
    #          Rep3 (BW16), HiFi Cas9 (BW28)
    # ------------------------------------------------------------------
    if "DISCOVER VEGFA" in xl.sheet_names:
        df_vegfa = xl.parse("DISCOVER VEGFA", header=2, engine="openpyxl")
        logger.info(
            "DISCOVER VEGFA: %d rows, columns: %s", len(df_vegfa), list(df_vegfa.columns)
        )
        parsed_vegfa = _parse_discover_sheet(
            df_vegfa,
            guide_id="VEGFA",
            coord_col="Chr:Start-End",
            strand_col="Strand",
            pam_col="PAM",
            guide_seq_col="Guide sequence",
            mismatches_col="Mismatches",
        )
        frames.append(parsed_vegfa)
        logger.info("VEGFA off-targets after filtering: %d", len(parsed_vegfa))
    else:
        logger.warning("Sheet 'DISCOVER VEGFA' not found in %s; skipping.", s1_path)

    # ------------------------------------------------------------------
    # DISCOVER RNF2  (header=1: row 0 is title, row 1 is header)
    # Columns: gRNA, gene, Hg38 coordinates, cut site, Strand, PAM,
    #          Guide sequence, mismatches, Rep1 (BW19), Rep2 (BW27)
    # All rows have mismatches=0 (on-target only) → 0 off-targets returned.
    # ------------------------------------------------------------------
    if "DISCOVER RNF2" in xl.sheet_names:
        df_rnf2 = xl.parse("DISCOVER RNF2", header=1, engine="openpyxl")
        logger.info(
            "DISCOVER RNF2: %d rows, columns: %s", len(df_rnf2), list(df_rnf2.columns)
        )
        parsed_rnf2 = _parse_discover_sheet(
            df_rnf2,
            guide_id="RNF2",
            coord_col="Hg38 coordinates",
            strand_col="Strand",
            pam_col="PAM",
            guide_seq_col="Guide sequence",
            mismatches_col="mismatches",
        )
        frames.append(parsed_rnf2)
        logger.info("RNF2 off-targets after filtering: %d", len(parsed_rnf2))
    else:
        logger.warning("Sheet 'DISCOVER RNF2' not found in %s; skipping.", s1_path)

    # ------------------------------------------------------------------
    # DISCOVER HBB  (header=1: row 0 is title, row 1 is header)
    # Columns: gRNA, gene, Cas9, Hg38 coordinates, cut site, (Unnamed),
    #          Strand, PAM, Guide sequence, mismatches, Rep1 (BW29), Rep2 (BW13)
    # Rows 3-4 (0-indexed in data) are NaN separator rows between WT and HiFi
    # Cas9 sections — dropped by _parse_discover_sheet via coord_col notna().
    # ------------------------------------------------------------------
    if "DISCOVER HBB" in xl.sheet_names:
        df_hbb = xl.parse("DISCOVER HBB", header=1, engine="openpyxl")
        logger.info(
            "DISCOVER HBB: %d rows, columns: %s", len(df_hbb), list(df_hbb.columns)
        )
        parsed_hbb = _parse_discover_sheet(
            df_hbb,
            guide_id="HBB",
            coord_col="Hg38 coordinates",
            strand_col="Strand",
            pam_col="PAM",
            guide_seq_col="Guide sequence",
            mismatches_col="mismatches",
        )
        frames.append(parsed_hbb)
        logger.info("HBB off-targets after filtering: %d", len(parsed_hbb))
    else:
        logger.warning("Sheet 'DISCOVER HBB' not found in %s; skipping.", s1_path)

    if not frames:
        raise ValueError(
            f"No DISCOVER sheets found in {s1_path}. "
            "Expected at least one of: 'DISCOVER VEGFA', 'DISCOVER RNF2', "
            "'DISCOVER HBB'."
        )

    # Filter out fully-empty frames before concat to avoid FutureWarning about
    # all-NA columns affecting dtype inference.
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        # All sheets had 0 off-targets (e.g. RNF2-only run)
        df_all = pd.DataFrame(columns=frames[0].columns if frames else [])
    else:
        df_all = pd.concat(non_empty, ignore_index=True)

    # ------------------------------------------------------------------
    # Normalise chromosome names
    # ------------------------------------------------------------------
    n_before = len(df_all)
    df_all, valid_mask = _apply_normalise_chrom(df_all, "chrom")
    df_all = df_all[valid_mask].copy()
    log_filter(logger, "non-standard contigs", n_before, len(df_all))

    # ------------------------------------------------------------------
    # Assemble standardised output
    # ------------------------------------------------------------------
    out = df_all[_OUTPUT_COLUMNS].copy()
    for col, dtype in _OUTPUT_DTYPES.items():
        out[col] = out[col].astype(dtype)

    logger.info(
        "DISCOVER-seq K562 final: %d rows, %d guides: %s",
        len(out),
        out["guide_id"].nunique(),
        sorted(out["guide_id"].unique()),
    )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--guide-seq",
        type=Path,
        default=_GUIDE_SEQ_RAW,
        help=f"Path to Tsai 2015 GUIDE-seq TSV (default: {_GUIDE_SEQ_RAW})",
    )
    p.add_argument(
        "--change-seq",
        type=Path,
        default=_CHANGE_SEQ_RAW,
        help=f"Path to Lazzarotto 2020 CHANGE-seq XLSX (default: {_CHANGE_SEQ_RAW})",
    )
    p.add_argument(
        "--deepcrispr",
        type=Path,
        default=_DEEPCRISPR_RAW,
        metavar="PATH",
        help=f"Path to Chuai 2018 DeepCRISPR Additional File 7 (default: {_DEEPCRISPR_RAW})",
    )
    p.add_argument(
        "--deepcrispr-genome-build",
        choices=["hg19", "hg38"],
        default="hg19",
        metavar="{hg19,hg38}",
        help="Genome build of DeepCRISPR source coordinates (default: hg19). "
             "Pass hg38 to skip liftover if the file was already lifted.",
    )
    p.add_argument(
        "--discover-seq-s1",
        type=Path,
        default=_DISCOVER_SEQ_S1_RAW,
        metavar="PATH",
        help=f"Path to Wienert 2019 Supplementary Table S1 (default: {_DISCOVER_SEQ_S1_RAW})",
    )
    p.add_argument(
        "--discover-seq-s3",
        type=Path,
        default=_DISCOVER_SEQ_S3_RAW,
        metavar="PATH",
        help=f"Path to Wienert 2019 Supplementary Table S3 (default: {_DISCOVER_SEQ_S3_RAW}). "
             "Pass the same path as --discover-seq-s1 if both tables are in one workbook.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_PROCESSED,
        help=f"Output directory for parquet files (default: {_PROCESSED})",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- GUIDE-seq ---
    gs_out = args.out_dir / "guide_seq_tsai2015.parquet"
    try:
        df_gs = parse_guide_seq(args.guide_seq)
        df_gs.to_parquet(gs_out, index=False)
        logger.info("Saved GUIDE-seq → %s  (%d rows)", gs_out, len(df_gs))
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    # --- CHANGE-seq (optional — skip with instructions if absent) ---
    cs_out = args.out_dir / "change_seq_lazzarotto2020.parquet"
    if not args.change_seq.exists():
        logger.warning(
            "CHANGE-seq file not found: %s\n"
            "Skipping CHANGE-seq parsing.\n"
            "Download instructions:\n"
            "  https://pmc.ncbi.nlm.nih.gov/articles/PMC7652380/\n"
            "  → Supplementary Material → SuppTables.xlsx\n"
            "  → save as data/raw/lazzarotto2020_change_seq_supp_tables.xlsx",
            args.change_seq,
        )
    else:
        try:
            df_cs = parse_change_seq(args.change_seq)
            df_cs.to_parquet(cs_out, index=False)
            logger.info("Saved CHANGE-seq → %s  (%d rows)", cs_out, len(df_cs))
        except (FileNotFoundError, ValueError) as e:
            logger.error("%s", e)

    # --- DeepCRISPR K562 (optional — skip with instructions if absent) ---
    dc_out = args.out_dir / "deepcrispr_k562.parquet"
    if not args.deepcrispr.exists():
        logger.warning(
            "DeepCRISPR file not found: %s\n"
            "Skipping DeepCRISPR parsing.\n"
            "Download instructions:\n"
            "  https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1459-4\n"
            "  → Additional files → Additional file 7\n"
            "  → save as data/raw/chuai2018_deepcrispr_k562.xlsx",
            args.deepcrispr,
        )
    else:
        try:
            df_dc = parse_deepcrispr_k562(
                args.deepcrispr,
                src_genome_build=args.deepcrispr_genome_build,
            )
            df_dc.to_parquet(dc_out, index=False)
            logger.info("Saved DeepCRISPR K562 → %s  (%d rows)", dc_out, len(df_dc))
        except (FileNotFoundError, ValueError) as e:
            logger.error("%s", e)

    # --- DISCOVER-seq K562 (optional — skip with instructions if absent) ---
    ds_out = args.out_dir / "discover_seq_k562.parquet"
    if not args.discover_seq_s1.exists():
        logger.warning(
            "DISCOVER-seq S1 file not found: %s\n"
            "Skipping DISCOVER-seq parsing.\n"
            "Download instructions:\n"
            "  https://pmc.ncbi.nlm.nih.gov/articles/PMC6589096/\n"
            "  → Supplementary Materials\n"
            "  → Table S1 → save as data/raw/wienert2019_discover_seq_S1.xlsx",
            args.discover_seq_s1,
        )
    else:
        try:
            df_ds = parse_discover_seq_k562(args.discover_seq_s1)
            df_ds.to_parquet(ds_out, index=False)
            logger.info("Saved DISCOVER-seq K562 → %s  (%d rows)", ds_out, len(df_ds))
        except (FileNotFoundError, ValueError) as e:
            logger.error("%s", e)


if __name__ == "__main__":
    main()
