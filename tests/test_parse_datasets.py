"""
tests/test_parse_datasets.py — Tests for scripts/02_parse_datasets.py

Strategy
--------
All tests use synthetic data written to temporary files so they run without
any real supplementary data downloaded.

DeepCRISPR tests cover both paths:
  * Two-sheet Excel format  (actual Chuai 2018 file: 'target' + 'off-target')
  * Flat-file CSV format    (fallback path with alias-based column detection)

Liftover is bypassed by using src_genome_build="hg38" throughout (the hg19
liftover path is tested with a mock patch on the module-level function).

DISCOVER-seq tests use synthetic Excel files that replicate the actual three-
sheet structure of wienert2019_discover_seq_S1.xlsx:
  * DISCOVER VEGFA  (header=2)
  * DISCOVER RNF2   (header=1)
  * DISCOVER HBB    (header=1, with NaN separator rows)
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Dynamic import: scripts/02_parse_datasets.py can't use "from scripts.02..."
# ---------------------------------------------------------------------------
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "02_parse_datasets.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("parse_datasets_mod", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PD = _load_module()

# Pull out the symbols under test
_pick_col = _PD._pick_col
_load_tabular = _PD._load_tabular
_apply_normalise_chrom = _PD._apply_normalise_chrom
parse_deepcrispr_k562 = _PD.parse_deepcrispr_k562
parse_discover_seq_k562 = _PD.parse_discover_seq_k562

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

GUIDE   = "GAGTCCGAGCAGAAGAAGAA"   # 20-nt protospacer
OT_1MM  = "GAGTCCGAGCAGAAGAACAA"   # 1 mismatch at pos 18 (G→C)
OT_2MM  = "GAGTCCGAGCAGAAGATCAA"   # 2 mismatches
OT_23NT = OT_1MM + "GGG"           # 23-nt with PAM

_SCHEMA_COLS = [
    "guide_id", "chrom", "start", "end", "strand",
    "guide_seq", "offtarget_seq", "pam_seq",
    "num_mismatches", "read_count", "cell_type", "dataset", "genome_build",
]
_SCHEMA_DTYPES: dict[str, str] = {
    "start": "int64",
    "end": "int64",
    "num_mismatches": "int64",
    "read_count": "float64",
}

# ---------------------------------------------------------------------------
# Helpers — generic
# ---------------------------------------------------------------------------


def _write_csv(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return p


def _write_xlsx(
    tmp_path: Path, name: str, data: dict[str, list[dict]]
) -> Path:
    """Write named sheets to an Excel workbook (all sheets: rows as dicts)."""
    p = tmp_path / name
    with pd.ExcelWriter(p, engine="openpyxl") as writer:
        for sheet, rows in data.items():
            pd.DataFrame(rows).to_excel(writer, sheet_name=sheet, index=False)
    return p


def _assert_schema(df: pd.DataFrame) -> None:
    assert list(df.columns) == _SCHEMA_COLS, f"Column mismatch: {list(df.columns)}"
    for col, dtype in _SCHEMA_DTYPES.items():
        assert df[col].dtype == dtype, (
            f"'{col}': expected {dtype}, got {df[col].dtype}"
        )


# ---------------------------------------------------------------------------
# Helpers — DeepCRISPR two-sheet Excel format
# ---------------------------------------------------------------------------

# Actual column names from chuai2018_deepcrispr_k562.xlsx
_DC_TARGET_COLS = ["Target ID", "Chromosome", "Start", "End", "Strand", "Target sgRNA"]
_DC_OT_COLS = [
    "OT-ID", "Target ID", "Chromosome", "Start", "End", "Strand",
    "OT", "Cleavage Frequency",
]


def _make_dc_target_row(
    tid: str = "sg1",
    chrom: str = "chr1",
    start: int = 1000,
    end: int = 1023,
    strand: str = "+",
    guide_seq_23: str | None = None,
) -> dict[str, Any]:
    """One row for the 'target' sheet.  guide_seq_23 defaults to GUIDE+NGG."""
    return {
        "Target ID": tid,
        "Chromosome": chrom,
        "Start": start,
        "End": end,
        "Strand": strand,
        "Target sgRNA": guide_seq_23 or (GUIDE + "NGG"),
    }


def _make_dc_ot_row(
    ot_id: str = "sg1_ot1",
    tid: str = "sg1",
    chrom: str = "chr1",
    start: int = 2000,
    end: int = 2023,
    strand: str = "+",
    ot_seq: str = OT_1MM,
    freq: float = 0.05,
) -> dict[str, Any]:
    """One row for the 'off-target' sheet."""
    return {
        "OT-ID": ot_id,
        "Target ID": tid,
        "Chromosome": chrom,
        "Start": start,
        "End": end,
        "Strand": strand,
        "OT": ot_seq,
        "Cleavage Frequency": freq,
    }


def _write_dc_excel(
    tmp_path: Path,
    target_rows: list[dict] | None = None,
    ot_rows: list[dict] | None = None,
    name: str = "dc.xlsx",
) -> Path:
    """Write a two-sheet DeepCRISPR Excel file."""
    p = tmp_path / name
    target = pd.DataFrame(target_rows or [_make_dc_target_row()])
    ot = pd.DataFrame(ot_rows or [_make_dc_ot_row()])
    with pd.ExcelWriter(p, engine="openpyxl") as w:
        target.to_excel(w, sheet_name="target", index=False)
        ot.to_excel(w, sheet_name="off-target", index=False)
    return p


# ---------------------------------------------------------------------------
# Helpers — DeepCRISPR flat-file (CSV) format  (backward-compat / fallback)
# ---------------------------------------------------------------------------


def _make_dc_row(
    *,
    cell_type: str | None = "K562",
    ot_seq: str = OT_1MM,
    n_mm: int | None = 1,
    freq: float = 0.05,
    chrom: str = "chr1",
    start: int = 1000,
    end: int = 1023,
    strand: str = "+",
) -> dict[str, Any]:
    """Minimal DeepCRISPR flat-file row using canonical column names."""
    row: dict[str, Any] = {
        "Target":             "VEGFA_site2",
        "Target_Sequence":    GUIDE,
        "Offtarget_Sequence": ot_seq,
        "Chromosome":         chrom,
        "Start":              start,
        "End":                end,
        "Strand":             strand,
        "Cleavage_Frequency": freq,
    }
    if cell_type is not None:
        row["Cell_type"] = cell_type
    if n_mm is not None:
        row["Mismatches"] = n_mm
    return row


# ---------------------------------------------------------------------------
# Helpers — DISCOVER-seq S1 Excel (actual three-sheet format)
# ---------------------------------------------------------------------------

# --- DISCOVER VEGFA sheet helpers ---
# header=2: rows 0,1 are title/span; row 2 is the actual column header.
_VEGFA_COLS = [
    None, "Chr:Start-End", "Cutsite", "Strand", "PAM",
    "Guide sequence", "Mismatches", "Rep1", "Rep2",
]


def _vegfa_raw_rows(data_rows: list[list]) -> list[list]:
    """Build the full set of rows for the VEGFA sheet (incl. 2 dummy rows)."""
    dummy0 = ["Table S1: DISCOVER scores"] + [None] * (len(_VEGFA_COLS) - 1)
    dummy1 = [None] * len(_VEGFA_COLS)
    header = _VEGFA_COLS  # this row becomes the pandas column names at header=2
    return [dummy0, dummy1, header] + data_rows


def _make_vegfa_data_row(
    gene: str = "PAPD7",
    coord: str = "chr5:6714989-6715008",
    cutsite: int = 6715005,
    strand: str = "sense",
    pam: str = "NGG",
    guide_seq: str = GUIDE,
    mismatches: int = 1,
    rep1: float = 100.0,
    rep2: float = 50.0,
) -> list:
    return [gene, coord, cutsite, strand, pam, guide_seq, mismatches, rep1, rep2]


def _make_vegfa_ontarget_row(guide_seq: str = GUIDE) -> list:
    return ["VEGFA", "chr6:43770822-43770841", 43770824, "antisense",
            "NGG", guide_seq, 0, 417, 321]


# --- DISCOVER RNF2 sheet helpers ---
# header=1: row 0 is title; row 1 is the column header.
_RNF2_COLS = [
    "gRNA", "gene", "Hg38 coordinates", "cut site", "Strand",
    "PAM", "Guide sequence", "mismatches", "Rep1", "Rep2",
]


def _rnf2_raw_rows(data_rows: list[list]) -> list[list]:
    dummy0 = ["Table S1: DISCOVER scores"] + [None] * (len(_RNF2_COLS) - 1)
    header = _RNF2_COLS
    return [dummy0, header] + data_rows


def _make_rnf2_ontarget_row(guide_seq: str = GUIDE) -> list:
    return ["RNF2", "RNF2", "chr1:185087638-185087657",
            185087640, "antisense", "NGG", guide_seq, 0, 44, 56]


def _make_rnf2_data_row(
    guide_id: str = "RNF2",
    coord: str = "chr1:100000-100019",
    strand: str = "sense",
    pam: str = "NGG",
    guide_seq: str = GUIDE,
    mismatches: int = 2,
    rep1: float = 10.0,
    rep2: float = 5.0,
) -> list:
    return [guide_id, "gene", coord, 100010, strand, pam, guide_seq,
            mismatches, rep1, rep2]


# --- DISCOVER HBB sheet helpers ---
# header=1: row 0 is title; row 1 is column header.
# An extra unnamed column (Unnamed: 5) appears between 'cut site' and 'Strand'.
_HBB_COLS = [
    "gRNA", "gene", "Cas9", "Hg38 coordinates", "cut site", None,
    "Strand", "PAM", "Guide sequence", "mismatches", "Rep1", "Rep2",
]


def _hbb_raw_rows(data_rows: list[list]) -> list[list]:
    dummy0 = ["Table S1: DISCOVER scores"] + [None] * (len(_HBB_COLS) - 1)
    header = _HBB_COLS
    return [dummy0, header] + data_rows


def _make_hbb_ontarget_row(guide_seq: str = GUIDE) -> list:
    return ["HBB", "HBB", "WT", "chr11:5226968-5226987",
            5226984, None, "sense", "NGG", guide_seq, 0, 124, 372]


def _make_hbb_data_row(
    coord: str = "chr9:101833584-101833603",
    strand: str = "sense",
    pam: str = "NGG",
    guide_seq: str = GUIDE,
    mismatches: int = 3,
    rep1: float = 49.0,
    rep2: float = 97.0,
    cas9: str = "WT",
) -> list:
    return ["HBB", "OT1 (NC)", cas9, coord, 101833600,
            None, strand, pam, guide_seq, mismatches, rep1, rep2]


def _hbb_separator_rows() -> list[list]:
    """Two NaN separator rows between WT and HiFi Cas9 sections."""
    nan_row = [None] * len(_HBB_COLS)
    return [nan_row, nan_row]


def _write_discover_s1(
    tmp_path: Path,
    vegfa_data: list[list] | None = None,
    rnf2_data: list[list] | None = None,
    hbb_data: list[list] | None = None,
    name: str = "s1.xlsx",
    include_vegfa: bool = True,
    include_rnf2: bool = True,
    include_hbb: bool = True,
) -> Path:
    """Write a synthetic S1 Excel with DISCOVER sheets matching the real format."""
    p = tmp_path / name

    # Default synthetic data
    if vegfa_data is None:
        vegfa_data = [_make_vegfa_ontarget_row(), _make_vegfa_data_row()]
    if rnf2_data is None:
        rnf2_data = [_make_rnf2_ontarget_row()]  # on-target only → 0 OTs
    if hbb_data is None:
        hbb_data = (
            [_make_hbb_ontarget_row(), _make_hbb_data_row()]
            + _hbb_separator_rows()
        )

    with pd.ExcelWriter(p, engine="openpyxl") as writer:
        if include_vegfa:
            raw_vegfa = pd.DataFrame(_vegfa_raw_rows(vegfa_data))
            raw_vegfa.to_excel(
                writer, sheet_name="DISCOVER VEGFA", index=False, header=False
            )
        if include_rnf2:
            raw_rnf2 = pd.DataFrame(_rnf2_raw_rows(rnf2_data))
            raw_rnf2.to_excel(
                writer, sheet_name="DISCOVER RNF2", index=False, header=False
            )
        if include_hbb:
            raw_hbb = pd.DataFrame(_hbb_raw_rows(hbb_data))
            raw_hbb.to_excel(
                writer, sheet_name="DISCOVER HBB", index=False, header=False
            )

    return p


# ===========================================================================
# _pick_col
# ===========================================================================

class TestPickCol:

    def test_finds_first_matching_alias(self):
        df = pd.DataFrame({"B": [1], "C": [2]})
        assert _pick_col(df, ["A", "B", "C"], "x") == "B"

    def test_prefers_earlier_alias(self):
        df = pd.DataFrame({"A": [1], "B": [2]})
        assert _pick_col(df, ["A", "B"], "x") == "A"

    def test_raises_when_no_alias_matches(self):
        df = pd.DataFrame({"X": [1]})
        with pytest.raises(ValueError, match="Actual columns"):
            _pick_col(df, ["A", "B"], "myfield")

    def test_error_lists_tried_aliases(self):
        df = pd.DataFrame({"Z": [1]})
        with pytest.raises(ValueError, match="Tried aliases"):
            _pick_col(df, ["alpha", "beta"], "foo")


# ===========================================================================
# _load_tabular
# ===========================================================================

class TestLoadTabular:

    def test_loads_tsv(self, tmp_path):
        p = tmp_path / "data.tsv"
        p.write_text("A\tB\n1\t2\n3\t4\n")
        df = _load_tabular(p)
        assert list(df.columns) == ["A", "B"]
        assert len(df) == 2

    def test_loads_csv(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("A,B\n1,2\n3,4\n")
        df = _load_tabular(p)
        assert list(df.columns) == ["A", "B"]
        assert len(df) == 2

    def test_loads_xlsx(self, tmp_path):
        p = _write_xlsx(tmp_path, "data.xlsx", {"Sheet1": [{"A": 1, "B": 2}]})
        df = _load_tabular(p)
        assert "A" in df.columns
        assert len(df) == 1


# ===========================================================================
# _apply_normalise_chrom
# ===========================================================================

class TestApplyNormaliseChrom:

    def test_valid_chroms_pass_through(self):
        df = pd.DataFrame({"chrom": ["chr1", "chrX", "chr22"]})
        df, mask = _apply_normalise_chrom(df, "chrom")
        assert all(mask)
        assert list(df["chrom"]) == ["chr1", "chrX", "chr22"]

    def test_bare_numbers_get_chr_prefix(self):
        df = pd.DataFrame({"chrom": ["1", "22", "X"]})
        df, mask = _apply_normalise_chrom(df, "chrom")
        assert all(mask)
        assert df["chrom"].tolist() == ["chr1", "chr22", "chrX"]

    def test_invalid_contig_flagged_false(self):
        df = pd.DataFrame({"chrom": ["chr1", "scaffold_99999"]})
        df, mask = _apply_normalise_chrom(df, "chrom")
        assert bool(mask.iloc[0]) is True
        assert bool(mask.iloc[1]) is False

    def test_returns_same_length(self):
        df = pd.DataFrame({"chrom": ["chr1", "chr2", "bad"]})
        df, mask = _apply_normalise_chrom(df, "chrom")
        assert len(mask) == 3


# ===========================================================================
# parse_deepcrispr_k562 — two-sheet Excel format (actual Chuai 2018 file)
# ===========================================================================

class TestParseDeepCRISPRK562Excel:
    """Tests for the two-sheet Excel path (_parse_deepcrispr_twosheets)."""

    # --- Error handling ---

    def test_missing_file_raises_with_download_url(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="genomebiology"):
            parse_deepcrispr_k562(tmp_path / "nonexistent.xlsx")

    # --- Happy path ---

    def test_basic_two_sheet_excel_correct_schema(self, tmp_path):
        p = _write_dc_excel(tmp_path)
        df = parse_deepcrispr_k562(p, src_genome_build="hg38")
        _assert_schema(df)
        assert len(df) == 1

    def test_metadata_fields(self, tmp_path):
        p = _write_dc_excel(tmp_path)
        df = parse_deepcrispr_k562(p, src_genome_build="hg38")
        assert df["cell_type"].iloc[0] == "K562"
        assert df["dataset"].iloc[0] == "deepcrispr_k562"
        assert df["genome_build"].iloc[0] == "hg38"

    def test_guide_id_is_target_id(self, tmp_path):
        p = _write_dc_excel(
            tmp_path,
            target_rows=[_make_dc_target_row(tid="sg42")],
            ot_rows=[_make_dc_ot_row(tid="sg42")],
        )
        df = parse_deepcrispr_k562(p, src_genome_build="hg38")
        assert df["guide_id"].iloc[0] == "sg42"

    # --- Sequence handling ---

    def test_guide_seq_trimmed_to_20nt_from_23nt_target_sgrna(self, tmp_path):
        """'Target sgRNA' is 23-nt; parser must take only first 20 nt."""
        p = _write_dc_excel(
            tmp_path,
            target_rows=[_make_dc_target_row(guide_seq_23=GUIDE + "NGG")],
            ot_rows=[_make_dc_ot_row(ot_seq=OT_1MM)],
        )
        df = parse_deepcrispr_k562(p, src_genome_build="hg38")
        assert len(df["guide_seq"].iloc[0]) == 20
        assert df["guide_seq"].iloc[0] == GUIDE

    def test_23nt_ot_extracts_pam(self, tmp_path):
        p = _write_dc_excel(
            tmp_path,
            ot_rows=[_make_dc_ot_row(ot_seq=OT_23NT)],
        )
        df = parse_deepcrispr_k562(p, src_genome_build="hg38")
        assert df["pam_seq"].iloc[0] == "GGG"
        assert len(df["offtarget_seq"].iloc[0]) == 20

    def test_20nt_ot_gets_nnn_pam(self, tmp_path):
        p = _write_dc_excel(
            tmp_path,
            ot_rows=[_make_dc_ot_row(ot_seq=OT_1MM)],
        )
        df = parse_deepcrispr_k562(p, src_genome_build="hg38")
        assert df["pam_seq"].iloc[0] == "NNN"

    def test_offtarget_seq_normalised_to_uppercase(self, tmp_path):
        p = _write_dc_excel(
            tmp_path,
            ot_rows=[_make_dc_ot_row(ot_seq=OT_1MM.lower())],
        )
        df = parse_deepcrispr_k562(p, src_genome_build="hg38")
        assert df["offtarget_seq"].iloc[0] == OT_1MM

    # --- Mismatch count (computed from sequences in two-sheet format) ---

    def test_mismatch_count_computed_from_sequences(self, tmp_path):
        p = _write_dc_excel(
            tmp_path,
            ot_rows=[_make_dc_ot_row(ot_seq=OT_2MM)],
        )
        df = parse_deepcrispr_k562(p, src_genome_build="hg38")
        assert df["num_mismatches"].iloc[0] == 2

    # --- On-target filtering ---

    def test_on_target_row_removed(self, tmp_path):
        """Row where OT matches guide exactly (0 mismatches) must be dropped."""
        on_target_ot = _make_dc_ot_row(ot_id="sg1_on", ot_seq=GUIDE + "NGG")
        off_target_ot = _make_dc_ot_row(ot_id="sg1_ot1", ot_seq=OT_1MM)
        p = _write_dc_excel(tmp_path, ot_rows=[on_target_ot, off_target_ot])
        df = parse_deepcrispr_k562(p, src_genome_build="hg38")
        assert len(df) == 1
        assert df["num_mismatches"].iloc[0] > 0

    # --- Read count ---

    def test_read_count_equals_cleavage_frequency(self, tmp_path):
        p = _write_dc_excel(
            tmp_path,
            ot_rows=[_make_dc_ot_row(freq=0.314)],
        )
        df = parse_deepcrispr_k562(p, src_genome_build="hg38")
        assert df["read_count"].iloc[0] == pytest.approx(0.314)

    # --- Coordinates ---

    def test_coordinates_stored_correctly(self, tmp_path):
        p = _write_dc_excel(
            tmp_path,
            ot_rows=[_make_dc_ot_row(chrom="chr7", start=5000, end=5023)],
        )
        df = parse_deepcrispr_k562(p, src_genome_build="hg38")
        assert df["chrom"].iloc[0] == "chr7"
        assert df["start"].iloc[0] == 5000
        assert df["end"].iloc[0] == 5023

    # --- Multiple guides ---

    def test_multiple_guides_from_two_target_rows(self, tmp_path):
        targets = [
            _make_dc_target_row(tid="sg1"),
            _make_dc_target_row(tid="sg2"),
        ]
        ots = [
            _make_dc_ot_row(ot_id="sg1_ot1", tid="sg1", ot_seq=OT_1MM),
            _make_dc_ot_row(ot_id="sg1_ot2", tid="sg1", ot_seq=OT_2MM),
            _make_dc_ot_row(ot_id="sg2_ot1", tid="sg2", ot_seq=OT_1MM),
        ]
        p = _write_dc_excel(tmp_path, target_rows=targets, ot_rows=ots)
        df = parse_deepcrispr_k562(p, src_genome_build="hg38")
        assert len(df) == 3
        assert df["guide_id"].nunique() == 2

    def test_unknown_target_id_in_ot_sheet_dropped(self, tmp_path):
        targets = [_make_dc_target_row(tid="sg1")]
        ots = [
            _make_dc_ot_row(tid="sg1", ot_seq=OT_1MM),
            _make_dc_ot_row(tid="sg99", ot_seq=OT_2MM),  # not in target sheet
        ]
        p = _write_dc_excel(tmp_path, target_rows=targets, ot_rows=ots)
        df = parse_deepcrispr_k562(p, src_genome_build="hg38")
        assert len(df) == 1
        assert df["guide_id"].iloc[0] == "sg1"

    # --- hg19 liftover triggered ---

    def test_hg19_triggers_liftover(self, tmp_path):
        p = _write_dc_excel(tmp_path)
        lifted = ("chr1", 2000)
        with patch.object(_PD, "liftover_hg19_to_hg38", return_value=lifted):
            df = parse_deepcrispr_k562(p, src_genome_build="hg19")
        assert len(df) == 1
        assert df["start"].iloc[0] == 2000

    def test_hg19_liftover_failure_drops_row(self, tmp_path):
        p = _write_dc_excel(tmp_path)
        with patch.object(_PD, "liftover_hg19_to_hg38", return_value=None):
            df = parse_deepcrispr_k562(p, src_genome_build="hg19")
        assert len(df) == 0


# ===========================================================================
# parse_deepcrispr_k562 — flat-file CSV format (fallback path)
# ===========================================================================

class TestParseDeepCRISPRK562FlatFile:
    """Tests for the flat-file fallback path (CSV / TSV format)."""

    def test_missing_required_column_raises_with_actual_columns(self, tmp_path):
        # Row present but guide_seq and offtarget_seq are missing
        rows = [{
            "Target": "X", "Chromosome": "chr1",
            "Start": 0, "End": 23, "Strand": "+",
            "Cleavage_Frequency": 0.1, "Mismatches": 1,
        }]
        path = _write_csv(tmp_path, "dc.tsv", rows)
        with pytest.raises(ValueError, match="Actual columns"):
            parse_deepcrispr_k562(path, src_genome_build="hg38")

    def test_basic_valid_csv_produces_correct_schema(self, tmp_path):
        path = _write_csv(tmp_path, "dc.tsv", [_make_dc_row()])
        df = parse_deepcrispr_k562(path, src_genome_build="hg38")
        _assert_schema(df)
        assert len(df) == 1

    def test_k562_filter_removes_hek293_rows(self, tmp_path):
        rows = [
            _make_dc_row(cell_type="K562"),
            _make_dc_row(cell_type="HEK293", ot_seq=OT_2MM),
        ]
        path = _write_csv(tmp_path, "dc.tsv", rows)
        df = parse_deepcrispr_k562(path, src_genome_build="hg38")
        assert len(df) == 1
        assert df["cell_type"].iloc[0] == "K562"

    def test_no_cell_type_col_keeps_all_rows(self, tmp_path):
        rows = [
            _make_dc_row(cell_type=None),
            _make_dc_row(cell_type=None, ot_seq=OT_2MM),
        ]
        path = _write_csv(tmp_path, "dc.tsv", rows)
        df = parse_deepcrispr_k562(path, src_genome_build="hg38")
        assert len(df) == 2

    def test_on_target_row_is_removed(self, tmp_path):
        on_target  = _make_dc_row(ot_seq=GUIDE, n_mm=0)
        off_target = _make_dc_row()
        path = _write_csv(tmp_path, "dc.tsv", [on_target, off_target])
        df = parse_deepcrispr_k562(path, src_genome_build="hg38")
        assert len(df) == 1
        assert df["num_mismatches"].iloc[0] > 0

    def test_20nt_ot_seq_gets_nnn_pam(self, tmp_path):
        path = _write_csv(tmp_path, "dc.tsv", [_make_dc_row(ot_seq=OT_1MM)])
        df = parse_deepcrispr_k562(path, src_genome_build="hg38")
        assert df["pam_seq"].iloc[0] == "NNN"

    def test_23nt_ot_seq_extracts_pam(self, tmp_path):
        path = _write_csv(tmp_path, "dc.tsv", [_make_dc_row(ot_seq=OT_23NT)])
        df = parse_deepcrispr_k562(path, src_genome_build="hg38")
        assert df["pam_seq"].iloc[0] == "GGG"
        assert len(df["offtarget_seq"].iloc[0]) == 20

    def test_mismatch_count_from_column(self, tmp_path):
        path = _write_csv(tmp_path, "dc.tsv", [_make_dc_row(n_mm=2)])
        df = parse_deepcrispr_k562(path, src_genome_build="hg38")
        assert df["num_mismatches"].iloc[0] == 2

    def test_mismatch_computed_from_sequences_when_col_absent(self, tmp_path):
        path = _write_csv(tmp_path, "dc.tsv", [_make_dc_row(n_mm=None)])
        df = parse_deepcrispr_k562(path, src_genome_build="hg38")
        assert df["num_mismatches"].iloc[0] == 1

    def test_alternative_column_names_accepted(self, tmp_path):
        rows = [{
            "sgRNA":         "RNF2",
            "guide_seq":     GUIDE,
            "offtarget_seq": OT_2MM,
            "chrom":         "chr1",
            "start":         5000,
            "end":           5023,
            "strand":        "-",
            "score":         0.02,
            "Mismatches":    2,
        }]
        path = _write_csv(tmp_path, "dc.tsv", rows)
        df = parse_deepcrispr_k562(path, src_genome_build="hg38")
        assert len(df) == 1
        assert df["guide_id"].iloc[0] == "RNF2"

    def test_read_count_equals_cleavage_freq(self, tmp_path):
        path = _write_csv(tmp_path, "dc.tsv", [_make_dc_row(freq=0.123)])
        df = parse_deepcrispr_k562(path, src_genome_build="hg38")
        assert df["read_count"].iloc[0] == pytest.approx(0.123)


# ===========================================================================
# parse_discover_seq_k562 — actual S1 three-sheet format
# ===========================================================================

class TestParseDiscoverSeqK562:

    # --- Error handling ---

    def test_missing_file_raises_with_download_url(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="PMC6589096"):
            parse_discover_seq_k562(tmp_path / "nonexistent.xlsx")

    def test_missing_required_column_raises_with_actual_columns(self, tmp_path):
        """VEGFA sheet missing 'Guide sequence' → ValueError with actual cols."""
        bad_vegfa_rows = [
            # Data row: has coord but no Guide sequence column
            _vegfa_raw_rows([[
                "PAPD7", "chr5:6714989-6715008", 6715005, "sense",
                "NGG", "MISSING_GUIDE", 1, 100.0, 50.0
            ]])
        ]
        # Write a VEGFA sheet with wrong column names by using raw DataFrame
        p = tmp_path / "s1.xlsx"
        # Sheet with no 'Guide sequence' column at all
        raw = pd.DataFrame([
            ["Title"] + [None]*4,
            [None]*5,
            [None, "Chr:Start-End", "Cutsite", "Strand", "PAM"],  # missing Guide sequence
            ["PAPD7", "chr5:100-200", 150, "sense", "NGG"],
        ])
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            raw.to_excel(w, sheet_name="DISCOVER VEGFA", index=False, header=False)
            pd.DataFrame([["RNF2"]]).to_excel(w, sheet_name="DISCOVER RNF2", index=False)
            pd.DataFrame([["HBB"]]).to_excel(w, sheet_name="DISCOVER HBB", index=False)
        with pytest.raises(ValueError, match="Actual columns"):
            parse_discover_seq_k562(p)

    def test_no_discover_sheets_raises_value_error(self, tmp_path):
        """File with no DISCOVER sheets at all → ValueError."""
        p = _write_xlsx(tmp_path, "s1.xlsx", {"Sheet1": [{"A": 1}]})
        with pytest.raises(ValueError):
            parse_discover_seq_k562(p)

    # --- Happy path: VEGFA sheet ---

    def test_vegfa_sheet_basic_output(self, tmp_path):
        p = _write_discover_s1(
            tmp_path,
            vegfa_data=[_make_vegfa_ontarget_row(), _make_vegfa_data_row()],
            rnf2_data=[],
            hbb_data=[],
        )
        df = parse_discover_seq_k562(p)
        assert len(df) == 1
        assert df["guide_id"].iloc[0] == "VEGFA"

    def test_vegfa_schema_correct(self, tmp_path):
        p = _write_discover_s1(
            tmp_path,
            vegfa_data=[_make_vegfa_ontarget_row(), _make_vegfa_data_row()],
            rnf2_data=[],
            hbb_data=[],
        )
        df = parse_discover_seq_k562(p)
        _assert_schema(df)

    def test_metadata_fields(self, tmp_path):
        p = _write_discover_s1(
            tmp_path,
            vegfa_data=[_make_vegfa_ontarget_row(), _make_vegfa_data_row()],
            rnf2_data=[],
            hbb_data=[],
        )
        df = parse_discover_seq_k562(p)
        assert df["cell_type"].iloc[0] == "K562"
        assert df["dataset"].iloc[0] == "discover_seq_k562"
        assert df["genome_build"].iloc[0] == "hg38"

    def test_vegfa_on_target_removed(self, tmp_path):
        """Row with Mismatches=0 must be dropped."""
        p = _write_discover_s1(
            tmp_path,
            # on-target (mismatches=0) + 1 off-target
            vegfa_data=[_make_vegfa_ontarget_row(), _make_vegfa_data_row()],
            rnf2_data=[],
            hbb_data=[],
        )
        df = parse_discover_seq_k562(p)
        assert len(df) == 1
        assert df["num_mismatches"].iloc[0] > 0

    def test_vegfa_offtarget_seq_is_nnn_placeholder(self, tmp_path):
        """DISCOVER sheets have no genomic OT sequence → 'N'*20."""
        p = _write_discover_s1(
            tmp_path,
            vegfa_data=[_make_vegfa_ontarget_row(), _make_vegfa_data_row()],
            rnf2_data=[],
            hbb_data=[],
        )
        df = parse_discover_seq_k562(p)
        assert df["offtarget_seq"].iloc[0] == "N" * 20

    def test_vegfa_pam_from_column(self, tmp_path):
        p = _write_discover_s1(
            tmp_path,
            vegfa_data=[
                _make_vegfa_ontarget_row(),
                _make_vegfa_data_row(pam="AGG"),
            ],
            rnf2_data=[],
            hbb_data=[],
        )
        df = parse_discover_seq_k562(p)
        assert df["pam_seq"].iloc[0] == "AGG"

    def test_vegfa_read_count_is_sum_of_reps(self, tmp_path):
        p = _write_discover_s1(
            tmp_path,
            vegfa_data=[
                _make_vegfa_ontarget_row(),
                _make_vegfa_data_row(rep1=100.0, rep2=50.0),
            ],
            rnf2_data=[],
            hbb_data=[],
        )
        df = parse_discover_seq_k562(p)
        assert df["read_count"].iloc[0] == pytest.approx(150.0)

    def test_vegfa_guide_seq_trimmed_to_20nt(self, tmp_path):
        p = _write_discover_s1(
            tmp_path,
            vegfa_data=[
                _make_vegfa_ontarget_row(),
                _make_vegfa_data_row(guide_seq=GUIDE + "NGG"),
            ],
            rnf2_data=[],
            hbb_data=[],
        )
        df = parse_discover_seq_k562(p)
        assert len(df["guide_seq"].iloc[0]) == 20
        assert df["guide_seq"].iloc[0] == GUIDE

    def test_vegfa_strand_normalised_sense_to_plus(self, tmp_path):
        p = _write_discover_s1(
            tmp_path,
            vegfa_data=[
                _make_vegfa_ontarget_row(),
                _make_vegfa_data_row(strand="sense"),
            ],
            rnf2_data=[],
            hbb_data=[],
        )
        df = parse_discover_seq_k562(p)
        assert df["strand"].iloc[0] == "+"

    def test_vegfa_strand_normalised_antisense_to_minus(self, tmp_path):
        p = _write_discover_s1(
            tmp_path,
            vegfa_data=[
                _make_vegfa_ontarget_row(),
                _make_vegfa_data_row(strand="antisense"),
            ],
            rnf2_data=[],
            hbb_data=[],
        )
        df = parse_discover_seq_k562(p)
        assert df["strand"].iloc[0] == "-"

    def test_vegfa_coordinate_parsed_from_string(self, tmp_path):
        p = _write_discover_s1(
            tmp_path,
            vegfa_data=[
                _make_vegfa_ontarget_row(),
                _make_vegfa_data_row(coord="chr5:6714989-6715008"),
            ],
            rnf2_data=[],
            hbb_data=[],
        )
        df = parse_discover_seq_k562(p)
        assert df["chrom"].iloc[0] == "chr5"
        assert df["start"].iloc[0] == 6714989
        assert df["end"].iloc[0] == 6715008

    def test_vegfa_mismatch_count_from_column(self, tmp_path):
        p = _write_discover_s1(
            tmp_path,
            vegfa_data=[
                _make_vegfa_ontarget_row(),
                _make_vegfa_data_row(mismatches=4),
            ],
            rnf2_data=[],
            hbb_data=[],
        )
        df = parse_discover_seq_k562(p)
        assert df["num_mismatches"].iloc[0] == 4

    # --- RNF2 sheet: 0 off-targets ---

    def test_rnf2_returns_empty_after_filtering_on_targets(self, tmp_path):
        """RNF2 has only on-target (mismatches=0) → 0 rows returned."""
        p = _write_discover_s1(
            tmp_path,
            vegfa_data=[],
            rnf2_data=[_make_rnf2_ontarget_row()],
            hbb_data=[],
        )
        df = parse_discover_seq_k562(p)
        # Only RNF2 sheet populated but it has 0 off-targets
        assert len(df) == 0
        _assert_schema(df)

    def test_rnf2_with_off_target_row_included(self, tmp_path):
        """If RNF2 actually has an off-target, it should appear in output."""
        p = _write_discover_s1(
            tmp_path,
            vegfa_data=[],
            rnf2_data=[
                _make_rnf2_ontarget_row(),
                _make_rnf2_data_row(mismatches=2),
            ],
            hbb_data=[],
        )
        df = parse_discover_seq_k562(p)
        assert len(df) == 1
        assert df["guide_id"].iloc[0] == "RNF2"

    # --- HBB sheet: NaN separator rows ---

    def test_hbb_nan_separator_rows_dropped(self, tmp_path):
        """Rows with NaN in coordinate column (separator rows) must be dropped."""
        hbb_data = [
            _make_hbb_ontarget_row(),    # on-target (filtered)
            _make_hbb_data_row(),         # off-target WT (kept)
            *_hbb_separator_rows(),       # NaN rows (dropped)
            _make_hbb_data_row(cas9="HiFi", rep1=4.0, rep2=2.0),  # HiFi (kept)
        ]
        p = _write_discover_s1(tmp_path, vegfa_data=[], rnf2_data=[], hbb_data=hbb_data)
        df = parse_discover_seq_k562(p)
        assert len(df) == 2
        assert df["guide_id"].unique().tolist() == ["HBB"]

    def test_hbb_read_count_is_sum_of_reps(self, tmp_path):
        hbb_data = [
            _make_hbb_ontarget_row(),
            _make_hbb_data_row(rep1=49.0, rep2=97.0),
        ]
        p = _write_discover_s1(tmp_path, vegfa_data=[], rnf2_data=[], hbb_data=hbb_data)
        df = parse_discover_seq_k562(p)
        assert df["read_count"].iloc[0] == pytest.approx(146.0)

    # --- Combined sheets ---

    def test_multiple_guides_combined(self, tmp_path):
        p = _write_discover_s1(
            tmp_path,
            vegfa_data=[_make_vegfa_ontarget_row(), _make_vegfa_data_row()],
            rnf2_data=[_make_rnf2_ontarget_row()],
            hbb_data=[_make_hbb_ontarget_row(), _make_hbb_data_row()],
        )
        df = parse_discover_seq_k562(p)
        assert len(df) == 2
        assert set(df["guide_id"].unique()) == {"VEGFA", "HBB"}

    def test_schema_with_combined_sheets(self, tmp_path):
        p = _write_discover_s1(
            tmp_path,
            vegfa_data=[_make_vegfa_ontarget_row(), _make_vegfa_data_row()],
            rnf2_data=[_make_rnf2_ontarget_row()],
            hbb_data=[_make_hbb_ontarget_row(), _make_hbb_data_row()],
        )
        df = parse_discover_seq_k562(p)
        _assert_schema(df)

    def test_all_three_guides_all_on_targets_returns_empty(self, tmp_path):
        """If all sheets have only on-target rows, output is empty."""
        p = _write_discover_s1(
            tmp_path,
            vegfa_data=[_make_vegfa_ontarget_row()],
            rnf2_data=[_make_rnf2_ontarget_row()],
            hbb_data=[_make_hbb_ontarget_row()],
        )
        df = parse_discover_seq_k562(p)
        assert len(df) == 0
        _assert_schema(df)
