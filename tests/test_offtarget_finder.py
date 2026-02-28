"""
tests/test_offtarget_finder.py — Tests for src/offtarget_finder.py

Strategy
--------
- _write_input, _parse_output, _compute_mismatch_positions:
    tested directly against real files / pure logic (no mocking).
- find_offtargets:
    binary_path / genome_path existence is checked first, so we test those
    error paths directly.  The full happy-path is tested by patching
    subprocess.run and writing a fake output file via a patched _run_binary.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.offtarget_finder import (
    _SPACER_LENGTH,
    _compute_mismatch_positions,
    _empty_dataframe,
    _parse_output,
    _write_input,
    find_offtargets,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

GUIDE = "GAGTCCGAGCAGAAGAAGAA"  # 20 nt, no PAM


def _make_output_file(tmp_path: Path, lines: list[str]) -> Path:
    """Write a fake Cas-OFFinder output file and return its path."""
    p = tmp_path / "output.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# One real off-target line: single mismatch at position 1 (T instead of G)
# Actual Cas-OFFinder v2 output format: guide_pam  chrom  pos  DNA+PAM  strand  num_mm
_OT_LINE = (
    f"{GUIDE}NGG\tchr7\t117548628\t"
    f"TAGTCCGAGCAGAAGAAGAANGG\t+\t1"
)

# Perfect-match (on-target) line: 0 mismatches
_ONTARGET_LINE = (
    f"{GUIDE}NGG\tchr7\t117548628\t"
    f"{GUIDE}NGG\t+\t0"
)


# ---------------------------------------------------------------------------
# _write_input
# ---------------------------------------------------------------------------


class TestWriteInput:
    def test_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "input.txt"
        _write_input(p, GUIDE, Path("/genome/hg38.fa"), "NGG", 4)
        assert p.exists()

    def test_line_count(self, tmp_path: Path) -> None:
        p = tmp_path / "input.txt"
        _write_input(p, GUIDE, Path("/genome/hg38.fa"), "NGG", 4)
        lines = p.read_text().splitlines()
        assert len(lines) == 3

    def test_line1_is_genome_path(self, tmp_path: Path) -> None:
        genome = tmp_path / "hg38.fa"
        genome.touch()
        p = tmp_path / "input.txt"
        _write_input(p, GUIDE, genome, "NGG", 4)
        lines = p.read_text().splitlines()
        assert str(genome.resolve()) == lines[0]

    def test_line2_crispr_pattern_length(self, tmp_path: Path) -> None:
        p = tmp_path / "input.txt"
        _write_input(p, GUIDE, Path("/g/hg38.fa"), "NGG", 4)
        pattern = p.read_text().splitlines()[1]
        # 20 N's + 3-char PAM = 23 chars
        assert len(pattern) == _SPACER_LENGTH + 3

    def test_line2_starts_with_n_spacer(self, tmp_path: Path) -> None:
        p = tmp_path / "input.txt"
        _write_input(p, GUIDE, Path("/g/hg38.fa"), "NGG", 4)
        pattern = p.read_text().splitlines()[1]
        assert pattern[:_SPACER_LENGTH] == "N" * _SPACER_LENGTH

    def test_line2_ends_with_pam(self, tmp_path: Path) -> None:
        p = tmp_path / "input.txt"
        _write_input(p, GUIDE, Path("/g/hg38.fa"), "NGG", 4)
        pattern = p.read_text().splitlines()[1]
        assert pattern.endswith("NGG")

    def test_line3_contains_guide(self, tmp_path: Path) -> None:
        p = tmp_path / "input.txt"
        _write_input(p, GUIDE, Path("/g/hg38.fa"), "NGG", 4)
        entry = p.read_text().splitlines()[2]
        assert entry.startswith(GUIDE)

    def test_line3_contains_max_mismatches(self, tmp_path: Path) -> None:
        p = tmp_path / "input.txt"
        _write_input(p, GUIDE, Path("/g/hg38.fa"), "NGG", 4)
        entry = p.read_text().splitlines()[2]
        assert entry.endswith("4")

    def test_custom_pam_reflected(self, tmp_path: Path) -> None:
        p = tmp_path / "input.txt"
        _write_input(p, GUIDE, Path("/g/hg38.fa"), "NRG", 4)
        lines = p.read_text().splitlines()
        assert lines[1].endswith("NRG")
        assert lines[2].startswith(GUIDE + "NRG")

    def test_lowercase_pam_uppercased(self, tmp_path: Path) -> None:
        p = tmp_path / "input.txt"
        _write_input(p, GUIDE, Path("/g/hg38.fa"), "ngg", 4)
        lines = p.read_text().splitlines()
        assert lines[1].endswith("NGG")


# ---------------------------------------------------------------------------
# _compute_mismatch_positions
# ---------------------------------------------------------------------------


class TestComputeMismatchPositions:
    def test_no_mismatch_returns_empty(self) -> None:
        assert _compute_mismatch_positions(GUIDE, GUIDE) == []

    def test_single_mismatch_position_1(self) -> None:
        ot = "T" + GUIDE[1:]
        assert _compute_mismatch_positions(GUIDE, ot) == [1]

    def test_single_mismatch_position_20(self) -> None:
        ot = GUIDE[:-1] + "C"
        assert _compute_mismatch_positions(GUIDE, ot) == [20]

    def test_two_mismatches(self) -> None:
        # GUIDE[5] = "C"; replace index 0 with "T" and index 5 with "T"
        ot = "T" + GUIDE[1:5] + "T" + GUIDE[6:]
        result = _compute_mismatch_positions(GUIDE, ot)
        assert result == [1, 6]

    def test_all_mismatches(self) -> None:
        ot = "T" * _SPACER_LENGTH
        result = _compute_mismatch_positions(GUIDE, ot)
        # All positions that differ (G→T)
        expected = [i + 1 for i, b in enumerate(GUIDE) if b != "T"]
        assert result == expected

    def test_case_insensitive(self) -> None:
        assert _compute_mismatch_positions(GUIDE, GUIDE.lower()) == []

    def test_returns_sorted_positions(self) -> None:
        ot = GUIDE[:2] + "T" + GUIDE[3:10] + "C" + GUIDE[11:]
        result = _compute_mismatch_positions(GUIDE, ot)
        assert result == sorted(result)

    def test_partial_overlap_uses_zip(self) -> None:
        # zip stops at shorter sequence — positions beyond min length are not reported
        result = _compute_mismatch_positions("AAAAAAAAAA", "TTTTTTTTTT")
        assert len(result) == 10


# ---------------------------------------------------------------------------
# _empty_dataframe
# ---------------------------------------------------------------------------


class TestEmptyDataframe:
    def test_zero_rows(self) -> None:
        assert len(_empty_dataframe()) == 0

    def test_expected_columns(self) -> None:
        df = _empty_dataframe()
        expected = {
            "guide",
            "chrom",
            "start",
            "strand",
            "off_target_seq",
            "pam_seq",
            "num_mismatches",
            "mismatch_positions",
        }
        assert set(df.columns) == expected


# ---------------------------------------------------------------------------
# _parse_output
# ---------------------------------------------------------------------------


class TestParseOutput:
    def test_empty_file_returns_empty_df(self, tmp_path: Path) -> None:
        p = tmp_path / "out.txt"
        p.write_text("")
        df = _parse_output(p, GUIDE)
        assert len(df) == 0
        assert set(df.columns) >= {"guide", "chrom", "start", "num_mismatches"}

    def test_missing_file_returns_empty_df(self, tmp_path: Path) -> None:
        p = tmp_path / "does_not_exist.txt"
        df = _parse_output(p, GUIDE)
        assert len(df) == 0

    def test_one_hit_parsed(self, tmp_path: Path) -> None:
        p = _make_output_file(tmp_path, [_OT_LINE])
        df = _parse_output(p, GUIDE)
        assert len(df) == 1

    def test_chrom_normalized(self, tmp_path: Path) -> None:
        # Cas-OFFinder might return "7" instead of "chr7"
        line = _OT_LINE.replace("\tchr7\t", "\t7\t")
        p = _make_output_file(tmp_path, [line])
        df = _parse_output(p, GUIDE)
        assert df.loc[0, "chrom"] == "chr7"

    def test_start_is_integer(self, tmp_path: Path) -> None:
        p = _make_output_file(tmp_path, [_OT_LINE])
        df = _parse_output(p, GUIDE)
        assert df["start"].dtype.kind == "i"

    def test_num_mismatches_is_integer(self, tmp_path: Path) -> None:
        p = _make_output_file(tmp_path, [_OT_LINE])
        df = _parse_output(p, GUIDE)
        assert df["num_mismatches"].dtype.kind == "i"

    def test_strand_preserved(self, tmp_path: Path) -> None:
        p = _make_output_file(tmp_path, [_OT_LINE])
        df = _parse_output(p, GUIDE)
        assert df.loc[0, "strand"] == "+"

    def test_minus_strand_preserved(self, tmp_path: Path) -> None:
        minus_line = _OT_LINE.replace("\t+\t", "\t-\t")
        p = _make_output_file(tmp_path, [minus_line])
        df = _parse_output(p, GUIDE)
        assert df.loc[0, "strand"] == "-"

    def test_pam_seq_extracted(self, tmp_path: Path) -> None:
        p = _make_output_file(tmp_path, [_OT_LINE])
        df = _parse_output(p, GUIDE)
        # off-target sequence in _OT_LINE ends with "AAGAANGG"
        assert len(df.loc[0, "pam_seq"]) == 3

    def test_off_target_seq_is_20nt(self, tmp_path: Path) -> None:
        p = _make_output_file(tmp_path, [_OT_LINE])
        df = _parse_output(p, GUIDE)
        assert len(df.loc[0, "off_target_seq"]) == 20

    def test_mismatch_positions_computed(self, tmp_path: Path) -> None:
        p = _make_output_file(tmp_path, [_OT_LINE])
        df = _parse_output(p, GUIDE)
        # off-target spacer starts with T, guide starts with G → mismatch at 1
        assert 1 in df.loc[0, "mismatch_positions"]

    def test_guide_column_populated(self, tmp_path: Path) -> None:
        p = _make_output_file(tmp_path, [_OT_LINE])
        df = _parse_output(p, GUIDE)
        assert df.loc[0, "guide"] == GUIDE

    def test_malformed_line_skipped(self, tmp_path: Path) -> None:
        p = _make_output_file(tmp_path, ["only\ttwo\tfields", _OT_LINE])
        df = _parse_output(p, GUIDE)
        assert len(df) == 1

    def test_scaffold_chrom_skipped(self, tmp_path: Path) -> None:
        scaffold_line = _OT_LINE.replace("\tchr7\t", "\tchrUn_gl000220\t")
        p = _make_output_file(tmp_path, [scaffold_line, _OT_LINE])
        df = _parse_output(p, GUIDE)
        # scaffold site dropped, real chr7 site kept
        assert len(df) == 1
        assert df.loc[0, "chrom"] == "chr7"

    def test_multiple_hits(self, tmp_path: Path) -> None:
        line2 = _OT_LINE.replace("117548628", "200000000")
        p = _make_output_file(tmp_path, [_OT_LINE, line2])
        df = _parse_output(p, GUIDE)
        assert len(df) == 2

    def test_all_expected_columns_present(self, tmp_path: Path) -> None:
        p = _make_output_file(tmp_path, [_OT_LINE])
        df = _parse_output(p, GUIDE)
        for col in (
            "guide",
            "chrom",
            "start",
            "strand",
            "off_target_seq",
            "pam_seq",
            "num_mismatches",
            "mismatch_positions",
        ):
            assert col in df.columns, f"Missing column: {col}"

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        p = _make_output_file(tmp_path, ["", _OT_LINE, ""])
        df = _parse_output(p, GUIDE)
        assert len(df) == 1

    def test_non_integer_fields_skipped(self, tmp_path: Path) -> None:
        bad_line = _OT_LINE.replace("\t117548628\t", "\tNOT_A_NUM\t")
        p = _make_output_file(tmp_path, [bad_line])
        df = _parse_output(p, GUIDE)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# find_offtargets — error paths (no binary needed)
# ---------------------------------------------------------------------------


class TestFindOfftargetsErrors:
    def test_missing_genome_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Genome not found"):
            find_offtargets(
                GUIDE,
                tmp_path / "no_genome.fa",
                binary_path=tmp_path / "fake_bin",
            )

    def test_missing_binary_raises(self, tmp_path: Path) -> None:
        genome = tmp_path / "hg38.fa"
        genome.touch()
        with pytest.raises(FileNotFoundError, match="Cas-OFFinder binary not found"):
            find_offtargets(
                GUIDE,
                genome,
                binary_path=tmp_path / "no_binary",
            )


# ---------------------------------------------------------------------------
# find_offtargets — happy path (mocked binary)
# ---------------------------------------------------------------------------


class TestFindOfftargetsHappyPath:
    """Tests that mock _run_binary so no real Cas-OFFinder is needed."""

    def _mock_run_binary(self, output_lines: list[str]):
        """Return a mock for _run_binary that writes fake output."""

        def _fake_run(
            input_path: Path,
            output_path: Path,
            binary_path: Path,
            mode: str,
        ) -> None:
            output_path.write_text(
                "\n".join(output_lines) + "\n", encoding="utf-8"
            )

        return _fake_run

    def test_returns_dataframe(self, tmp_path: Path) -> None:
        genome = tmp_path / "hg38.fa"
        genome.touch()
        binary = tmp_path / "cas-offinder"
        binary.touch()

        with patch(
            "src.offtarget_finder._run_binary",
            side_effect=self._mock_run_binary([_OT_LINE]),
        ):
            df = find_offtargets(GUIDE, genome, binary_path=binary)

        assert len(df) == 1

    def test_ontarget_removed_by_default(self, tmp_path: Path) -> None:
        genome = tmp_path / "hg38.fa"
        genome.touch()
        binary = tmp_path / "cas-offinder"
        binary.touch()

        with patch(
            "src.offtarget_finder._run_binary",
            side_effect=self._mock_run_binary([_ONTARGET_LINE, _OT_LINE]),
        ):
            df = find_offtargets(GUIDE, genome, binary_path=binary)

        # on-target (0 mm) should be removed
        assert len(df) == 1
        assert (df["num_mismatches"] > 0).all()

    def test_ontarget_kept_when_disabled(self, tmp_path: Path) -> None:
        genome = tmp_path / "hg38.fa"
        genome.touch()
        binary = tmp_path / "cas-offinder"
        binary.touch()

        with patch(
            "src.offtarget_finder._run_binary",
            side_effect=self._mock_run_binary([_ONTARGET_LINE, _OT_LINE]),
        ):
            df = find_offtargets(
                GUIDE, genome, binary_path=binary, remove_ontarget=False
            )

        assert len(df) == 2

    def test_guide_normalized_23nt_accepted(self, tmp_path: Path) -> None:
        """find_offtargets should strip PAM from 23-nt input."""
        genome = tmp_path / "hg38.fa"
        genome.touch()
        binary = tmp_path / "cas-offinder"
        binary.touch()

        with patch(
            "src.offtarget_finder._run_binary",
            side_effect=self._mock_run_binary([_OT_LINE]),
        ):
            df = find_offtargets(
                GUIDE + "NGG",  # 23-nt with PAM
                genome,
                binary_path=binary,
            )

        assert df.loc[0, "guide"] == GUIDE

    def test_empty_output_returns_empty_df(self, tmp_path: Path) -> None:
        genome = tmp_path / "hg38.fa"
        genome.touch()
        binary = tmp_path / "cas-offinder"
        binary.touch()

        with patch(
            "src.offtarget_finder._run_binary",
            side_effect=self._mock_run_binary([]),
        ):
            df = find_offtargets(GUIDE, genome, binary_path=binary)

        assert len(df) == 0
        assert set(df.columns) >= {"guide", "chrom", "start", "num_mismatches"}

    def test_binary_called_with_correct_mode(self, tmp_path: Path) -> None:
        genome = tmp_path / "hg38.fa"
        genome.touch()
        binary = tmp_path / "cas-offinder"
        binary.touch()

        captured_mode: list[str] = []

        def _capture(input_path, output_path, binary_path, mode):
            captured_mode.append(mode)
            output_path.write_text("")

        with patch("src.offtarget_finder._run_binary", side_effect=_capture):
            find_offtargets(GUIDE, genome, binary_path=binary, mode="C")

        assert captured_mode == ["C"]

    def test_casoffinder_error_propagates(self, tmp_path: Path) -> None:
        genome = tmp_path / "hg38.fa"
        genome.touch()
        binary = tmp_path / "cas-offinder"
        binary.touch()

        def _failing_run(input_path, output_path, binary_path, mode):
            raise subprocess.CalledProcessError(1, ["cas-offinder"])

        with patch("src.offtarget_finder._run_binary", side_effect=_failing_run):
            with pytest.raises(subprocess.CalledProcessError):
                find_offtargets(GUIDE, genome, binary_path=binary)
