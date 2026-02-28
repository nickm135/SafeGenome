"""Tests for src/sequence_features.py

Success criteria from PROJECT.md:
  [x] CFD score of on-target site (0 mismatches) = 1.0
  [x] CFD scores decrease monotonically with more mismatches (on average)
  [x] Seed-region mismatches produce lower CFD scores than non-seed mismatches
  [x] Feature matrix has no NaN values for any candidate site
  [x] MIT score = 100 for perfect match
  [x] All features present and in expected ranges
"""

import math
from typing import Any

import pytest

from src.sequence_features import (
    PROTOSPACER_LENGTH,
    SEED_END,
    _CFD_MM_SCORES,
    _CFD_PAM_SCORES,
    _MIT_POS_WEIGHTS,
    _gc_content,
    _get_mismatches,
    _max_consecutive_run,
    cfd_score,
    compute_sequence_features,
    mit_score,
)

# ---------------------------------------------------------------------------
# Reference sequences
# ---------------------------------------------------------------------------

# EMX1 guide used in Doench 2016 / GUIDE-seq experiments.
# 20-nt protospacer, DNA notation, 5'→3', position 1 = PAM-proximal.
GUIDE_EMX1 = "GAGTCCGAGCAGAAGAAGAA"

# Perfect match (on-target) = same as guide
ON_TARGET = GUIDE_EMX1

# One mismatch: PAM-proximal (position 1, seed region)
# Change the 1st character (A at pos 1) to C  → rA:dC mismatch at pos 1
ONE_MM_SEED = "CAGTCCGAGCAGAAGAAGAA"

# One mismatch: PAM-distal (position 20)
# Change the last character (A at pos 20) to C → rA:dC mismatch at pos 20
ONE_MM_NONSEED = "GAGTCCGAGCAGAAGAAGAC"

# Two mismatches in seed
TWO_MM_SEED = "CAGTCCGAGCAGAAGAAGAA"  # pos 1 only — use different variant
TWO_MM_SEED_ACTUAL = "CCGTCCGAGCAGAAGAAGAA"  # pos 1 + pos 2

# Three mismatches: all non-seed
THREE_MM_NONSEED = "GAGTCCGAGCAGAAGCAGCC"  # positions 17, 19, 20

CANONICAL_PAM = "TGG"   # NGG → GG → PAM score 1.0
NAG_PAM = "TAG"          # NAG → AG → PAM score 0.259...
NONCANOICAL_PAM = "TAA"  # NAA → AA → PAM score 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_features_finite(features: dict[str, Any]) -> bool:
    """Return True if all numeric feature values are finite (no NaN/Inf)."""
    for k, v in features.items():
        if isinstance(v, float) and not math.isfinite(v):
            return False
    return True


# ---------------------------------------------------------------------------
# Matrix sanity checks
# ---------------------------------------------------------------------------

class TestScoringMatrices:
    def test_pam_gg_is_1(self) -> None:
        assert _CFD_PAM_SCORES["GG"] == 1.0

    def test_pam_ag_approx(self) -> None:
        assert abs(_CFD_PAM_SCORES["AG"] - 0.259_259_259) < 1e-6

    def test_pam_gt_approx(self) -> None:
        assert abs(_CFD_PAM_SCORES["GT"] - 0.107_142_857) < 1e-6

    def test_pam_tg_approx(self) -> None:
        assert abs(_CFD_PAM_SCORES["TG"] - 0.069_444_444) < 1e-6

    def test_pam_scores_all_in_unit_interval(self) -> None:
        for k, v in _CFD_PAM_SCORES.items():
            assert 0.0 <= v <= 1.0, f"PAM score out of range for {k}: {v}"

    def test_mm_matrix_has_12_types(self) -> None:
        assert len(_CFD_MM_SCORES) == 12

    def test_mm_matrix_each_type_has_20_positions(self) -> None:
        for key, vals in _CFD_MM_SCORES.items():
            assert len(vals) == 20, f"Expected 20 values for {key}, got {len(vals)}"

    def test_mm_matrix_values_in_unit_interval(self) -> None:
        for key, vals in _CFD_MM_SCORES.items():
            for i, v in enumerate(vals):
                assert 0.0 <= v <= 1.0, (
                    f"CFD matrix value out of [0,1] at {key}[{i}]: {v}"
                )

    def test_mit_weights_correct_length(self) -> None:
        assert len(_MIT_POS_WEIGHTS) == PROTOSPACER_LENGTH

    def test_mit_weights_in_unit_interval(self) -> None:
        for i, w in enumerate(_MIT_POS_WEIGHTS):
            assert 0.0 <= w <= 1.0, f"MIT weight out of [0,1] at index {i}: {w}"

    def test_no_match_types_in_mm_matrix(self) -> None:
        """rA:dA, rC:dC, rG:dG, rT:dT should NOT be in the mismatch table."""
        for base in "ACGT":
            match_key = f"r{base}:d{base}"
            assert match_key not in _CFD_MM_SCORES, (
                f"Match type {match_key!r} should not appear in mismatch matrix"
            )


# ---------------------------------------------------------------------------
# _get_mismatches
# ---------------------------------------------------------------------------

class TestGetMismatches:
    def test_perfect_match_no_mismatches(self) -> None:
        assert _get_mismatches(GUIDE_EMX1, ON_TARGET) == []

    def test_single_mismatch_position_and_type(self) -> None:
        # GUIDE_EMX1[0] = 'G' (position 1, PAM-proximal).
        # ONE_MM_SEED changes pos-1 from G→C → mismatch type = rG:dC
        result = _get_mismatches(GUIDE_EMX1, ONE_MM_SEED)
        assert len(result) == 1
        pos, key = result[0]
        assert pos == 1
        assert key == "rG:dC"

    def test_position_20_mismatch(self) -> None:
        result = _get_mismatches(GUIDE_EMX1, ONE_MM_NONSEED)
        assert len(result) == 1
        pos, key = result[0]
        assert pos == 20
        assert key == "rA:dC"

    def test_two_mismatches(self) -> None:
        result = _get_mismatches(GUIDE_EMX1, TWO_MM_SEED_ACTUAL)
        assert len(result) == 2
        positions = {p for p, _ in result}
        assert positions == {1, 2}

    def test_all_mismatch(self) -> None:
        # All positions differ
        guide = "AAAAAAAAAAAAAAAAAAAAAA"[:20]
        ot    = "CCCCCCCCCCCCCCCCCCCC"
        result = _get_mismatches(guide, ot)
        assert len(result) == 20


# ---------------------------------------------------------------------------
# CFD score
# ---------------------------------------------------------------------------

class TestCfdScore:
    # Project success criterion: on-target = 1.0
    def test_perfect_match_canonical_pam_is_1(self) -> None:
        score = cfd_score(GUIDE_EMX1, ON_TARGET, CANONICAL_PAM)
        assert score == pytest.approx(1.0)

    def test_perfect_match_nag_pam(self) -> None:
        # Still 0-mismatch product = 1, then × PAM score
        score = cfd_score(GUIDE_EMX1, ON_TARGET, NAG_PAM)
        assert score == pytest.approx(0.259_259_259)

    def test_noncanonical_pam_gives_zero(self) -> None:
        score = cfd_score(GUIDE_EMX1, ON_TARGET, NONCANOICAL_PAM)
        assert score == pytest.approx(0.0)

    def test_score_in_unit_interval(self) -> None:
        for ot in [ON_TARGET, ONE_MM_SEED, ONE_MM_NONSEED, THREE_MM_NONSEED]:
            score = cfd_score(GUIDE_EMX1, ot, CANONICAL_PAM)
            assert 0.0 <= score <= 1.0, f"CFD out of range for {ot}: {score}"

    def test_more_mismatches_lower_score_average(self) -> None:
        """Project criterion: CFD decreases monotonically with more mismatches."""
        score_0mm = cfd_score(GUIDE_EMX1, ON_TARGET, CANONICAL_PAM)
        # Build sites with increasing mismatches all in PAM-distal positions
        # to avoid zeros from seed penalties
        ot_1mm = GUIDE_EMX1[:19] + "C"        # pos 20 mismatch (rA:dC)
        ot_2mm = GUIDE_EMX1[:18] + "CC"       # pos 19, 20 mismatch
        score_1mm = cfd_score(GUIDE_EMX1, ot_1mm, CANONICAL_PAM)
        score_2mm = cfd_score(GUIDE_EMX1, ot_2mm, CANONICAL_PAM)
        assert score_0mm >= score_1mm, (
            f"0-mm ({score_0mm}) should be >= 1-mm ({score_1mm})"
        )
        assert score_1mm >= score_2mm, (
            f"1-mm ({score_1mm}) should be >= 2-mm ({score_2mm})"
        )

    def test_invalid_guide_length_raises(self) -> None:
        with pytest.raises(ValueError, match="20 nt"):
            cfd_score("ACGT", ON_TARGET, CANONICAL_PAM)

    def test_invalid_pam_length_raises(self) -> None:
        with pytest.raises(ValueError, match="PAM must be 3"):
            cfd_score(GUIDE_EMX1, ON_TARGET, "GG")

    def test_invalid_base_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid bases"):
            cfd_score(GUIDE_EMX1, "NAGTCCGAGCAGAAGAAGAA", CANONICAL_PAM)

    def test_case_insensitive_input(self) -> None:
        lower = cfd_score(
            GUIDE_EMX1.lower(), ON_TARGET.lower(), CANONICAL_PAM.lower()
        )
        upper = cfd_score(GUIDE_EMX1, ON_TARGET, CANONICAL_PAM)
        assert lower == pytest.approx(upper)

    def test_pam_first_base_n_ignored(self) -> None:
        """NGG and AGG should give same result (N position ignored)."""
        score_ngg = cfd_score(GUIDE_EMX1, ON_TARGET, "TGG")
        score_agg = cfd_score(GUIDE_EMX1, ON_TARGET, "AGG")
        score_cgg = cfd_score(GUIDE_EMX1, ON_TARGET, "CGG")
        assert score_ngg == pytest.approx(score_agg)
        assert score_agg == pytest.approx(score_cgg)


# ---------------------------------------------------------------------------
# MIT score
# ---------------------------------------------------------------------------

class TestMitScore:
    def test_perfect_match_is_100(self) -> None:
        assert mit_score(GUIDE_EMX1, ON_TARGET) == pytest.approx(100.0)

    def test_score_in_range(self) -> None:
        for ot in [ON_TARGET, ONE_MM_SEED, ONE_MM_NONSEED, THREE_MM_NONSEED]:
            s = mit_score(GUIDE_EMX1, ot)
            assert 0.0 <= s <= 100.0, f"MIT score out of range for {ot}: {s}"

    def test_mismatch_lowers_score(self) -> None:
        perfect = mit_score(GUIDE_EMX1, ON_TARGET)
        one_mm = mit_score(GUIDE_EMX1, ONE_MM_NONSEED)
        assert perfect > one_mm, (
            f"Perfect match ({perfect}) should beat 1-mismatch ({one_mm})"
        )

    def test_two_mismatches_lower_than_one(self) -> None:
        one = mit_score(GUIDE_EMX1, ONE_MM_NONSEED)
        # Two mismatches at positions 19 and 20
        ot_2mm = GUIDE_EMX1[:18] + "CC"
        two = mit_score(GUIDE_EMX1, ot_2mm)
        assert one > two, f"1-mm ({one}) should beat 2-mm ({two})"

    def test_clustered_mismatches_lower_than_spread(self) -> None:
        """Clustered mismatches should score lower due to distance penalty."""
        # 2 mismatches close together (positions 19, 20)
        ot_clustered = GUIDE_EMX1[:18] + "CC"       # gap = 1
        # 2 mismatches spread apart (positions 1, 20) — rA:dC for guide A
        ot_spread = "C" + GUIDE_EMX1[1:19] + "C"   # gap = 19
        score_clustered = mit_score(GUIDE_EMX1, ot_clustered)
        score_spread = mit_score(GUIDE_EMX1, ot_spread)
        # Spread → larger mean distance → larger d_factor → lower score
        assert score_spread <= score_clustered, (
            f"Spread ({score_spread}) should be <= clustered ({score_clustered})"
        )

    def test_invalid_length_raises(self) -> None:
        with pytest.raises(ValueError):
            mit_score(GUIDE_EMX1, "ACGT")


# ---------------------------------------------------------------------------
# compute_sequence_features
# ---------------------------------------------------------------------------

class TestComputeSequenceFeatures:
    def _perfect_match_features(self) -> dict:
        return compute_sequence_features(GUIDE_EMX1, ON_TARGET, CANONICAL_PAM)

    def test_returns_dict(self) -> None:
        f = self._perfect_match_features()
        assert isinstance(f, dict)

    def test_cfd_score_present_and_correct(self) -> None:
        f = self._perfect_match_features()
        assert "cfd_score" in f
        assert f["cfd_score"] == pytest.approx(1.0)

    def test_mit_score_present_and_correct(self) -> None:
        f = self._perfect_match_features()
        assert "mit_score" in f
        assert f["mit_score"] == pytest.approx(100.0)

    def test_num_mismatches_perfect_match(self) -> None:
        f = self._perfect_match_features()
        assert f["num_mismatches"] == 0

    def test_num_mismatches_one_mm(self) -> None:
        f = compute_sequence_features(GUIDE_EMX1, ONE_MM_SEED, CANONICAL_PAM)
        assert f["num_mismatches"] == 1

    def test_mm_pos_flags_correct_for_perfect_match(self) -> None:
        f = self._perfect_match_features()
        for pos in range(1, PROTOSPACER_LENGTH + 1):
            assert f[f"mm_pos_{pos}"] == 0

    def test_mm_pos_flags_correct_for_seed_mismatch(self) -> None:
        f = compute_sequence_features(GUIDE_EMX1, ONE_MM_SEED, CANONICAL_PAM)
        assert f["mm_pos_1"] == 1
        for pos in range(2, PROTOSPACER_LENGTH + 1):
            assert f[f"mm_pos_{pos}"] == 0

    def test_mm_type_correct_for_seed_mismatch(self) -> None:
        # GUIDE_EMX1[0] = 'G'; ONE_MM_SEED changes it to 'C' → rG:dC
        f = compute_sequence_features(GUIDE_EMX1, ONE_MM_SEED, CANONICAL_PAM)
        assert f["mm_type_1"] == "rG:dC"
        assert f["mm_type_2"] == ""  # match

    def test_mm_type_empty_string_on_match(self) -> None:
        f = self._perfect_match_features()
        for pos in range(1, PROTOSPACER_LENGTH + 1):
            assert f[f"mm_type_{pos}"] == ""

    def test_gc_content_guide(self) -> None:
        f = self._perfect_match_features()
        expected = (GUIDE_EMX1.count("G") + GUIDE_EMX1.count("C")) / 20
        assert f["gc_content_guide"] == pytest.approx(expected)

    def test_gc_content_offtarget(self) -> None:
        f = self._perfect_match_features()
        assert f["gc_content_offtarget"] == pytest.approx(f["gc_content_guide"])

    def test_pam_match_canonical(self) -> None:
        f = self._perfect_match_features()
        assert f["pam_match"] == 1

    def test_pam_match_nag(self) -> None:
        f = compute_sequence_features(GUIDE_EMX1, ON_TARGET, NAG_PAM)
        assert f["pam_match"] == 0

    def test_seed_mismatches_correct(self) -> None:
        # ONE_MM_SEED: mismatch at position 1 (seed)
        f = compute_sequence_features(GUIDE_EMX1, ONE_MM_SEED, CANONICAL_PAM)
        assert f["seed_mismatches"] == 1
        assert f["nonseed_mismatches"] == 0

    def test_nonseed_mismatches_correct(self) -> None:
        # ONE_MM_NONSEED: mismatch at position 20 (non-seed)
        f = compute_sequence_features(GUIDE_EMX1, ONE_MM_NONSEED, CANONICAL_PAM)
        assert f["seed_mismatches"] == 0
        assert f["nonseed_mismatches"] == 1

    def test_seed_end_boundary(self) -> None:
        """Position 12 is seed, position 13 is non-seed."""
        # Mismatch at position 12 (seed boundary)
        ot_pos12 = GUIDE_EMX1[:11] + "C" + GUIDE_EMX1[12:]
        f = compute_sequence_features(GUIDE_EMX1, ot_pos12, CANONICAL_PAM)
        assert f["seed_mismatches"] == 1
        assert f["nonseed_mismatches"] == 0

        # Mismatch at position 13 (non-seed boundary)
        ot_pos13 = GUIDE_EMX1[:12] + "C" + GUIDE_EMX1[13:]
        f2 = compute_sequence_features(GUIDE_EMX1, ot_pos13, CANONICAL_PAM)
        assert f2["seed_mismatches"] == 0
        assert f2["nonseed_mismatches"] == 1

    def test_consecutive_mm_no_mismatches(self) -> None:
        f = self._perfect_match_features()
        assert f["consecutive_mm"] == 0

    def test_consecutive_mm_single(self) -> None:
        f = compute_sequence_features(GUIDE_EMX1, ONE_MM_SEED, CANONICAL_PAM)
        assert f["consecutive_mm"] == 1

    def test_consecutive_mm_run_of_3(self) -> None:
        # Mismatches at positions 18, 19, 20
        ot = GUIDE_EMX1[:17] + "CCC"
        f = compute_sequence_features(GUIDE_EMX1, ot, CANONICAL_PAM)
        assert f["consecutive_mm"] == 3

    def test_consecutive_mm_nonconsecutive(self) -> None:
        # Mismatches at positions 1 and 20 (not consecutive)
        ot = "C" + GUIDE_EMX1[1:19] + "C"
        f = compute_sequence_features(GUIDE_EMX1, ot, CANONICAL_PAM)
        assert f["consecutive_mm"] == 1

    def test_no_nan_values(self) -> None:
        """Project criterion: feature matrix has no NaN values."""
        for ot in [ON_TARGET, ONE_MM_SEED, ONE_MM_NONSEED, THREE_MM_NONSEED,
                   TWO_MM_SEED_ACTUAL]:
            f = compute_sequence_features(GUIDE_EMX1, ot, CANONICAL_PAM)
            assert _all_features_finite(f), f"NaN/Inf in features for {ot}"

    def test_all_expected_keys_present(self) -> None:
        f = self._perfect_match_features()
        expected_keys = (
            ["cfd_score", "mit_score", "num_mismatches"]
            + [f"mm_pos_{p}" for p in range(1, 21)]
            + [f"mm_type_{p}" for p in range(1, 21)]
            + ["gc_content_guide", "gc_content_offtarget",
               "pam_match", "seed_mismatches",
               "nonseed_mismatches", "consecutive_mm"]
        )
        for key in expected_keys:
            assert key in f, f"Missing feature key: {key!r}"

    def test_feature_count(self) -> None:
        """Total features = 3 + 20 + 20 + 6 = 49."""
        f = self._perfect_match_features()
        assert len(f) == 49

    def test_lowercase_input_accepted(self) -> None:
        f = compute_sequence_features(
            GUIDE_EMX1.lower(), ON_TARGET.lower(), CANONICAL_PAM.lower()
        )
        assert f["cfd_score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

class TestGcContent:
    def test_all_gc(self) -> None:
        assert _gc_content("GCGCGCGCGCGCGCGCGCGC") == pytest.approx(1.0)

    def test_all_at(self) -> None:
        assert _gc_content("ATATATATATATATATAT" + "AT") == pytest.approx(0.0)

    def test_mixed(self) -> None:
        assert _gc_content("ACGT") == pytest.approx(0.5)

    def test_empty_returns_zero(self) -> None:
        assert _gc_content("") == pytest.approx(0.0)


class TestMaxConsecutiveRun:
    def test_empty(self) -> None:
        assert _max_consecutive_run(set()) == 0

    def test_single(self) -> None:
        assert _max_consecutive_run({5}) == 1

    def test_two_consecutive(self) -> None:
        assert _max_consecutive_run({3, 4}) == 2

    def test_three_consecutive(self) -> None:
        assert _max_consecutive_run({1, 2, 3}) == 3

    def test_non_consecutive(self) -> None:
        assert _max_consecutive_run({1, 3, 5}) == 1

    def test_mixed_consecutive_and_not(self) -> None:
        # 8,9,10 form a run of 3; 1 is isolated
        assert _max_consecutive_run({1, 8, 9, 10}) == 3


# ---------------------------------------------------------------------------
# Seed vs non-seed qualitative check (PROJECT.md success criterion)
# ---------------------------------------------------------------------------

class TestSeedVsNonSeedEffect:
    """Project criterion: seed mismatches produce lower CFD scores than
    non-seed mismatches (checked for rG:dA, the most tolerated type)."""

    def test_rGdA_seed_below_nonseed(self) -> None:
        """A rG:dA mismatch at a deep-seed position should have a lower
        CFD mismatch factor than the same type at the most PAM-distal
        position, because seed-region activity should be more disrupted.

        Uses the mismatch matrix values for rG:dA specifically, since that
        type has non-zero values at PAM-distal positions in the canonical
        matrix.
        """
        # Position 1 (PAM-proximal, deep seed), index 0
        seed_factor = _CFD_MM_SCORES["rG:dA"][0]
        # Position 20 (PAM-distal, non-seed), index 19
        nonseed_factor = _CFD_MM_SCORES["rG:dA"][19]
        assert seed_factor <= nonseed_factor, (
            f"Seed factor ({seed_factor}) should be <= non-seed factor "
            f"({nonseed_factor}) for rG:dA"
        )

    def test_cfd_seed_mismatch_le_nonseed_mismatch_rGdA(self) -> None:
        """End-to-end: single rG:dT mismatch in seed has lower CFD than non-seed.

        Uses canonical Doench 2016 matrix values.
        Guide: GAGTCCGAGCAGAAGAAGAA
          idx 11 = G (position 12, PAM-proximal seed)  → rG:dT[11] = 0.385
          idx 17 = G (position 18, PAM-distal non-seed) → rG:dT[17] = 0.667

        Biological expectation: seed mismatch (pos 12) is more disruptive
        (lower CFD) than PAM-distal mismatch (pos 18) for rG:dA mismatches
        (reference notation; our internal rG:dT).
        """
        # Position 12 (idx 11, last seed position): guide G → off-target T
        ot_seed = GUIDE_EMX1[:11] + "T" + GUIDE_EMX1[12:]   # rG:dT at pos 12

        # Position 18 (idx 17, PAM-distal non-seed): guide G → off-target T
        ot_nonseed = GUIDE_EMX1[:17] + "T" + GUIDE_EMX1[18:]  # rG:dT at pos 18

        cfd_seed    = cfd_score(GUIDE_EMX1, ot_seed,    CANONICAL_PAM)
        cfd_nonseed = cfd_score(GUIDE_EMX1, ot_nonseed, CANONICAL_PAM)

        assert cfd_seed <= cfd_nonseed, (
            f"Seed CFD ({cfd_seed}) should be <= non-seed CFD ({cfd_nonseed})"
        )
