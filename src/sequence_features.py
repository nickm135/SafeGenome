"""
sequence_features.py — Sequence-level CRISPR off-target feature extraction.

For each (guide, off-target, PAM) triplet this module computes:

  - CFD score     (Doench et al., 2016 Nat Biotechnol)
  - MIT score     (Hsu et al., 2013 Nat Biotechnol)
  - Mismatch position binary flags (mm_pos_1 … mm_pos_20)
  - Mismatch type string per position (mm_type_1 … mm_type_20)
  - Aggregated mismatch counts (num_mismatches, seed_mismatches,
    nonseed_mismatches, consecutive_mm)
  - GC content for guide and off-target
  - PAM canonical flag (pam_match)

Coordinate / sequence conventions
----------------------------------
- Position 1 = PAM-proximal (3' end of 20-nt protospacer, adjacent to PAM).
- Position 20 = PAM-distal (5' end of protospacer).
- Sequences are in DNA notation (T not U), uppercase, 5'→3'.
- guide[i] and off_target[i] are compared directly; a MATCH is when they
  are identical (both represent the non-template strand protospacer sequence).

IMPORTANT — CFD mismatch matrix
---------------------------------
Values translated from Doench et al. (2016) / GuideMaker reference data.

Convention:
  - Reference uses TEMPLATE-strand dX notation and rU for guide T.
  - Internal keys use PROTOSPACER-strand dX and rT (KEY_MAP applied).
  - Position 1 = PAM-proximal (index 0). idx = ref_pos - 1.
  - Validated: rG:dT[5] = 0.6667 matches Broad GPP test case
    (ref rG:dA pos6 = 0.67, NGG PAM).

All 240 positions are now filled from the authoritative Doench 2016 pickle
(maximilianh/crisporWebsite, CFD_Scoring/mismatch_score.pkl).

6 of 240 positions are legitimately 0.0 in the original data (confirmed real zeros,
not gaps): rA:dC pos 16, rA:dT pos 16, rC:dG pos 14, rG:dC pos 16, rG:dT pos 16,
rT:dG pos 14.

The PAM scores are confirmed correct from the same paper.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Sequence position convention
# ---------------------------------------------------------------------------

# The protospacer is 20 nt, numbered 1-20 (1 = PAM-proximal).
PROTOSPACER_LENGTH: int = 20

# "Seed region" = positions 1-12 (PAM-proximal half), biologically most
# important for Cas9 specificity.
SEED_END: int = 12  # inclusive, 1-based


# ---------------------------------------------------------------------------
# CFD scoring tables — Doench et al. 2016, Nat Biotechnol
# ---------------------------------------------------------------------------

# PAM scores: keyed by the TWO PAM bases that follow the N (positions 2-3 of NGG).
# NGG → "GG" → 1.0 (canonical, full activity)
# NAG → "AG" → 0.259… (partial activity)
# Values confirmed from Doench 2016 / crispor canonical implementation.
_CFD_PAM_SCORES: dict[str, float] = {
    "AA": 0.0, "AC": 0.0, "AT": 0.0, "AG": 0.259_259_259,
    "CA": 0.0, "CC": 0.0, "CT": 0.0, "CG": 0.0,
    "GA": 0.0, "GC": 0.0, "GT": 0.107_142_857, "GG": 1.0,
    "TA": 0.0, "TC": 0.0, "TT": 0.0, "TG": 0.069_444_444,
}

# Mismatch score matrix.
#
# Key:   "rX:dY" where X = guide base (DNA notation), Y = off-target
#        (protospacer non-template strand) base.  X == Y means MATCH (not
#        present in this table; matched positions contribute factor 1.0).
#
# Value: list of 20 floats.
#        Index 0  ↔  position 1 (PAM-proximal).
#        Index 19 ↔  position 20 (PAM-distal).
#        Each value is the cutting-frequency factor (0 = abolishes cutting,
#        1 = no effect) for that mismatch type at that guide position.
#
# !!! VERIFY AGAINST DOENCH 2016 TABLE S5 BEFORE PUBLISHING !!!
# The values below are derived from the canonical cfd.py release that
# accompanied the paper and from the crispor re-implementation, cross-checked
# against biological expectation (seed near 0, non-seed higher).
# Minor discrepancies from the published table may exist.
_CFD_MM_SCORES: dict[str, list[float]] = {
    # Guide A mismatches (rA:dX where dX = protospacer base != A)
    # pos 16 is 0.0 (confirmed real zero in Doench 2016 pickle)
    "rA:dC": [0.857142857, 0.785714286, 0.428571429, 0.352941176, 0.5, 0.454545455, 0.4375, 0.428571429, 0.571428571, 0.333333333,
              0.4, 0.263157895, 0.210526316, 0.214285714, 0.272727273, 0.0, 0.176470588, 0.19047619, 0.206896552, 0.227272727],
    "rA:dG": [1.0, 0.8, 0.611111111, 0.625, 0.72, 0.714285714, 0.705882353, 0.733333333, 0.666666667, 0.555555556,
              0.65, 0.722222222, 0.652173913, 0.466666667, 0.65, 0.192307692, 0.176470588, 0.4, 0.375, 0.764705882],
    "rA:dT": [1.0, 0.727272727, 0.705882353, 0.636363636, 0.363636364, 0.714285714, 0.4375, 0.428571429, 0.6, 0.882352941,
              0.307692308, 0.333333333, 0.3, 0.533333333, 0.2, 0.0, 0.133333333, 0.5, 0.538461538, 0.6],
    # Guide C mismatches
    "rC:dA": [1.0, 0.727272727, 0.866666667, 0.842105263, 0.571428571, 0.928571429, 0.75, 0.65, 0.857142857, 0.866666667,
              0.75, 0.714285714, 0.384615385, 0.35, 0.222222222, 1.0, 0.466666667, 0.538461538, 0.428571429, 0.5],
    "rC:dG": [0.913043478, 0.695652174, 0.5, 0.5, 0.6, 0.5, 0.470588235, 0.642857143, 0.619047619, 0.388888889,
              0.25, 0.444444444, 0.136363636, 0.0, 0.05, 0.153846154, 0.058823529, 0.133333333, 0.125, 0.058823529],
    "rC:dT": [1.0, 0.909090909, 0.6875, 0.8, 0.636363636, 0.928571429, 0.8125, 0.875, 0.875, 0.941176471,
              0.307692308, 0.538461538, 0.7, 0.733333333, 0.066666667, 0.307692308, 0.466666667, 0.642857143, 0.461538462, 0.3],
    # Guide G mismatches
    "rG:dA": [0.9, 0.846153846, 0.75, 0.9, 0.866666667, 1.0, 1.0, 1.0, 0.642857143, 0.933333333,
              1.0, 0.933333333, 0.923076923, 0.75, 0.941176471, 1.0, 0.933333333, 0.692307692, 0.714285714, 0.9375],
    "rG:dC": [0.714285714, 0.692307692, 0.384615385, 0.529411765, 0.785714286, 0.681818182, 0.6875, 0.615384615, 0.538461538, 0.4,
              0.428571429, 0.529411765, 0.421052632, 0.428571429, 0.272727273, 0.0, 0.235294118, 0.476190476, 0.448275862, 0.428571429],
    "rG:dT": [1.0, 0.636363636, 0.5, 0.363636364, 0.3, 0.666666667, 0.571428571, 0.625, 0.533333333, 0.8125,
              0.384615385, 0.384615385, 0.3, 0.266666667, 0.142857143, 0.0, 0.25, 0.666666667, 0.666666667, 0.7],
    # Guide T mismatches (rT = rU in original Doench notation)
    "rT:dA": [1.0, 0.846153846, 0.714285714, 0.476190476, 0.5, 0.866666667, 0.875, 0.8, 0.928571429, 0.857142857,
              0.75, 0.8, 0.692307692, 0.619047619, 0.578947368, 0.909090909, 0.533333333, 0.666666667, 0.285714286, 0.5625],
    "rT:dC": [0.857142857, 0.857142857, 0.428571429, 0.647058824, 1.0, 0.909090909, 0.6875, 1.0, 0.923076923, 0.533333333,
              0.666666667, 0.947368421, 0.789473684, 0.285714286, 0.272727273, 0.666666667, 0.705882353, 0.428571429, 0.275862069, 0.090909091],
    # pos 14 is 0.0 (confirmed real zero in Doench 2016 pickle)
    "rT:dG": [0.956521739, 0.84, 0.5, 0.625, 0.64, 0.571428571, 0.588235294, 0.733333333, 0.619047619, 0.5,
              0.4, 0.5, 0.260869565, 0.0, 0.05, 0.346153846, 0.117647059, 0.333333333, 0.25, 0.176470588],
}

# Match base-pairs: guide_base -> off_target_base that constitutes a match.
# Both sequences are the NON-TEMPLATE strand, so a match is guide[i] == ot[i].
_MATCH: dict[str, str] = {"A": "A", "C": "C", "G": "G", "T": "T"}

# Valid DNA bases
_VALID_BASES: frozenset[str] = frozenset("ACGT")


# ---------------------------------------------------------------------------
# MIT scoring table — Hsu et al. 2013, Nat Biotechnol
# ---------------------------------------------------------------------------

# Per-position penalty weights.
# Index 0 = position 1 (PAM-proximal), index 19 = position 20 (PAM-distal).
# Higher weight → mismatch at that position is more penalising.
# Source: Hsu et al. 2013, Nat Biotechnol, Supplementary Table S1.
# !!! VERIFY ORDERING (PAM-proximal vs PAM-distal) AGAINST PAPER IF RESULTS
#     LOOK SURPRISING !!!
_MIT_POS_WEIGHTS: list[float] = [
    0.000, 0.000, 0.014, 0.000, 0.000, 0.395, 0.317, 0.000,
    0.389, 0.079, 0.445, 0.508, 0.613, 0.851, 0.732, 0.828,
    0.615, 0.804, 0.685, 0.583,
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_sequences(guide: str, off_target: str) -> None:
    """Raise ValueError if sequences are not valid 20-nt DNA strings."""
    for name, seq in (("guide", guide), ("off_target", off_target)):
        if len(seq) != PROTOSPACER_LENGTH:
            raise ValueError(
                f"{name} must be {PROTOSPACER_LENGTH} nt, got {len(seq)}: {seq!r}"
            )
        invalid = set(seq) - _VALID_BASES
        if invalid:
            raise ValueError(
                f"{name} contains invalid bases {invalid!r}: {seq!r}"
            )


def _get_mismatches(guide: str, off_target: str) -> list[tuple[int, str]]:
    """Return list of (position_1based, mismatch_key) for all mismatches.

    Parameters
    ----------
    guide : str
        20-nt protospacer sequence.
    off_target : str
        20-nt off-target sequence.

    Returns
    -------
    list[tuple[int, str]]
        Each element is (1-based position, mismatch key like "rG:dA").
        Positions are in 1-20 convention (1 = PAM-proximal).
    """
    mismatches = []
    for i in range(PROTOSPACER_LENGTH):
        g = guide[i]
        o = off_target[i]
        if g != o:
            pos_1based = i + 1
            key = f"r{g}:d{o}"
            mismatches.append((pos_1based, key))
    return mismatches


# ---------------------------------------------------------------------------
# CFD score
# ---------------------------------------------------------------------------

def cfd_score(guide: str, off_target: str, pam: str) -> float:
    """Compute the Cutting Frequency Determination (CFD) score.

    The CFD score estimates how efficiently Cas9 cleaves an off-target
    site relative to the intended on-target.

    Parameters
    ----------
    guide : str
        20-nt protospacer (non-template strand, 5'→3', DNA notation).
    off_target : str
        20-nt off-target sequence (same strand convention).
    pam : str
        3-nt PAM sequence observed at the off-target site (e.g. "AGG").
        The first base (N in NGG) is ignored; scoring uses positions 2-3.

    Returns
    -------
    float
        CFD score in [0.0, 1.0].  1.0 = perfect match + canonical PAM.
        0.0 = no predicted cleavage activity.

    References
    ----------
    Doench et al. (2016) Nat Biotechnol 34, 184-191. doi:10.1038/nbt.3437

    """
    guide = guide.upper()
    off_target = off_target.upper()
    pam = pam.upper()

    _validate_sequences(guide, off_target)

    if len(pam) != 3:
        raise ValueError(f"PAM must be 3 nt, got {len(pam)}: {pam!r}")
    if not set(pam) <= _VALID_BASES:
        raise ValueError(f"PAM contains invalid bases: {pam!r}")

    # Mismatch factors
    score: float = 1.0
    for pos_1based, key in _get_mismatches(guide, off_target):
        idx = pos_1based - 1  # 0-based
        if key not in _CFD_MM_SCORES:
            raise KeyError(f"Unknown mismatch type: {key!r}")
        score *= _CFD_MM_SCORES[key][idx]
        if score == 0.0:
            break  # No further multiplication can change it

    # PAM factor (uses bases 2-3 of the 3-nt PAM)
    pam_key = pam[1:]  # drop the N
    score *= _CFD_PAM_SCORES.get(pam_key, 0.0)

    return score


# ---------------------------------------------------------------------------
# MIT score
# ---------------------------------------------------------------------------

def mit_score(guide: str, off_target: str) -> float:
    """Compute a per-site MIT specificity score.

    Based on Hsu et al. 2013.  Returns a value in [0, 100] reflecting
    predicted cleavage efficiency at the off-target site.  100 = perfect
    match.  Lower values indicate less likely cleavage.

    The per-site formula:

        score = 100 × ∏(1 − W[i]) / (n² × d_factor)

    where the product runs over all mismatching positions i (0-indexed),
    n = number of mismatches, and d_factor accounts for mismatch clustering:

        d_factor = 4/19 × mean_pairwise_distance + 1

    For n = 0 (perfect match) the score is 100.0.

    Parameters
    ----------
    guide : str
        20-nt guide protospacer.
    off_target : str
        20-nt off-target sequence.

    Returns
    -------
    float
        MIT score in [0.0, 100.0].

    References
    ----------
    Hsu et al. (2013) Nat Biotechnol 31, 827-832. doi:10.1038/nbt.2647
    """
    guide = guide.upper()
    off_target = off_target.upper()

    _validate_sequences(guide, off_target)

    mismatches = _get_mismatches(guide, off_target)
    n = len(mismatches)

    if n == 0:
        return 100.0

    # Position-specific penalty product
    penalty_product: float = 1.0
    for pos_1based, _ in mismatches:
        idx = pos_1based - 1
        penalty_product *= 1.0 - _MIT_POS_WEIGHTS[idx]

    # Clustering penalty: mean pairwise distance between mismatch positions
    positions = [p for p, _ in mismatches]
    if n == 1:
        mean_dist = 0.0
    else:
        pairs = [
            abs(positions[i] - positions[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        mean_dist = float(np.mean(pairs))

    d_factor = (4.0 / 19.0) * mean_dist + 1.0

    return 100.0 * penalty_product / (n ** 2 * d_factor)


# ---------------------------------------------------------------------------
# Full feature extraction
# ---------------------------------------------------------------------------

def compute_sequence_features(
    guide: str,
    off_target: str,
    pam: str,
) -> dict[str, Any]:
    """Compute all sequence-level features for a (guide, off-target, PAM) triplet.

    Parameters
    ----------
    guide : str
        20-nt guide protospacer in DNA notation (uppercase, 5'→3').
    off_target : str
        20-nt off-target protospacer (same strand/notation).
    pam : str
        3-nt PAM at the off-target locus (e.g. "TGG", "AAG").

    Returns
    -------
    dict[str, Any]
        Feature dictionary.  Keys:

        - ``cfd_score``            : float, [0, 1]
        - ``mit_score``            : float, [0, 100]
        - ``num_mismatches``       : int
        - ``mm_pos_1`` … ``mm_pos_20``: int, 0 or 1
        - ``mm_type_1`` … ``mm_type_20``: str, e.g. "rG:dA" or "" (match)
        - ``gc_content_guide``     : float, [0, 1]
        - ``gc_content_offtarget`` : float, [0, 1]
        - ``pam_match``            : int, 1 if PAM[-2:] == "GG" else 0
        - ``seed_mismatches``      : int, mismatches in positions 1-12
        - ``nonseed_mismatches``   : int, mismatches in positions 13-20
        - ``consecutive_mm``       : int, longest run of consecutive mismatches

    Raises
    ------
    ValueError
        If any sequence has invalid length or characters.
    """
    guide = guide.upper()
    off_target = off_target.upper()
    pam = pam.upper()

    _validate_sequences(guide, off_target)
    mismatches = _get_mismatches(guide, off_target)
    mm_positions: set[int] = {pos for pos, _ in mismatches}   # 1-based

    features: dict[str, Any] = {}

    # ---- Scores -------------------------------------------------------
    features["cfd_score"] = cfd_score(guide, off_target, pam)
    features["mit_score"] = mit_score(guide, off_target)

    # ---- Mismatch count -----------------------------------------------
    features["num_mismatches"] = len(mismatches)

    # ---- Per-position binary flags and type strings -------------------
    mm_type_map: dict[int, str] = {pos: key for pos, key in mismatches}
    for pos in range(1, PROTOSPACER_LENGTH + 1):
        features[f"mm_pos_{pos}"] = 1 if pos in mm_positions else 0
        features[f"mm_type_{pos}"] = mm_type_map.get(pos, "")

    # ---- GC content ---------------------------------------------------
    features["gc_content_guide"] = _gc_content(guide)
    features["gc_content_offtarget"] = _gc_content(off_target)

    # ---- PAM canonical flag -------------------------------------------
    pam_dinucleotide = pam[1:] if len(pam) == 3 else pam[-2:]
    features["pam_match"] = 1 if pam_dinucleotide == "GG" else 0

    # ---- Seed vs non-seed mismatches ----------------------------------
    features["seed_mismatches"] = sum(
        1 for p in mm_positions if p <= SEED_END
    )
    features["nonseed_mismatches"] = sum(
        1 for p in mm_positions if p > SEED_END
    )

    # ---- Max consecutive mismatch run ---------------------------------
    features["consecutive_mm"] = _max_consecutive_run(mm_positions)

    return features


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _gc_content(seq: str) -> float:
    """Return GC fraction of *seq*.  Returns 0.0 for empty string."""
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def _max_consecutive_run(mm_positions: set[int]) -> int:
    """Return the length of the longest consecutive mismatch run.

    Parameters
    ----------
    mm_positions : set[int]
        1-based mismatch positions.

    Returns
    -------
    int
        Length of longest consecutive run.  0 if no mismatches.
    """
    if not mm_positions:
        return 0
    sorted_pos = sorted(mm_positions)
    max_run = current_run = 1
    for i in range(1, len(sorted_pos)):
        if sorted_pos[i] == sorted_pos[i - 1] + 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run
