"""Feature matrix assembly for the SafeGenome three-model CRISPR off-target comparison.

Combines sequence features, AlphaGenome chromatin predictions, and ENCODE
experimental chromatin signals with ground-truth labels into a single
training-ready DataFrame.

Three-model feature sets
------------------------
- **Model A** (sequence-only baseline): sequence features derived from guide
  and off-target sequences — CFD score, MIT score, mismatch positions/types,
  GC content, PAM, seed statistics.
- **Model B** (gold standard): Model A features + ENCODE experimental
  chromatin signals.  Trained on K562 + HEK293 only.
- **Model C** (our method): Model A features + AlphaGenome predicted
  chromatin signals.  Trained on K562 + HEK293 only.

Critical rules enforced here
-----------------------------
1. **On-target removal**: candidates with ``num_mismatches == 0`` are removed
   before any downstream step.
2. **Label assignment via match window**: positives are matched to candidates
   by (guide_id, chrom, |start_cand − start_pos| ≤ match_window_bp).  All
   unmatched candidates are labeled negative (y = 0).
3. **NaN imputation**: chromatin features may be NaN (missing cell-type
   tracks, missing BigWig files, chromosome boundaries).  Each NaN column is
   filled with its within-dataset median and a binary missingness indicator
   column (``{feature}_missing``) is added.  See ``impute_chromatin_features``.
4. **All data drops are logged** via ``utils.log_filter`` for the paper's
   Methods section.

Cross-validation and model training are handled in ``src/model.py``, not here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import src.alphagenome_client as _ag_mod
import src.encode_client as _enc_mod
from src.sequence_features import compute_sequence_features
from src.utils import log_filter, normalize_chrom

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Maps mismatch-type strings (CFD notation) to integer codes for XGBoost.
#: "" (match at a position) → 0.  Twelve canonical rX:dY mismatches → 1–12.
#: Any unrecognised value → -1 (logged as a warning).
MM_TYPE_TO_INT: dict[str, int] = {
    "":       0,
    "rA:dC":  1,  "rA:dG":  2,  "rA:dT":  3,
    "rC:dA":  4,  "rC:dG":  5,  "rC:dT":  6,
    "rG:dA":  7,  "rG:dC":  8,  "rG:dT":  9,
    "rT:dA": 10,  "rT:dC": 11,  "rT:dG": 12,
}

#: All sequence feature names (numeric-compatible, suitable for XGBoost).
#: mm_type columns are encoded as integers via MM_TYPE_TO_INT.
SEQUENCE_FEATURE_NAMES: list[str] = (
    ["cfd_score", "mit_score", "num_mismatches"]
    + [f"mm_pos_{i}" for i in range(1, 21)]
    + [f"mm_type_{i}" for i in range(1, 21)]
    + [
        "gc_content_guide", "gc_content_offtarget",
        "pam_match", "seed_mismatches", "nonseed_mismatches", "consecutive_mm",
    ]
)

#: AlphaGenome predicted chromatin feature names (from alphagenome_client).
AG_FEATURE_NAMES: list[str] = list(_ag_mod.FEATURE_NAMES)

#: ENCODE experimental chromatin feature names (from encode_client).
EXP_FEATURE_NAMES: list[str] = list(_enc_mod.FEATURE_NAMES)

#: Feature sets for each model.
MODEL_A_FEATURES: list[str] = list(SEQUENCE_FEATURE_NAMES)
MODEL_B_FEATURES: list[str] = list(SEQUENCE_FEATURE_NAMES) + list(EXP_FEATURE_NAMES)
MODEL_C_FEATURES: list[str] = list(SEQUENCE_FEATURE_NAMES) + list(AG_FEATURE_NAMES)

#: Output metadata column names (never used as model features).
METADATA_COLS: list[str] = [
    "guide_id", "chrom", "start", "strand",
    "guide_seq", "off_target_seq", "pam_seq",
    "cell_type", "dataset", "label",
]

#: Default position tolerance for matching candidates to experimental positives.
DEFAULT_MATCH_WINDOW_BP: int = 5

#: Default half-window for chromatin feature extraction (matches other modules).
DEFAULT_WINDOW_BP: int = 500

#: Suffix appended to imputed feature columns to flag missingness.
MISSING_INDICATOR_SUFFIX: str = "_missing"

#: Required columns in the candidates DataFrame.
_REQUIRED_CANDIDATE_COLS: tuple[str, ...] = (
    "guide_id", "guide_seq", "chrom", "start",
    "strand", "off_target_seq", "pam_seq", "num_mismatches",
)

#: Required columns in the positives DataFrame.
_REQUIRED_POSITIVE_COLS: tuple[str, ...] = ("guide_id", "chrom", "start")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, cols: tuple[str, ...], name: str) -> None:
    """Raise ValueError if any of *cols* are absent from *df*."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}. "
            f"Present columns: {list(df.columns)}"
        )


def get_model_features(
    model_type: str,
    include_missingness_indicators: bool = False,
) -> list[str]:
    """Return the feature column names for a given model type.

    Parameters
    ----------
    model_type : str
        One of ``"A"``, ``"B"``, ``"C"`` (case-insensitive).
    include_missingness_indicators : bool
        If ``True``, append ``{feature}_missing`` indicator columns for all
        chromatin features (relevant for Models B and C).

    Returns
    -------
    list[str]
        Ordered list of feature column names.

    Raises
    ------
    ValueError
        If *model_type* is not ``"A"``, ``"B"``, or ``"C"``.
    """
    mapping: dict[str, list[str]] = {
        "a": MODEL_A_FEATURES,
        "b": MODEL_B_FEATURES,
        "c": MODEL_C_FEATURES,
    }
    key = model_type.strip().lower()
    if key not in mapping:
        raise ValueError(
            f"model_type must be 'A', 'B', or 'C', got {model_type!r}"
        )
    feats = list(mapping[key])

    if include_missingness_indicators:
        if key == "b":
            chromatin = EXP_FEATURE_NAMES
        elif key == "c":
            chromatin = AG_FEATURE_NAMES
        else:
            chromatin = []
        feats += [f + MISSING_INDICATOR_SUFFIX for f in chromatin]

    return feats


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------

def label_candidates(
    candidates: pd.DataFrame,
    positives: pd.DataFrame,
    *,
    match_window_bp: int = DEFAULT_MATCH_WINDOW_BP,
) -> pd.DataFrame:
    """Assign binary labels to off-target candidates by matching to experimental positives.

    A candidate is labeled positive (``label=1``) if a site in *positives*
    satisfies all three criteria:

    - Same ``guide_id``
    - Same chromosome (after normalisation via :func:`~src.utils.normalize_chrom`)
    - ``|candidate.start − positive.start| ≤ match_window_bp``

    All other candidates receive ``label=0`` (negative).

    Parameters
    ----------
    candidates : pd.DataFrame
        Must contain ``guide_id``, ``chrom``, ``start``.
    positives : pd.DataFrame
        Must contain ``guide_id``, ``chrom``, ``start``.
    match_window_bp : int
        Tolerance in base pairs for position matching.  Default 5 bp.

    Returns
    -------
    pd.DataFrame
        Copy of *candidates* with an additional ``label`` column (int, 0 or 1).
    """
    result = candidates.copy()

    if len(candidates) == 0:
        result["label"] = pd.array([], dtype=int)
        return result

    if len(positives) == 0:
        logger.warning(
            "label_candidates: positives DataFrame is empty — "
            "all %d candidates labeled negative", len(candidates)
        )
        result["label"] = 0
        return result

    # Normalise positives: deduplicate (guide_id, chrom, start) triples.
    pos = positives[["guide_id", "chrom", "start"]].copy()
    pos = pos.assign(chrom=pos["chrom"].apply(normalize_chrom))
    pos = pos.drop_duplicates(subset=["guide_id", "chrom", "start"])
    pos = pos.rename(columns={"start": "_pos_start"})

    # Build a working copy of candidates with normalised chrom and an
    # explicit column holding the original DataFrame index for later alignment.
    cands_work = candidates[["guide_id", "chrom", "start"]].copy()
    cands_work["_orig_idx"] = candidates.index
    cands_work = cands_work.assign(chrom=cands_work["chrom"].apply(normalize_chrom))

    # Left-merge: each candidate row joins all positives for same guide+chrom.
    merged = cands_work.merge(pos, on=["guide_id", "chrom"], how="left")
    merged["_within"] = (
        (merged["start"] - merged["_pos_start"]).abs() <= match_window_bp
    )
    # NaN distance (no positive for this guide+chrom) → no match.
    merged["_within"] = merged["_within"].fillna(False)

    # A candidate is positive if ANY of its joined rows matched.
    any_match = merged.groupby("_orig_idx")["_within"].any()

    # Align back to original index; unrepresented rows (impossible after
    # left-merge, but defensive) → False.
    result["label"] = (
        any_match.reindex(candidates.index, fill_value=False).astype(int).values
    )

    n_pos = int(result["label"].sum())
    n_total = len(result)
    logger.info(
        "label_candidates: %d / %d positive (%.1f%%), window=%d bp",
        n_pos, n_total,
        100.0 * n_pos / n_total if n_total else 0.0,
        match_window_bp,
    )
    return result


# ---------------------------------------------------------------------------
# Sequence feature extraction
# ---------------------------------------------------------------------------

def extract_sequence_features_batch(candidates: pd.DataFrame) -> pd.DataFrame:
    """Compute sequence features for every candidate site.

    Calls :func:`~src.sequence_features.compute_sequence_features` row by row
    and encodes ``mm_type_*`` string columns to integers via
    :data:`MM_TYPE_TO_INT`.

    Parameters
    ----------
    candidates : pd.DataFrame
        Must contain ``guide_seq``, ``off_target_seq``, ``pam_seq``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with all :data:`SEQUENCE_FEATURE_NAMES` columns
        appended.  Rows with invalid sequences get NaN for all feature
        columns.
    """
    result = candidates.copy()
    feature_dicts: list[dict[str, Any]] = []

    for row_idx, row in candidates.iterrows():
        try:
            feats = compute_sequence_features(
                guide=str(row["guide_seq"]),
                off_target=str(row["off_target_seq"]),
                pam=str(row["pam_seq"]),
            )
            # Encode string mismatch types to integers in-place.
            for i in range(1, 21):
                key = f"mm_type_{i}"
                raw = feats.get(key, "")
                if isinstance(raw, str):
                    encoded = MM_TYPE_TO_INT.get(raw, -1)
                    if encoded == -1 and raw != "":
                        logger.warning(
                            "Unknown mm_type %r at row %s pos %d — encoding as -1",
                            raw, row_idx, i,
                        )
                    feats[key] = encoded
        except Exception as exc:
            logger.warning(
                "Sequence feature error at index %s (%s:%s): %s",
                row_idx, row.get("chrom"), row.get("start"), exc,
            )
            feats = {k: float("nan") for k in SEQUENCE_FEATURE_NAMES}

        feature_dicts.append(feats)

    feats_df = pd.DataFrame(feature_dicts, index=candidates.index)
    for col in feats_df.columns:
        result[col] = feats_df[col]

    return result


# ---------------------------------------------------------------------------
# NaN imputation
# ---------------------------------------------------------------------------

def impute_chromatin_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Median-impute NaN chromatin features and add binary missingness indicators.

    For each column in *feature_cols* that contains NaN values:

    1. Adds a new column ``{feature}_missing`` (1 where value was NaN, 0 otherwise).
    2. Fills NaN values with the column's within-DataFrame median.
       If the entire column is NaN (e.g., U2OS with no AlphaGenome tracks),
       fills with ``0.0`` and logs a WARNING.

    Non-NaN values are never modified.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
        Column names to impute.  Columns absent from *df* are skipped.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with imputed values and added ``*_missing`` columns.

    Notes
    -----
    Do NOT apply imputation to sequence features — they are always available.
    Apply only to chromatin features (AlphaGenome and ENCODE columns).
    """
    result = df.copy()

    for col in feature_cols:
        if col not in result.columns:
            logger.debug("impute_chromatin_features: column %r not in DataFrame, skipping", col)
            continue

        is_nan = result[col].isna()
        result[col + MISSING_INDICATOR_SUFFIX] = is_nan.astype(int)

        if not is_nan.any():
            continue  # No NaN values — nothing to impute.

        median_val = result[col].median()  # pandas median() ignores NaN.

        if pd.isna(median_val):
            logger.warning(
                "Feature %r is entirely NaN (e.g., no tracks for this cell type); "
                "imputing with 0.0",
                col,
            )
            median_val = 0.0

        n_imputed = int(is_nan.sum())
        logger.info(
            "Imputing %d / %d NaN values in %r with median=%.4f",
            n_imputed, len(result), col, median_val,
        )
        result[col] = result[col].fillna(median_val)

    return result


# ---------------------------------------------------------------------------
# Master assembly function
# ---------------------------------------------------------------------------

def build_feature_matrix(
    candidates: pd.DataFrame,
    positives: pd.DataFrame,
    *,
    cell_type: str,
    encode_manifest: Optional[Any] = None,
    ag_model: Optional[Any] = None,
    ag_ontology_term: Optional[str] = None,
    ag_cache_dir: Optional[Path] = None,
    window_bp: int = DEFAULT_WINDOW_BP,
    match_window_bp: int = DEFAULT_MATCH_WINDOW_BP,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Assemble the full feature matrix for model training and evaluation.

    Orchestrates label assignment, sequence feature extraction, AlphaGenome
    chromatin prediction, ENCODE experimental chromatin extraction, and NaN
    imputation into a single training-ready DataFrame.

    Parameters
    ----------
    candidates : pd.DataFrame
        All candidate off-target sites (both future positives and negatives).
        Required columns: ``guide_id``, ``guide_seq``, ``chrom``, ``start``
        (0-based), ``strand``, ``off_target_seq``, ``pam_seq``,
        ``num_mismatches``.
    positives : pd.DataFrame
        Confirmed positive off-target sites from GUIDE-seq / CHANGE-seq.
        Required columns: ``guide_id``, ``chrom``, ``start`` (0-based).
    cell_type : str
        Human-readable cell-type name, e.g. ``"K562"``.  Used for ENCODE
        BigWig lookup and stored in the output ``cell_type`` metadata column.
    encode_manifest : BigWigManifest | None
        ENCODE BigWig manifest from :func:`~src.encode_client.build_manifest`.
        Pass ``None`` to skip experimental chromatin features (Model A or C).
    ag_model : DnaClient | None
        Pre-initialised AlphaGenome client.  Pass ``None`` to skip AlphaGenome
        chromatin features (Model A or B).
    ag_ontology_term : str | None
        AlphaGenome cell-type ontology term, e.g. ``"EFO:0002067"`` for K562.
        Required when *ag_model* is provided.
    ag_cache_dir : Path | None
        Cache directory for AlphaGenome API responses.  Required when
        *ag_model* is provided.
    window_bp : int
        Half-width of chromatin feature extraction window in bp (default 500).
    match_window_bp : int
        Maximum distance (bp) for matching a candidate to an experimental
        positive.  Default 5 bp.
    output_path : Path | None
        If given, save the assembled DataFrame as a parquet file at this path
        (parent directories are created automatically).

    Returns
    -------
    pd.DataFrame
        Feature matrix with columns:

        - Metadata: ``guide_id``, ``chrom``, ``start``, ``strand``,
          ``guide_seq``, ``off_target_seq``, ``pam_seq``, ``cell_type``,
          ``dataset``, ``label``
        - Sequence features: :data:`SEQUENCE_FEATURE_NAMES`
        - AlphaGenome features (if *ag_model* provided): :data:`AG_FEATURE_NAMES`
          + ``{feat}_missing`` indicator columns
        - ENCODE features (if *encode_manifest* provided): :data:`EXP_FEATURE_NAMES`
          + ``{feat}_missing`` indicator columns

    Raises
    ------
    ValueError
        If required columns are absent from *candidates* or *positives*, or
        if *ag_model* is provided without *ag_ontology_term*.
    """
    # ---- Validate inputs --------------------------------------------------
    _require_columns(candidates, _REQUIRED_CANDIDATE_COLS, "candidates")
    _require_columns(positives, _REQUIRED_POSITIVE_COLS, "positives")

    if ag_model is not None and ag_ontology_term is None:
        raise ValueError(
            "ag_ontology_term is required when ag_model is provided. "
            "Example: ag_ontology_term='EFO:0002067' for K562."
        )
    if ag_model is not None and ag_cache_dir is None:
        raise ValueError("ag_cache_dir is required when ag_model is provided.")

    logger.info(
        "build_feature_matrix: %d candidates, %d positives, cell_type=%s",
        len(candidates), len(positives), cell_type,
    )

    # ---- Step 1: Remove on-target sites (num_mismatches == 0) ------------
    n_before = len(candidates)
    work = candidates[candidates["num_mismatches"] > 0].copy()
    log_filter(logger, "remove on-target sites (num_mismatches == 0)", n_before, len(work))

    if len(work) == 0:
        logger.error("No candidates remain after on-target removal — check input data.")
        return work

    # ---- Step 2: Label candidates -----------------------------------------
    work = label_candidates(work, positives, match_window_bp=match_window_bp)

    n_pos = int(work["label"].sum())
    n_neg = len(work) - n_pos
    if n_pos == 0:
        logger.warning(
            "Zero positive candidates after label assignment. "
            "Check coordinate systems and guide_id mapping."
        )
    elif n_pos / len(work) > 0.50:
        logger.warning(
            "Positive rate is %.1f%% (>50%%) — match window may be too large.",
            100.0 * n_pos / len(work),
        )
    logger.info(
        "Class balance: %d positive (%.1f%%), %d negative",
        n_pos, 100.0 * n_pos / len(work), n_neg,
    )

    # ---- Step 3: Sequence features ----------------------------------------
    logger.info("Extracting sequence features for %d candidates ...", len(work))
    work = extract_sequence_features_batch(work)

    # ---- Step 4: AlphaGenome chromatin features ----------------------------
    if ag_model is not None:
        logger.info(
            "Extracting AlphaGenome features for %d sites, ontology=%s ...",
            len(work), ag_ontology_term,
        )
        work = _ag_mod.batch_get_chromatin_features(
            work,
            cell_type=ag_ontology_term,
            model=ag_model,
            cache_dir=Path(ag_cache_dir),
            window_bp=window_bp,
        )
        ag_cols = [c for c in AG_FEATURE_NAMES if c in work.columns]
        work = impute_chromatin_features(work, ag_cols)

    # ---- Step 5: ENCODE experimental chromatin features -------------------
    if encode_manifest is not None:
        logger.info(
            "Extracting ENCODE features for %d sites, cell_type=%s ...",
            len(work), cell_type,
        )
        work = _enc_mod.batch_get_chromatin_features(
            work,
            cell_type=cell_type,
            manifest=encode_manifest,
            window_bp=window_bp,
        )
        exp_cols = [c for c in EXP_FEATURE_NAMES if c in work.columns]
        work = impute_chromatin_features(work, exp_cols)

    # ---- Step 6: Add metadata columns if not already present --------------
    if "cell_type" not in work.columns:
        work["cell_type"] = cell_type
    if "dataset" not in work.columns:
        work["dataset"] = "unknown"

    # ---- Step 7: Summary log ---------------------------------------------
    logger.info(
        "Feature matrix complete: %d rows × %d columns | "
        "%d positive (%.1f%%) | %d negative",
        len(work), len(work.columns),
        n_pos, 100.0 * n_pos / len(work) if len(work) else 0.0,
        n_neg,
    )

    # ---- Step 8: Optional save -------------------------------------------
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        work.to_parquet(output_path, index=False)
        logger.info("Saved feature matrix → %s", output_path)

    return work
