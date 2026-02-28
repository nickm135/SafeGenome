"""Cross-cell-type CRISPR off-target risk re-ranking using AlphaGenome.

This module implements the "killer feature" of SafeGenome: re-ranking the same
set of CRISPR off-target candidates across multiple cell types using only
AlphaGenome's predicted chromatin.  Traditional tools (Elevation, DeepCRISPR)
require experimental data per cell type; SafeGenome can score any cell type
from sequence alone.

Workflow
--------
1. Start with off-target candidates (from Cas-OFFinder) with sequence
   features pre-computed (or compute them here).
2. For each target cell type, query AlphaGenome for chromatin features.
3. Score candidates with the trained **Model C** (sequence + AlphaGenome
   features).
4. Aggregate into a site × cell-type risk-score matrix.
5. Visualise as a heatmap (Figure 3 in the paper) and compute rank-change
   statistics.

Critical note on imputation during inference
--------------------------------------------
During training, NaN chromatin features were imputed with within-dataset
medians (or 0.0 for all-NaN columns such as U2OS AlphaGenome features).
At inference time for a new cell type we apply the same strategy: NaN values
are imputed with the within-inference-batch median (or 0.0 for all-NaN
columns, consistent with the U2OS/HEK293 DNase handling during training).
This is appropriate for the visualisation.  For publication-quality metrics,
store training-set medians and apply them here.

Functions
---------
make_site_id                     → canonical site identifier string
score_candidates_for_cell_type   → score + rank sites for one cell type
transfer_across_cell_types       → long-format scores across all cell types
build_score_matrix               → reshape long → wide score matrix
compute_rank_correlation_matrix  → Spearman rank correlations between cell types
validate_transfer_rankings       → AUPRC vs. experimental labels for one cell type
compute_chromatin_correlation    → Spearman r (AlphaGenome vs ENCODE, Validation 1)
plot_transfer_heatmap            → Figure 3: site × cell-type risk heatmap
plot_rank_correlation_heatmap    → cell-type × cell-type Spearman correlation heatmap
run_cell_type_transfer           → main pipeline entry point
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, rankdata

import src.alphagenome_client as _ag_mod
from src.feature_matrix import (
    AG_FEATURE_NAMES,
    SEQUENCE_FEATURE_NAMES,
    extract_sequence_features_batch,
    impute_chromatin_features,
    label_candidates,
)
from src.model import compute_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default recommended cell types and their AlphaGenome ontology terms.
RECOMMENDED_CELL_TYPES: dict[str, str] = {
    "K562": "EFO:0002067",
    "HEK293": "EFO:0001182",
    "CD4_T_cell": "CL:0000624",
    "Hepatocyte": "CL:0000182",
}

#: Required site columns for AlphaGenome queries.
_REQUIRED_SITE_COLS: tuple[str, ...] = ("chrom", "start")

#: Column used to indicate computed sequence features are present.
_SEQ_FEATURE_SENTINEL: str = "cfd_score"

#: Column name for the canonical site identifier.
SITE_ID_COL: str = "site_id"


# ---------------------------------------------------------------------------
# Site ID helper
# ---------------------------------------------------------------------------

def make_site_id(chrom: str, start: int, strand: str = ".") -> str:
    """Return a canonical site identifier string.

    Parameters
    ----------
    chrom : str
        Chromosome name, e.g. ``"chr7"``.
    start : int
        0-based start position.
    strand : str
        Strand (``"+"``, ``"-"``, or ``"."`` if unknown).

    Returns
    -------
    str
        Identifier of the form ``"chr7:100000:+"``.
    """
    return f"{chrom}:{start}:{strand}"


def _add_site_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``site_id`` column if not already present."""
    if SITE_ID_COL not in df.columns:
        strand = df["strand"] if "strand" in df.columns else "."
        df = df.copy()
        df[SITE_ID_COL] = [
            make_site_id(str(c), int(s), str(st))
            for c, s, st in zip(df["chrom"], df["start"], strand)
        ]
    return df


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_candidates_for_cell_type(
    candidates: pd.DataFrame,
    cell_type: str,
    ag_ontology_term: str,
    model,
    feature_cols: list[str],
    *,
    ag_client,
    ag_cache_dir: Path,
    window_bp: int = 500,
) -> pd.DataFrame:
    """Score and rank off-target candidates for a single cell type.

    Sequence features are computed if absent.  AlphaGenome chromatin features
    are queried for *cell_type*, NaN values are imputed, then all features in
    *feature_cols* are assembled and scored with *model*.

    Parameters
    ----------
    candidates : pd.DataFrame
        Off-target candidate sites.  Required columns: ``chrom``, ``start``
        (0-based).  Sequence features (``cfd_score`` etc.) are computed
        automatically if not present.  If present they are reused.
    cell_type : str
        Human-readable cell-type label, e.g. ``"K562"`` (used for logging).
    ag_ontology_term : str
        AlphaGenome ontology term, e.g. ``"EFO:0002067"``.
    model : xgb.XGBClassifier
        Trained Model C (sequence + AlphaGenome features).
    feature_cols : list[str]
        Feature columns the model expects (in training order).  Typically
        ``MODEL_C_FEATURES`` from ``src.feature_matrix``.
    ag_client : DnaClient
        Initialised AlphaGenome client.
    ag_cache_dir : Path
        Cache directory for API responses.
    window_bp : int
        Feature extraction half-window around each site (default 500 bp).

    Returns
    -------
    pd.DataFrame
        Copy of *candidates* with two additional columns:

        - ``risk_score`` (float in [0, 1]): probability of off-target cleavage
          in *cell_type*.
        - ``rank`` (int, 1-indexed): rank by descending risk score (1 = highest
          risk).
        - ``cell_type`` (str): *cell_type* label.
        - ``site_id`` (str): canonical ``"{chrom}:{start}:{strand}"``.
    """
    for col in _REQUIRED_SITE_COLS:
        if col not in candidates.columns:
            raise ValueError(
                f"score_candidates_for_cell_type: candidates is missing column "
                f"{col!r}. Required: {list(_REQUIRED_SITE_COLS)}"
            )

    work = candidates.copy()
    work = _add_site_ids(work)

    # --- Step 1: Sequence features (compute if absent) ---
    if _SEQ_FEATURE_SENTINEL not in work.columns:
        seq_cols_needed = [
            c for c in feature_cols if c in SEQUENCE_FEATURE_NAMES
        ]
        if seq_cols_needed:
            logger.info(
                "score_candidates_for_cell_type: sequence features missing — "
                "computing for %d candidates",
                len(work),
            )
            work = extract_sequence_features_batch(work)

    # --- Step 2: AlphaGenome chromatin features ---
    logger.info(
        "Querying AlphaGenome for %d sites, cell_type=%s (%s) ...",
        len(work), cell_type, ag_ontology_term,
    )
    work = _ag_mod.batch_get_chromatin_features(
        work,
        cell_type=ag_ontology_term,
        model=ag_client,
        cache_dir=Path(ag_cache_dir),
        window_bp=window_bp,
    )

    # --- Step 3: Impute NaN chromatin features ---
    ag_cols_present = [c for c in AG_FEATURE_NAMES if c in work.columns]
    work = impute_chromatin_features(work, ag_cols_present)

    # --- Step 4: Build X and score ---
    available_feat_cols = [c for c in feature_cols if c in work.columns]
    missing_feat_cols = [c for c in feature_cols if c not in work.columns]
    if missing_feat_cols:
        logger.warning(
            "score_candidates_for_cell_type: %d feature column(s) missing from "
            "candidates for cell_type=%s — filling with 0: %s%s",
            len(missing_feat_cols), cell_type,
            missing_feat_cols[:5],
            " ..." if len(missing_feat_cols) > 5 else "",
        )
        for col in missing_feat_cols:
            work[col] = 0.0
        available_feat_cols = feature_cols  # now all present (filled with 0)

    X = work[available_feat_cols].to_numpy(dtype=np.float32)
    risk_scores = model.predict_proba(X)[:, 1]

    # Rank: 1 = highest risk (largest risk_score).
    ranks = rankdata(-risk_scores, method="min").astype(int)

    result = candidates.copy()
    result = _add_site_ids(result)
    result["risk_score"] = risk_scores
    result["rank"] = ranks
    result["cell_type"] = cell_type

    n_pos = int(result["label"].sum()) if "label" in result.columns else None
    logger.info(
        "score_candidates_for_cell_type done: cell_type=%s, %d sites%s",
        cell_type, len(result),
        f", {n_pos} labeled positives" if n_pos is not None else "",
    )
    return result


# ---------------------------------------------------------------------------
# Multi-cell-type transfer
# ---------------------------------------------------------------------------

def transfer_across_cell_types(
    candidates: pd.DataFrame,
    cell_type_map: dict[str, str],
    model,
    feature_cols: list[str],
    *,
    ag_client,
    ag_cache_dir: Path,
    window_bp: int = 500,
) -> pd.DataFrame:
    """Score and rank candidates across multiple cell types.

    For each cell type in *cell_type_map*, calls
    :func:`score_candidates_for_cell_type` and concatenates results into a
    single long-format DataFrame.

    Parameters
    ----------
    candidates : pd.DataFrame
        Off-target candidates.  Required columns: ``chrom``, ``start``.
    cell_type_map : dict[str, str]
        Mapping from human-readable cell-type name to AlphaGenome ontology
        term.  Example: ``{"K562": "EFO:0002067", "HEK293": "EFO:0001182"}``.
    model : xgb.XGBClassifier
        Trained Model C.
    feature_cols : list[str]
        Feature columns used during training (e.g. ``MODEL_C_FEATURES``).
    ag_client : DnaClient
        Initialised AlphaGenome client.
    ag_cache_dir : Path
        Cache directory for API responses.
    window_bp : int
        Feature extraction half-window (default 500 bp).

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with one row per (site, cell_type) pair.
        Contains all original candidate columns plus ``risk_score``,
        ``rank``, ``cell_type``, and ``site_id``.

    Raises
    ------
    ValueError
        If *cell_type_map* is empty.
    """
    if not cell_type_map:
        raise ValueError("cell_type_map must contain at least one entry.")

    logger.info(
        "transfer_across_cell_types: %d sites × %d cell types",
        len(candidates), len(cell_type_map),
    )

    parts: list[pd.DataFrame] = []
    for cell_type, ontology_term in cell_type_map.items():
        logger.info("--- Processing cell type: %s (%s) ---", cell_type, ontology_term)
        scored = score_candidates_for_cell_type(
            candidates,
            cell_type=cell_type,
            ag_ontology_term=ontology_term,
            model=model,
            feature_cols=feature_cols,
            ag_client=ag_client,
            ag_cache_dir=Path(ag_cache_dir),
            window_bp=window_bp,
        )
        parts.append(scored)

    transfer_df = pd.concat(parts, ignore_index=True)
    logger.info(
        "transfer_across_cell_types complete: %d total rows (%d sites × %d cell types)",
        len(transfer_df), len(candidates), len(cell_type_map),
    )
    return transfer_df


# ---------------------------------------------------------------------------
# Score matrix
# ---------------------------------------------------------------------------

def build_score_matrix(
    transfer_df: pd.DataFrame,
    *,
    sort_by: Optional[str] = "cfd_score",
    ascending: bool = True,
    site_id_col: str = SITE_ID_COL,
) -> pd.DataFrame:
    """Reshape a long-format transfer DataFrame into a wide score matrix.

    Parameters
    ----------
    transfer_df : pd.DataFrame
        Long-format output of :func:`transfer_across_cell_types`.  Must
        contain ``site_id``, ``cell_type``, and ``risk_score`` columns.
    sort_by : str | None
        Column in *transfer_df* to use for row sorting.  If ``None`` or the
        column is not found, rows are sorted by mean risk score descending.
        Typical choices: ``"cfd_score"`` (sequence-only score, ascending =
        lowest-sequence-score first, showing chromatin-driven re-ranking).
    ascending : bool
        Sort direction for *sort_by* (default ``True`` = ascending).
    site_id_col : str
        Name of the site identifier column (default ``"site_id"``).

    Returns
    -------
    pd.DataFrame
        Wide matrix with:

        - Index: site identifiers (``site_id``).
        - Columns: cell-type names.
        - Values: risk scores in [0, 1].
    """
    for col in (site_id_col, "cell_type", "risk_score"):
        if col not in transfer_df.columns:
            raise ValueError(
                f"build_score_matrix: required column {col!r} not in transfer_df."
            )

    matrix = transfer_df.pivot_table(
        index=site_id_col,
        columns="cell_type",
        values="risk_score",
        aggfunc="first",
    )
    matrix.columns.name = None

    # Determine row order.
    if sort_by is not None and sort_by in transfer_df.columns:
        sort_series = (
            transfer_df.groupby(site_id_col)[sort_by].first()
            .reindex(matrix.index)
        )
        ordered_idx = sort_series.sort_values(ascending=ascending).index
        matrix = matrix.loc[ordered_idx]
    else:
        if sort_by is not None:
            logger.warning(
                "build_score_matrix: sort_by=%r not found in transfer_df; "
                "sorting by mean risk score descending",
                sort_by,
            )
        mean_scores = matrix.mean(axis=1)
        matrix = matrix.loc[mean_scores.sort_values(ascending=False).index]

    return matrix


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------

def compute_rank_correlation_matrix(
    score_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Spearman rank correlation between all pairs of cell types.

    Uses :meth:`pandas.DataFrame.corr` with ``method="spearman"``.  NaN
    values are handled pairwise (sites with NaN in either cell type are
    excluded from that pair's computation).

    Parameters
    ----------
    score_matrix : pd.DataFrame
        Wide matrix of risk scores (rows = sites, columns = cell types), as
        returned by :func:`build_score_matrix`.

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix with shape
        ``(n_cell_types, n_cell_types)``.  Values are in [−1, 1].
        Diagonal entries are 1.0.
    """
    if score_matrix.empty:
        raise ValueError("compute_rank_correlation_matrix: score_matrix is empty.")

    corr = score_matrix.corr(method="spearman")
    return corr


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_transfer_rankings(
    transfer_df: pd.DataFrame,
    positives: pd.DataFrame,
    cell_type: str,
    *,
    match_window_bp: int = 5,
) -> dict[str, float]:
    """Evaluate ranked predictions against experimental positive labels.

    For the given *cell_type*, retrieves predicted risk scores from
    *transfer_df*, matches them to known positives in *positives*, and
    computes AUPRC and AUROC.

    Parameters
    ----------
    transfer_df : pd.DataFrame
        Long-format output of :func:`transfer_across_cell_types`.  Must
        contain ``cell_type``, ``risk_score``, ``guide_id`` (if available),
        ``chrom``, and ``start`` columns.
    positives : pd.DataFrame
        Experimentally verified positive off-target sites for *cell_type*.
        Must contain ``guide_id``, ``chrom``, ``start``.
    cell_type : str
        Cell type to validate.
    match_window_bp : int
        Tolerance (bp) for position matching between candidates and positives.

    Returns
    -------
    dict[str, float]
        Keys: ``auprc``, ``auroc``, ``precision_at_top``, ``recall_at_top``,
        ``enrichment_at_top``, ``n_sites``, ``n_positives``.
    """
    ct_df = transfer_df[transfer_df["cell_type"] == cell_type].copy()
    if len(ct_df) == 0:
        raise ValueError(
            f"validate_transfer_rankings: no sites found for cell_type={cell_type!r}. "
            f"Available cell types: {transfer_df['cell_type'].unique().tolist()}"
        )

    # Ensure guide_id exists (needed by label_candidates).
    if "guide_id" not in ct_df.columns:
        logger.warning(
            "validate_transfer_rankings: 'guide_id' missing — adding a "
            "placeholder. All sites will be treated as from the same guide."
        )
        ct_df = ct_df.copy()
        ct_df["guide_id"] = "unknown_guide"
        positives = positives.copy()
        if "guide_id" not in positives.columns:
            positives["guide_id"] = "unknown_guide"

    labeled = label_candidates(ct_df, positives, match_window_bp=match_window_bp)
    y_true = labeled["label"].to_numpy(dtype=np.int32)
    y_score = labeled["risk_score"].to_numpy(dtype=np.float64)

    metrics = compute_metrics(y_true, y_score)
    metrics["n_sites"] = float(len(labeled))
    metrics["n_positives"] = float(int(y_true.sum()))

    logger.info(
        "validate_transfer_rankings (%s): AUPRC=%.4f, AUROC=%.4f, "
        "n_sites=%d, n_positives=%d",
        cell_type,
        metrics.get("auprc", float("nan")),
        metrics.get("auroc", float("nan")),
        int(metrics["n_sites"]),
        int(metrics["n_positives"]),
    )
    return metrics


# ---------------------------------------------------------------------------
# Chromatin correlation (Validation 1)
# ---------------------------------------------------------------------------

def compute_chromatin_correlation(
    feature_matrix: pd.DataFrame,
    *,
    ag_feature: str = "ag_dnase_mean",
    exp_feature: str = "exp_dnase_mean",
    cell_type: Optional[str] = None,
) -> dict[str, float]:
    """Spearman correlation between AlphaGenome predictions and ENCODE signals.

    This is **Validation 1** from the project spec: we expect r > 0.5 for K562
    (preferably r > 0.7) to confirm that AlphaGenome's predictions are
    meaningful at CRISPR off-target loci.

    Per the project's critical rules, Spearman (not Pearson) is used because
    the two signals are on different scales.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Feature matrix containing both AlphaGenome (``ag_*``) and ENCODE
        (``exp_*``) feature columns.
    ag_feature : str
        AlphaGenome feature column name (default ``"ag_dnase_mean"``).
    exp_feature : str
        ENCODE feature column name (default ``"exp_dnase_mean"``).
    cell_type : str | None
        If provided, filter to rows where ``cell_type == cell_type``.

    Returns
    -------
    dict[str, float]
        Keys: ``spearman_r``, ``p_value``, ``n``.  Any value is ``nan`` if
        fewer than 3 paired non-NaN observations are available.

    Raises
    ------
    ValueError
        If *ag_feature* or *exp_feature* are not present in *feature_matrix*.
    """
    for col in (ag_feature, exp_feature):
        if col not in feature_matrix.columns:
            raise ValueError(
                f"compute_chromatin_correlation: column {col!r} not in "
                f"feature_matrix. Available columns: "
                f"{[c for c in feature_matrix.columns if c.startswith(('ag_', 'exp_'))]}"
            )

    df = feature_matrix
    if cell_type is not None and "cell_type" in df.columns:
        df = df[df["cell_type"] == cell_type]
        if len(df) == 0:
            logger.warning(
                "compute_chromatin_correlation: no rows for cell_type=%r", cell_type
            )
            return {"spearman_r": float("nan"), "p_value": float("nan"), "n": 0}

    valid = df[[ag_feature, exp_feature]].dropna()
    n = len(valid)

    if n < 3:
        logger.warning(
            "compute_chromatin_correlation: only %d valid pairs; "
            "Spearman r cannot be reliably estimated",
            n,
        )
        return {"spearman_r": float("nan"), "p_value": float("nan"), "n": n}

    r, p = spearmanr(valid[ag_feature], valid[exp_feature])
    logger.info(
        "Spearman correlation (%s vs %s%s): r=%.4f, p=%.4g, n=%d",
        ag_feature, exp_feature,
        f", cell_type={cell_type!r}" if cell_type else "",
        r, p, n,
    )
    return {"spearman_r": float(r), "p_value": float(p), "n": n}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_transfer_heatmap(
    score_matrix: pd.DataFrame,
    *,
    output_path: Optional[Path] = None,
    title: str = "CRISPR Off-Target Risk Scores Across Cell Types",
    top_n: Optional[int] = None,
    normalize_rows: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    cmap: str = "RdBu_r",
):
    """Plot the site × cell-type risk score heatmap (Figure 3).

    Parameters
    ----------
    score_matrix : pd.DataFrame
        Wide risk-score matrix (rows = sites, columns = cell types), as
        returned by :func:`build_score_matrix`.
    output_path : Path | None
        Save destination.  Extension determines format (PNG, SVG, PDF).
    title : str
        Figure title.
    top_n : int | None
        If given, show only the top-n rows (by mean risk score).
    normalize_rows : bool
        If ``True``, z-score each row (highlights relative differences
        between cell types for the same site).  Useful for visualization
        when absolute risk levels differ substantially between sites.
    figsize : tuple | None
        Figure size ``(width, height)`` in inches.  Auto-sized if ``None``.
    cmap : str
        Matplotlib colormap.  ``"RdBu_r"`` (red=high risk, blue=low risk).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ImportError
        If matplotlib or seaborn are not installed.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "matplotlib and seaborn are required for plot_transfer_heatmap. "
            "Install with: pip install matplotlib>=3.7 seaborn>=0.12"
        )

    plot_matrix = score_matrix.copy()
    if top_n is not None:
        mean_scores = plot_matrix.mean(axis=1)
        plot_matrix = plot_matrix.loc[
            mean_scores.nlargest(top_n).index
        ]

    if normalize_rows:
        plot_matrix = plot_matrix.sub(plot_matrix.mean(axis=1), axis=0)
        std = plot_matrix.std(axis=1).replace(0.0, 1.0)
        plot_matrix = plot_matrix.div(std, axis=0)
        vmin, vmax, center = None, None, 0.0
        colorbar_label = "Row-normalised risk score (z-score)"
    else:
        vmin, vmax, center = 0.0, 1.0, 0.5
        colorbar_label = "Risk score (Model C probability)"

    n_sites, n_ct = plot_matrix.shape
    if figsize is None:
        figsize = (max(4.0, n_ct * 1.2 + 2.0), max(4.0, n_sites * 0.18 + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        plot_matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        xticklabels=True,
        yticklabels=False,   # too many sites to label individually
        cbar_kws={"label": colorbar_label, "shrink": 0.8},
    )
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Cell type", fontsize=11)
    ax.set_ylabel(f"Off-target sites (n={n_sites})", fontsize=11)
    ax.tick_params(axis="x", labelsize=10, rotation=30)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Transfer heatmap saved → %s", output_path)

    return fig


def plot_rank_correlation_heatmap(
    rank_corr: pd.DataFrame,
    *,
    output_path: Optional[Path] = None,
    title: str = "Spearman Rank Correlation of Off-Target Risk Across Cell Types",
    figsize: tuple[float, float] = (6.0, 5.5),
    cmap: str = "coolwarm",
    annot: bool = True,
):
    """Plot Spearman rank-correlation matrix between cell types.

    A high correlation between two cell types indicates that the off-target
    risk rankings are similar (e.g. both have open chromatin at the same
    loci).  Low correlation demonstrates cell-type-specific re-ranking.

    Parameters
    ----------
    rank_corr : pd.DataFrame
        Square Spearman correlation matrix from
        :func:`compute_rank_correlation_matrix`.
    output_path : Path | None
        Save destination.
    title : str
        Figure title.
    figsize : tuple[float, float]
        Figure size in inches.
    cmap : str
        Diverging colormap centred at zero.
    annot : bool
        Annotate each cell with its correlation value.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "matplotlib and seaborn are required. "
            "Install with: pip install matplotlib seaborn"
        )

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.zeros_like(rank_corr.values, dtype=bool)
    np.fill_diagonal(mask, True)  # hide diagonal (always 1.0)

    sns.heatmap(
        rank_corr,
        ax=ax,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        annot=annot,
        fmt=".2f",
        mask=mask,
        square=True,
        cbar_kws={"label": "Spearman r", "shrink": 0.8},
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Rank correlation heatmap saved → %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_cell_type_transfer(
    candidates: pd.DataFrame,
    cell_type_map: dict[str, str],
    model,
    feature_cols: list[str],
    *,
    ag_client,
    ag_cache_dir: Path,
    window_bp: int = 500,
    output_dir: Optional[Path] = None,
    top_n_heatmap: int = 50,
    normalize_heatmap_rows: bool = False,
    sort_heatmap_by: Optional[str] = "cfd_score",
    positives_by_cell_type: Optional[dict[str, pd.DataFrame]] = None,
    match_window_bp: int = 5,
) -> dict:
    """Run the complete cell-type transfer analysis pipeline.

    Scores candidates across all cell types in *cell_type_map*, builds the
    score matrix, computes rank correlations, optionally validates against
    experimental labels, and generates figures.

    Parameters
    ----------
    candidates : pd.DataFrame
        Off-target candidates (chrom + start required; seq features optional).
    cell_type_map : dict[str, str]
        ``{cell_type_name: ontology_term}`` mapping.
    model : xgb.XGBClassifier
        Trained Model C final model.
    feature_cols : list[str]
        Feature columns used during Model C training (e.g. ``MODEL_C_FEATURES``).
    ag_client : DnaClient
        Initialised AlphaGenome client.
    ag_cache_dir : Path
        Cache directory for API responses.
    window_bp : int
        Feature extraction half-window (default 500 bp).
    output_dir : Path | None
        Directory for saving figures and CSV results.
    top_n_heatmap : int
        Number of top-risk sites to show in the heatmap.
    normalize_heatmap_rows : bool
        If ``True``, z-score normalise heatmap rows.
    sort_heatmap_by : str | None
        Column for heatmap row ordering (default ``"cfd_score"``).
    positives_by_cell_type : dict[str, pd.DataFrame] | None
        Mapping from cell-type name to positives DataFrame for validation.
        Only cell types with GUIDE-seq or equivalent data need entries.
    match_window_bp : int
        Position tolerance for matching candidates to experimental positives.

    Returns
    -------
    dict with keys:
        - ``"transfer_df"`` : pd.DataFrame — long-format risk scores.
        - ``"score_matrix"`` : pd.DataFrame — wide site × cell-type matrix.
        - ``"rank_corr"`` : pd.DataFrame — Spearman rank correlation matrix.
        - ``"validation"`` : dict[str, dict] — per-cell-type AUPRC/AUROC
          (only for cell types in *positives_by_cell_type*).
    """
    logger.info(
        "=== Cell-Type Transfer Analysis ===\n"
        "  %d candidates × %d cell types",
        len(candidates), len(cell_type_map),
    )

    # --- Score across cell types ---
    transfer_df = transfer_across_cell_types(
        candidates,
        cell_type_map=cell_type_map,
        model=model,
        feature_cols=feature_cols,
        ag_client=ag_client,
        ag_cache_dir=Path(ag_cache_dir),
        window_bp=window_bp,
    )

    # --- Build score matrix ---
    score_matrix = build_score_matrix(
        transfer_df,
        sort_by=sort_heatmap_by,
        ascending=True,
    )

    # --- Rank correlation ---
    rank_corr = compute_rank_correlation_matrix(score_matrix)
    logger.info("Rank correlation matrix:\n%s", rank_corr.to_string())

    # --- Validation ---
    validation: dict[str, dict] = {}
    if positives_by_cell_type:
        for ct, positives in positives_by_cell_type.items():
            if ct not in cell_type_map:
                logger.warning(
                    "run_cell_type_transfer: validation cell type %r not in "
                    "cell_type_map — skipping",
                    ct,
                )
                continue
            try:
                val_metrics = validate_transfer_rankings(
                    transfer_df, positives, ct,
                    match_window_bp=match_window_bp,
                )
                validation[ct] = val_metrics
                logger.info(
                    "Validation (%s): AUPRC=%.4f, n_positives=%d",
                    ct, val_metrics.get("auprc", float("nan")),
                    int(val_metrics.get("n_positives", 0)),
                )
            except Exception as exc:
                logger.warning(
                    "Validation for %r failed: %s", ct, exc
                )

    # --- Save outputs ---
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Long-format CSV
        transfer_csv = output_dir / "cell_type_transfer_scores.csv"
        transfer_df.to_csv(transfer_csv, index=False)
        logger.info("Transfer scores saved → %s", transfer_csv)

        # Score matrix CSV
        matrix_csv = output_dir / "cell_type_score_matrix.csv"
        score_matrix.to_csv(matrix_csv)
        logger.info("Score matrix saved → %s", matrix_csv)

        # Rank correlation CSV
        corr_csv = output_dir / "cell_type_rank_correlation.csv"
        rank_corr.to_csv(corr_csv)
        logger.info("Rank correlation saved → %s", corr_csv)

        # Heatmap figure
        try:
            plot_transfer_heatmap(
                score_matrix,
                output_path=output_dir / "fig3_cell_type_heatmap.png",
                top_n=top_n_heatmap,
                normalize_rows=normalize_heatmap_rows,
            )
        except Exception as exc:
            logger.warning("Transfer heatmap plotting failed: %s", exc)

        # Rank correlation figure
        try:
            plot_rank_correlation_heatmap(
                rank_corr,
                output_path=output_dir / "cell_type_rank_correlation_heatmap.png",
            )
        except Exception as exc:
            logger.warning("Rank correlation heatmap plotting failed: %s", exc)

        # Validation summary
        if validation:
            val_df = pd.DataFrame(
                [{"cell_type": ct, **v} for ct, v in validation.items()]
            )
            val_csv = output_dir / "transfer_validation.csv"
            val_df.to_csv(val_csv, index=False)
            logger.info("Validation summary saved → %s", val_csv)

    return {
        "transfer_df": transfer_df,
        "score_matrix": score_matrix,
        "rank_corr": rank_corr,
        "validation": validation,
    }
