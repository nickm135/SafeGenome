"""AlphaGenome API wrapper for chromatin feature extraction at CRISPR off-target sites.

This module wraps the AlphaGenome SDK to predict chromatin accessibility,
histone modifications, and TF binding at candidate off-target loci for a
specified cell type.  It is the core novel contribution of SafeGenome.

Key implementation notes
------------------------
- **Interval size** ``131 072`` bp (2**17).  The project spec described 100 kb but the
  actual SDK only accepts ``[16384, 131072, 524288, 1048576]``.  We use 131 072 as the
  closest option.
- **Resolution** differs by output type.  DNase and ATAC are 1 bp (shape
  ``(131072, n_tracks)``).  CHIP_HISTONE and CHIP_TF are 128 bp / bin
  (shape ``(1024, n_tracks)``).
- **Tiling**: sites are grouped into non-overlapping 131 072-bp tiles so that all
  sites within the same tile share a single API call.
- **Caching**: raw API responses are pickled to *cache_dir* via ``utils.save_cache``.
- **Rate limiting**: exponential back-off on rate-limit errors, plus a 1-second
  courtesy pause after every uncached call.
- **Missing tracks**: if a track type has no data for the requested cell type, the
  corresponding features are ``NaN`` rather than raising an error.

Cell-type track availability (queried 2026-02-24)
--------------------------------------------------
================== ============== ======= ====  ====================  =====
Cell type          Ontology term  DNase   ATAC  Histone marks         CTCF
================== ============== ======= ====  ====================  =====
K562               EFO:0002067    yes     yes   10 marks (no H3K9me3) yes
HEK293             EFO:0001182    **no**  **no** 3 marks (limited)    yes
U2OS               (none found)   **no**  **no** none                  no
================== ============== ======= ====  ====================  =====

DNase/ATAC features are NaN for HEK293; all features are NaN for U2OS.
This is a known limitation documented in P2.2 of PROJECT.md.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import GenomicInterval, normalize_chrom, load_cache, save_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Interval size used for all API calls.  Must be a power-of-2 value
#: accepted by the AlphaGenome SDK.
INTERVAL_SIZE: int = 131_072

#: Base-pairs per bin for CHIP_HISTONE / CHIP_TF outputs.
BIN_SIZE: int = 128                           # 131072 / 1024 = 128

#: Number of bins per interval.
N_BINS: int = INTERVAL_SIZE // BIN_SIZE       # = 1024

#: Default half-window in bp for wide feature extraction.
DEFAULT_WINDOW_BP: int = 500

#: Half-window in bp for the narrow feature extraction (P2.5).
NARROW_WINDOW_BP: int = 50

#: DNase mean threshold to call a site "accessible" (ag_accessibility_binary).
DNASE_OPEN_THRESHOLD: float = 1.0

#: Maximum API retry attempts before raising.
DEFAULT_MAX_ATTEMPTS: int = 5

#: Seconds to sleep after each uncached API call (courtesy rate-limit).
COURTESY_SLEEP_S: float = 1.0

#: Ordered list of feature column names produced per site.
FEATURE_NAMES: list[str] = [
    "ag_dnase_mean",          # Mean DNase over +/-window_bp
    "ag_dnase_max",           # Max DNase over +/-window_bp
    "ag_dnase_at_site",       # DNase signal at exact cut site
    "ag_dnase_mean_narrow",   # Mean DNase over +/-50 bp (P2.5)
    "ag_atac_mean",           # Mean ATAC over +/-window_bp
    "ag_atac_at_site",        # ATAC signal at exact cut site
    "ag_h3k4me3_mean",        # Active promoter mark
    "ag_h3k27me3_mean",       # Polycomb repression mark
    "ag_h3k9me3_mean",        # Constitutive heterochromatin mark
    "ag_h3k27ac_mean",        # Active enhancer / promoter mark
    "ag_ctcf_mean",           # CTCF insulator binding
    "ag_accessibility_binary", # 1 if ag_dnase_mean > DNASE_OPEN_THRESHOLD
]


# ---------------------------------------------------------------------------
# TrackAvailability — per-cell-type track inventory
# ---------------------------------------------------------------------------

@dataclass
class TrackAvailability:
    """Describes which AlphaGenome output types are available for one cell type.

    Parameters
    ----------
    ontology_term : str
        Cell-type ontology identifier, e.g. ``"EFO:0002067"`` for K562.
    has_dnase : bool
        At least one DNase track exists for this cell type.
    has_atac : bool
        At least one ATAC track exists for this cell type.
    histone_marks : list[str]
        Mark names present, e.g. ``["H3K27ac", "H3K27me3", "H3K4me3"]``.
    has_ctcf : bool
        A CTCF TF ChIP track exists for this cell type.
    """

    ontology_term: str
    has_dnase: bool
    has_atac: bool
    histone_marks: list[str]
    has_ctcf: bool

    @property
    def has_any_tracks(self) -> bool:
        """True if at least one output type has data for this cell type."""
        return (
            self.has_dnase
            or self.has_atac
            or bool(self.histone_marks)
            or self.has_ctcf
        )

    @property
    def requested_output_types(self) -> list:
        """Build the ``requested_outputs`` list for ``model.predict_interval``.

        Only includes ``OutputType`` values that have at least one track for
        this cell type, avoiding unnecessary data transfer.

        Returns
        -------
        list[dna_client.OutputType]
        """
        from alphagenome.models import dna_client  # local import — SDK optional at module level

        types: list = []
        if self.has_dnase:
            types.append(dna_client.OutputType.DNASE)
        if self.has_atac:
            types.append(dna_client.OutputType.ATAC)
        if self.histone_marks:
            types.append(dna_client.OutputType.CHIP_HISTONE)
        if self.has_ctcf:
            types.append(dna_client.OutputType.CHIP_TF)
        return types


# ---------------------------------------------------------------------------
# Client factory and metadata helpers
# ---------------------------------------------------------------------------

def create_client(api_key: str):
    """Create and return an AlphaGenome ``DnaClient``.

    Parameters
    ----------
    api_key : str
        AlphaGenome API key (from ``config.ALPHAGENOME_API_KEY``).

    Returns
    -------
    dna_client.DnaClient
    """
    from alphagenome.models import dna_client
    return dna_client.create(api_key)


def get_track_availability(model, ontology_term: str) -> TrackAvailability:
    """Query AlphaGenome metadata to discover tracks available for a cell type.

    Parameters
    ----------
    model : dna_client.DnaClient
        Pre-initialised AlphaGenome client.
    ontology_term : str
        Cell-type ontology identifier, e.g. ``"EFO:0002067"``.

    Returns
    -------
    TrackAvailability

    Notes
    -----
    Logs a warning if no tracks are found (all features will be NaN for
    sites queried with this cell type).
    """
    from alphagenome.models import dna_client

    meta = model.output_metadata(dna_client.Organism.HOMO_SAPIENS)

    has_dnase = bool((meta.dnase["ontology_curie"] == ontology_term).any())
    has_atac = bool((meta.atac["ontology_curie"] == ontology_term).any())

    histone_df = meta.chip_histone[
        meta.chip_histone["ontology_curie"] == ontology_term
    ]
    # Extract mark name as the last token of the track name string, e.g.
    # "EFO:0002067 Histone ChIP-seq H3K4me3" → "H3K4me3"
    marks: list[str] = sorted(
        {name.split()[-1] for name in histone_df["name"].tolist()}
    )

    tf_df = meta.chip_tf[meta.chip_tf["ontology_curie"] == ontology_term]
    has_ctcf = bool(tf_df["name"].str.contains("CTCF", na=False).any())

    avail = TrackAvailability(
        ontology_term=ontology_term,
        has_dnase=has_dnase,
        has_atac=has_atac,
        histone_marks=marks,
        has_ctcf=has_ctcf,
    )
    logger.info(
        "Track availability for %s: DNase=%s ATAC=%s histones=%s CTCF=%s",
        ontology_term,
        avail.has_dnase,
        avail.has_atac,
        avail.histone_marks,
        avail.has_ctcf,
    )
    if not avail.has_any_tracks:
        logger.warning(
            "No tracks found for ontology term %r — all features will be NaN",
            ontology_term,
        )
    return avail


def list_cell_types(model) -> pd.DataFrame:
    """Return a summary DataFrame of all cell types with at least one track.

    Useful for discovering which ontology terms to pass to
    :func:`batch_get_chromatin_features`.

    Parameters
    ----------
    model : dna_client.DnaClient

    Returns
    -------
    pd.DataFrame
        Columns: ``ontology_curie``, ``biosample_name``, ``has_dnase``,
        ``has_atac``, ``n_histone_tracks``, ``has_ctcf``.
        Sorted by ``ontology_curie``.
    """
    from alphagenome.models import dna_client

    meta = model.output_metadata(dna_client.Organism.HOMO_SAPIENS)

    # Collect every unique ontology term across all output types
    all_terms: pd.DataFrame = pd.concat([
        meta.dnase[["ontology_curie", "biosample_name"]],
        meta.atac[["ontology_curie", "biosample_name"]],
        meta.chip_histone[["ontology_curie", "biosample_name"]],
        meta.chip_tf[["ontology_curie", "biosample_name"]],
    ]).drop_duplicates("ontology_curie")

    rows = []
    for _, row in all_terms.iterrows():
        term = row["ontology_curie"]
        has_dnase = bool((meta.dnase["ontology_curie"] == term).any())
        has_atac = bool((meta.atac["ontology_curie"] == term).any())
        n_histone = int((meta.chip_histone["ontology_curie"] == term).sum())
        has_ctcf = bool(
            meta.chip_tf[meta.chip_tf["ontology_curie"] == term]["name"]
            .str.contains("CTCF", na=False)
            .any()
        )
        rows.append({
            "ontology_curie": term,
            "biosample_name": row["biosample_name"],
            "has_dnase": has_dnase,
            "has_atac": has_atac,
            "n_histone_tracks": n_histone,
            "has_ctcf": has_ctcf,
        })
    return (
        pd.DataFrame(rows)
        .sort_values("ontology_curie")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _site_to_tile(site_pos: int, chrom: str) -> tuple[str, int, int]:
    """Return the ``(chrom, tile_start, tile_end)`` for a 0-based site position.

    Sites are assigned to non-overlapping ``INTERVAL_SIZE``-bp tiles aligned
    to multiples of ``INTERVAL_SIZE``.  This guarantees:

    - Every tile is exactly ``INTERVAL_SIZE`` bp (a valid API interval).
    - Every site is strictly inside its tile (never at an edge by more than
      the tile size).
    - Cache keys are deterministic regardless of call order.

    Parameters
    ----------
    site_pos : int
        0-based off-target site position.
    chrom : str
        Normalised chromosome name (``chr``-prefixed).

    Returns
    -------
    tuple[str, int, int]
        ``(chrom, tile_start, tile_end)`` in 0-based half-open coordinates.
    """
    tile_start = (site_pos // INTERVAL_SIZE) * INTERVAL_SIZE
    return chrom, tile_start, tile_start + INTERVAL_SIZE


def _predict_with_retry(
    model,
    interval,
    requested_outputs: list,
    ontology_terms: list[str],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
):
    """Call ``model.predict_interval`` with exponential back-off on errors.

    Retries on any exception that looks like a rate-limit (HTTP 429,
    ``ResourceExhausted``, ``quota``).  Other errors are retried up to
    *max_attempts* times with the same back-off schedule before raising.

    Parameters
    ----------
    model : dna_client.DnaClient
    interval : genome.Interval
    requested_outputs : list[OutputType]
    ontology_terms : list[str]
    max_attempts : int

    Returns
    -------
    Prediction output object from ``model.predict_interval``.

    Raises
    ------
    RuntimeError
        If all attempts are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return model.predict_interval(
                interval=interval,
                requested_outputs=requested_outputs,
                ontology_terms=ontology_terms,
            )
        except Exception as exc:
            last_exc = exc
            exc_lower = str(exc).lower()
            is_rate_limit = any(
                kw in exc_lower
                for kw in ("rate", "429", "resourceexhausted", "quota", "resource_exhausted")
            )
            sleep_s = 2 ** attempt
            if is_rate_limit:
                logger.warning(
                    "Rate-limit on attempt %d/%d, sleeping %ds: %s",
                    attempt + 1, max_attempts, sleep_s, exc,
                )
            else:
                logger.warning(
                    "API error on attempt %d/%d, sleeping %ds: %s",
                    attempt + 1, max_attempts, sleep_s, exc,
                )
            if attempt < max_attempts - 1:
                time.sleep(sleep_s)

    raise RuntimeError(
        f"All {max_attempts} attempts failed for interval {interval}"
    ) from last_exc


def _extract_bp_signal(
    signal_2d: np.ndarray,
    site_offset: int,
    window_bp: int,
    narrow_window_bp: int,
) -> tuple[float, float, float, float]:
    """Extract summary statistics from a 1-bp-resolution signal array.

    Parameters
    ----------
    signal_2d : np.ndarray
        Shape ``(interval_bp, n_tracks)``.  Multiple tracks are averaged.
    site_offset : int
        0-based position of the cut site within the interval.
    window_bp : int
        Half-width of the wide window in bp.
    narrow_window_bp : int
        Half-width of the narrow window in bp.

    Returns
    -------
    tuple[float, float, float, float]
        ``(wide_mean, wide_max, at_site, narrow_mean)``
    """
    signal_1d: np.ndarray = signal_2d.mean(axis=1)
    n = len(signal_1d)

    w_start = max(0, site_offset - window_bp)
    w_end = min(n, site_offset + window_bp)
    nw_start = max(0, site_offset - narrow_window_bp)
    nw_end = min(n, site_offset + narrow_window_bp)

    at_site = (
        float(signal_1d[site_offset]) if 0 <= site_offset < n else float("nan")
    )
    wide_window = signal_1d[w_start:w_end]
    narrow_window = signal_1d[nw_start:nw_end]

    wide_mean = float(np.mean(wide_window)) if len(wide_window) else float("nan")
    wide_max = float(np.max(wide_window)) if len(wide_window) else float("nan")
    narrow_mean = (
        float(np.mean(narrow_window)) if len(narrow_window) else float("nan")
    )
    return wide_mean, wide_max, at_site, narrow_mean


def _extract_bin_signal(
    signal_2d: np.ndarray,
    site_offset: int,
    window_bp: int,
    track_mask: Optional[np.ndarray] = None,
) -> float:
    """Extract mean signal from a binned-resolution (128 bp/bin) track array.

    Parameters
    ----------
    signal_2d : np.ndarray
        Shape ``(n_bins, n_tracks)``.
    site_offset : int
        0-based position of the cut site within the interval in bp.
    window_bp : int
        Half-width in bp, converted to bins via ``// BIN_SIZE``.
    track_mask : np.ndarray | None
        Boolean mask selecting specific columns (tracks).  None = use all.

    Returns
    -------
    float
        Mean signal over the bin window, or NaN if mask selects nothing.
    """
    if track_mask is not None:
        if not np.any(track_mask):
            return float("nan")
        data = signal_2d[:, track_mask]
    else:
        data = signal_2d

    signal_1d: np.ndarray = data.mean(axis=1)
    n_bins = len(signal_1d)

    site_bin = site_offset // BIN_SIZE
    half_bins = max(1, window_bp // BIN_SIZE)
    b_start = max(0, site_bin - half_bins)
    b_end = min(n_bins, site_bin + half_bins)

    window = signal_1d[b_start:b_end]
    return float(np.mean(window)) if len(window) else float("nan")


def _nan_features() -> dict[str, float]:
    """Return a feature dict with all values set to NaN."""
    return {name: float("nan") for name in FEATURE_NAMES}


def _extract_chromatin_features(
    output,
    site_offset: int,
    avail: TrackAvailability,
    window_bp: int = DEFAULT_WINDOW_BP,
) -> dict[str, float]:
    """Extract the full chromatin feature vector from an AlphaGenome output.

    Parameters
    ----------
    output
        AlphaGenome prediction object returned by ``model.predict_interval``.
    site_offset : int
        0-based position of the off-target site within the interval.
    avail : TrackAvailability
        Which output types are present.
    window_bp : int
        Half-width of the wide extraction window in bp.

    Returns
    -------
    dict[str, float]
        Feature dictionary with keys matching :data:`FEATURE_NAMES`.
    """
    feats: dict[str, float] = {}

    # ------------------------------------------------------------------ DNase
    if avail.has_dnase:
        d_mean, d_max, d_site, d_narrow = _extract_bp_signal(
            output.dnase.values, site_offset, window_bp, NARROW_WINDOW_BP
        )
        feats["ag_dnase_mean"] = d_mean
        feats["ag_dnase_max"] = d_max
        feats["ag_dnase_at_site"] = d_site
        feats["ag_dnase_mean_narrow"] = d_narrow
        feats["ag_accessibility_binary"] = float(d_mean > DNASE_OPEN_THRESHOLD)
    else:
        feats["ag_dnase_mean"] = float("nan")
        feats["ag_dnase_max"] = float("nan")
        feats["ag_dnase_at_site"] = float("nan")
        feats["ag_dnase_mean_narrow"] = float("nan")
        feats["ag_accessibility_binary"] = float("nan")

    # ------------------------------------------------------------------- ATAC
    if avail.has_atac:
        a_mean, _, a_site, _ = _extract_bp_signal(
            output.atac.values, site_offset, window_bp, NARROW_WINDOW_BP
        )
        feats["ag_atac_mean"] = a_mean
        feats["ag_atac_at_site"] = a_site
    else:
        feats["ag_atac_mean"] = float("nan")
        feats["ag_atac_at_site"] = float("nan")

    # -------------------------------------------------------------- Histones
    _MARK_TO_FEAT: dict[str, str] = {
        "H3K4me3":  "ag_h3k4me3_mean",
        "H3K27me3": "ag_h3k27me3_mean",
        "H3K9me3":  "ag_h3k9me3_mean",
        "H3K27ac":  "ag_h3k27ac_mean",
    }
    if avail.histone_marks:
        histone_vals = output.chip_histone.values
        histone_meta = output.chip_histone.metadata
        for mark, feat_name in _MARK_TO_FEAT.items():
            mask = histone_meta["name"].str.contains(mark, na=False).values
            feats[feat_name] = _extract_bin_signal(
                histone_vals, site_offset, window_bp, track_mask=mask
            )
    else:
        for feat_name in _MARK_TO_FEAT.values():
            feats[feat_name] = float("nan")

    # ------------------------------------------------------------------- CTCF
    if avail.has_ctcf:
        ctcf_mask = (
            output.chip_tf.metadata["name"].str.contains("CTCF", na=False).values
        )
        feats["ag_ctcf_mean"] = _extract_bin_signal(
            output.chip_tf.values, site_offset, window_bp, track_mask=ctcf_mask
        )
    else:
        feats["ag_ctcf_mean"] = float("nan")

    return feats


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------

def get_chromatin_features(
    site: GenomicInterval,
    cell_type: str,
    model,
    cache_dir: Path,
    avail: Optional[TrackAvailability] = None,
    window_bp: int = DEFAULT_WINDOW_BP,
) -> dict[str, float]:
    """Extract AlphaGenome chromatin features for a single off-target site.

    Uses the tile cache if an API response for this locus + cell type has
    already been fetched.

    Parameters
    ----------
    site : GenomicInterval
        Off-target locus in 0-based half-open coordinates (hg38).
    cell_type : str
        Cell-type ontology term, e.g. ``"EFO:0002067"`` for K562.
    model : dna_client.DnaClient
        Pre-initialised AlphaGenome client.
    cache_dir : Path
        Directory for caching raw API responses.
    avail : TrackAvailability | None
        Track availability for *cell_type*.  Queried automatically if ``None``.
    window_bp : int
        Half-width of the feature extraction window in bp.

    Returns
    -------
    dict[str, float]
        Feature dictionary with keys matching :data:`FEATURE_NAMES`.
        Missing tracks yield NaN values.
    """
    from alphagenome.data import genome as ag_genome

    if avail is None:
        avail = get_track_availability(model, cell_type)

    if not avail.has_any_tracks:
        logger.warning(
            "No tracks for cell type %r — returning NaN for %s", cell_type, site
        )
        return _nan_features()

    chrom = normalize_chrom(site.chrom)
    _, tile_start, tile_end = _site_to_tile(site.start, chrom)
    site_offset = site.start - tile_start

    cached = load_cache(
        Path(cache_dir), chrom, str(tile_start), str(tile_end), cell_type
    )
    if cached is None:
        interval = ag_genome.Interval(
            chromosome=chrom, start=tile_start, end=tile_end
        )
        logger.debug(
            "API call: %s:%d-%d for %s", chrom, tile_start, tile_end, cell_type
        )
        output = _predict_with_retry(
            model, interval, avail.requested_output_types, [cell_type]
        )
        save_cache(
            Path(cache_dir), output,
            chrom, str(tile_start), str(tile_end), cell_type,
        )
        time.sleep(COURTESY_SLEEP_S)
    else:
        logger.debug(
            "Cache hit: %s:%d-%d for %s", chrom, tile_start, tile_end, cell_type
        )
        output = cached

    return _extract_chromatin_features(output, site_offset, avail, window_bp)


def batch_get_chromatin_features(
    sites: pd.DataFrame,
    cell_type: str,
    model,
    cache_dir: Path,
    avail: Optional[TrackAvailability] = None,
    window_bp: int = DEFAULT_WINDOW_BP,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Batch AlphaGenome feature extraction with tiling, caching, and rate limiting.

    Groups sites into ``INTERVAL_SIZE``-bp tiles so that all sites within the
    same tile share a single API call.  Sites from different chromosomes are
    batched separately.  Processes chromosomes in sorted order.

    Parameters
    ----------
    sites : pd.DataFrame
        Must contain columns ``chrom`` (str) and ``start`` (int, 0-based).
        All other columns are preserved in the output.
    cell_type : str
        Cell-type ontology term, e.g. ``"EFO:0002067"``.
    model : dna_client.DnaClient
    cache_dir : Path
        Directory for caching raw API responses (created if absent).
    avail : TrackAvailability | None
        Track availability.  Queried automatically if ``None``.
    window_bp : int
        Feature extraction half-window in bp.
    show_progress : bool
        Log a progress message after each tile (useful for long jobs).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with one additional column per feature in
        :data:`FEATURE_NAMES`.  Row order matches the input.

    Notes
    -----
    If *cell_type* has no tracks (e.g. U2OS), all feature columns are NaN
    and no API calls are made.
    """
    from alphagenome.data import genome as ag_genome

    cache_dir = Path(cache_dir)

    if avail is None:
        avail = get_track_availability(model, cell_type)

    if not avail.has_any_tracks:
        logger.warning(
            "No tracks for %r — all %d sites will have NaN features",
            cell_type, len(sites),
        )
        result = sites.copy()
        for feat in FEATURE_NAMES:
            result[feat] = float("nan")
        return result

    # Compute tile assignments without modifying caller's DataFrame
    work = sites.copy()
    work["_chrom_norm"] = work["chrom"].apply(normalize_chrom)
    work["_tile_start"] = (work["start"] // INTERVAL_SIZE) * INTERVAL_SIZE
    work["_tile_end"] = work["_tile_start"] + INTERVAL_SIZE
    work["_site_offset"] = work["start"] - work["_tile_start"]

    # Unique tiles sorted for reproducibility
    tiles = (
        work[["_chrom_norm", "_tile_start", "_tile_end"]]
        .drop_duplicates()
        .sort_values(["_chrom_norm", "_tile_start"])
        .reset_index(drop=True)
    )
    n_tiles = len(tiles)
    logger.info(
        "batch_get_chromatin_features: %d sites → %d unique tiles, cell_type=%s",
        len(sites), n_tiles, cell_type,
    )

    # feature_rows maps DataFrame index → feature dict
    feature_rows: dict[int, dict[str, float]] = {}

    for tile_idx, tile_row in tiles.iterrows():
        chrom: str = tile_row["_chrom_norm"]
        tile_start: int = int(tile_row["_tile_start"])
        tile_end: int = int(tile_row["_tile_end"])

        cached = load_cache(
            cache_dir, chrom, str(tile_start), str(tile_end), cell_type
        )
        if cached is None:
            interval = ag_genome.Interval(
                chromosome=chrom, start=tile_start, end=tile_end
            )
            if show_progress:
                logger.info(
                    "Tile %d/%d API call: %s:%d-%d for %s",
                    int(tile_idx) + 1, n_tiles, chrom, tile_start, tile_end, cell_type,
                )
            output = _predict_with_retry(
                model, interval, avail.requested_output_types, [cell_type]
            )
            save_cache(
                cache_dir, output,
                chrom, str(tile_start), str(tile_end), cell_type,
            )
            time.sleep(COURTESY_SLEEP_S)
        else:
            if show_progress:
                logger.info(
                    "Tile %d/%d cache hit: %s:%d-%d for %s",
                    int(tile_idx) + 1, n_tiles, chrom, tile_start, tile_end, cell_type,
                )
            output = cached

        # Extract features for all sites in this tile
        tile_mask = (
            (work["_chrom_norm"] == chrom)
            & (work["_tile_start"] == tile_start)
        )
        for row_idx, site_row in work[tile_mask].iterrows():
            feature_rows[row_idx] = _extract_chromatin_features(
                output, int(site_row["_site_offset"]), avail, window_bp
            )

    # Assemble result preserving input row order
    result = sites.copy()
    for feat in FEATURE_NAMES:
        result[feat] = [feature_rows[idx][feat] for idx in result.index]

    return result
