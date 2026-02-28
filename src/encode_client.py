"""ENCODE experimental chromatin data retrieval for gold-standard model comparison.

Reads BigWig files from the ENCODE portal to extract experimental chromatin
accessibility (DNase-seq) and histone modification (ChIP-seq) signals at
CRISPR off-target loci.

This module provides the "gold standard" experimental features used to train
Model B in the SafeGenome three-model comparison. Model B uses actual
experimental measurements from ENCODE rather than AlphaGenome predictions.

Data source
-----------
ENCODE Project: https://www.encodeproject.org/
Files should be downloaded and placed in ``data/experimental_chromatin/``.
See :func:`list_required_files` for required filenames and
the project documentation for download instructions.

Expected filenames
------------------
::

    data/experimental_chromatin/
    ├── K562_DNase_hg38.bigWig
    ├── K562_H3K4me3_hg38.bigWig
    ├── K562_H3K27me3_hg38.bigWig
    ├── K562_H3K9me3_hg38.bigWig
    ├── K562_H3K27ac_hg38.bigWig
    ├── HEK293_DNase_hg38.bigWig      ← may not exist
    ├── HEK293_H3K4me3_hg38.bigWig   ← may not exist
    └── ...

Feature window
--------------
All features use a +/-500 bp window around the cut site, matching the
AlphaGenome feature extraction window in ``src/alphagenome_client.py``.

Scale mismatch note (P2.6 in PROJECT.md)
-----------------------------------------
AlphaGenome outputs and ENCODE BigWig files use DIFFERENT scales and
normalizations.  Never directly compare feature values across the two
modules.  Compare only model PERFORMANCE (AUPRC) between Model B (ENCODE
features) and Model C (AlphaGenome features).  For correlation checks,
use Spearman rank correlation (not Pearson).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils import GenomicInterval, normalize_chrom

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default half-window in bp for feature extraction (matches AlphaGenome client).
DEFAULT_WINDOW_BP: int = 500

#: Half-window in bp for the narrow at-site feature (mirrors AlphaGenome).
NARROW_WINDOW_BP: int = 50

#: DNase mean threshold to call a site "accessible" (exp_accessibility_binary).
#: Matches the same threshold used in alphagenome_client.DNASE_OPEN_THRESHOLD.
DNASE_OPEN_THRESHOLD: float = 1.0

#: Ordered list of feature column names produced per site.
FEATURE_NAMES: list[str] = [
    "exp_dnase_mean",           # Mean DNase signal over +/-window_bp
    "exp_dnase_max",            # Max DNase signal over +/-window_bp
    "exp_dnase_at_site",        # DNase signal at exact cut site (0-based start)
    "exp_h3k4me3_mean",         # Mean H3K4me3 signal (active promoter mark)
    "exp_h3k27me3_mean",        # Mean H3K27me3 signal (Polycomb repression)
    "exp_h3k9me3_mean",         # Mean H3K9me3 signal (constitutive heterochromatin)
    "exp_h3k27ac_mean",         # Mean H3K27ac signal (active enhancer/promoter)
    "exp_accessibility_binary", # 1 if exp_dnase_mean > DNASE_OPEN_THRESHOLD
]

#: Recognised assay names — determines which BigWig files are scanned.
ASSAY_NAMES: tuple[str, ...] = ("DNase", "H3K4me3", "H3K27me3", "H3K9me3", "H3K27ac")

#: Filename template used when scanning the BigWig directory.
FILE_TEMPLATE: str = "{cell_type}_{assay}_hg38.bigWig"

#: Map from assay name to the single feature name it populates (histones only).
_HISTONE_MAP: dict[str, str] = {
    "H3K4me3":  "exp_h3k4me3_mean",
    "H3K27me3": "exp_h3k27me3_mean",
    "H3K9me3":  "exp_h3k9me3_mean",
    "H3K27ac":  "exp_h3k27ac_mean",
}


# ---------------------------------------------------------------------------
# BigWigManifest — file registry
# ---------------------------------------------------------------------------

@dataclass
class BigWigManifest:
    """Registry mapping ``(cell_type, assay)`` pairs to BigWig file paths.

    Build this once at pipeline startup via :func:`build_manifest` and pass
    it to :func:`get_chromatin_features` / :func:`batch_get_chromatin_features`.

    Parameters
    ----------
    bigwig_dir : Path
        Root directory that was scanned to build the manifest.
    files : dict[tuple[str, str], Path]
        Mapping of ``(cell_type, assay)`` to absolute ``Path`` objects.
        Populated by :func:`build_manifest`.

    Notes
    -----
    Missing assays are represented by their absence from *files*, not by a
    sentinel value.  Missing assays yield NaN features — not an error.
    """

    bigwig_dir: Path
    files: dict[tuple[str, str], Path] = field(default_factory=dict)

    def get(self, cell_type: str, assay: str) -> Optional[Path]:
        """Return the Path for ``(cell_type, assay)``, or ``None`` if absent."""
        return self.files.get((cell_type, assay))

    def has_any(self, cell_type: str) -> bool:
        """Return ``True`` if at least one BigWig file exists for *cell_type*."""
        return any(ct == cell_type for ct, _ in self.files)

    def available_cell_types(self) -> list[str]:
        """Sorted list of cell types with at least one BigWig file."""
        return sorted({ct for ct, _ in self.files})

    def available_assays(self, cell_type: str) -> list[str]:
        """Sorted list of assays available for *cell_type*."""
        return sorted(assay for ct, assay in self.files if ct == cell_type)


def build_manifest(bigwig_dir: Path) -> BigWigManifest:
    """Scan *bigwig_dir* for BigWig files and build a :class:`BigWigManifest`.

    Recognises files matching ``{cell_type}_{assay}_hg38.bigWig`` for
    ``cell_type`` in ``("K562", "HEK293")`` and ``assay`` in
    :data:`ASSAY_NAMES`.  Unknown files are silently ignored.

    Parameters
    ----------
    bigwig_dir : Path
        Directory containing BigWig files (e.g. ``data/experimental_chromatin/``).

    Returns
    -------
    BigWigManifest
        Populated manifest.  If the directory does not exist or has no
        recognised files, an empty manifest is returned and a WARNING is logged.
    """
    bigwig_dir = Path(bigwig_dir)
    manifest = BigWigManifest(bigwig_dir=bigwig_dir)

    if not bigwig_dir.exists():
        logger.warning(
            "BigWig directory does not exist: %s — all features will be NaN",
            bigwig_dir,
        )
        return manifest

    found = 0
    for cell_type in ("K562", "HEK293"):
        for assay in ASSAY_NAMES:
            fname = FILE_TEMPLATE.format(cell_type=cell_type, assay=assay)
            path = bigwig_dir / fname
            if path.exists():
                manifest.files[(cell_type, assay)] = path
                logger.info(
                    "BigWig manifest: %s/%s → %s", cell_type, assay, fname
                )
                found += 1

    if found == 0:
        logger.warning(
            "No recognised BigWig files found in %s — all features will be NaN",
            bigwig_dir,
        )
    else:
        logger.info("BigWig manifest: %d file(s) indexed", found)

    return manifest


def list_required_files() -> list[str]:
    """Return expected filenames for all BigWig files.

    Use this to check which files still need to be downloaded.

    Returns
    -------
    list[str]
        Filenames (relative to ``data/experimental_chromatin/``), one per
        cell-type × assay combination.
    """
    return [
        FILE_TEMPLATE.format(cell_type=ct, assay=assay)
        for ct in ("K562", "HEK293")
        for assay in ASSAY_NAMES
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open_bigwig(path: Path) -> Any:
    """Open a BigWig file with pybigtools.

    Parameters
    ----------
    path : Path
        Absolute path to the ``.bigWig`` file.

    Returns
    -------
    pybigtools.BBIRead
        Open file handle.  Caller is responsible for calling ``.close()``.

    Raises
    ------
    ImportError
        If ``pybigtools`` is not installed.
    FileNotFoundError
        If *path* does not exist.
    """
    try:
        import pybigtools  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pybigtools is required for ENCODE feature extraction. "
            "Install with: pip install pybigtools"
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"BigWig file not found: {path}")

    return pybigtools.open(str(path))


def _query_region(bw: Any, chrom: str, start: int, end: int) -> np.ndarray:
    """Query a BigWig file over ``[start, end)``, returning NaN for gaps.

    pybigtools returns a ``numpy.ndarray`` where uncovered positions are filled
    with the ``missing`` value (set to NaN here).

    Chromosome boundary clamping is handled: if ``start < 0`` or
    ``end > chrom_length``, the out-of-bounds region is filled with NaN.

    Parameters
    ----------
    bw : pybigtools.BBIRead
        Open BigWig file handle.
    chrom : str
        Chromosome name (``chr``-prefixed, e.g. ``"chr7"``).
    start : int
        0-based inclusive start position.
    end : int
        0-based exclusive end position.

    Returns
    -------
    np.ndarray
        Float64 array of length ``end - start``, with NaN for uncovered
        positions.  Returns an all-NaN array if *chrom* is absent from
        the BigWig or if the requested range is entirely out of bounds.
    """
    n = end - start
    if n <= 0:
        return np.array([], dtype=np.float64)

    chroms = bw.chroms()
    if chrom not in chroms:
        logger.debug("Chrom %s not in BigWig; returning NaN array", chrom)
        return np.full(n, np.nan, dtype=np.float64)

    chrom_len: int = chroms[chrom]
    actual_start = max(0, start)
    actual_end = min(chrom_len, end)

    if actual_start >= actual_end:
        return np.full(n, np.nan, dtype=np.float64)

    # pybigtools returns a numpy array directly; missing=nan fills uncovered positions
    arr = np.asarray(
        bw.values(chrom, actual_start, actual_end, missing=float("nan")),
        dtype=np.float64,
    )

    # Pad with NaN if the query was clamped at chromosome boundaries.
    if actual_start > start or actual_end < end:
        padded = np.full(n, np.nan, dtype=np.float64)
        pad_offset = actual_start - start
        padded[pad_offset : pad_offset + len(arr)] = arr
        return padded

    return arr


def _extract_signal_stats(
    arr: np.ndarray,
    site_offset: int,
    window_bp: int,
) -> tuple[float, float, float]:
    """Extract summary statistics from a signal array.

    Parameters
    ----------
    arr : np.ndarray
        1D float64 signal array (may contain NaN).
    site_offset : int
        0-based index of the cut site within *arr*.
    window_bp : int
        Half-width of the extraction window in base pairs.

    Returns
    -------
    tuple[float, float, float]
        ``(mean, max, at_site)`` over the window.  All NaN if *arr* is
        empty, the window is empty, or all values are NaN.
        ``at_site`` is NaN if *site_offset* is out of bounds.

    Notes
    -----
    Uses ``np.nanmean`` / ``np.nanmax`` so that sparse coverage (isolated
    None values from pyBigWig) does not dominate the statistics.
    """
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    w_start = max(0, site_offset - window_bp)
    w_end = min(n, site_offset + window_bp)
    window = arr[w_start:w_end]

    if len(window) == 0:
        return float("nan"), float("nan"), float("nan")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_v = float(np.nanmean(window))
        max_v = float(np.nanmax(window))

    at_site = float(arr[site_offset]) if 0 <= site_offset < n else float("nan")
    return mean_v, max_v, at_site


def _nan_features() -> dict[str, float]:
    """Return a feature dict with all values set to NaN."""
    return {name: float("nan") for name in FEATURE_NAMES}


def _extract_from_open_bws(
    open_bws: dict[str, Any],
    chrom: str,
    site_pos: int,
    window_bp: int,
) -> dict[str, float]:
    """Extract the full chromatin feature vector from already-open BigWig handles.

    This is the core extraction routine shared by both
    :func:`get_chromatin_features` (single-site) and
    :func:`batch_get_chromatin_features` (batch, files opened once).

    Parameters
    ----------
    open_bws : dict[str, Any]
        Mapping of assay name (e.g. ``"DNase"``) to open pyBigWig handle.
    chrom : str
        Normalised chromosome name (``chr``-prefixed).
    site_pos : int
        0-based position of the cut site on *chrom*.
    window_bp : int
        Half-width of the feature extraction window in bp.

    Returns
    -------
    dict[str, float]
        Feature dictionary with keys matching :data:`FEATURE_NAMES`.
        Assays absent from *open_bws* yield NaN values.
    """
    feats = _nan_features()

    # The query window starts at max(0, site_pos - window_bp).
    # site_offset is the index of the cut site within the returned array.
    q_start = max(0, site_pos - window_bp)
    q_end = site_pos + window_bp
    site_offset = site_pos - q_start  # always >= 0

    # ---- DNase: produces mean, max, at_site, and accessibility_binary
    if "DNase" in open_bws:
        arr = _query_region(open_bws["DNase"], chrom, q_start, q_end)
        mean_v, max_v, at_site = _extract_signal_stats(arr, site_offset, window_bp)
        feats["exp_dnase_mean"] = mean_v
        feats["exp_dnase_max"] = max_v
        feats["exp_dnase_at_site"] = at_site
        # Match alphagenome_client convention: nan > threshold = False → 0.0
        feats["exp_accessibility_binary"] = float(mean_v > DNASE_OPEN_THRESHOLD)

    # ---- Histone marks: each assay → single mean feature
    for assay, feat_name in _HISTONE_MAP.items():
        if assay in open_bws:
            arr = _query_region(open_bws[assay], chrom, q_start, q_end)
            mean_v, _, _ = _extract_signal_stats(arr, site_offset, window_bp)
            feats[feat_name] = mean_v

    return feats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_chromatin_features(
    site: GenomicInterval,
    cell_type: str,
    manifest: BigWigManifest,
    window_bp: int = DEFAULT_WINDOW_BP,
) -> dict[str, float]:
    """Extract ENCODE experimental chromatin features for a single off-target site.

    Opens each BigWig file, queries the region, and closes the file.  For
    processing many sites, prefer :func:`batch_get_chromatin_features` which
    keeps files open across all queries.

    Parameters
    ----------
    site : GenomicInterval
        Off-target locus in 0-based half-open coordinates (hg38).
        The cut site is taken as ``site.start``.
    cell_type : str
        Cell type name exactly as indexed in the manifest, e.g. ``"K562"``
        or ``"HEK293"``.
    manifest : BigWigManifest
        BigWig file manifest built by :func:`build_manifest`.
    window_bp : int
        Half-width of the feature extraction window in bp.

    Returns
    -------
    dict[str, float]
        Feature dictionary with keys matching :data:`FEATURE_NAMES`.
        Missing files or missing assays yield NaN values.

    Notes
    -----
    Logs a WARNING if no BigWig files are found for *cell_type*.
    Per-assay open errors are also warned and result in NaN for that assay.
    """
    chrom = normalize_chrom(site.chrom)

    if not manifest.has_any(cell_type):
        logger.warning(
            "No BigWig files for cell type %r — returning NaN for %s",
            cell_type,
            site,
        )
        return _nan_features()

    open_bws: dict[str, Any] = {}
    try:
        for assay in ASSAY_NAMES:
            path = manifest.get(cell_type, assay)
            if path is not None:
                try:
                    open_bws[assay] = _open_bigwig(path)
                except Exception as exc:
                    logger.warning(
                        "Could not open %s/%s BigWig: %s", cell_type, assay, exc
                    )

        return _extract_from_open_bws(open_bws, chrom, site.start, window_bp)

    finally:
        for bw in open_bws.values():
            try:
                bw.close()
            except Exception:
                pass


def batch_get_chromatin_features(
    sites: pd.DataFrame,
    cell_type: str,
    manifest: BigWigManifest,
    window_bp: int = DEFAULT_WINDOW_BP,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Batch ENCODE feature extraction for a DataFrame of off-target sites.

    Opens BigWig files **once** for the entire batch (unlike
    :func:`get_chromatin_features` which opens/closes per site).  This is
    the recommended function for processing more than a handful of sites.

    Parameters
    ----------
    sites : pd.DataFrame
        Must contain columns ``chrom`` (str) and ``start`` (int, 0-based).
        All other columns are preserved in the output.
    cell_type : str
        Cell type name, e.g. ``"K562"`` or ``"HEK293"``.
    manifest : BigWigManifest
        BigWig file manifest built by :func:`build_manifest`.
    window_bp : int
        Feature extraction half-window in bp.
    show_progress : bool
        Log a progress message every 1 000 sites.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with one additional column per feature in
        :data:`FEATURE_NAMES`.  Row order matches the input.

    Notes
    -----
    If *cell_type* has no BigWig files, all feature columns are NaN and
    no files are opened.
    """
    n_sites = len(sites)
    logger.info(
        "batch_get_chromatin_features: %d sites, cell_type=%s", n_sites, cell_type
    )

    result = sites.copy()

    if not manifest.has_any(cell_type):
        logger.warning(
            "No BigWig files for %r — all %d sites will have NaN features",
            cell_type,
            n_sites,
        )
        for feat in FEATURE_NAMES:
            result[feat] = float("nan")
        return result

    # Open all available BigWig files once for the entire batch.
    open_bws: dict[str, Any] = {}
    try:
        for assay in ASSAY_NAMES:
            path = manifest.get(cell_type, assay)
            if path is not None:
                try:
                    open_bws[assay] = _open_bigwig(path)
                    logger.debug("Opened %s/%s BigWig", cell_type, assay)
                except Exception as exc:
                    logger.warning(
                        "Could not open %s/%s BigWig: %s", cell_type, assay, exc
                    )

        feature_rows: list[dict[str, float]] = []
        for i, (_, row) in enumerate(sites.iterrows()):
            if show_progress and (i % 1000 == 0 or i == n_sites - 1):
                logger.info(
                    "batch_get_chromatin_features: site %d/%d for %s",
                    i + 1,
                    n_sites,
                    cell_type,
                )
            chrom = normalize_chrom(str(row["chrom"]))
            site_pos = int(row["start"])
            feature_rows.append(
                _extract_from_open_bws(open_bws, chrom, site_pos, window_bp)
            )

    finally:
        for bw in open_bws.values():
            try:
                bw.close()
            except Exception:
                pass

    for feat in FEATURE_NAMES:
        result[feat] = [fr[feat] for fr in feature_rows]

    return result
