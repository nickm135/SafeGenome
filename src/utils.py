"""
utils.py — Shared helpers for the SafeGenome pipeline.

Provides:
- Logging configuration (get_logger, log_filter)
- GenomicInterval dataclass in 0-based half-open (BED) coordinates
- Chromosome name normalisation and hg38 validation
- hg19 → hg38 coordinate liftover via pyliftover
- Disk-based pickle cache (used by alphagenome_client)
- Guide RNA sequence normalisation
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Standard hg38 chromosome names, chr-prefixed.
VALID_CHROMS_HG38: frozenset[str] = frozenset(
    [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with a standard timestamp format.

    Parameters
    ----------
    name : str
        Logger name — pass ``__name__`` from the calling module.
    level : int
        Logging level.  Default is ``logging.INFO``.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_filter(
    logger: logging.Logger,
    reason: str,
    n_before: int,
    n_after: int,
) -> None:
    """Log a data-filtering step with row counts.

    Required by project rules: every row drop must be logged so the
    counts can appear in the paper's Methods section.

    Parameters
    ----------
    logger : logging.Logger
    reason : str
        Human-readable explanation, e.g. ``"remove on-target sites"``.
    n_before : int
        Row count before the filter.
    n_after : int
        Row count after the filter.
    """
    n_dropped = n_before - n_after
    pct = 100.0 * n_dropped / n_before if n_before > 0 else 0.0
    logger.info(
        "FILTER | %s | dropped %d / %d rows (%.1f%%) | %d remaining",
        reason,
        n_dropped,
        n_before,
        pct,
        n_after,
    )


# ---------------------------------------------------------------------------
# GenomicInterval
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenomicInterval:
    """A genomic locus in 0-based half-open (BED) coordinates.

    This is the **only** coordinate representation used internally.
    All inputs are converted to this form on ingestion; all outputs are
    converted back to 1-based inclusive when written for human consumption.

    The interval ``[start, end)`` spans ``end - start`` bases.

    Parameters
    ----------
    chrom : str
        Chromosome name with ``chr`` prefix, e.g. ``"chr1"``.
    start : int
        0-based inclusive start position.
    end : int
        0-based exclusive end position.
    strand : str
        ``"+"`` or ``"-"``; use ``"."`` when strand is unknown.
    """

    chrom: str
    start: int
    end: int
    strand: str = "."

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")
        if self.strand not in ("+", "-", "."):
            raise ValueError(
                f"strand must be '+', '-', or '.', got {self.strand!r}"
            )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_1based(
        cls,
        chrom: str,
        start: int,
        end: int,
        strand: str = ".",
    ) -> "GenomicInterval":
        """Create from 1-based inclusive coordinates (VCF / GFF / most papers).

        Converts: ``start_0based = start_1based - 1``, end is unchanged.

        Parameters
        ----------
        chrom : str
        start : int
            1-based inclusive start.
        end : int
            1-based inclusive end.
        strand : str

        Returns
        -------
        GenomicInterval
        """
        return cls(chrom=chrom, start=start - 1, end=end, strand=strand)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def length(self) -> int:
        """Number of bases in the interval."""
        return self.end - self.start

    @property
    def center(self) -> int:
        """0-based center position (rounds down for even-length intervals)."""
        return (self.start + self.end) // 2

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_1based(self) -> tuple[str, int, int, str]:
        """Return ``(chrom, start, end, strand)`` in 1-based inclusive form.

        Returns
        -------
        tuple[str, int, int, str]
        """
        return self.chrom, self.start + 1, self.end, self.strand

    def expand(self, half_width: int) -> "GenomicInterval":
        """Return a new interval centred on ``self.center`` ± ``half_width``.

        Clamps start to 0 if the expansion would go negative.

        Parameters
        ----------
        half_width : int
            Number of bases to extend in each direction from centre.

        Returns
        -------
        GenomicInterval
        """
        c = self.center
        new_start = max(0, c - half_width)
        new_end = c + half_width
        return GenomicInterval(self.chrom, new_start, new_end, self.strand)

    def __str__(self) -> str:
        return f"{self.chrom}:{self.start}-{self.end}({self.strand})"


# ---------------------------------------------------------------------------
# Chromosome normalisation and hg38 validation
# ---------------------------------------------------------------------------

def normalize_chrom(chrom: str) -> str:
    """Normalise a chromosome name to standard hg38 form (``chr``-prefixed).

    Handles inputs such as ``"1"``, ``"chr1"``, ``"Chr1"``, ``"chrx"``,
    ``"MT"``, ``"chrMT"``.

    Parameters
    ----------
    chrom : str
        Raw chromosome identifier from any source.

    Returns
    -------
    str
        Normalised name, e.g. ``"chr1"``, ``"chrX"``, ``"chrM"``.

    Raises
    ------
    ValueError
        If the normalised name is not among the 25 standard hg38 sequences.
    """
    chrom = chrom.strip()

    # Ensembl / UCSC mitochondrial aliases
    if chrom.upper() in ("MT", "CHRMT"):
        return "chrM"

    # Add prefix if missing
    if not chrom.lower().startswith("chr"):
        chrom = "chr" + chrom

    # Normalise X/Y/M suffix capitalisation; leave numeric suffixes as-is
    suffix = chrom[3:]
    if suffix.upper() in ("X", "Y", "M"):
        chrom = "chr" + suffix.upper()
    else:
        chrom = "chr" + suffix

    if chrom not in VALID_CHROMS_HG38:
        raise ValueError(
            f"'{chrom}' is not a valid hg38 chromosome. "
            f"Valid values: chr1–chr22, chrX, chrY, chrM."
        )
    return chrom


def is_valid_hg38_chrom(chrom: str) -> bool:
    """Return ``True`` if *chrom* resolves to a valid hg38 chromosome.

    Parameters
    ----------
    chrom : str

    Returns
    -------
    bool
    """
    try:
        normalize_chrom(chrom)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# hg19 → hg38 liftover
# ---------------------------------------------------------------------------

# Module-level singleton — expensive to initialise (downloads chain file once).
_liftover_instance: Any = None


def _get_liftover_instance() -> Any:
    """Return a cached pyliftover ``LiftOver`` instance (hg19 → hg38).

    Downloads the chain file on the first call; subsequent calls are free.

    Raises
    ------
    ImportError
        If ``pyliftover`` is not installed.
    """
    global _liftover_instance
    if _liftover_instance is None:
        try:
            from pyliftover import LiftOver  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pyliftover is required for coordinate liftover. "
                "Install with: pip install pyliftover"
            ) from exc
        _liftover_instance = LiftOver("hg19", "hg38")
    return _liftover_instance


def liftover_hg19_to_hg38(chrom: str, pos: int) -> tuple[str, int] | None:
    """Convert a single 0-based position from hg19 to hg38.

    Parameters
    ----------
    chrom : str
        hg19 chromosome name (``"chr"`` prefix is added if absent).
    pos : int
        0-based position on the chromosome.

    Returns
    -------
    tuple[str, int] | None
        ``(hg38_chrom, hg38_pos_0based)``, or ``None`` if the position
        cannot be mapped (e.g. falls in a gap or deleted region).

    Raises
    ------
    ImportError
        If ``pyliftover`` is not installed.
    """
    if not chrom.startswith("chr"):
        chrom = "chr" + chrom

    lo = _get_liftover_instance()
    result = lo.convert_coordinate(chrom, pos)
    if not result:
        return None

    new_chrom, new_pos, _strand, _score = result[0]
    return new_chrom, new_pos


# ---------------------------------------------------------------------------
# Guide RNA sequence normalisation
# ---------------------------------------------------------------------------

def normalize_guide(sequence: str) -> str:
    """Normalise a guide RNA spacer to a 20-nt uppercase DNA string.

    Handles:
    - Whitespace stripping
    - Uppercase conversion
    - RNA → DNA (U → T)
    - PAM suffix stripping (accepts 20-nt or 23-nt input; strips last 3 if 23)

    Parameters
    ----------
    sequence : str
        Raw guide sequence — 20-nt spacer or 23-nt spacer+PAM.

    Returns
    -------
    str
        20-nt normalised DNA spacer sequence.

    Raises
    ------
    ValueError
        If the resulting spacer is not exactly 20 nucleotides, or if it
        contains characters other than A, C, G, T.
    """
    seq = sequence.strip().upper().replace("U", "T")

    if len(seq) == 23:
        # Strip PAM (last 3 nt)
        seq = seq[:20]

    if len(seq) != 20:
        raise ValueError(
            f"Guide sequence must be 20 nt (or 23 nt with PAM) after "
            f"normalisation, got {len(seq)} nt: {sequence!r}"
        )

    invalid = set(seq) - set("ACGT")
    if invalid:
        raise ValueError(
            f"Guide sequence contains invalid characters {invalid!r}: {seq!r}"
        )

    return seq


# ---------------------------------------------------------------------------
# Disk-based pickle cache
# ---------------------------------------------------------------------------

def _make_cache_filename(*key_parts: str) -> str:
    """Create a filesystem-safe filename from arbitrary key parts.

    Uses a short human-readable prefix plus a 12-character SHA-256 digest
    to guarantee uniqueness without exceeding path length limits.

    Parameters
    ----------
    *key_parts : str

    Returns
    -------
    str
        Filename without extension, e.g.
        ``"chr1__1000__2000__K562__a3f9b812c441"``.
    """
    raw = "__".join(str(p) for p in key_parts)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    prefix = re.sub(r"[^\w-]", "_", raw)[:64]
    return f"{prefix}__{digest}"


def load_cache(cache_dir: Path, *key_parts: str) -> Any | None:
    """Load a cached object from disk.

    Parameters
    ----------
    cache_dir : Path
        Directory containing ``.pkl`` cache files.
    *key_parts : str
        Strings that together uniquely identify the cached item.

    Returns
    -------
    Any | None
        The deserialised object, or ``None`` if no cache entry exists.
    """
    path = cache_dir / (_make_cache_filename(*key_parts) + ".pkl")
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except (EOFError, pickle.UnpicklingError, Exception):
        # File was truncated or corrupted (e.g. process killed mid-write).
        # Treat as a cache miss and remove the bad file so it gets rewritten.
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "Corrupt cache file removed (will refetch): %s", path.name
        )
        try:
            path.unlink()
        except OSError:
            pass
        return None


def save_cache(cache_dir: Path, data: Any, *key_parts: str) -> Path:
    """Serialise *data* to disk as a pickle file.

    Creates *cache_dir* (and any parents) if it does not already exist.

    Parameters
    ----------
    cache_dir : Path
        Directory for cache files.
    data : Any
        Object to cache — must be picklable.
    *key_parts : str
        Strings that together uniquely identify the item.

    Returns
    -------
    Path
        Absolute path of the written cache file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (_make_cache_filename(*key_parts) + ".pkl")
    with path.open("wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path
