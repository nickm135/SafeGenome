"""
scripts/09_bootstrap_test.py — Paired bootstrap significance test for AUPRC.

Uses pooled out-of-fold predictions from LOGO CV to compare Models A, B, C.
Resamples sites with replacement (10,000 iterations), computes AUPRC on each
resample, and reports p-value as the fraction of resamples where the inferior
model's AUPRC >= the superior model's.

Requires
--------
* data/models/cv_results.pkl  (from script 05)

Outputs
-------
    data/results/bootstrap_test.csv
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import get_logger

logger = get_logger("bootstrap_test")

_ROOT = Path(__file__).resolve().parents[1]
_MODELS = _ROOT / "data" / "models"
_RESULTS = _ROOT / "data" / "results"

N_BOOTSTRAP = 10_000
RANDOM_STATE = 42


def paired_bootstrap_auprc(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Paired bootstrap test comparing AUPRC of two score vectors.

    Parameters
    ----------
    y_true : np.ndarray
        Shared ground truth labels (binary).
    scores_a : np.ndarray
        Predicted scores from model A (the inferior / reference model).
    scores_b : np.ndarray
        Predicted scores from model B (the superior / test model).
    n_bootstrap : int
        Number of bootstrap resamples.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        auprc_a : float — observed AUPRC for model A
        auprc_b : float — observed AUPRC for model B
        delta : float — auprc_b - auprc_a (positive = B better)
        p_value : float — fraction of resamples where A's AUPRC >= B's AUPRC
        n_resamples : int
        n_sites : int
    """
    n = len(y_true)
    assert len(scores_a) == n and len(scores_b) == n

    observed_a = float(average_precision_score(y_true, scores_a))
    observed_b = float(average_precision_score(y_true, scores_b))

    rng = np.random.RandomState(random_state)
    count_a_geq_b = 0

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_boot = y_true[idx]

        # Skip degenerate resamples (all one class)
        if y_boot.sum() == 0 or y_boot.sum() == len(y_boot):
            continue

        ap_a = average_precision_score(y_boot, scores_a[idx])
        ap_b = average_precision_score(y_boot, scores_b[idx])

        if ap_a >= ap_b:
            count_a_geq_b += 1

    p_value = count_a_geq_b / n_bootstrap

    return {
        "auprc_a": observed_a,
        "auprc_b": observed_b,
        "delta": observed_b - observed_a,
        "p_value": p_value,
        "n_resamples": n_bootstrap,
        "n_sites": n,
    }


def main() -> None:
    _RESULTS.mkdir(parents=True, exist_ok=True)

    # Load CV results
    cv_path = _MODELS / "cv_results.pkl"
    logger.info("Loading CV results: %s", cv_path)
    with open(cv_path, "rb") as f:
        cv_results = pickle.load(f)

    # Extract OOF predictions
    oof = {}
    for key in ("A", "B", "C"):
        r = cv_results[key]
        oof[key] = {
            "y_true": r.oof_y_true,
            "y_score": r.oof_y_score,
        }
        logger.info(
            "Model %s OOF: %d sites, %d positives, AUPRC=%.4f",
            key, len(r.oof_y_true), int(r.oof_y_true.sum()),
            float(average_precision_score(r.oof_y_true, r.oof_y_score)),
        )

    # Verify OOF arrays are aligned (same y_true across models)
    assert np.array_equal(oof["A"]["y_true"], oof["B"]["y_true"]), \
        "OOF y_true mismatch between A and B"
    assert np.array_equal(oof["A"]["y_true"], oof["C"]["y_true"]), \
        "OOF y_true mismatch between A and C"
    logger.info("OOF y_true arrays verified identical across all models")

    y_true = oof["A"]["y_true"]

    # Run bootstrap tests
    comparisons = [
        ("A", "C"),  # A is inferior, C is superior
        ("B", "C"),  # B is inferior, C is superior
        ("A", "B"),  # A is inferior, B is superior
    ]

    rows: list[dict] = []
    for ref_key, test_key in comparisons:
        logger.info(
            "Bootstrap test: %s vs %s (%d resamples) ...",
            ref_key, test_key, N_BOOTSTRAP,
        )
        result = paired_bootstrap_auprc(
            y_true,
            oof[ref_key]["y_score"],
            oof[test_key]["y_score"],
            n_bootstrap=N_BOOTSTRAP,
            random_state=RANDOM_STATE,
        )
        rows.append({
            "comparison": f"{ref_key}_vs_{test_key}",
            "auprc_inferior": result["auprc_a"],
            "auprc_superior": result["auprc_b"],
            "delta_auprc": result["delta"],
            "bootstrap_p": result["p_value"],
            "n_resamples": result["n_resamples"],
            "n_sites": result["n_sites"],
            "significant_005": result["p_value"] < 0.05,
        })
        logger.info(
            "  %s vs %s: delta=%.4f, p=%.4f (%s)",
            ref_key, test_key, result["delta"], result["p_value"],
            "significant" if result["p_value"] < 0.05 else "not significant",
        )

    df = pd.DataFrame(rows)
    out_path = _RESULTS / "bootstrap_test.csv"
    df.to_csv(out_path, index=False)
    logger.info("Results saved -> %s", out_path)

    # Print summary
    print("\n" + "=" * 80)
    print("Paired Bootstrap Significance Test (AUPRC, 10,000 resamples, seed=42)")
    print("=" * 80)
    print(f"  Sites: {len(y_true)}, Positives: {int(y_true.sum())} ({100*y_true.mean():.1f}%)")
    print()
    print(f"{'Comparison':<12} {'AUPRC_ref':>10} {'AUPRC_test':>11} {'Delta':>8} {'p-value':>10} {'Sig':>5}")
    print("-" * 62)
    for _, row in df.iterrows():
        sig = "*" if row["significant_005"] else ""
        print(
            f"{row['comparison']:<12} "
            f"{row['auprc_inferior']:>10.4f} "
            f"{row['auprc_superior']:>11.4f} "
            f"{row['delta_auprc']:>+8.4f} "
            f"{row['bootstrap_p']:>10.4f} "
            f"{sig:>5}"
        )
    print("-" * 62)
    print("  p = fraction of resamples where inferior model AUPRC >= superior")
    print("  * = p < 0.05")
    print("=" * 80)


if __name__ == "__main__":
    main()
