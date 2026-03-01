"""
Power analysis for SafeGenome guide-level paired t-tests.

Quantifies the impact of expanding from 16 guides (current) to ~50+ guides
(with Lazzarotto 2020 T cell GUIDE-seq data) on statistical power for
detecting AUPRC differences between models A, B, and C.

Uses the canonical no-DISCOVER-seq results (16 guides).
"""

import numpy as np
from scipy import stats
import csv
from pathlib import Path


def paired_ttest_power(
    delta: float, sd_delta: float, n: int, alpha: float = 0.05
) -> float:
    """
    Compute power of a two-sided paired t-test.

    Uses the noncentral t-distribution:
      - t-statistic under H1: noncentrality parameter = delta / (sd_delta / sqrt(n))
      - Reject H0 if |t| > t_crit(alpha/2, df=n-1)
      - Power = P(|T_nc| > t_crit) = 1 - P(-t_crit < T_nc < t_crit)

    Parameters
    ----------
    delta : float
        True mean of paired differences (effect).
    sd_delta : float
        Standard deviation of paired differences.
    n : int
        Number of paired observations (guides).
    alpha : float
        Significance level (two-sided).

    Returns
    -------
    float
        Statistical power.
    """
    if n < 2:
        return np.nan
    df = n - 1
    # Noncentrality parameter
    ncp = delta / (sd_delta / np.sqrt(n))
    # Critical value
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    # Power: P(T_nc > t_crit) + P(T_nc < -t_crit)
    # Using the noncentral t distribution
    power = 1 - (stats.nct.cdf(t_crit, df, ncp) - stats.nct.cdf(-t_crit, df, ncp))
    return power


def required_effect_for_power(
    sd_delta: float, n: int, target_power: float = 0.80, alpha: float = 0.05
) -> float:
    """
    Find the minimum delta needed to achieve target_power at sample size n.

    Uses bisection search.

    Parameters
    ----------
    sd_delta : float
        Standard deviation of paired differences.
    n : int
        Number of paired observations.
    target_power : float
        Desired power level.
    alpha : float
        Significance level (two-sided).

    Returns
    -------
    float
        Minimum delta for target_power.
    """
    lo, hi = 0.0, 2.0  # AUPRC differences are bounded [0, 2]
    for _ in range(200):  # bisection converges fast
        mid = (lo + hi) / 2
        p = paired_ttest_power(mid, sd_delta, n, alpha)
        if p < target_power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> None:
    # ---- Load actual per-guide AUPRCs ----
    data_path = Path("data/results/per_guide_auprc_no_discoverseq.csv")
    guide_ids = []
    auprc_a = []
    auprc_b = []
    auprc_c = []

    with open(data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = row["guide_id"]
            if gid in ("median", "IQR_25", "IQR_75"):
                continue
            guide_ids.append(gid)
            auprc_a.append(float(row["auprc_A"]))
            auprc_b.append(float(row["auprc_B"]))
            auprc_c.append(float(row["auprc_C"]))

    auprc_a = np.array(auprc_a)
    auprc_b = np.array(auprc_b)
    auprc_c = np.array(auprc_c)

    n_current = len(guide_ids)

    # ---- Observed statistics ----
    delta_ac = auprc_c - auprc_a  # per-guide C minus A
    delta_bc = auprc_c - auprc_b  # per-guide C minus B

    mean_delta_ac = np.mean(delta_ac)
    sd_delta_ac = np.std(delta_ac, ddof=1)
    cohens_d_ac = mean_delta_ac / sd_delta_ac

    mean_delta_bc = np.mean(delta_bc)
    sd_delta_bc = np.std(delta_bc, ddof=1)
    cohens_d_bc = mean_delta_bc / sd_delta_bc

    # Verify against reported statistics
    t_ac, p_ac = stats.ttest_rel(auprc_c, auprc_a)
    t_bc, p_bc = stats.ttest_rel(auprc_c, auprc_b)

    print("=" * 80)
    print("POWER ANALYSIS: SafeGenome Guide-Level Paired t-Tests")
    print("Canonical results: 16-guide, no-DISCOVER-seq")
    print("=" * 80)

    print("\n--- OBSERVED STATISTICS (n=16 guides) ---\n")
    print(f"A vs C comparison (C - A):")
    print(f"  Mean delta      = {mean_delta_ac:.6f}")
    print(f"  SD of deltas    = {sd_delta_ac:.6f}")
    print(f"  Cohen's d       = {cohens_d_ac:.4f}")
    print(f"  t-statistic     = {t_ac:.4f}")
    print(f"  p-value (2-sid) = {p_ac:.4f}")
    print()
    print(f"B vs C comparison (C - B):")
    print(f"  Mean delta      = {mean_delta_bc:.6f}")
    print(f"  SD of deltas    = {sd_delta_bc:.6f}")
    print(f"  Cohen's d       = {cohens_d_bc:.4f}")
    print(f"  t-statistic     = {t_bc:.4f}")
    print(f"  p-value (2-sid) = {p_bc:.4f}")

    # ---- Power analysis for A vs C ----
    sample_sizes = [16, 30, 40, 50, 60, 70, 80, 100]

    # Scenario 1: same effect size as observed (delta=0.011, very small)
    # Scenario 2: moderate improvement in T cells (delta=0.03)
    # Scenario 3: larger improvement (delta=0.05)
    # Scenario 4: substantial improvement (delta=0.08)
    deltas_to_test = [mean_delta_ac, 0.03, 0.05, 0.08]
    delta_labels = [
        f"Observed ({mean_delta_ac:.3f})",
        "Moderate (0.030)",
        "Larger (0.050)",
        "Substantial (0.080)",
    ]

    print("\n\n" + "=" * 80)
    print("SECTION 1: Power of Paired t-Test for A vs C (two-sided, alpha=0.05)")
    print(f"SD of deltas held constant at {sd_delta_ac:.4f} (observed)")
    print("=" * 80)

    # Header
    header = f"{'n':>5s}"
    for label in delta_labels:
        header += f"  {label:>22s}"
    print(f"\n{header}")
    print("-" * len(header))

    for n in sample_sizes:
        row = f"{n:>5d}"
        for d in deltas_to_test:
            pwr = paired_ttest_power(d, sd_delta_ac, n)
            row += f"  {pwr:>22.4f}"
        print(row)

    # ---- Power analysis for B vs C ----
    print("\n\n" + "=" * 80)
    print("SECTION 2: Power of Paired t-Test for B vs C (two-sided, alpha=0.05)")
    print(f"SD of deltas held constant at {sd_delta_bc:.4f} (observed)")
    print("=" * 80)

    deltas_bc = [mean_delta_bc, 0.03, 0.05, 0.08]
    delta_labels_bc = [
        f"Observed ({mean_delta_bc:.3f})",
        "Moderate (0.030)",
        "Larger (0.050)",
        "Substantial (0.080)",
    ]

    header = f"{'n':>5s}"
    for label in delta_labels_bc:
        header += f"  {label:>22s}"
    print(f"\n{header}")
    print("-" * len(header))

    for n in sample_sizes:
        row = f"{n:>5d}"
        for d in deltas_bc:
            pwr = paired_ttest_power(d, sd_delta_bc, n)
            row += f"  {pwr:>22.4f}"
        print(row)

    # ---- Required effect size for 80% power ----
    print("\n\n" + "=" * 80)
    print("SECTION 3: Minimum Effect Size (delta) for 80% Power")
    print("=" * 80)

    print(f"\n{'n':>5s}  {'delta_AC (sd={:.3f})':>22s}  {'delta_BC (sd={:.3f})':>22s}  {'Cohen d (AC)':>14s}  {'Cohen d (BC)':>14s}".format(
        sd_delta_ac, sd_delta_bc
    ))
    print("-" * 85)

    for n in sample_sizes:
        req_ac = required_effect_for_power(sd_delta_ac, n, 0.80)
        req_bc = required_effect_for_power(sd_delta_bc, n, 0.80)
        d_ac = req_ac / sd_delta_ac
        d_bc = req_bc / sd_delta_bc
        print(f"{n:>5d}  {req_ac:>22.4f}  {req_bc:>22.4f}  {d_ac:>14.4f}  {d_bc:>14.4f}")

    # ---- How many guides needed for 80% power at observed effect? ----
    print("\n\n" + "=" * 80)
    print("SECTION 4: Sample Size Needed for 80% Power at Various Effect Sizes")
    print("=" * 80)

    for delta, label in zip(deltas_to_test, delta_labels):
        # Search for n
        for n_try in range(3, 10000):
            if paired_ttest_power(delta, sd_delta_ac, n_try) >= 0.80:
                break
        if n_try >= 9999:
            n_str = ">9999"
        else:
            n_str = str(n_try)
        d_eff = delta / sd_delta_ac
        print(f"  delta={delta:.3f} (d={d_eff:.3f}): n = {n_str} guides for 80% power")

    # ---- Simulation: mixed population (K562/HEK293 + T cells) ----
    print("\n\n" + "=" * 80)
    print("SECTION 5: Sensitivity Analysis — Mixed Cell-Type Population")
    print("=" * 80)
    print("""
Scenario: Current 16 guides (K562/HEK293) + 34 new T cell guides = 50 total.
The T cell guides may show a different effect size. We model:
  - 16 guides with observed delta ~ N(0.011, 0.167)
  - 34 guides with T cell delta ~ N(delta_T, sd_T)

We compute the overall power for three assumptions about T cell data:
  (a) Same as current: delta_T = 0.011, sd_T = 0.167  (no improvement)
  (b) Slightly better: delta_T = 0.03, sd_T = 0.167   (AG helps in T cells)
  (c) Worse/no effect: delta_T = 0.00, sd_T = 0.20    (noisier, no benefit)
""")

    n_sim = 100_000
    np.random.seed(42)

    scenarios = [
        ("Same as current", 0.011, sd_delta_ac),
        ("AG slightly better in T cells", 0.03, sd_delta_ac),
        ("No effect, noisier in T cells", 0.00, 0.20),
        ("AG moderately better in T cells", 0.05, sd_delta_ac),
    ]

    n_existing = 16
    n_new = 34
    n_total = n_existing + n_new

    print(f"{'Scenario':<40s}  {'Power':>8s}  {'Mean reject rate':>16s}")
    print("-" * 70)

    for label, delta_t, sd_t in scenarios:
        reject_count = 0
        for _ in range(n_sim):
            # Existing guides: draw from observed distribution
            existing = np.random.normal(mean_delta_ac, sd_delta_ac, n_existing)
            # New T cell guides
            new = np.random.normal(delta_t, sd_t, n_new)
            # Combined
            combined = np.concatenate([existing, new])
            # Paired t-test on combined
            t_stat = np.mean(combined) / (np.std(combined, ddof=1) / np.sqrt(n_total))
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_total - 1))
            if p_val < 0.05:
                reject_count += 1
        power = reject_count / n_sim
        print(f"{label:<40s}  {power:>8.4f}  {power:>16.4f}")

    # ---- Summary and interpretation ----
    print("\n\n" + "=" * 80)
    print("SECTION 6: Summary and Interpretation")
    print("=" * 80)
    print(f"""
CURRENT STATE (n={n_current} guides):
  - A vs C: Cohen's d = {cohens_d_ac:.3f}, power = {paired_ttest_power(mean_delta_ac, sd_delta_ac, n_current):.4f}
  - B vs C: Cohen's d = {cohens_d_bc:.3f}, power = {paired_ttest_power(mean_delta_bc, sd_delta_bc, n_current):.4f}
  - Both comparisons are SEVERELY underpowered.

EFFECT SIZE CONTEXT:
  - Cohen's d = {cohens_d_ac:.3f} for A vs C is FAR below the "small effect" threshold (d=0.20).
  - The observed mean AUPRC improvement of {mean_delta_ac:.3f} over 16 guides is negligible
    relative to the guide-to-guide variability (SD = {sd_delta_ac:.3f}).
  - Even at d=0.20 (conventional "small"), you need ~199 guides for 80% power.

EXPANSION TO ~50 GUIDES:
  - At the OBSERVED effect size (d={cohens_d_ac:.3f}), power at n=50 = {paired_ttest_power(mean_delta_ac, sd_delta_ac, 50):.4f}.
    This is essentially no improvement.
  - You would need n > {next(n for n in range(3, 10000) if paired_ttest_power(mean_delta_ac, sd_delta_ac, n) >= 0.80)} guides
    to reach 80% power at the observed effect size.
  - HOWEVER: if the true effect in T cells is delta=0.05 (plausible if AlphaGenome
    predicts T cell chromatin well), then n=50 gives power = {paired_ttest_power(0.05, sd_delta_ac, 50):.4f}.

KEY INSIGHT:
  The fundamental problem is NOT sample size — it is effect size.
  The guide-to-guide variability in AUPRC (SD ~ {sd_delta_ac:.2f}) is ~15x larger than
  the mean delta ({mean_delta_ac:.3f}). This means chromatin features provide a tiny
  per-guide improvement that is completely swamped by guide-specific factors
  (target site accessibility, mismatch patterns, local chromatin context).

  Expanding to 50 guides helps only if:
    (a) The T cell data reveals a LARGER effect of chromatin (delta >= 0.05), OR
    (b) The variability is lower in T cells (unlikely — biology is noisy), OR
    (c) You abandon the guide-level test and rely on site-level pooled analysis
        (which already shows p < 0.0001 at n=4068 sites).

RECOMMENDATION:
  1. The guide-level paired test will remain underpowered unless delta >> 0.05.
     Do NOT expand the dataset PRIMARILY to achieve guide-level significance.
  2. The site-level bootstrap (already p < 0.0001) demonstrates the effect exists
     at the observation level, though it doesn't account for within-guide correlation.
  3. A mixed-effects model (random intercepts for guides, fixed effect for model type)
     would be a more appropriate framework. It uses all site-level observations while
     properly accounting for guide-level clustering.
  4. The Lazzarotto expansion is valuable for GENERALIZABILITY (new cell type, new guides),
     not for statistical power on a near-zero guide-level effect.
""")

    # ---- Bonus: what if we use a mixed-effects framing? ----
    print("=" * 80)
    print("SECTION 7: Effective Sample Size Comparison")
    print("=" * 80)
    n_sites_current = 4068
    n_guides_current = 16
    avg_sites_per_guide = n_sites_current / n_guides_current
    # Approximate ICC from the data: high guide-level variance relative to total
    # ICC ~ var_between / (var_between + var_within)
    # With guide-level SD of AUPRC ~0.17 and site-level effects much smaller,
    # the design effect is substantial
    icc_estimate = 0.10  # conservative estimate
    deff = 1 + (avg_sites_per_guide - 1) * icc_estimate
    n_effective = n_sites_current / deff

    print(f"""
  Guide-level test: n_eff = {n_guides_current} (one observation per guide)
  Site-level test:  n_obs = {n_sites_current} (ignores clustering)
  Mixed model:      n_eff ~ {n_effective:.0f} (design effect = {deff:.1f}, ICC ~ {icc_estimate})

  Even with a conservative ICC of {icc_estimate}, the mixed model provides
  ~{n_effective:.0f} effective observations — far more than {n_guides_current} guides.
  This is why site-level bootstrap detects the effect (p<0.0001) while
  guide-level tests do not (p=0.75).

  Adding ~34 T cell guides (~200 sites each) would give:
    Guide-level: n = 50
    Site-level:  n ~ {n_sites_current + 34 * 200}
    Mixed model: n_eff ~ {(n_sites_current + 34 * 200) / deff:.0f}
""")


if __name__ == "__main__":
    main()
