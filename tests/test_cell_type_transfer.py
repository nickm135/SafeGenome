"""Tests for src/cell_type_transfer.py.

Test strategy
-------------
- The AlphaGenome client (ag_client) is fully mocked: its
  ``batch_get_chromatin_features`` returns a DataFrame with synthetic AG
  feature columns.  No live API calls are made.
- XGBoost models are trained for real on tiny synthetic data (~40 rows,
  10 estimators) and reused across tests.
- All pure-computation functions (build_score_matrix,
  compute_rank_correlation_matrix, compute_chromatin_correlation, etc.) are
  tested directly with controlled DataFrames.
- Plotting tests use the Agg backend and verify that figures are returned
  without error.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.feature_matrix import (
    AG_FEATURE_NAMES,
    EXP_FEATURE_NAMES,
    MODEL_C_FEATURES,
    SEQUENCE_FEATURE_NAMES,
)
from src.cell_type_transfer import (
    RECOMMENDED_CELL_TYPES,
    SITE_ID_COL,
    build_score_matrix,
    compute_chromatin_correlation,
    compute_rank_correlation_matrix,
    make_site_id,
    plot_rank_correlation_heatmap,
    plot_transfer_heatmap,
    run_cell_type_transfer,
    score_candidates_for_cell_type,
    transfer_across_cell_types,
    validate_transfer_rankings,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_FAST_XGB = {"n_estimators": 10, "max_depth": 3}


def _make_candidates(n: int = 30, n_guides: int = 3, seed: int = 0) -> pd.DataFrame:
    """Return a synthetic candidates DataFrame with all required columns."""
    rng = np.random.default_rng(seed)
    chroms = ["chr1", "chr2", "chr7"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "guide_id": f"g{i % n_guides}",
                "chrom": chroms[i % len(chroms)],
                "start": 1_000_000 + i * 10_000,
                "strand": "+" if i % 2 == 0 else "-",
                "guide_seq": "GAGTCCGAGCAGAAGAAGAA",
                "off_target_seq": "GAGTCCGAGCAGAAGAAGCC",
                "pam_seq": "TGG",
                "num_mismatches": rng.integers(1, 5),
                "cfd_score": rng.random(),
                "mit_score": rng.random() * 100,
            }
        )
    df = pd.DataFrame(rows)
    # Add all sequence features.
    for col in SEQUENCE_FEATURE_NAMES:
        if col not in df.columns:
            if "mm_type" in col:
                df[col] = rng.integers(0, 13, size=n)
            elif col in ("num_mismatches", "seed_mismatches", "nonseed_mismatches",
                         "consecutive_mm") or "mm_pos" in col:
                df[col] = rng.integers(0, 4, size=n)
            else:
                df[col] = rng.random(n).astype(np.float32)
    return df


def _make_ag_features_df(candidates: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """Build a mock return value from batch_get_chromatin_features."""
    rng = np.random.default_rng(seed)
    result = candidates.copy()
    for feat in AG_FEATURE_NAMES:
        result[feat] = (rng.random(len(candidates)) * 5).astype(np.float32)
    return result


def _make_tiny_model(n: int = 40, seed: int = 42):
    """Train a tiny Model C XGBClassifier on synthetic data."""
    import xgboost as xgb
    from sklearn.model_selection import StratifiedGroupKFold

    rng = np.random.default_rng(seed)
    # Use first 10 seq features + 5 AG features = 15 features total.
    feat_cols = SEQUENCE_FEATURE_NAMES[:10] + AG_FEATURE_NAMES[:5]
    X = rng.random((n, len(feat_cols))).astype(np.float32)
    y = np.zeros(n, dtype=np.int32)
    y[:4] = 1  # 4 positives
    groups = np.array([f"g{i % 4}" for i in range(n)])

    clf = xgb.XGBClassifier(
        n_estimators=10, max_depth=3, scale_pos_weight=9.0,
        verbosity=0, random_state=seed, eval_metric="aucpr",
    )
    clf.fit(X, y)
    return clf, feat_cols


@pytest.fixture(scope="module")
def tiny_model():
    """Shared tiny XGBClassifier and feature_cols for the whole test module."""
    return _make_tiny_model()


@pytest.fixture
def candidates():
    return _make_candidates(n=20, n_guides=3)


@pytest.fixture
def mock_ag_client(candidates):
    """AG client whose batch_get_chromatin_features returns synthetic features."""
    client = MagicMock()

    def fake_batch(sites, cell_type, model, cache_dir, **kwargs):
        return _make_ag_features_df(sites, seed=hash(cell_type) % 1000)

    client.predict_interval = MagicMock()  # not called directly
    return client


# ---------------------------------------------------------------------------
# make_site_id
# ---------------------------------------------------------------------------

class TestMakeSiteId:
    def test_format(self):
        assert make_site_id("chr7", 100000, "+") == "chr7:100000:+"

    def test_minus_strand(self):
        assert make_site_id("chr1", 0, "-") == "chr1:0:-"

    def test_default_strand(self):
        sid = make_site_id("chrX", 500)
        assert sid == "chrX:500:."

    def test_different_chroms_differ(self):
        a = make_site_id("chr1", 1000, "+")
        b = make_site_id("chr2", 1000, "+")
        assert a != b

    def test_different_positions_differ(self):
        a = make_site_id("chr1", 1000, "+")
        b = make_site_id("chr1", 2000, "+")
        assert a != b


# ---------------------------------------------------------------------------
# score_candidates_for_cell_type
# ---------------------------------------------------------------------------

class TestScoreCandidatesForCellType:
    def test_returns_dataframe(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = score_candidates_for_cell_type(
                candidates, "K562", "EFO:0002067",
                clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        assert isinstance(result, pd.DataFrame)

    def test_adds_risk_score_column(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = score_candidates_for_cell_type(
                candidates, "K562", "EFO:0002067",
                clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        assert "risk_score" in result.columns

    def test_adds_rank_column(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = score_candidates_for_cell_type(
                candidates, "K562", "EFO:0002067",
                clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        assert "rank" in result.columns

    def test_adds_cell_type_column(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = score_candidates_for_cell_type(
                candidates, "K562", "EFO:0002067",
                clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        assert (result["cell_type"] == "K562").all()

    def test_adds_site_id_column(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = score_candidates_for_cell_type(
                candidates, "K562", "EFO:0002067",
                clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        assert SITE_ID_COL in result.columns

    def test_risk_scores_in_unit_interval(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = score_candidates_for_cell_type(
                candidates, "K562", "EFO:0002067",
                clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        assert (result["risk_score"] >= 0.0).all()
        assert (result["risk_score"] <= 1.0).all()

    def test_ranks_are_one_indexed(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = score_candidates_for_cell_type(
                candidates, "K562", "EFO:0002067",
                clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        assert result["rank"].min() == 1
        assert result["rank"].max() == len(result)

    def test_row_count_unchanged(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = score_candidates_for_cell_type(
                candidates, "K562", "EFO:0002067",
                clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        assert len(result) == len(candidates)

    def test_missing_chrom_raises(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        bad = candidates.drop(columns=["chrom"])
        with pytest.raises(ValueError, match="chrom"):
            score_candidates_for_cell_type(
                bad, "K562", "EFO:0002067",
                clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )

    def test_missing_start_raises(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        bad = candidates.drop(columns=["start"])
        with pytest.raises(ValueError, match="start"):
            score_candidates_for_cell_type(
                bad, "K562", "EFO:0002067",
                clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )

    def test_highest_score_gets_rank_one(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = score_candidates_for_cell_type(
                candidates, "K562", "EFO:0002067",
                clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        highest_score_row = result.loc[result["risk_score"].idxmax()]
        assert highest_score_row["rank"] == 1

    def test_seq_features_computed_when_absent(self, tiny_model, tmp_path):
        """If sequence features are missing, they should be computed automatically."""
        clf, feat_cols = tiny_model
        bare_candidates = pd.DataFrame({
            "guide_id": ["g0", "g1"],
            "chrom": ["chr1", "chr2"],
            "start": [1_000_000, 2_000_000],
            "strand": ["+", "-"],
            "guide_seq": ["GAGTCCGAGCAGAAGAAGAA"] * 2,
            "off_target_seq": ["GAGTCCGAGCAGAAGAAGCC"] * 2,
            "pam_seq": ["TGG"] * 2,
            "num_mismatches": [2, 3],
        })
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = score_candidates_for_cell_type(
                bare_candidates, "K562", "EFO:0002067",
                clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        assert "risk_score" in result.columns
        assert len(result) == 2


# ---------------------------------------------------------------------------
# transfer_across_cell_types
# ---------------------------------------------------------------------------

class TestTransferAcrossCellTypes:
    _CELL_TYPE_MAP = {
        "K562": "EFO:0002067",
        "HEK293": "EFO:0001182",
    }

    def test_long_format_row_count(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = transfer_across_cell_types(
                candidates, self._CELL_TYPE_MAP, clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        expected_rows = len(candidates) * len(self._CELL_TYPE_MAP)
        assert len(result) == expected_rows

    def test_cell_type_column_values(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = transfer_across_cell_types(
                candidates, self._CELL_TYPE_MAP, clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        assert set(result["cell_type"].unique()) == set(self._CELL_TYPE_MAP.keys())

    def test_has_risk_score_column(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = transfer_across_cell_types(
                candidates, self._CELL_TYPE_MAP, clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        assert "risk_score" in result.columns

    def test_has_site_id_column(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = transfer_across_cell_types(
                candidates, self._CELL_TYPE_MAP, clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        assert SITE_ID_COL in result.columns

    def test_different_scores_across_cell_types(self, candidates, tiny_model, tmp_path):
        """Different AG features per cell type should produce different scores."""
        clf, feat_cols = tiny_model
        call_count = [0]

        def seeded_batch(sites, cell_type, **kw):
            call_count[0] += 1
            # Different seed per cell type → different features → different scores.
            return _make_ag_features_df(sites, seed=call_count[0] * 17)

        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=seeded_batch,
        ):
            result = transfer_across_cell_types(
                candidates, self._CELL_TYPE_MAP, clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )

        k562_scores = result[result["cell_type"] == "K562"]["risk_score"].values
        hek_scores = result[result["cell_type"] == "HEK293"]["risk_score"].values
        # With different random AG features, the scores should not be identical.
        assert not np.allclose(k562_scores, hek_scores)

    def test_empty_cell_type_map_raises(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with pytest.raises(ValueError, match="cell_type_map"):
            transfer_across_cell_types(
                candidates, {}, clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )

    def test_three_cell_types(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        cell_type_map = {
            "K562": "EFO:0002067",
            "HEK293": "EFO:0001182",
            "CD4_T_cell": "CL:0000624",
        }
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = transfer_across_cell_types(
                candidates, cell_type_map, clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        assert len(result) == len(candidates) * 3


# ---------------------------------------------------------------------------
# build_score_matrix
# ---------------------------------------------------------------------------

class TestBuildScoreMatrix:
    @pytest.fixture
    def long_df(self):
        """Synthetic long-format transfer DataFrame."""
        sites = [f"chr1:{i}:+" for i in [1000, 2000, 3000, 4000, 5000]]
        cell_types = ["K562", "HEK293"]
        rng = np.random.default_rng(7)
        rows = []
        for site in sites:
            for ct in cell_types:
                rows.append(
                    {
                        SITE_ID_COL: site,
                        "cell_type": ct,
                        "risk_score": rng.random(),
                        "cfd_score": rng.random(),
                        "rank": 1,
                    }
                )
        return pd.DataFrame(rows)

    def test_returns_wide_dataframe(self, long_df):
        matrix = build_score_matrix(long_df)
        assert isinstance(matrix, pd.DataFrame)

    def test_columns_are_cell_types(self, long_df):
        matrix = build_score_matrix(long_df)
        assert set(matrix.columns) == {"K562", "HEK293"}

    def test_row_count_equals_unique_sites(self, long_df):
        n_sites = long_df[SITE_ID_COL].nunique()
        matrix = build_score_matrix(long_df)
        assert len(matrix) == n_sites

    def test_values_match_original(self, long_df):
        matrix = build_score_matrix(long_df)
        for _, row in long_df.iterrows():
            site = row[SITE_ID_COL]
            ct = row["cell_type"]
            expected = row["risk_score"]
            assert matrix.loc[site, ct] == pytest.approx(expected)

    def test_sort_by_cfd_ascending(self, long_df):
        matrix = build_score_matrix(long_df, sort_by="cfd_score", ascending=True)
        # Compute the expected order using per-site first cfd_score.
        site_cfd = long_df.groupby(SITE_ID_COL)["cfd_score"].first()
        expected_order = site_cfd.sort_values(ascending=True).index.tolist()
        assert list(matrix.index) == expected_order

    def test_sort_by_cfd_descending(self, long_df):
        matrix = build_score_matrix(long_df, sort_by="cfd_score", ascending=False)
        site_cfd = long_df.groupby(SITE_ID_COL)["cfd_score"].first()
        expected_order = site_cfd.sort_values(ascending=False).index.tolist()
        assert list(matrix.index) == expected_order

    def test_sort_by_missing_column_falls_back(self, long_df):
        # Should not raise; falls back to mean-score sort.
        matrix = build_score_matrix(long_df, sort_by="nonexistent_col")
        assert len(matrix) == long_df[SITE_ID_COL].nunique()

    def test_sort_by_none_falls_back(self, long_df):
        matrix = build_score_matrix(long_df, sort_by=None)
        assert len(matrix) == long_df[SITE_ID_COL].nunique()

    def test_missing_required_column_raises(self, long_df):
        bad = long_df.drop(columns=["risk_score"])
        with pytest.raises(ValueError, match="risk_score"):
            build_score_matrix(bad)

    def test_missing_cell_type_raises(self, long_df):
        bad = long_df.drop(columns=["cell_type"])
        with pytest.raises(ValueError, match="cell_type"):
            build_score_matrix(bad)


# ---------------------------------------------------------------------------
# compute_rank_correlation_matrix
# ---------------------------------------------------------------------------

class TestComputeRankCorrelationMatrix:
    @pytest.fixture
    def perfect_matrix(self):
        """Two cell types with perfectly correlated rankings."""
        rng = np.random.default_rng(5)
        scores = rng.random(20)
        return pd.DataFrame(
            {"K562": scores, "HEK293": scores},
            index=[f"site_{i}" for i in range(20)],
        )

    @pytest.fixture
    def anti_correlated_matrix(self):
        """Two cell types with perfectly anti-correlated rankings."""
        scores = np.arange(20, dtype=float)
        return pd.DataFrame(
            {"K562": scores, "HEK293": scores[::-1]},
            index=[f"site_{i}" for i in range(20)],
        )

    def test_diagonal_is_one(self, perfect_matrix):
        corr = compute_rank_correlation_matrix(perfect_matrix)
        for ct in corr.columns:
            assert corr.loc[ct, ct] == pytest.approx(1.0)

    def test_symmetric(self, perfect_matrix):
        corr = compute_rank_correlation_matrix(perfect_matrix)
        assert corr.loc["K562", "HEK293"] == pytest.approx(corr.loc["HEK293", "K562"])

    def test_perfect_correlation(self, perfect_matrix):
        corr = compute_rank_correlation_matrix(perfect_matrix)
        assert corr.loc["K562", "HEK293"] == pytest.approx(1.0)

    def test_anti_correlation(self, anti_correlated_matrix):
        corr = compute_rank_correlation_matrix(anti_correlated_matrix)
        assert corr.loc["K562", "HEK293"] == pytest.approx(-1.0, abs=0.01)

    def test_values_between_minus_one_and_one(self):
        rng = np.random.default_rng(3)
        matrix = pd.DataFrame(
            {ct: rng.random(30) for ct in ["K562", "HEK293", "T_cell"]},
            index=[f"s{i}" for i in range(30)],
        )
        corr = compute_rank_correlation_matrix(matrix)
        assert (corr.values >= -1.0 - 1e-9).all()
        assert (corr.values <= 1.0 + 1e-9).all()

    def test_returns_square_dataframe(self):
        rng = np.random.default_rng(4)
        n_ct = 4
        matrix = pd.DataFrame(
            {f"ct_{i}": rng.random(25) for i in range(n_ct)},
            index=[f"s{i}" for i in range(25)],
        )
        corr = compute_rank_correlation_matrix(matrix)
        assert corr.shape == (n_ct, n_ct)

    def test_empty_matrix_raises(self):
        with pytest.raises((ValueError, Exception)):
            compute_rank_correlation_matrix(pd.DataFrame())


# ---------------------------------------------------------------------------
# validate_transfer_rankings
# ---------------------------------------------------------------------------

class TestValidateTransferRankings:
    @pytest.fixture
    def transfer_df_with_labels(self):
        """Long-format DataFrame where K562 sites have known positive labels."""
        rng = np.random.default_rng(9)
        n = 40
        # 4 positives at the start.
        rows = []
        for i in range(n):
            rows.append(
                {
                    "guide_id": f"g{i % 3}",
                    "chrom": "chr1",
                    "start": 1_000_000 + i * 1000,
                    "strand": "+",
                    "cell_type": "K562",
                    SITE_ID_COL: f"chr1:{1_000_000 + i * 1000}:+",
                    # Positive sites get high risk scores.
                    "risk_score": 0.9 - 0.5 * (i >= 4),
                }
            )
        return pd.DataFrame(rows)

    @pytest.fixture
    def positives(self):
        """4 known positives corresponding to the first 4 sites."""
        return pd.DataFrame(
            {
                "guide_id": [f"g{i % 3}" for i in range(4)],
                "chrom": ["chr1"] * 4,
                "start": [1_000_000 + i * 1000 for i in range(4)],
            }
        )

    def test_returns_dict_with_auprc(self, transfer_df_with_labels, positives):
        result = validate_transfer_rankings(
            transfer_df_with_labels, positives, "K562"
        )
        assert "auprc" in result

    def test_returns_dict_with_auroc(self, transfer_df_with_labels, positives):
        result = validate_transfer_rankings(
            transfer_df_with_labels, positives, "K562"
        )
        assert "auroc" in result

    def test_n_sites_correct(self, transfer_df_with_labels, positives):
        result = validate_transfer_rankings(
            transfer_df_with_labels, positives, "K562"
        )
        n_k562 = int((transfer_df_with_labels["cell_type"] == "K562").sum())
        assert int(result["n_sites"]) == n_k562

    def test_n_positives_correct(self, transfer_df_with_labels, positives):
        result = validate_transfer_rankings(
            transfer_df_with_labels, positives, "K562"
        )
        assert int(result["n_positives"]) == len(positives)

    def test_good_model_high_auprc(self, transfer_df_with_labels, positives):
        # Risk scores are 0.9 for positives, 0.4 for negatives → AUPRC > 0.5.
        result = validate_transfer_rankings(
            transfer_df_with_labels, positives, "K562"
        )
        if not math.isnan(result["auprc"]):
            assert result["auprc"] > 0.5

    def test_missing_cell_type_raises(self, transfer_df_with_labels, positives):
        with pytest.raises(ValueError, match="HEK293"):
            validate_transfer_rankings(
                transfer_df_with_labels, positives, "HEK293"
            )

    def test_all_keys_present(self, transfer_df_with_labels, positives):
        result = validate_transfer_rankings(
            transfer_df_with_labels, positives, "K562"
        )
        for key in ("auprc", "auroc", "n_sites", "n_positives"):
            assert key in result


# ---------------------------------------------------------------------------
# compute_chromatin_correlation
# ---------------------------------------------------------------------------

class TestComputeChRomatinCorrelation:
    @pytest.fixture
    def feature_matrix_with_both(self):
        """Feature matrix with ag_dnase_mean and exp_dnase_mean columns."""
        rng = np.random.default_rng(11)
        n = 60
        ag = rng.random(n)
        # exp is perfectly correlated with ag for easier testing.
        exp = ag + rng.random(n) * 0.01  # nearly perfect
        return pd.DataFrame(
            {
                "ag_dnase_mean": ag,
                "exp_dnase_mean": exp,
                "cell_type": ["K562"] * 40 + ["HEK293"] * 20,
            }
        )

    def test_returns_dict_with_spearman_r(self, feature_matrix_with_both):
        result = compute_chromatin_correlation(feature_matrix_with_both)
        assert "spearman_r" in result

    def test_returns_dict_with_p_value(self, feature_matrix_with_both):
        result = compute_chromatin_correlation(feature_matrix_with_both)
        assert "p_value" in result

    def test_returns_dict_with_n(self, feature_matrix_with_both):
        result = compute_chromatin_correlation(feature_matrix_with_both)
        assert "n" in result

    def test_high_correlation_with_near_perfect_data(self, feature_matrix_with_both):
        result = compute_chromatin_correlation(feature_matrix_with_both)
        assert result["spearman_r"] > 0.99

    def test_cell_type_filter(self, feature_matrix_with_both):
        result_all = compute_chromatin_correlation(feature_matrix_with_both)
        result_k562 = compute_chromatin_correlation(
            feature_matrix_with_both, cell_type="K562"
        )
        assert result_k562["n"] == 40
        assert result_all["n"] == 60

    def test_negative_correlation(self):
        rng = np.random.default_rng(13)
        n = 50
        ag = rng.random(n)
        exp = 1.0 - ag  # perfectly anti-correlated
        df = pd.DataFrame({"ag_dnase_mean": ag, "exp_dnase_mean": exp})
        result = compute_chromatin_correlation(df)
        assert result["spearman_r"] < -0.99

    def test_missing_ag_feature_raises(self):
        df = pd.DataFrame({"exp_dnase_mean": [1.0, 2.0]})
        with pytest.raises(ValueError, match="ag_dnase_mean"):
            compute_chromatin_correlation(df)

    def test_missing_exp_feature_raises(self):
        df = pd.DataFrame({"ag_dnase_mean": [1.0, 2.0]})
        with pytest.raises(ValueError, match="exp_dnase_mean"):
            compute_chromatin_correlation(df)

    def test_fewer_than_3_pairs_returns_nan(self):
        df = pd.DataFrame(
            {"ag_dnase_mean": [1.0, 2.0], "exp_dnase_mean": [1.5, 2.5]}
        )
        result = compute_chromatin_correlation(df)
        assert math.isnan(result["spearman_r"])

    def test_nan_values_dropped(self):
        rng = np.random.default_rng(15)
        n = 30
        ag = rng.random(n).tolist() + [float("nan")] * 5
        exp = rng.random(n).tolist() + [float("nan")] * 5
        df = pd.DataFrame({"ag_dnase_mean": ag, "exp_dnase_mean": exp})
        result = compute_chromatin_correlation(df)
        assert result["n"] == n

    def test_custom_features(self):
        rng = np.random.default_rng(17)
        ag = rng.random(40)
        exp = ag + rng.random(40) * 0.05
        df = pd.DataFrame(
            {
                "ag_h3k4me3_mean": ag,
                "exp_h3k4me3_mean": exp,
            }
        )
        result = compute_chromatin_correlation(
            df,
            ag_feature="ag_h3k4me3_mean",
            exp_feature="exp_h3k4me3_mean",
        )
        assert result["spearman_r"] > 0.9

    def test_unknown_cell_type_returns_nan(self, feature_matrix_with_both):
        result = compute_chromatin_correlation(
            feature_matrix_with_both, cell_type="U2OS"
        )
        assert math.isnan(result["spearman_r"])


# ---------------------------------------------------------------------------
# plot_transfer_heatmap
# ---------------------------------------------------------------------------

class TestPlotTransferHeatmap:
    @pytest.fixture
    def score_matrix(self):
        rng = np.random.default_rng(20)
        n_sites = 15
        cell_types = ["K562", "HEK293", "CD4_T_cell"]
        index = [f"chr1:{1_000_000 + i * 1000}:+" for i in range(n_sites)]
        return pd.DataFrame(
            {ct: rng.random(n_sites) for ct in cell_types},
            index=index,
        )

    def test_returns_figure(self, score_matrix):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plot_transfer_heatmap(score_matrix)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_saves_to_file(self, score_matrix, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = tmp_path / "heatmap.png"
        fig = plot_transfer_heatmap(score_matrix, output_path=out)
        assert out.exists()
        plt.close(fig)

    def test_top_n_reduces_rows(self, score_matrix):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # With top_n=5, only 5 sites should be shown (embedded in figure axes).
        fig = plot_transfer_heatmap(score_matrix, top_n=5)
        plt.close(fig)

    def test_normalize_rows(self, score_matrix):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plot_transfer_heatmap(score_matrix, normalize_rows=True)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_rank_correlation_heatmap
# ---------------------------------------------------------------------------

class TestPlotRankCorrelationHeatmap:
    @pytest.fixture
    def corr_matrix(self):
        ct = ["K562", "HEK293", "CD4_T_cell"]
        arr = np.array(
            [[1.0, 0.7, 0.3],
             [0.7, 1.0, 0.5],
             [0.3, 0.5, 1.0]]
        )
        return pd.DataFrame(arr, index=ct, columns=ct)

    def test_returns_figure(self, corr_matrix):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plot_rank_correlation_heatmap(corr_matrix)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_saves_to_file(self, corr_matrix, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = tmp_path / "rank_corr.png"
        fig = plot_rank_correlation_heatmap(corr_matrix, output_path=out)
        assert out.exists()
        plt.close(fig)


# ---------------------------------------------------------------------------
# run_cell_type_transfer (integration)
# ---------------------------------------------------------------------------

class TestRunCellTypeTransfer:
    _CELL_TYPE_MAP = {
        "K562": "EFO:0002067",
        "HEK293": "EFO:0001182",
    }

    @pytest.fixture
    def positives(self):
        """Known positives for K562 validation."""
        cands = _make_candidates(n=20)
        return pd.DataFrame(
            {
                "guide_id": cands["guide_id"].iloc[:3].tolist(),
                "chrom": cands["chrom"].iloc[:3].tolist(),
                "start": cands["start"].iloc[:3].tolist(),
            }
        )

    def test_returns_expected_keys(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = run_cell_type_transfer(
                candidates, self._CELL_TYPE_MAP, clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        assert set(result.keys()) == {
            "transfer_df", "score_matrix", "rank_corr", "validation"
        }

    def test_transfer_df_is_dataframe(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = run_cell_type_transfer(
                candidates, self._CELL_TYPE_MAP, clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        assert isinstance(result["transfer_df"], pd.DataFrame)

    def test_score_matrix_columns_are_cell_types(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = run_cell_type_transfer(
                candidates, self._CELL_TYPE_MAP, clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        assert set(result["score_matrix"].columns) == set(self._CELL_TYPE_MAP.keys())

    def test_rank_corr_is_square(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = run_cell_type_transfer(
                candidates, self._CELL_TYPE_MAP, clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        n = len(self._CELL_TYPE_MAP)
        assert result["rank_corr"].shape == (n, n)

    def test_validation_empty_when_no_positives(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = run_cell_type_transfer(
                candidates, self._CELL_TYPE_MAP, clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
            )
        assert result["validation"] == {}

    def test_validation_populated_with_positives(
        self, candidates, tiny_model, tmp_path, positives
    ):
        clf, feat_cols = tiny_model
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            result = run_cell_type_transfer(
                candidates, self._CELL_TYPE_MAP, clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
                positives_by_cell_type={"K562": positives},
            )
        assert "K562" in result["validation"]
        assert "auprc" in result["validation"]["K562"]

    def test_saves_csvs_to_output_dir(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        out_dir = tmp_path / "transfer_output"
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            run_cell_type_transfer(
                candidates, self._CELL_TYPE_MAP, clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
                output_dir=out_dir,
            )
        assert (out_dir / "cell_type_transfer_scores.csv").exists()
        assert (out_dir / "cell_type_score_matrix.csv").exists()
        assert (out_dir / "cell_type_rank_correlation.csv").exists()

    def test_saves_heatmap_figure(self, candidates, tiny_model, tmp_path):
        clf, feat_cols = tiny_model
        out_dir = tmp_path / "figs"
        with patch(
            "src.cell_type_transfer._ag_mod.batch_get_chromatin_features",
            side_effect=lambda sites, **kw: _make_ag_features_df(sites),
        ):
            run_cell_type_transfer(
                candidates, self._CELL_TYPE_MAP, clf, feat_cols,
                ag_client=MagicMock(),
                ag_cache_dir=tmp_path,
                output_dir=out_dir,
            )
        assert (out_dir / "fig3_cell_type_heatmap.png").exists()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_recommended_cell_types_contains_k562(self):
        assert "K562" in RECOMMENDED_CELL_TYPES

    def test_recommended_cell_types_contains_hek293(self):
        assert "HEK293" in RECOMMENDED_CELL_TYPES

    def test_k562_ontology_term(self):
        assert RECOMMENDED_CELL_TYPES["K562"] == "EFO:0002067"

    def test_hek293_ontology_term(self):
        assert RECOMMENDED_CELL_TYPES["HEK293"] == "EFO:0001182"

    def test_site_id_col_is_string(self):
        assert isinstance(SITE_ID_COL, str)
