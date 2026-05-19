"""Sanity tests for the LS and Kalman estimators on synthetic data."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_pairs import (  # noqa: E402
    kalman_basic,
    kalman_momentum,
    kalman_partial_coint,
    rolling_ls_hedge,
    synthetic_pair,
)


def _make_pair(n=1500, gamma=0.7, mu=5.0, sigma=0.3, seed=42):
    rng = np.random.default_rng(seed)
    y2 = pd.Series(20.0 + np.cumsum(rng.normal(0, 0.3, n)))
    y1 = pd.Series(mu + gamma * y2 + rng.normal(0, sigma, n))
    return y1, y2


def test_rolling_ls_recovers_truth():
    y1, y2 = _make_pair(gamma=0.7, mu=5.0)
    out = rolling_ls_hedge(y1, y2, window=252)
    assert out["gamma"].iloc[-1] == pytest.approx(0.7, abs=0.02)
    assert out["mu"].iloc[-1] == pytest.approx(5.0, abs=0.5)


def test_kalman_basic_tracks_constant_hedge():
    y1, y2 = _make_pair(gamma=0.8, mu=3.0, n=2000)
    res = kalman_basic(y1, y2, alpha=1e-5, t_ls=252)
    gamma_tail = res.states_filt["gamma"].iloc[-200:].mean()
    mu_tail = res.states_filt["mu"].iloc[-200:].mean()
    assert gamma_tail == pytest.approx(0.8, abs=0.05)
    assert mu_tail == pytest.approx(3.0, abs=0.8)


def test_kalman_smoother_than_short_rolling_ls():
    """Both Kalman variants should track gamma far more smoothly than a short rolling LS."""
    y1, y2 = _make_pair(gamma=0.6, mu=4.0, sigma=0.5, n=2000)
    ls = rolling_ls_hedge(y1, y2, window=60)["gamma"].dropna()  # short window = noisy
    basic = kalman_basic(y1, y2, alpha=1e-5, t_ls=252).states_filt["gamma"].iloc[300:]
    mom = kalman_momentum(y1, y2, alpha=1e-6, t_ls=252).states_filt["gamma"].iloc[300:]
    assert basic.diff().std() < 0.5 * ls.diff().std()
    assert mom.diff().std() < 0.5 * ls.diff().std()
    # Both Kalman variants should hit the right neighbourhood at the tail.
    assert basic.iloc[-200:].mean() == pytest.approx(0.6, abs=0.05)
    assert mom.iloc[-200:].mean() == pytest.approx(0.6, abs=0.10)


def test_kalman_partial_coint_runs():
    y1, y2 = _make_pair(gamma=0.5, mu=10.0, n=1500)
    res = kalman_partial_coint(y1, y2, rho=0.9, alpha=1e-5, t_ls=252)
    assert res.states_filt.shape[1] == 3
    assert np.isfinite(res.states_filt.iloc[-1]).all()


def test_synthetic_pairs_have_expected_shape():
    ewa_ewc = synthetic_pair("EWA-EWC", start="2013-01-01", end="2022-12-31")
    ko_pep = synthetic_pair("KO-PEP", start="2013-01-01", end="2022-12-31")
    assert ewa_ewc.shape[1] == 2 and ko_pep.shape[1] == 2
    assert "EWA" in ewa_ewc.columns and "EWC" in ewa_ewc.columns
    assert "KO" in ko_pep.columns and "PEP" in ko_pep.columns
    assert len(ewa_ewc) > 2000  # ~10 years of business days


def test_kalman_basic_index_matches():
    y1, y2 = _make_pair()
    res = kalman_basic(y1, y2, t_ls=252)
    assert res.states_pred.index.equals(y1.index)
    assert res.spread.index.equals(y1.index)
