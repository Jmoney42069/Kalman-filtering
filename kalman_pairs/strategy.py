"""Pairs-trading strategy: z-score, thresholded signal, P&L.

Section 15.5 / 15.6.4 of the chapter:
* Compute z-score of the spread on a rolling-window basis (6 months).
* Enter long when ``z < -s0``, short when ``z > +s0``; exit at ``z=0``.
* Return is ``signal_{t-1} * spread_return_t`` (no transaction costs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def spread_zscore(spread: pd.Series, window: int = 126) -> pd.Series:
    """Rolling z-score of a spread series.

    A six-month lookback (~126 business days) is the chapter's default.
    """
    mean = spread.rolling(window=window, min_periods=window).mean()
    std = spread.rolling(window=window, min_periods=window).std(ddof=1)
    return ((spread - mean) / std).rename("z")


def thresholded_signal(z: pd.Series, s0: float = 1.0) -> pd.Series:
    """Convert a z-score to a position ``+1 / 0 / -1``.

    A long position (+1) opens when ``z`` crosses below ``-s0`` and
    closes when ``z`` crosses 0; a short position (-1) opens when ``z``
    crosses above ``+s0`` and closes when ``z`` crosses 0. This is the
    standard "thresholded strategy" of Section 15.5.
    """
    pos = np.zeros(len(z))
    state = 0  # +1 long, -1 short, 0 flat
    zv = z.to_numpy()
    for t, zt in enumerate(zv):
        if np.isnan(zt):
            pos[t] = 0
            continue
        if state == 0:
            if zt > s0:
                state = -1
            elif zt < -s0:
                state = 1
        elif state == 1 and zt >= 0:
            state = 0
        elif state == -1 and zt <= 0:
            state = 0
        pos[t] = state
    return pd.Series(pos, index=z.index, name="signal")


def backtest_pairs(
    spread: pd.Series,
    window: int = 126,
    s0: float = 1.0,
) -> pd.DataFrame:
    """Backtest the thresholded pairs strategy on a spread.

    Returns a DataFrame with columns ``z``, ``signal``, ``spread_return``,
    ``strategy_return``, and ``cumulative_return``. Positions are
    applied with a one-step lag to avoid look-ahead.
    """
    z = spread_zscore(spread, window=window)
    sig = thresholded_signal(z, s0=s0)
    spread_ret = spread.diff()
    strat_ret = sig.shift(1) * spread_ret
    cum = strat_ret.fillna(0.0).cumsum()
    return pd.DataFrame({
        "z": z,
        "signal": sig,
        "spread_return": spread_ret,
        "strategy_return": strat_ret,
        "cumulative_return": cum,
    })
