"""Rolling least-squares hedge-ratio estimation.

Matches Section 15.6.1 of the chapter. The closed-form solution

    [mu_hat, gamma_hat]^T = (X^T X)^{-1} X^T y_1,   X = [1, y_2]

is computed inside a rolling window of length ``window`` and produces a
hedge ratio and intercept at every step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_ls_hedge(
    y1: pd.Series,
    y2: pd.Series,
    window: int,
) -> pd.DataFrame:
    """Rolling least-squares estimate of ``mu`` and ``gamma`` in
    ``y1 ≈ mu + gamma * y2`` using a trailing window of ``window`` rows.

    Returns a DataFrame with columns ``mu`` and ``gamma`` aligned with
    ``y1.index``. Rows before the window fills up are NaN.
    """
    if len(y1) != len(y2):
        raise ValueError("y1 and y2 must have the same length")

    y1v = np.asarray(y1, dtype=float)
    y2v = np.asarray(y2, dtype=float)
    n = len(y1v)

    mu = np.full(n, np.nan)
    gamma = np.full(n, np.nan)

    for t in range(window - 1, n):
        s = slice(t - window + 1, t + 1)
        y2_win = y2v[s]
        y1_win = y1v[s]
        y2_bar = y2_win.mean()
        y1_bar = y1_win.mean()
        y2c = y2_win - y2_bar
        denom = float(y2c @ y2c)
        if denom <= 0:
            continue
        g = float(y2c @ (y1_win - y1_bar)) / denom
        gamma[t] = g
        mu[t] = y1_bar - g * y2_bar

    return pd.DataFrame({"mu": mu, "gamma": gamma}, index=y1.index)
