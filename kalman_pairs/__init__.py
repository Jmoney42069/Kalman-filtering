"""Kalman filtering for pairs trading.

Implementation of the methods from Section 15.6 of "Portfolio Optimization:
Theory and Application" (Palomar): rolling least squares, basic Kalman
(model 15.3), Kalman with momentum (model 15.4), and partial-cointegration
Kalman with an AR residual (model 15.5).
"""

from kalman_pairs.least_squares import rolling_ls_hedge
from kalman_pairs.kalman import (
    KalmanResult,
    kalman_basic,
    kalman_momentum,
    kalman_partial_coint,
)
from kalman_pairs.strategy import (
    spread_zscore,
    thresholded_signal,
    backtest_pairs,
)
from kalman_pairs.data import load_pair, synthetic_pair

__all__ = [
    "rolling_ls_hedge",
    "KalmanResult",
    "kalman_basic",
    "kalman_momentum",
    "kalman_partial_coint",
    "spread_zscore",
    "thresholded_signal",
    "backtest_pairs",
    "load_pair",
    "synthetic_pair",
]
