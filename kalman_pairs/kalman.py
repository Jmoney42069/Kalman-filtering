"""Kalman filtering for time-varying spread modeling.

Three models from Section 15.6.3 of the chapter are implemented:

* :func:`kalman_basic` — model (15.3), hidden state ``(mu_t, gamma_t)``.
* :func:`kalman_momentum` — model (15.4), hidden state
  ``(mu_t, gamma_t, gamma_dot_t)``; ``gamma`` evolves as a local-linear
  trend, which produces a smoother hedge ratio.
* :func:`kalman_partial_coint` — model (15.5), hidden state
  ``(mu_t, gamma_t, epsilon_t)``; ``epsilon_t`` follows an AR(1) process
  ``epsilon_{t+1} = rho * epsilon_t + eta`` (partial cointegration of
  Clegg and Krauss, 2018).

All recursions follow Durbin and Koopman (2012) for the linear Gaussian
state-space model::

    y_t       = Z_t alpha_t + epsilon_t,   epsilon_t ~ N(0, H)
    alpha_{t+1} = T alpha_t + eta_t,         eta_t ~ N(0, Q)

The forward pass is::

    v_t   = y_t - Z_t a_t
    F_t   = Z_t P_t Z_t' + H
    K_t   = T P_t Z_t' / F_t
    a_{t+1} = T a_t + K_t v_t
    P_{t+1} = T P_t (T - K_t Z_t)' + Q

``a_t`` is the predicted state ``alpha_{t|t-1}`` and ``a_t + P_t Z_t' v_t / F_t``
is the filtered state ``alpha_{t|t}``. The chapter's normalized spread
uses the predicted state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kalman_pairs.least_squares import rolling_ls_hedge


@dataclass
class KalmanResult:
    """Container for a forward-pass run.

    Attributes
    ----------
    states_pred : DataFrame
        Predicted hidden state ``alpha_{t|t-1}`` at every step.
    states_filt : DataFrame
        Filtered hidden state ``alpha_{t|t}`` at every step.
    spread : Series
        Normalized spread ``z_t`` (leverage one) using the predicted
        hedge ratio, as in equation just above (15.4) in the chapter.
    innovation : Series
        Innovation ``v_t = y_{1t} - Z_t a_t``.
    innovation_var : Series
        Innovation variance ``F_t``.
    """

    states_pred: pd.DataFrame
    states_filt: pd.DataFrame
    spread: pd.Series
    innovation: pd.Series
    innovation_var: pd.Series


def _ls_warmup(y1: pd.Series, y2: pd.Series, t_ls: int) -> tuple[float, float, float]:
    """Run an initial least-squares fit on the first ``t_ls`` samples.

    Returns ``(mu_ls, gamma_ls, var_eps_ls)`` used as priors for the
    Kalman filter, per the heuristic in Section 15.6.3.
    """
    if t_ls < 5 or t_ls > len(y1):
        raise ValueError(f"t_ls={t_ls} must be in [5, {len(y1)}]")
    ls = rolling_ls_hedge(y1.iloc[:t_ls], y2.iloc[:t_ls], window=t_ls)
    mu_ls = float(ls["mu"].iloc[-1])
    gamma_ls = float(ls["gamma"].iloc[-1])
    resid = y1.iloc[:t_ls].values - mu_ls - gamma_ls * y2.iloc[:t_ls].values
    var_eps = float(np.var(resid, ddof=1))
    return mu_ls, gamma_ls, var_eps


def _run_filter(
    y1: np.ndarray,
    Z_rows: np.ndarray,
    T_mat: np.ndarray,
    H: float,
    Q: np.ndarray,
    a1: np.ndarray,
    P1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generic forward pass for a univariate observation.

    Parameters
    ----------
    y1 : (n,) array, observations.
    Z_rows : (n, k) array, one ``Z_t`` per time step (rows are 1xk).
    T_mat : (k, k) state-transition matrix (time-invariant here).
    H : scalar observation noise variance.
    Q : (k, k) state noise covariance.
    a1, P1 : prior mean and covariance of ``alpha_1``.

    Returns
    -------
    a_pred : (n, k) predicted states ``alpha_{t|t-1}``.
    a_filt : (n, k) filtered states ``alpha_{t|t}``.
    v : (n,) innovations.
    F : (n,) innovation variances.
    """
    n, k = Z_rows.shape
    a_pred = np.zeros((n, k))
    a_filt = np.zeros((n, k))
    v = np.zeros(n)
    F = np.zeros(n)

    a = a1.copy()
    P = P1.copy()

    for t in range(n):
        z = Z_rows[t]
        a_pred[t] = a
        v_t = y1[t] - float(z @ a)
        Pz = P @ z
        F_t = float(z @ Pz) + H
        K = (T_mat @ Pz) / F_t

        a_filt[t] = a + Pz * (v_t / F_t)

        a = T_mat @ a + K * v_t
        P = T_mat @ P @ (T_mat - np.outer(K, z)).T + Q

        v[t] = v_t
        F[t] = F_t

    return a_pred, a_filt, v, F


def _normalized_spread(y1: np.ndarray, y2: np.ndarray, mu: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Compute z_t = (y_1t - gamma * y_2t - mu) / (1 + gamma) with leverage one."""
    denom = 1.0 + gamma
    denom = np.where(np.abs(denom) < 1e-8, np.nan, denom)
    return (y1 - gamma * y2 - mu) / denom


def kalman_basic(
    y1: pd.Series,
    y2: pd.Series,
    alpha: float = 1e-5,
    t_ls: int = 252,
) -> KalmanResult:
    """Run the basic Kalman model (15.3) on the pair ``(y1, y2)``.

    Parameters
    ----------
    y1, y2 : pd.Series
        Aligned price series. Both must have the same index.
    alpha : float
        Ratio of hidden-state variability to spread variability. Smaller
        values yield smoother estimates of ``mu_t`` and ``gamma_t``.
    t_ls : int
        Number of initial observations used for the least-squares
        warm-up that sets the priors and noise variances.
    """
    if not y1.index.equals(y2.index):
        raise ValueError("y1 and y2 must share the same index")

    mu_ls, gamma_ls, var_eps = _ls_warmup(y1, y2, t_ls)
    var_y2 = float(np.var(y2.iloc[:t_ls].values, ddof=1))

    sigma2_eps = var_eps
    sigma2_mu = alpha * var_eps
    sigma2_gamma = alpha * var_eps / var_y2

    a1 = np.array([mu_ls, gamma_ls])
    P1 = np.diag([var_eps / t_ls, var_eps / (t_ls * var_y2)])
    T_mat = np.eye(2)
    Q = np.diag([sigma2_mu, sigma2_gamma])

    y2v = y2.to_numpy(dtype=float)
    y1v = y1.to_numpy(dtype=float)
    Z = np.column_stack([np.ones_like(y2v), y2v])

    a_pred, a_filt, v, F = _run_filter(y1v, Z, T_mat, sigma2_eps, Q, a1, P1)

    idx = y1.index
    states_pred = pd.DataFrame(a_pred, index=idx, columns=["mu", "gamma"])
    states_filt = pd.DataFrame(a_filt, index=idx, columns=["mu", "gamma"])
    spread = pd.Series(
        _normalized_spread(y1v, y2v, a_pred[:, 0], a_pred[:, 1]),
        index=idx,
        name="spread",
    )
    return KalmanResult(
        states_pred=states_pred,
        states_filt=states_filt,
        spread=spread,
        innovation=pd.Series(v, index=idx, name="v"),
        innovation_var=pd.Series(F, index=idx, name="F"),
    )


def kalman_momentum(
    y1: pd.Series,
    y2: pd.Series,
    alpha: float = 1e-6,
    t_ls: int = 252,
) -> KalmanResult:
    """Run the Kalman-with-momentum model (15.4).

    The hidden state is ``(mu_t, gamma_t, gamma_dot_t)``. Only
    ``gamma_dot_t`` has process noise; ``gamma_t`` evolves
    deterministically as ``gamma_{t+1} = gamma_t + gamma_dot_t``, which
    is the standard "local linear trend" structure for gamma.
    """
    if not y1.index.equals(y2.index):
        raise ValueError("y1 and y2 must share the same index")

    mu_ls, gamma_ls, var_eps = _ls_warmup(y1, y2, t_ls)
    var_y2 = float(np.var(y2.iloc[:t_ls].values, ddof=1))

    sigma2_eps = var_eps
    sigma2_mu = alpha * var_eps
    sigma2_gdot = alpha * var_eps / var_y2

    a1 = np.array([mu_ls, gamma_ls, 0.0])
    P1 = np.diag([var_eps / t_ls, var_eps / (t_ls * var_y2), sigma2_gdot])
    T_mat = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 1.0],
                      [0.0, 0.0, 1.0]])
    Q = np.diag([sigma2_mu, 0.0, sigma2_gdot])

    y2v = y2.to_numpy(dtype=float)
    y1v = y1.to_numpy(dtype=float)
    Z = np.column_stack([np.ones_like(y2v), y2v, np.zeros_like(y2v)])

    a_pred, a_filt, v, F = _run_filter(y1v, Z, T_mat, sigma2_eps, Q, a1, P1)

    idx = y1.index
    states_pred = pd.DataFrame(a_pred, index=idx, columns=["mu", "gamma", "gamma_dot"])
    states_filt = pd.DataFrame(a_filt, index=idx, columns=["mu", "gamma", "gamma_dot"])
    spread = pd.Series(
        _normalized_spread(y1v, y2v, a_pred[:, 0], a_pred[:, 1]),
        index=idx,
        name="spread",
    )
    return KalmanResult(
        states_pred=states_pred,
        states_filt=states_filt,
        spread=spread,
        innovation=pd.Series(v, index=idx, name="v"),
        innovation_var=pd.Series(F, index=idx, name="F"),
    )


def kalman_partial_coint(
    y1: pd.Series,
    y2: pd.Series,
    rho: float = 0.9,
    alpha: float = 1e-5,
    t_ls: int = 252,
) -> KalmanResult:
    """Run the partial-cointegration Kalman model (15.5).

    The hidden state is ``(mu_t, gamma_t, epsilon_t)`` with
    ``epsilon_{t+1} = rho * epsilon_t + eta`` and the observation
    ``y_{1t} = mu_t + gamma_t * y_{2t} + epsilon_t``. ``|rho| < 1``
    enforces mean reversion of the residual component.
    """
    if not 0 <= abs(rho) < 1:
        raise ValueError("rho must satisfy |rho| < 1")
    if not y1.index.equals(y2.index):
        raise ValueError("y1 and y2 must share the same index")

    mu_ls, gamma_ls, var_eps = _ls_warmup(y1, y2, t_ls)
    var_y2 = float(np.var(y2.iloc[:t_ls].values, ddof=1))

    sigma2_mu = alpha * var_eps
    sigma2_gamma = alpha * var_eps / var_y2
    sigma2_eta_eps = (1 - rho * rho) * var_eps

    a1 = np.array([mu_ls, gamma_ls, 0.0])
    P1 = np.diag([var_eps / t_ls, var_eps / (t_ls * var_y2), var_eps])
    T_mat = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, rho]])
    Q = np.diag([sigma2_mu, sigma2_gamma, sigma2_eta_eps])

    y2v = y2.to_numpy(dtype=float)
    y1v = y1.to_numpy(dtype=float)
    Z = np.column_stack([np.ones_like(y2v), y2v, np.ones_like(y2v)])
    H = 1e-8  # observation noise is absorbed into the AR component

    a_pred, a_filt, v, F = _run_filter(y1v, Z, T_mat, H, Q, a1, P1)

    idx = y1.index
    cols = ["mu", "gamma", "epsilon"]
    states_pred = pd.DataFrame(a_pred, index=idx, columns=cols)
    states_filt = pd.DataFrame(a_filt, index=idx, columns=cols)
    spread = pd.Series(
        _normalized_spread(y1v, y2v, a_pred[:, 0], a_pred[:, 1]),
        index=idx,
        name="spread",
    )
    return KalmanResult(
        states_pred=states_pred,
        states_filt=states_filt,
        spread=spread,
        innovation=pd.Series(v, index=idx, name="v"),
        innovation_var=pd.Series(F, index=idx, name="F"),
    )
