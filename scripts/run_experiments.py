"""Reproduce the EWA-EWC and KO-PEP pairs-trading experiments from
Section 15.6.4 of the chapter.

For each pair we estimate the hedge ratio three ways:

1. Rolling least squares with a two-year window.
2. Basic Kalman (model 15.3) with ``alpha = 1e-5``.
3. Kalman with momentum (model 15.4) with ``alpha = 1e-6``.

Then we build the normalized spread, run the z-score thresholded
strategy, and plot:

* Figure A — hedge ratio over time.
* Figure B — spread (z_t) over time.
* Figure C — cumulative return over time.

All figures are written to ``figures/`` as PNG.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kalman_pairs import (  # noqa: E402
    backtest_pairs,
    kalman_basic,
    kalman_momentum,
    load_pair,
    rolling_ls_hedge,
)
from kalman_pairs.plots import (  # noqa: E402
    plot_cumulative_returns,
    plot_hedge_ratios,
    plot_spreads,
    plot_spreads_stacked,
)

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def normalized_spread(y1, y2, mu, gamma):
    return (y1 - gamma * y2 - mu) / (1.0 + gamma)


def run_pair(ticker_a: str, ticker_b: str, fig_prefix: str, title_suffix: str) -> None:
    print(f"\n=== {ticker_a}-{ticker_b} ===")
    prices = load_pair(ticker_a, ticker_b)
    y1 = prices[ticker_a]
    y2 = prices[ticker_b]
    print(f"  loaded {len(prices)} rows from {prices.index[0].date()} to {prices.index[-1].date()}")

    # 1. Rolling LS, 2-year window (~504 business days).
    ls = rolling_ls_hedge(y1, y2, window=504)
    spread_ls = normalized_spread(y1, y2, ls["mu"], ls["gamma"])

    # 2. Basic Kalman.
    kb = kalman_basic(y1, y2, alpha=1e-5, t_ls=252)
    spread_kb = kb.spread

    # 3. Kalman with momentum.
    km = kalman_momentum(y1, y2, alpha=1e-6, t_ls=252)
    spread_km = km.spread

    # Backtests.
    bt_ls = backtest_pairs(spread_ls)
    bt_kb = backtest_pairs(spread_kb)
    bt_km = backtest_pairs(spread_km)
    for name, bt in [("LS", bt_ls), ("Kalman basic", bt_kb), ("Kalman momentum", bt_km)]:
        final = bt["cumulative_return"].iloc[-1]
        print(f"  {name:18s} final cumulative return: {final:8.3f}")

    # Figure A — hedge ratio.
    fig, ax = plt.subplots(figsize=(11, 4))
    plot_hedge_ratios(
        {
            "Rolling LS (2y)": ls["gamma"],
            "Kalman basic": kb.states_filt["gamma"],
            "Kalman + momentum": km.states_filt["gamma"],
        },
        title=f"Hedge ratio tracking — {title_suffix}",
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fig_prefix}_hedge_ratio.png", dpi=130)
    plt.close(fig)

    # Figure B — spread (stacked rows so the three series can be compared).
    fig = plot_spreads_stacked(
        {
            "Rolling LS (2y)": spread_ls,
            "Kalman basic": spread_kb,
            "Kalman + momentum": spread_km,
        },
        title=f"Normalized spread $z_t$ — {title_suffix}",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fig_prefix}_spread.png", dpi=130)
    plt.close(fig)

    # Figure C — cumulative returns.
    fig, ax = plt.subplots(figsize=(11, 4))
    plot_cumulative_returns(
        {
            "Rolling LS (2y)": bt_ls["cumulative_return"],
            "Kalman basic": bt_kb["cumulative_return"],
            "Kalman + momentum": bt_km["cumulative_return"],
        },
        title=f"Cumulative return (no transaction costs) — {title_suffix}",
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fig_prefix}_cumret.png", dpi=130)
    plt.close(fig)

    # Combined dashboard.
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    plot_hedge_ratios(
        {
            "Rolling LS (2y)": ls["gamma"],
            "Kalman basic": kb.states_filt["gamma"],
            "Kalman + momentum": km.states_filt["gamma"],
        },
        title=f"Hedge ratio — {title_suffix}",
        ax=axes[0],
    )
    plot_spreads(
        {
            "Rolling LS (2y)": spread_ls,
            "Kalman basic": spread_kb,
            "Kalman + momentum": spread_km,
        },
        title="Spread",
        ax=axes[1],
    )
    plot_cumulative_returns(
        {
            "Rolling LS (2y)": bt_ls["cumulative_return"],
            "Kalman basic": bt_kb["cumulative_return"],
            "Kalman + momentum": bt_km["cumulative_return"],
        },
        title="Cumulative return",
        ax=axes[2],
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fig_prefix}_dashboard.png", dpi=130)
    plt.close(fig)


def main() -> None:
    run_pair("EWA", "EWC", "ewa_ewc", "EWA-EWC")
    run_pair("KO", "PEP", "ko_pep", "KO-PEP")
    print(f"\nFigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
