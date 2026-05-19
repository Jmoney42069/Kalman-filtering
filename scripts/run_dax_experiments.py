"""Reproduce pairs-trading experiments on DAX 30 / GER30 names.

Three German pairs that each tell a different story:

* **Allianz vs Munich Re** — classic stable cointegration.
* **E.ON vs RWE** — 2019/2020 asset-swap inverted their exposure;
  perfect demo of Kalman handling a regime shift.
* **BMW vs Mercedes-Benz** — Feb 2022 Daimler spin-off broke the
  long-run relationship.

Writes a dashboard PNG per pair to ``figures/`` plus a combined
summary chart comparing all three.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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
    plot_spreads_stacked,
)

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def normalized_spread(y1, y2, mu, gamma):
    return (y1 - gamma * y2 - mu) / (1.0 + gamma)


def run_pair(ticker_a: str, ticker_b: str, fig_prefix: str, title: str, regime_note: str | None = None):
    print(f"\n=== {ticker_a} vs {ticker_b} ===")
    prices = load_pair(ticker_a, ticker_b)
    y1, y2 = prices[ticker_a], prices[ticker_b]
    print(f"  {len(prices)} rows  {prices.index[0].date()} -> {prices.index[-1].date()}")

    ls = rolling_ls_hedge(y1, y2, window=504)
    kb = kalman_basic(y1, y2, alpha=1e-5, t_ls=252)
    km = kalman_momentum(y1, y2, alpha=1e-6, t_ls=252)

    spread_ls = normalized_spread(y1, y2, ls["mu"], ls["gamma"])
    spread_kb = kb.spread
    spread_km = km.spread

    bt_ls = backtest_pairs(spread_ls)
    bt_kb = backtest_pairs(spread_kb)
    bt_km = backtest_pairs(spread_km)
    for name, bt in [("LS", bt_ls), ("Kalman basic", bt_kb), ("Kalman + mom", bt_km)]:
        print(f"  {name:14s} final P&L: {bt['cumulative_return'].iloc[-1]:8.2f}")

    # 4-panel dashboard.
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)

    axes[0].plot(prices.index, y1, label=ticker_a, color="#222", linewidth=1.0)
    axes[0].set_ylabel(ticker_a, color="#222")
    ax2 = axes[0].twinx()
    ax2.plot(prices.index, y2, label=ticker_b, color="#888", linewidth=1.0, linestyle="--")
    ax2.set_ylabel(ticker_b, color="#888")
    axes[0].set_title(f"{title} — prices")
    axes[0].grid(True, alpha=0.3)

    plot_hedge_ratios(
        {"Rolling LS (2y)": ls["gamma"], "Kalman basic": kb.states_filt["gamma"], "Kalman + momentum": km.states_filt["gamma"]},
        title="Hedge ratio γₜ",
        ax=axes[1],
    )

    axes[2].plot(spread_kb.index, spread_kb.values, label="Kalman basic", color="#ff7f0e", linewidth=0.7, alpha=0.85)
    axes[2].plot(spread_km.index, spread_km.values, label="Kalman + momentum", color="#2ca02c", linewidth=0.7, alpha=0.85)
    axes[2].axhline(0, color="black", linewidth=0.4)
    axes[2].set_title("Normalized spread zₜ (Kalman)")
    axes[2].set_ylabel("zₜ")
    axes[2].legend(loc="best", fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plot_cumulative_returns(
        {"Rolling LS (2y)": bt_ls["cumulative_return"], "Kalman basic": bt_kb["cumulative_return"], "Kalman + momentum": bt_km["cumulative_return"]},
        title="Cumulative P&L (no transaction costs)",
        ax=axes[3],
    )

    if regime_note:
        for ax in axes:
            ax.text(
                0.5, 0.95, regime_note,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color="#999",
            ) if ax is axes[0] else None

    fig.tight_layout()
    out = FIG_DIR / f"dax_{fig_prefix}_dashboard.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out.name}")

    return {
        "title": title,
        "cum": {
            "Rolling LS (2y)": bt_ls["cumulative_return"],
            "Kalman basic": bt_kb["cumulative_return"],
            "Kalman + momentum": bt_km["cumulative_return"],
        },
        "gamma": {
            "Rolling LS (2y)": ls["gamma"],
            "Kalman basic": kb.states_filt["gamma"],
            "Kalman + momentum": km.states_filt["gamma"],
        },
    }


def main():
    pairs = [
        ("ALV.DE", "MUV2.DE", "alv_muv2", "Allianz vs Munich Re — stable DAX insurers", None),
        ("EOAN.DE", "RWE.DE", "eoan_rwe", "E.ON vs RWE — 2019 asset-swap regime shift", "asset swap completes Sept 2019"),
        ("BMW.DE", "MBG.DE", "bmw_mbg", "BMW vs Mercedes-Benz — 2022 Daimler spin-off", "Daimler -> Mercedes-Benz Group Feb 2022"),
    ]
    results = [run_pair(*p) for p in pairs]

    # Combined comparison chart.
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex="col")
    for col, (ticker_a, ticker_b, _, title, _) in enumerate(pairs):
        r = results[col]
        plot_hedge_ratios(r["gamma"], title=f"γₜ — {ticker_a.split('.')[0]} / {ticker_b.split('.')[0]}", ax=axes[0, col])
        plot_cumulative_returns(r["cum"], title=f"P&L — {ticker_a.split('.')[0]} / {ticker_b.split('.')[0]}", ax=axes[1, col])
    fig.suptitle("DAX 30 pairs trading — three case studies", fontsize=14)
    fig.tight_layout()
    out = FIG_DIR / "dax_overview.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
