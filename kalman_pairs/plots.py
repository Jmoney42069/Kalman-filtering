"""Matplotlib helpers that reproduce Figures 15.21–15.26 of the chapter."""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd


def plot_hedge_ratios(
    hedge_ratios: Mapping[str, pd.Series],
    title: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    for label, series in hedge_ratios.items():
        ax.plot(series.index, series.values, label=label, linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.4, alpha=0.4)
    ax.set_title(title)
    ax.set_ylabel(r"Hedge ratio $\gamma_t$")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def plot_spreads(
    spreads: Mapping[str, pd.Series],
    title: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    for label, series in spreads.items():
        ax.plot(series.index, series.values, label=label, linewidth=0.7, alpha=0.7)
    ax.axhline(0.0, color="black", linewidth=0.4, alpha=0.4)
    ax.set_title(title)
    ax.set_ylabel(r"Spread $z_t$")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def plot_spreads_stacked(
    spreads: Mapping[str, pd.Series],
    title: str,
    fig: plt.Figure | None = None,
) -> plt.Figure:
    """Plot each spread in its own row (shared x-axis), for readability
    when the spreads overlap heavily on a single panel."""
    n = len(spreads)
    if fig is None:
        fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n), sharex=True)
    else:
        axes = fig.subplots(n, 1, sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (label, series) in zip(axes, spreads.items()):
        ax.plot(series.index, series.values, linewidth=0.6, alpha=0.85)
        ax.axhline(0.0, color="black", linewidth=0.4, alpha=0.4)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(title)
    return fig


def plot_cumulative_returns(
    cumrets: Mapping[str, pd.Series],
    title: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    for label, series in cumrets.items():
        final = series.dropna().iloc[-1] if series.dropna().size else float("nan")
        ax.plot(series.index, series.values, label=f"{label} (final: {final:.2f})", linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.4, alpha=0.4)
    ax.set_title(title)
    ax.set_ylabel("Cumulative return (spread units)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax
