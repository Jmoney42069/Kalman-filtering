"""Build a self-contained interactive Plotly HTML dashboard.

The output is a single HTML file you can open in any browser without
running a server. It shows three panels for EWA-EWC and KO-PEP:

* Hedge ratio γₜ for the three estimators.
* Normalized spread + z-score (zoom and pan; hover for values).
* Cumulative return (drawdown shaded for the best performer).

A dropdown lets you switch between pairs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kalman_pairs import (  # noqa: E402
    backtest_pairs,
    kalman_basic,
    kalman_momentum,
    load_pair,
    rolling_ls_hedge,
    spread_zscore,
)

OUTPUT = ROOT / "figures" / "demo_dashboard.html"

COLORS = {
    "Rolling LS (2y)": "#1f77b4",
    "Kalman basic": "#ff7f0e",
    "Kalman + momentum": "#2ca02c",
}


def normalized_spread(y1, y2, mu, gamma):
    return (y1 - gamma * y2 - mu) / (1.0 + gamma)


def compute(ticker_a: str, ticker_b: str):
    prices = load_pair(ticker_a, ticker_b, start="2013-01-01", end="2022-12-31")
    y1 = prices[ticker_a]
    y2 = prices[ticker_b]
    ls = rolling_ls_hedge(y1, y2, window=504)
    kb = kalman_basic(y1, y2, alpha=1e-5, t_ls=252)
    km = kalman_momentum(y1, y2, alpha=1e-6, t_ls=252)
    spreads = {
        "Rolling LS (2y)": normalized_spread(y1, y2, ls["mu"], ls["gamma"]),
        "Kalman basic": kb.spread,
        "Kalman + momentum": km.spread,
    }
    gammas = {
        "Rolling LS (2y)": ls["gamma"],
        "Kalman basic": kb.states_filt["gamma"],
        "Kalman + momentum": km.states_filt["gamma"],
    }
    bts = {name: backtest_pairs(s) for name, s in spreads.items()}
    zscores = {name: spread_zscore(s) for name, s in spreads.items()}
    return {
        "prices": prices,
        "gammas": gammas,
        "spreads": spreads,
        "zscores": zscores,
        "bts": bts,
    }


def build_pair_figure(data: dict, title: str) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.18, 0.25, 0.28, 0.29],
        vertical_spacing=0.04,
        subplot_titles=(
            "Prices (left axis: y1; right axis: y2)",
            "γₜ — hedge ratio",
            "Normalized spread zₜ",
            "Cumulative P&L (no transaction costs)",
        ),
        specs=[[{"secondary_y": True}], [{}], [{}], [{}]],
    )

    # Row 1: prices
    p = data["prices"]
    a, b = p.columns
    fig.add_trace(
        go.Scatter(x=p.index, y=p[a], name=a, line=dict(color="#444", width=1.0)),
        row=1, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=p.index, y=p[b], name=b, line=dict(color="#999", width=1.0, dash="dot")),
        row=1, col=1, secondary_y=True,
    )

    # Row 2: gammas
    for name, series in data["gammas"].items():
        fig.add_trace(
            go.Scatter(x=series.index, y=series.values, name=name, line=dict(color=COLORS[name], width=1.4),
                        legendgroup=name),
            row=2, col=1,
        )

    # Row 3: spreads
    for name, series in data["spreads"].items():
        fig.add_trace(
            go.Scatter(x=series.index, y=series.values, name=name, line=dict(color=COLORS[name], width=0.7),
                        opacity=0.7, legendgroup=name, showlegend=False),
            row=3, col=1,
        )
    fig.add_hline(y=0, line_color="black", line_width=0.6, row=3, col=1)

    # Row 4: cumulative P&L + drawdown shading on the winner
    finals = {n: bt["cumulative_return"].iloc[-1] for n, bt in data["bts"].items()}
    winner = max(finals, key=finals.get)
    for name, bt in data["bts"].items():
        cum = bt["cumulative_return"]
        fig.add_trace(
            go.Scatter(x=cum.index, y=cum.values, name=f"{name} (final {finals[name]:.1f})",
                        line=dict(color=COLORS[name], width=1.6), legendgroup=name, showlegend=False),
            row=4, col=1,
        )
    dd = data["bts"][winner]["cumulative_return"]
    dd_curve = dd - dd.cummax()
    fig.add_trace(
        go.Scatter(x=dd_curve.index, y=dd_curve.values, name=f"{winner} drawdown",
                    fill="tozeroy", line=dict(color=COLORS[winner], width=0), opacity=0.15, showlegend=False),
        row=4, col=1,
    )

    fig.update_yaxes(title_text=a, row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text=b, row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="γₜ", row=2, col=1)
    fig.update_yaxes(title_text="zₜ", row=3, col=1)
    fig.update_yaxes(title_text="cum P&L", row=4, col=1)
    fig.update_layout(
        height=950,
        title=title,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=60, r=60, t=80, b=40),
    )
    return fig


def build_combined_figure(pairs_data: dict[str, dict]) -> go.Figure:
    """Single figure with a dropdown to switch between pairs."""
    figs = {pair: build_pair_figure(d, title=pair) for pair, d in pairs_data.items()}

    pair_names = list(pairs_data.keys())
    base_fig = figs[pair_names[0]]
    base_traces = list(base_fig.data)
    n_traces_per_pair = [len(figs[p].data) for p in pair_names]

    all_data = []
    for p in pair_names:
        all_data.extend(figs[p].data)

    combined = go.Figure(layout=base_fig.layout)
    combined.update_layout(annotations=base_fig.layout.annotations)
    for tr in all_data:
        combined.add_trace(tr)

    visibility = []
    offset = 0
    for i, p in enumerate(pair_names):
        n = n_traces_per_pair[i]
        vis = [False] * len(all_data)
        for j in range(offset, offset + n):
            vis[j] = True
        visibility.append((p, vis, figs[p].layout.annotations))
        offset += n

    for tr_idx, tr in enumerate(combined.data):
        tr.visible = tr_idx < n_traces_per_pair[0]

    buttons = []
    for p, vis, annotations in visibility:
        buttons.append(dict(
            label=p,
            method="update",
            args=[{"visible": vis},
                  {"title": p, "annotations": annotations}],
        ))

    combined.update_layout(
        title=pair_names[0],
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.0, xanchor="left",
            y=1.13, yanchor="top",
            buttons=buttons,
            pad=dict(r=10, l=10, t=6, b=6),
            showactive=True,
            bgcolor="#ffffff",
            bordercolor="#ccc",
        )],
        annotations=list(base_fig.layout.annotations) + [dict(
            text="Select pair:", x=-0.005, xref="paper", y=1.135, yref="paper",
            xanchor="right", yanchor="middle", showarrow=False, font=dict(size=12, color="#555"),
        )],
    )
    return combined


def main() -> None:
    pairs_data = {
        "Allianz - Munich Re  (DAX, stable insurers)": compute("ALV.DE", "MUV2.DE"),
        "E.ON - RWE  (DAX, 2019 asset-swap regime shift)": compute("EOAN.DE", "RWE.DE"),
        "BMW - Mercedes-Benz  (DAX, 2022 Daimler spin-off)": compute("BMW.DE", "MBG.DE"),
        "SAP - Siemens  (DAX, weak cointegration)": compute("SAP.DE", "SIE.DE"),
        "EWA - EWC  (US ETFs, textbook stable pair)": compute("EWA", "EWC"),
        "KO - PEP  (US, 2020 COVID dislocation)": compute("KO", "PEP"),
    }

    fig = build_combined_figure(pairs_data)
    OUTPUT.parent.mkdir(exist_ok=True)
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>Kalman Pairs Trading — interactive demo</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; margin: 0; padding: 20px 28px 60px; background: #fafafa; color: #222; }}
  h1 {{ margin: 0 0 4px; font-size: 22px; }}
  .lede {{ color: #555; margin: 0 0 18px; font-size: 14px; max-width: 920px; line-height: 1.5; }}
  .legend-help {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; margin-top: 22px; font-size: 13px; color: #444; max-width: 920px; line-height: 1.55; }}
  code {{ background: #eef; padding: 1px 5px; border-radius: 3px; font-size: 12px; }}
</style>
</head>
<body>
<h1>Kalman filtering for pairs trading — interactive demo</h1>
<p class=\"lede\">
Pick a pair with the buttons. Drag-select on any panel to zoom; double-click to reset.
Hover for values. Click items in the legend to hide/show series.
The y₁ price uses the left axis on the top panel; y₂ uses the right axis.
</p>
{fig.to_html(include_plotlyjs='cdn', full_html=False)}
<div class=\"legend-help\">
  <strong>What you are looking at:</strong> three estimators of the time-varying hedge ratio γₜ in
  y₁ ≈ μₜ + γₜ · y₂. <code>Rolling LS (2y)</code> recomputes ordinary least squares on a sliding
  two-year window. <code>Kalman basic</code> treats (μₜ, γₜ) as a hidden state with random-walk
  dynamics (model 15.3 in the textbook). <code>Kalman + momentum</code> adds γ̇ₜ so γₜ evolves as
  a local linear trend (model 15.4). The spread zₜ is the normalized residual; cumulative P&amp;L
  is the integral of position × spread change for a threshold-z strategy.
</div>
</body>
</html>"""
    OUTPUT.write_text(html)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
