# Kalman Filtering for Pairs Trading

Implementation of Section 15.6 of Palomar's *Portfolio Optimization: Theory and
Application* — time-varying spread modeling for pairs trading with the Kalman
filter.

## What's here

```
kalman_pairs/
  data.py            # yfinance loader with deterministic synthetic fallback
  least_squares.py   # rolling LS hedge ratio (baseline)
  kalman.py          # three state-space models (15.3 / 15.4 / 15.5)
  strategy.py        # z-score, thresholded signal, backtest
  plots.py           # matplotlib helpers for the chapter's figures

dashboard/app.py                 # Streamlit dashboard with sliders
scripts/run_experiments.py       # reproduces EWA-EWC and KO-PEP figures (PNG)
scripts/run_dax_experiments.py   # DAX 30 / GER30 demo: ALV-MUV2, EOAN-RWE, BMW-MBG
scripts/build_demo_dashboard.py  # builds figures/demo_dashboard.html
notebooks/pairs_trading.ipynb    # annotated walkthrough with inline plots
tests/test_kalman.py             # sanity tests on synthetic data
figures/                          # generated PNGs and interactive HTML demo
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Interactive dashboard with sliders for alpha, rho, z-window, threshold
streamlit run dashboard/app.py

# Or a self-contained interactive HTML (no server, no install)
python scripts/build_demo_dashboard.py
open figures/demo_dashboard.html

# Reproduce Figures 15.21-15.26 as static PNGs
python scripts/run_experiments.py

# Run the tests
python -m pytest tests/ -v

# Open the walkthrough notebook
jupyter notebook notebooks/pairs_trading.ipynb
```

## Three Kalman models

* **Basic (15.3)** — state is `(mu_t, gamma_t)`, both random-walk.
  `kalman_basic(y1, y2, alpha=1e-5)`
* **With momentum (15.4)** — state is `(mu_t, gamma_t, gamma_dot_t)`;
  `gamma_t` evolves as a local linear trend so it adapts to regime
  changes more smoothly. `kalman_momentum(y1, y2, alpha=1e-6)`
* **Partial cointegration (15.5)** — state is `(mu_t, gamma_t, eps_t)`
  with `eps_{t+1} = rho * eps_t + eta`, so the residual is mean-reverting.
  `kalman_partial_coint(y1, y2, rho=0.9, alpha=1e-5)`

`alpha` is the chapter's single hyperparameter: it sets the ratio of
hidden-state variability to spread variability. Smaller `alpha` →
smoother `gamma_t`. Too small fails to track regime changes; too large
makes the spread noisy and transaction costs eat the P&L.

## Data

`load_pair` tries Yahoo Finance first. When the environment blocks
network access, it falls back to `synthetic_pair`, which generates
deterministic cointegrated series for these built-in pairs:

| Pair | Story |
| --- | --- |
| `EWA-EWC` | Australian / Canadian ETFs, stable cointegration (textbook 15.6.4) |
| `KO-PEP` | Coca-Cola / Pepsi, 2020 COVID dislocation (textbook 15.6.4) |
| `ALV.DE-MUV2.DE` | **DAX**: Allianz / Munich Re, stable insurer pair |
| `EOAN.DE-RWE.DE` | **DAX**: E.ON / RWE, 2019 asset-swap regime shift |
| `BMW.DE-MBG.DE` | **DAX**: BMW / Mercedes-Benz, 2022 Daimler spin-off |
| `SAP.DE-SIE.DE` | **DAX**: SAP / Siemens, weak cointegration (different sectors) |

Run `python scripts/run_dax_experiments.py` for the German market demo.
