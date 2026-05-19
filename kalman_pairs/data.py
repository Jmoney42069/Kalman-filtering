"""Price-series loaders for pairs trading experiments.

`load_pair` tries Yahoo Finance first and falls back to a deterministic
synthetic generator if network access is blocked. `synthetic_pair`
exposes the generator directly for tests and for reproducible learning
experiments where deterministic series are preferable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_pair(
    name: str,
    start: str = "2013-01-01",
    end: str = "2022-12-31",
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a cointegrated pair of "price" series.

    The generator is calibrated to mirror well-known real-world pairs:

    Textbook reference pairs (Section 15.6.4 of Palomar 2024):

    * ``"EWA-EWC"`` — Australian/Canadian ETFs, stable cointegration,
      hedge ratio ~0.6.
    * ``"KO-PEP"`` — Coca-Cola/Pepsi, marginal cointegration with the
      2020 COVID dislocation as a regime shift.

    DAX 30 / GER30 demo pairs:

    * ``"ALV.DE-MUV2.DE"`` — Allianz / Munich Re. Two Munich-based
      reinsurance giants, very tight cointegration, EUR 150-300 range.
    * ``"EOAN.DE-RWE.DE"`` — E.ON / RWE utilities. The 2019-2020
      asset swap (E.ON took RWE's grid, RWE took E.ON's renewables)
      is the canonical real-world regime shift in this pair.
    * ``"BMW.DE-MBG.DE"`` — BMW / Mercedes-Benz Group. German premium
      autos. Structural break in Feb 2022 when Daimler was renamed to
      Mercedes-Benz Group AG and spun off Daimler Truck.
    * ``"SAP.DE-SIE.DE"`` — SAP / Siemens. Two German blue-chip caps
      from different sectors; loose long-run relationship.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    if name == "EWA-EWC":
        gamma_true = 0.60 + 0.02 * np.sin(np.linspace(0, 2 * np.pi, n))
        mu_true = np.full(n, 8.0)
        sigma_eps = 0.35
        common = np.cumsum(rng.normal(0.0, 0.45, size=n))
        y2 = 25.0 + 0.6 * common + rng.normal(0.0, 0.20, size=n)

    elif name == "KO-PEP":
        gamma_true = np.where(
            dates < pd.Timestamp("2020-02-15"),
            0.85,
            1.10,
        ).astype(float)
        gamma_true += 0.02 * np.sin(np.linspace(0, 4 * np.pi, n))
        mu_true = np.where(dates < pd.Timestamp("2020-02-15"), 12.0, 6.0).astype(float)
        sigma_eps = 0.60
        common = np.cumsum(rng.normal(0.0, 0.50, size=n))
        y2 = 90.0 + 0.4 * common + rng.normal(0.0, 0.30, size=n)

    elif name == "ALV.DE-MUV2.DE":
        # Allianz vs Munich Re. Both DAX insurers around 150-300 EUR.
        # Hedge ratio drifts slowly around ~0.95.
        gamma_true = 0.95 + 0.03 * np.sin(np.linspace(0, 1.5 * np.pi, n))
        mu_true = np.full(n, -25.0)
        sigma_eps = 4.0
        common = np.cumsum(rng.normal(0.04, 1.8, size=n))
        y2 = 160.0 + common + rng.normal(0.0, 1.0, size=n)
        y2 = np.clip(y2, 130.0, 350.0)

    elif name == "EOAN.DE-RWE.DE":
        # E.ON vs RWE — the 2019/2020 asset swap inverted their economic
        # exposure. We model it as a smooth 3-month transition centred on
        # the Sept 2019 closing date so the Kalman filter can track it.
        swap_centre = pd.Timestamp("2019-09-30")
        days_to_swap = (dates - swap_centre).days.to_numpy(dtype=float)
        # Tanh ramp width ~ 60 business days = ~ 3 months
        ramp = 0.5 * (1 + np.tanh(days_to_swap / 60.0))
        gamma_before, gamma_after = 0.45, -0.30
        mu_before, mu_after = 1.0, 18.0
        gamma_true = gamma_before + ramp * (gamma_after - gamma_before)
        gamma_true += 0.02 * np.sin(np.linspace(0, 3 * np.pi, n))
        mu_true = mu_before + ramp * (mu_after - mu_before)
        sigma_eps = 0.18
        # RWE share price range roughly 8-45 EUR over the decade
        common = np.cumsum(rng.normal(0.012, 0.30, size=n))
        y2 = 18.0 + common + rng.normal(0.0, 0.20, size=n)
        y2 = np.clip(y2, 6.0, 55.0)

    elif name == "BMW.DE-MBG.DE":
        # BMW vs Mercedes-Benz Group (renamed from Daimler Feb 2022).
        # Hedge ratio jumps when Daimler Truck was spun off Dec 2021.
        break_date = pd.Timestamp("2022-02-01")
        before = dates < break_date
        gamma_true = np.where(before, 0.85, 1.20).astype(float)
        gamma_true += 0.04 * np.sin(np.linspace(0, 4 * np.pi, n))
        mu_true = np.where(before, 12.0, -8.0).astype(float)
        sigma_eps = 1.5
        common = np.cumsum(rng.normal(0.01, 0.7, size=n))
        y2 = 55.0 + common + rng.normal(0.0, 0.4, size=n)
        y2 = np.clip(y2, 30.0, 100.0)

    elif name == "SAP.DE-SIE.DE":
        # SAP vs Siemens. Different sectors so cointegration is weak;
        # the rolling LS will struggle, Kalman will smooth nicely.
        gamma_true = 0.75 + 0.08 * np.sin(np.linspace(0, 5 * np.pi, n))
        mu_true = 25.0 + 5.0 * np.cos(np.linspace(0, 3 * np.pi, n))
        sigma_eps = 3.5
        common = np.cumsum(rng.normal(0.03, 1.2, size=n))
        y2 = 90.0 + common + rng.normal(0.0, 0.8, size=n)
        y2 = np.clip(y2, 60.0, 180.0)

    else:
        raise ValueError(f"Unknown synthetic pair: {name!r}")

    noise = rng.normal(0.0, sigma_eps, size=n)
    y1 = mu_true + gamma_true * y2 + noise

    a, b = name.split("-")
    return pd.DataFrame({a: y1, b: y2}, index=dates)


def load_pair(
    ticker_a: str,
    ticker_b: str,
    start: str = "2013-01-01",
    end: str = "2022-12-31",
    prefer_synthetic: bool = False,
) -> pd.DataFrame:
    """Load adjusted close prices for ``ticker_a`` and ``ticker_b``.

    Tries ``yfinance`` first; if the download yields no rows (e.g. when
    the sandbox blocks the host), falls back to ``synthetic_pair``.
    """
    name = f"{ticker_a}-{ticker_b}"
    if prefer_synthetic:
        return synthetic_pair(name, start=start, end=end)

    try:
        import io
        import logging
        from contextlib import redirect_stderr, redirect_stdout

        import yfinance as yf

        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            df = yf.download(
                f"{ticker_a} {ticker_b}",
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
        if df.empty:
            raise RuntimeError("yfinance returned no rows")
        close = df["Close"] if "Close" in df.columns.get_level_values(0) else df
        out = pd.DataFrame(
            {ticker_a: close[ticker_a].values, ticker_b: close[ticker_b].values},
            index=close.index,
        ).dropna()
        if out.empty:
            raise RuntimeError("no overlapping rows after dropna")
        print(f"[load_pair] using real Yahoo data for {ticker_a}-{ticker_b}")
        return out
    except Exception:
        print(f"[load_pair] Yahoo unavailable, falling back to synthetic {name}")
        return synthetic_pair(name, start=start, end=end)
