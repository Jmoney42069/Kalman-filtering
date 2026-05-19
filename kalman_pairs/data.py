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

    Two flavours are produced to mirror the textbook experiments:

    * ``"EWA-EWC"`` — a stable cointegrated pair with hedge ratio ~0.6.
    * ``"KO-PEP"`` — a marginally cointegrated pair with a structural
      break in early 2020 that imitates the COVID dislocation discussed
      in the chapter.
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
