"""
data_pull.py
Step 1 of the John Doe model portfolio project.

Downloads ~10 years of adjusted daily prices for the ETF universe,
checks that every fund has a full history, and saves clean price and
return files for the optimizer and backtest.

Install:  pip install yfinance pandas
Run:      python data_pull.py
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
UNIVERSE = {
    "VOO": "US large-cap stocks",
    "IJR": "US small-cap stocks",
    "VEA": "Developed international stocks",
    "VWO": "Emerging market stocks",
    "BND": "US aggregate bonds",
    "VGLT": "Long-term Treasuries",
    "VCIT": "Corporate bonds",
    "SCHP": "Inflation-protected bonds",
    "VNQ": "Real estate",
    "IAU": "Gold",
    "SHV": "Short-term Treasuries (cash-like)",
}

YEARS = 10
TOLERANCE_DAYS = 7  # a fund may start a few days after the window opens
OUT_DIR = Path("data")


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
def download_prices(tickers, start, end):
    """Return a DataFrame of adjusted close prices, one column per ticker.

    auto_adjust=True folds dividends and splits into the price, so
    returns computed from it are total returns.
    """
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("No data came back. Check your internet connection.")

    prices = raw["Close"]
    if isinstance(prices, pd.Series):  # only happens with a single ticker
        prices = prices.to_frame(tickers[0])
    return prices[tickers]  # keep the column order stable


def check_history(prices, start, tolerance_days):
    """Print a per-ticker history report and return tickers that fail."""
    cutoff = pd.Timestamp(start) + pd.Timedelta(days=tolerance_days)
    failed = []

    print(f"{'Ticker':<8}{'First price':<14}{'Missing days':<14}Status")
    print("-" * 46)
    for ticker in prices.columns:
        series = prices[ticker]
        first_valid = series.first_valid_index()
        if first_valid is None:
            print(f"{ticker:<8}{'none':<14}{'-':<14}FAIL (no data)")
            failed.append(ticker)
            continue

        missing = int(series.loc[first_valid:].isna().sum())
        ok = first_valid <= cutoff
        status = "OK" if ok else "FAIL (starts too late)"
        print(f"{ticker:<8}{first_valid.date()!s:<14}{missing:<14}{status}")
        if not ok:
            failed.append(ticker)
    return failed


def main():
    end = date.today()
    start = end - timedelta(days=365 * YEARS)
    tickers = list(UNIVERSE)

    print(f"Downloading {len(tickers)} ETFs from {start} to {end}...\n")
    prices = download_prices(tickers, start.isoformat(), end.isoformat())

    failed = check_history(prices, start, TOLERANCE_DAYS)
    if failed:
        print(f"\nWARNING: {failed} do not cover the full window.")
        print("Drop them, swap them, or shorten the backtest before continuing.")

    # Forward-fill tiny gaps (e.g., a missing print on one day), then drop
    # any leading rows where some fund has not started trading yet.
    prices = prices.ffill().dropna()

    returns = prices.pct_change().dropna()

    OUT_DIR.mkdir(exist_ok=True)
    prices.to_csv(OUT_DIR / "prices.csv")
    returns.to_csv(OUT_DIR / "returns.csv")

    print(f"\nSaved {len(prices)} trading days "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")
    print(f"Files: {OUT_DIR / 'prices.csv'}, {OUT_DIR / 'returns.csv'}")

    # Quick sanity summary: annualized return and volatility per fund
    summary = pd.DataFrame(
        {
            "Asset class": pd.Series(UNIVERSE),
            "Ann. return": returns.mean() * 252,
            "Ann. volatility": returns.std() * (252 ** 0.5),
        }
    ).round(3)
    print("\n", summary.to_string())


if __name__ == "__main__":
    main()
