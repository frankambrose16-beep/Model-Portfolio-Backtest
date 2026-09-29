"""
backtest.py
Step 2 of the John Doe model portfolio project.

Walk-forward backtest:
  Each year, look only at the previous 3 years of data, find the
  max-Sharpe weights that obey the IPS rules (including the 20% max
  drawdown as a hard constraint), hold them for one year, then repeat.
  A 60% VOO / 40% BND benchmark is rebalanced on the same dates.

Needs data/returns.csv from data_pull.py.

Install:  pip install numpy pandas scipy matplotlib
Run:      python backtest.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# IPS rules and settings
# ---------------------------------------------------------------------------
STOCKS = ["VOO", "IJR", "VEA", "VWO"]
BONDS = ["BND", "VGLT", "VCIT", "SCHP", "SHV"]

MAX_POSITION = 0.20      # no single holding above 20%
MAX_STOCKS = 0.70        # stocks (total) at most 70%
MIN_BONDS = 0.20         # bonds (total) at least 20%
IPS_MAX_DRAWDOWN = 0.20  # the client's hard limit

# The optimizer aims a bit under the IPS limit on its training data, because
# the future rarely matches the past. Set to 0.20 to use the limit exactly.
TRAIN_DD_LIMIT = 0.15

RISK_FREE = 0.02         # assumption used in the Sharpe ratio
LOOKBACK = 756           # about 3 years of trading days
REBAL_DAYS = 252         # about 1 year
N_STARTS = 8             # random starting points for the optimizer
SEED = 42

BENCHMARK = {"VOO": 0.60, "BND": 0.40}

STRESS_PERIODS = {
    "COVID crash (Feb 19 - Mar 23, 2020)": ("2020-02-19", "2020-03-23"),
    "2022 (full year)": ("2022-01-01", "2022-12-31"),
}

DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def max_drawdown(daily_returns):
    """Largest peak-to-trough fall, as a positive number (0.20 = 20%)."""
    wealth = np.cumprod(1 + np.asarray(daily_returns))
    peak = np.maximum.accumulate(wealth)
    return float(np.max(1 - wealth / peak))


def optimize(train):
    """Max-Sharpe weights on the training window, obeying all IPS rules."""
    cols = list(train.columns)
    n = len(cols)
    stock_idx = [cols.index(t) for t in STOCKS]
    bond_idx = [cols.index(t) for t in BONDS]

    values = train.values
    mu = train.mean().values * 252
    cov = train.cov().values * 252

    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu - RISK_FREE) / vol

    def train_drawdown(w):
        return max_drawdown(values @ w)

    base_constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: MAX_STOCKS - np.sum(w[stock_idx])},
        {"type": "ineq", "fun": lambda w: np.sum(w[bond_idx]) - MIN_BONDS},
    ]
    dd_constraint = {"type": "ineq",
                     "fun": lambda w: TRAIN_DD_LIMIT - train_drawdown(w)}
    bounds = [(0, MAX_POSITION)] * n

    rng = np.random.default_rng(SEED)
    starts = [np.ones(n) / n] + [rng.dirichlet(np.ones(n)) for _ in range(N_STARTS)]

    def is_valid(w, check_dd):
        ok = (abs(w.sum() - 1) < 1e-4 and w.min() > -1e-6
              and w.max() < MAX_POSITION + 1e-4
              and w[stock_idx].sum() < MAX_STOCKS + 1e-4
              and w[bond_idx].sum() > MIN_BONDS - 1e-4)
        if check_dd:
            ok = ok and train_drawdown(w) <= TRAIN_DD_LIMIT + 1e-4
        return ok

    best = None
    for x0 in starts:
        res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds,
                       constraints=base_constraints + [dd_constraint],
                       options={"maxiter": 300, "ftol": 1e-9})
        w = np.clip(res.x, 0, None)
        w = w / w.sum()
        if is_valid(w, check_dd=True):
            if best is None or neg_sharpe(w) < neg_sharpe(best):
                best = w

    if best is None:
        # No portfolio met the drawdown limit: fall back to the one with the
        # smallest drawdown that still obeys the other rules.
        print("  WARNING: drawdown limit not reachable, using the "
              "lowest-drawdown portfolio instead.")
        for x0 in starts:
            res = minimize(train_drawdown, x0, method="SLSQP", bounds=bounds,
                           constraints=base_constraints,
                           options={"maxiter": 300})
            w = np.clip(res.x, 0, None)
            w = w / w.sum()
            if is_valid(w, check_dd=False):
                if best is None or train_drawdown(w) < train_drawdown(best):
                    best = w

    return pd.Series(best, index=cols)


def hold_one_period(test, weights, start_wealth):
    """Buy and hold for one period; weights drift with prices."""
    growth = (1 + test).cumprod().values
    values = start_wealth * (growth @ weights.reindex(test.columns).values)
    return pd.Series(values, index=test.index)


def to_daily_returns(wealth):
    r = wealth.pct_change()
    r.iloc[0] = wealth.iloc[0] / 1.0 - 1.0
    return r


def summarize(r):
    years = len(r) / 252
    total = (1 + r).prod()
    return {
        "Annual return": total ** (1 / years) - 1,
        "Volatility": r.std() * np.sqrt(252),
        "Sharpe ratio": (r.mean() * 252 - RISK_FREE) / (r.std() * np.sqrt(252)),
        "Max drawdown": -max_drawdown(r),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    returns = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True)

    port_wealth, bench_wealth = [], []
    port_start = bench_start = 1.0
    weights_by_date = {}

    bench_w = pd.Series(BENCHMARK).reindex(returns.columns).fillna(0.0)

    print("Running walk-forward backtest...\n")
    for i in range(LOOKBACK, len(returns), REBAL_DAYS):
        train = returns.iloc[i - LOOKBACK:i]
        test = returns.iloc[i:i + REBAL_DAYS]
        if len(test) < 20:
            break

        print(f"Rebalance {test.index[0].date()}: "
              f"training on {train.index[0].date()} to {train.index[-1].date()}")
        w = optimize(train)
        weights_by_date[test.index[0].date()] = w

        p = hold_one_period(test, w, port_start)
        b = hold_one_period(test, bench_w, bench_start)
        port_wealth.append(p)
        bench_wealth.append(b)
        port_start, bench_start = p.iloc[-1], b.iloc[-1]

    port_w = pd.concat(port_wealth)
    bench_w_series = pd.concat(bench_wealth)
    port_r = to_daily_returns(port_w)
    bench_r = to_daily_returns(bench_w_series)

    # ---- weights table
    weights_df = pd.DataFrame(weights_by_date).T.round(3)
    print("\nWeights chosen each year:\n")
    print(weights_df.to_string())

    # ---- performance table
    perf = pd.DataFrame({"Model portfolio": summarize(port_r),
                         "60/40 benchmark": summarize(bench_r)})
    print(f"\nPerformance ({port_r.index[0].date()} to {port_r.index[-1].date()}):\n")
    print(perf.round(3).to_string())

    dd = perf.loc["Max drawdown", "Model portfolio"]
    verdict = "WITHIN" if -dd <= IPS_MAX_DRAWDOWN else "BREACHED"
    print(f"\nIPS drawdown limit of {IPS_MAX_DRAWDOWN:.0%}: {verdict} "
          f"(actual {-dd:.1%})")

    # ---- stress periods
    rows = {}
    for name, (a, b) in STRESS_PERIODS.items():
        p, m = port_r.loc[a:b], bench_r.loc[a:b]
        if len(p) == 0:
            continue
        rows[name] = {
            "Portfolio return": (1 + p).prod() - 1,
            "Benchmark return": (1 + m).prod() - 1,
            "Portfolio max drawdown": -max_drawdown(p),
            "Benchmark max drawdown": -max_drawdown(m),
        }
    print("\nStress periods:\n")
    print(pd.DataFrame(rows).T.round(3).to_string())

    # ---- save files
    out = pd.DataFrame({"portfolio": port_r, "benchmark": bench_r})
    out.to_csv(DATA_DIR / "backtest_returns.csv")
    weights_df.to_csv(DATA_DIR / "weights_by_year.csv")
    perf.to_csv(DATA_DIR / "performance_summary.csv")

    # ---- chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        (10000 * (1 + port_r).cumprod()).plot(ax=ax1, label="Model portfolio")
        (10000 * (1 + bench_r).cumprod()).plot(ax=ax1, label="60/40 benchmark")
        ax1.set_title("Growth of $10,000")
        ax1.legend()

        for series, label in [(port_r, "Model portfolio"), (bench_r, "60/40 benchmark")]:
            wealth = (1 + series).cumprod()
            (wealth / wealth.cummax() - 1).plot(ax=ax2, label=label)
        ax2.axhline(-IPS_MAX_DRAWDOWN, color="red", linestyle="--",
                    label="IPS limit (-20%)")
        ax2.set_title("Drawdown from peak")
        ax2.legend()

        fig.tight_layout()
        fig.savefig(DATA_DIR / "backtest.png", dpi=150)
        print(f"\nSaved chart to {DATA_DIR / 'backtest.png'}")
    except ImportError:
        print("\n(Install matplotlib to also get a chart: pip install matplotlib)")


if __name__ == "__main__":
    main()
