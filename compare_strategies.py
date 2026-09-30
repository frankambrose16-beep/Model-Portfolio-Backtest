"""
compare_strategies.py
Step 4 of the John Doe model portfolio project: Version 2.

Version 1 (backtest.py) used max-Sharpe optimization and trailed the 60/40
benchmark. This script asks: was the problem the optimization METHOD?

It re-runs the exact same experiment with five methods side by side:
  same client, same IPS rules, same 11 ETFs, same 3-year lookback,
  same yearly rebalance dates, same 60/40 benchmark.
Only the way the weights are chosen changes.

  1. Max-Sharpe (Version 1)   - the original optimizer, unchanged
  2. Max-Sharpe (shrunk)      - same goal, but with steadier inputs:
                                Ledoit-Wolf covariance and expected returns
                                pulled halfway toward the average
  3. Minimum variance         - smoothest ride; ignores return forecasts
  4. Risk parity              - each fund adds a similar share of total risk
  5. Equal weight             - 1/11 in every fund (a no-forecast baseline)

Two return-seeking strategies were added after Version 2 showed that the
real gap was too little stock. They use a TREND FILTER checked once a
month (a change from the IPS's yearly rebalancing):
  6. 60/40 + trend filter     - 60% VOO / 40% BND, but when VOO is below its
                                200-day average price, the stock part moves
                                to cash (SHV) until the trend turns back up
  7. 70/30 + trend filter     - same rule with 70% VOO / 30% BND

Every optimized method must obey the same IPS rules as Version 1, including
the drawdown limit on its training data. A small trading cost is charged on
every rebalance for every portfolio, including the benchmark. (Fund expense
ratios are already inside the adjusted prices, so they are not charged again.)

Needs data/returns.csv from data_pull.py and backtest.py in the same folder.

Install:  pip install numpy pandas scipy matplotlib
Run:      python compare_strategies.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Reuse Version 1's rules and helpers so both versions follow identical rules.
import backtest as v1
from backtest import (BENCHMARK, BONDS, DATA_DIR, IPS_MAX_DRAWDOWN, LOOKBACK,
                      MAX_POSITION, MAX_STOCKS, MIN_BONDS, N_STARTS,
                      REBAL_DAYS, RISK_FREE, SEED, STOCKS, STRESS_PERIODS,
                      TRAIN_DD_LIMIT, max_drawdown)

# ---------------------------------------------------------------------------
# Version 2 settings
# ---------------------------------------------------------------------------
TRADING_COST = 0.0005   # 0.05% of the amount traded at each rebalance
RETURN_SHRINK = 0.50    # pull each fund's expected return halfway to the average

STRATEGIES = ["Max-Sharpe (Version 1)", "Max-Sharpe (shrunk)",
              "Minimum variance", "Risk parity", "Equal weight"]
BENCH_NAME = "60/40 benchmark"

# Trend-filter strategies: base mix, checked every CHECK_DAYS trading days
TREND_STRATEGIES = {
    "60/40 + trend filter": {"VOO": 0.60, "BND": 0.40},
    "70/30 + trend filter": {"VOO": 0.70, "BND": 0.30},
}
TREND_FUND = "VOO"      # the fund whose trend is watched
TREND_DAYS = 200        # moving-average length (about 10 months)
CHECK_DAYS = 21         # about once a month
SAFE_FUND = "SHV"       # where the stock money waits when the trend is down


# ---------------------------------------------------------------------------
# Steadier inputs
# ---------------------------------------------------------------------------
def ledoit_wolf_cov(train):
    """
    Ledoit-Wolf (2003) shrinkage toward a constant-correlation matrix.

    Plain idea: the raw covariance from 3 years of data is noisy. We blend it
    with a simple, stable version that assumes every pair of funds has the
    same (average) correlation. The data decides how much to blend.
    Returns an annualized covariance matrix.
    """
    x = train.values
    t, n = x.shape
    y = x - x.mean(axis=0)
    sample = y.T @ y / t
    var = np.diag(sample)
    sd = np.sqrt(var)
    corr = sample / np.outer(sd, sd)
    rbar = (corr.sum() - n) / (n * (n - 1))
    target = rbar * np.outer(sd, sd)
    np.fill_diagonal(target, var)

    # How noisy is the sample? (pi), how much of the noise the target
    # shares (rho), and how far apart the two matrices are (gamma).
    y2 = y ** 2
    pi_mat = y2.T @ y2 / t - sample ** 2
    pi_hat = pi_mat.sum()
    theta = (y ** 3).T @ y / t - var[:, None] * sample
    ratio = np.outer(1 / sd, sd)          # ratio[i, j] = sd_j / sd_i
    off = ~np.eye(n, dtype=bool)
    rho_hat = np.trace(pi_mat) + rbar * (ratio * theta)[off].sum()
    gamma_hat = ((target - sample) ** 2).sum()
    shrink = max(0.0, min(1.0, (pi_hat - rho_hat) / gamma_hat / t))

    return (shrink * target + (1 - shrink) * sample) * 252, shrink


def shrunk_means(train):
    """Expected returns pulled halfway toward the average of all funds."""
    mu = train.mean().values * 252
    return (1 - RETURN_SHRINK) * mu + RETURN_SHRINK * mu.mean()


# ---------------------------------------------------------------------------
# One optimizer, many objectives, same IPS rules
# ---------------------------------------------------------------------------
def ips_optimize(train, objective):
    """
    Minimize `objective(w)` subject to every IPS rule, including the drawdown
    limit on the training data. Same rules and fallback as Version 1.
    """
    cols = list(train.columns)
    n = len(cols)
    stock_idx = [cols.index(t) for t in STOCKS]
    bond_idx = [cols.index(t) for t in BONDS]
    values = train.values

    def train_dd(w):
        return max_drawdown(values @ w)

    base = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: MAX_STOCKS - np.sum(w[stock_idx])},
        {"type": "ineq", "fun": lambda w: np.sum(w[bond_idx]) - MIN_BONDS},
    ]
    dd = {"type": "ineq", "fun": lambda w: TRAIN_DD_LIMIT - train_dd(w)}
    bounds = [(0, MAX_POSITION)] * n

    def valid(w, check_dd):
        ok = (abs(w.sum() - 1) < 1e-4 and w.min() > -1e-6
              and w.max() < MAX_POSITION + 1e-4
              and w[stock_idx].sum() < MAX_STOCKS + 1e-4
              and w[bond_idx].sum() > MIN_BONDS - 1e-4)
        return ok and (not check_dd or train_dd(w) <= TRAIN_DD_LIMIT + 1e-4)

    rng = np.random.default_rng(SEED)
    starts = [np.ones(n) / n] + [rng.dirichlet(np.ones(n)) for _ in range(N_STARTS)]

    best = None
    for x0 in starts:
        res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                       constraints=base + [dd],
                       options={"maxiter": 500, "ftol": 1e-12})
        w = np.clip(res.x, 0, None)
        w = w / w.sum()
        if valid(w, True) and (best is None or objective(w) < objective(best)):
            best = w

    fallback = best is None
    if fallback:
        for x0 in starts:
            res = minimize(train_dd, x0, method="SLSQP", bounds=bounds,
                           constraints=base, options={"maxiter": 300})
            w = np.clip(res.x, 0, None)
            w = w / w.sum()
            if valid(w, False) and (best is None or train_dd(w) < train_dd(best)):
                best = w

    return pd.Series(best, index=cols), fallback


def weights_for(strategy, train):
    """Return (weights, note) for one strategy on one training window."""
    n = train.shape[1]
    cov = train.cov().values * 252

    if strategy == "Max-Sharpe (Version 1)":
        return v1.optimize(train), ""

    if strategy == "Max-Sharpe (shrunk)":
        cov_s, amount = ledoit_wolf_cov(train)
        mu_s = shrunk_means(train)

        def neg_sharpe(w):
            return -(w @ mu_s - RISK_FREE) / np.sqrt(w @ cov_s @ w)
        w, fb = ips_optimize(train, neg_sharpe)
        return w, f"covariance shrinkage {amount:.0%}" + (", FALLBACK" if fb else "")

    if strategy == "Minimum variance":
        w, fb = ips_optimize(train, lambda w: w @ cov @ w * 100)
        return w, "FALLBACK" if fb else ""

    if strategy == "Risk parity":
        def uneven_risk(w):
            total = w @ cov @ w
            share = w * (cov @ w) / total          # each fund's share of risk
            return np.sum((share - 1 / n) ** 2) * 1000
        w, fb = ips_optimize(train, uneven_risk)
        return w, "FALLBACK" if fb else ""

    if strategy == "Equal weight":
        return pd.Series(np.ones(n) / n, index=train.columns), ""

    raise ValueError(strategy)


# ---------------------------------------------------------------------------
# Backtest engine with trading costs
# ---------------------------------------------------------------------------
def run_period(test, target, current, wealth):
    """
    Trade from `current` weights to `target` (paying costs), then buy and
    hold for the period. Returns daily wealth, weights at period end, and
    the turnover (share of the portfolio traded, one way).
    """
    turnover = float(np.abs(target.values - current.values).sum()) / 2
    wealth = wealth * (1 - TRADING_COST * turnover * 2)
    growth = (1 + test).cumprod()
    holdings = wealth * growth.values * target.values      # $ in each fund
    path = pd.Series(holdings.sum(axis=1), index=test.index)
    end_w = pd.Series(holdings[-1] / holdings[-1].sum(), index=test.columns)
    return path, end_w, turnover


def run_all(returns):
    names = STRATEGIES + [BENCH_NAME]
    cash = pd.Series(0.0, index=returns.columns)
    state = {s: {"wealth": 1.0, "weights": cash, "paths": [], "turnover": []}
             for s in names}
    weight_rows, notes = [], []
    bench_w = pd.Series(BENCHMARK).reindex(returns.columns).fillna(0.0)

    for i in range(LOOKBACK, len(returns), REBAL_DAYS):
        train = returns.iloc[i - LOOKBACK:i]
        test = returns.iloc[i:i + REBAL_DAYS]
        if len(test) < 20:
            break
        date = test.index[0].date()
        print(f"Rebalance {date}")

        for s in names:
            if s == BENCH_NAME:
                w, note = bench_w, ""
            else:
                w, note = weights_for(s, train)
                if note:
                    notes.append(f"  {date} {s}: {note}")
                for ticker, value in w.items():
                    weight_rows.append({"Date": date, "Strategy": s,
                                        "Ticker": ticker, "Weight": round(value, 4)})
            st = state[s]
            path, end_w, turn = run_period(test, w, st["weights"], st["wealth"])
            st["paths"].append(path)
            st["turnover"].append(turn)
            st["wealth"], st["weights"] = path.iloc[-1], end_w

    daily = pd.DataFrame({s: v1.to_daily_returns(pd.concat(state[s]["paths"]))
                          for s in names})
    # Average turnover skips the first purchase (buying from cash).
    turnover = {s: np.mean(state[s]["turnover"][1:]) for s in names}
    return daily, pd.DataFrame(weight_rows), turnover, notes


def run_trend(returns, base_mix, start):
    """
    Monthly trend filter. On each check day, using only prices up to the day
    before, compare the stock fund's price with its 200-day average:
      above -> hold the normal mix
      below -> move the stock part to cash (SHV) until the next check
    """
    prices = (1 + returns).cumprod()
    base = pd.Series(base_mix).reindex(returns.columns).fillna(0.0)
    stock_part = base[TREND_FUND]
    weights = pd.Series(0.0, index=returns.columns)
    wealth, paths, turns, rows, switches, last = 1.0, [], [], [], 0, None

    for i in range(start, len(returns), CHECK_DAYS):
        history = prices[TREND_FUND].iloc[:i]
        uptrend = history.iloc[-1] > history.iloc[-TREND_DAYS:].mean()
        target = base.copy()
        if not uptrend:
            target[TREND_FUND] = 0.0
            target[SAFE_FUND] += stock_part
        if last is not None and uptrend != last:
            switches += 1
        last = uptrend

        test = returns.iloc[i:i + CHECK_DAYS]
        path, weights, turn = run_period(test, target, weights, wealth)
        wealth = path.iloc[-1]
        paths.append(path)
        turns.append(turn)
        for ticker, value in target.items():
            rows.append({"Date": test.index[0].date(), "Ticker": ticker,
                         "Weight": round(value, 4)})

    daily = v1.to_daily_returns(pd.concat(paths))
    years = len(daily) / 252
    yearly_turnover = sum(turns[1:]) / years
    return daily, pd.DataFrame(rows), yearly_turnover, switches


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
def longest_underwater_months(r):
    """Longest stretch (in months) spent below a previous peak."""
    wealth = (1 + r).cumprod()
    under = (wealth < wealth.cummax()).values
    longest = run = 0
    for flag in under:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return longest / 21


def scorecard(r, turnover, weights=None):
    years = len(r) / 252
    vol = r.std() * np.sqrt(252)
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(252)
    excess = r.mean() * 252 - RISK_FREE
    dd = max_drawdown(r)
    row = {
        "Annual return": (1 + r).prod() ** (1 / years) - 1,
        "Volatility": vol,
        "Sharpe ratio": excess / vol,
        "Sortino ratio": excess / downside,
        "Max drawdown": -dd,
        "Longest time underwater (months)": longest_underwater_months(r),
        "Avg yearly turnover": turnover,
        "IPS drawdown limit": "PASS" if dd <= IPS_MAX_DRAWDOWN else "FAIL",
    }
    for name, (a, b) in STRESS_PERIODS.items():
        row[name] = (1 + r.loc[a:b]).prod() - 1
    if weights is not None:
        row["Avg weight in stocks"] = weights.get("_stocks", np.nan)
        row["Avg weight in cash (SHV)"] = weights.get("SHV", np.nan)
        row["Avg number of funds held"] = weights.get("_count", np.nan)
    return row


def pretty(table):
    """Format the scorecard for printing: % where it makes sense."""
    plain = {"Sharpe ratio", "Sortino ratio", "Avg number of funds held"}
    out = table.astype(object).copy()
    for row in out.index:
        for col in out.columns:
            v = table.loc[row, col]
            if isinstance(v, str):
                continue
            if row in plain:
                out.loc[row, col] = f"{v:.2f}"
            elif row.startswith("Longest"):
                out.loc[row, col] = f"{v:.0f}"
            else:
                out.loc[row, col] = f"{v:.1%}"
    return out


def main():
    returns = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True)
    print("Round 1: running five methods through the same yearly walk-forward test...\n")
    daily, weights, turnover, notes = run_all(returns)

    print("\nRunning the trend-filter strategies (checked monthly)...")
    bench = daily.pop(BENCH_NAME)
    for name, mix in TREND_STRATEGIES.items():
        d, w, turn, switches = run_trend(returns, mix, LOOKBACK)
        daily[name] = d
        w.insert(1, "Strategy", name)
        weights = pd.concat([weights, w], ignore_index=True)
        turnover[name] = turn
        notes.append(f"  {name}: switched between stocks and cash {switches} times")
    daily[BENCH_NAME] = bench

    if notes:
        print("\nNotes:")
        print("\n".join(notes))

    # Average holdings per strategy (for the scorecard)
    avg = {}
    for s, grp in weights.groupby("Strategy"):
        wide = grp.pivot(index="Date", columns="Ticker", values="Weight")
        avg[s] = {"SHV": wide["SHV"].mean(),
                  "_stocks": wide[STOCKS].sum(axis=1).mean(),
                  "_count": (wide > 0.005).sum(axis=1).mean()}

    avg[BENCH_NAME] = {"SHV": 0.0, "_stocks": BENCHMARK.get("VOO", 0.0),
                       "_count": len(BENCHMARK)}
    table = pd.DataFrame({s: scorecard(daily[s], turnover[s], avg.get(s))
                          for s in daily.columns})

    print(f"\nResults ({daily.index[0].date()} to {daily.index[-1].date()}), "
          f"after {TRADING_COST:.2%} trading costs:\n")
    print(pretty(table).to_string())

    daily.to_csv(DATA_DIR / "strategy_returns.csv")
    weights.to_csv(DATA_DIR / "strategy_weights.csv", index=False)
    table.to_csv(DATA_DIR / "strategy_comparison.csv")
    print(f"\nSaved strategy_returns.csv, strategy_weights.csv and "
          f"strategy_comparison.csv in {DATA_DIR}/")

    make_chart(daily, table)


# Strategies highlighted in the chart; the rest are drawn in light gray.
FEATURED = ["Max-Sharpe (shrunk)", "60/40 + trend filter", "70/30 + trend filter"]


def make_chart(daily, table):
    """
    Three clean views instead of one crowded one:
      1. Growth of $10,000: featured strategies in color, the rest in gray
      2. Drawdown: one small panel per featured strategy, each against 60/40
      3. Risk vs return: every strategy as one labeled dot
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter, StrMethodFormatter
    except ImportError:
        print("(Install matplotlib to also get a chart: pip install matplotlib)")
        return

    ink, muted, grid, gray = "#0b0b0b", "#52514e", "#e6e5e0", "#c9c8c2"
    limit_red = "#d03b3b"
    colors = {"Max-Sharpe (shrunk)": "#eb6834", "60/40 + trend filter": "#008300",
              "70/30 + trend filter": "#4a3aa7", BENCH_NAME: "#52514e"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#b5b4ae",
                         "xtick.color": muted, "ytick.color": muted})

    fig = plt.figure(figsize=(12, 13))
    layout = fig.add_gridspec(3, 3, height_ratios=[3.2, 1.6, 2.6],
                              hspace=0.45, wspace=0.12)
    wealth = 10000 * (1 + daily).cumprod()

    # ---- 1. Growth ------------------------------------------------------
    ax = fig.add_subplot(layout[0, :])
    others = [c for c in daily.columns if c not in FEATURED + [BENCH_NAME]]
    for c in others:
        ax.plot(wealth.index, wealth[c], color=gray, lw=1)
    for c in FEATURED + [BENCH_NAME]:
        ax.plot(wealth.index, wealth[c], color=colors[c], lw=2.2,
                ls="--" if c == BENCH_NAME else "-")

    # Direct labels at the right end, nudged apart so they never overlap
    ends = sorted(((wealth[c].iloc[-1], c) for c in FEATURED + [BENCH_NAME]))
    gap, placed = 700, []
    for value, c in ends:
        y = max(value, placed[-1] + gap) if placed else value
        placed.append(y)
        ax.annotate(f"{c}  ${value:,.0f}", xy=(wealth.index[-1], value),
                    xytext=(wealth.index[-1] + pd.Timedelta(days=40), y),
                    color=colors[c], fontsize=10, fontweight="bold", va="center",
                    arrowprops=dict(arrowstyle="-", color=colors[c], lw=0.8))
    ax.text(wealth.index[-1] + pd.Timedelta(days=40),
            min(wealth[others].iloc[-1]) - 900,
            "Other Round 1 methods\n(gray lines)", color="#8a8983", fontsize=9,
            va="top")
    ax.set_xlim(wealth.index[0], wealth.index[-1] + pd.Timedelta(days=900))
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in
                   range(wealth.index[0].year + 1, wealth.index[-1].year + 1)])
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Growth of $10,000 (after trading costs)", loc="left",
                 fontsize=13, color=ink, pad=10)

    # ---- 2. Drawdown small multiples -----------------------------------
    def drawdown(c):
        w = (1 + daily[c]).cumprod()
        return w / w.cummax() - 1

    bench_dd = drawdown(BENCH_NAME)
    for k, c in enumerate(FEATURED):
        ax = fig.add_subplot(layout[1, k])
        ax.fill_between(bench_dd.index, bench_dd, 0, color=gray, lw=0,
                        label="60/40 benchmark")
        ax.plot(bench_dd.index, drawdown(c), color=colors[c], lw=1.4, label=c)
        ax.axhline(-IPS_MAX_DRAWDOWN, color=limit_red, ls=":", lw=1.2)
        worst = -table.loc["Max drawdown", c]
        ok = table.loc["IPS drawdown limit", c] == "PASS"
        ax.set_title(f"{c}\nworst {worst:.1%}  ·  {'within' if ok else 'breaks'} 20% limit",
                     loc="left", fontsize=10, color=colors[c] if ok else limit_red)
        ax.set_ylim(-0.25, 0.01)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        if k:
            ax.set_yticklabels([])
        else:
            ax.text(-0.02, 1.42, "Drawdown from peak   (gray area = 60/40 benchmark, "
                    "red dotted line = John's 20% limit)", transform=ax.transAxes,
                    fontsize=13, color=ink)

    # ---- 3. Risk vs return scatter -------------------------------------
    ax = fig.add_subplot(layout[2, :])
    ax.axvspan(IPS_MAX_DRAWDOWN * 100, 24.8, color="#fbe9e9", lw=0)
    ax.axvline(IPS_MAX_DRAWDOWN * 100, color=limit_red, ls=":", lw=1.2)
    ax.text(IPS_MAX_DRAWDOWN * 100 + 0.15, 3.3, "Breaks John's 20% limit",
            color=limit_red, fontsize=9, va="bottom")
    offsets = {"Max-Sharpe (Version 1)": (-9, 6), "Max-Sharpe (shrunk)": (9, 6),
               "Minimum variance": (9, -4), "Risk parity": (-9, 6),
               "Equal weight": (9, -4), "60/40 + trend filter": (-9, 8),
               "70/30 + trend filter": (-10, 6), BENCH_NAME: (9, -4)}
    for c in daily.columns:
        x = -table.loc["Max drawdown", c] * 100
        y = table.loc["Annual return", c] * 100
        color = colors.get(c, "#8a8983")
        ax.scatter(x, y, s=90 if c in colors else 55, color=color, zorder=3,
                   edgecolor="white", linewidth=1.5)
        dx, dy = offsets.get(c, (8, 0))
        ax.annotate(c, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", fontsize=9.5,
                    color=color if c in colors else muted,
                    fontweight="bold" if c in colors else "normal")
    ax.set_xlim(13, 24.8)
    ax.set_ylim(3, 11.5)
    ax.set_xlabel("Worst drop from peak (%)  →  more pain", color=muted)
    ax.set_ylabel("Annual return (%)", color=muted)
    ax.grid(color=grid, lw=0.8)
    ax.set_title("Risk vs return: up and to the left is better", loc="left",
                 fontsize=13, color=ink, pad=10)

    fig.savefig(DATA_DIR / "strategy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart to {DATA_DIR / 'strategy_comparison.png'}")


if __name__ == "__main__":
    main()
