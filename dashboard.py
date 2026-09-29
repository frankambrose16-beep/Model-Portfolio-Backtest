"""
dashboard.py
Step 3 of the John Doe model portfolio project: an interactive dashboard.

Reads the files that backtest.py saved in the data folder.

Install:  pip install streamlit pandas numpy
Run:      py -m streamlit run dashboard.py
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path("data")
RISK_FREE = 0.02          # same assumption as backtest.py
IPS_MAX_DRAWDOWN = 0.20   # the client's hard limit

STRESS_PERIODS = {
    "COVID crash (Feb 19 - Mar 23, 2020)": ("2020-02-19", "2020-03-23"),
    "2022 (full year)": ("2022-01-01", "2022-12-31"),
}

st.set_page_config(page_title="Model Portfolio: John Doe", layout="wide")


# ---------------------------------------------------------------------------
# Data and calculations
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    returns = pd.read_csv(DATA_DIR / "backtest_returns.csv",
                          index_col=0, parse_dates=True)
    weights = pd.read_csv(DATA_DIR / "weights_by_year.csv", index_col=0)
    return returns, weights


def max_drawdown(r):
    """Largest peak-to-trough fall, as a positive number."""
    wealth = (1 + r).cumprod()
    return float((1 - wealth / wealth.cummax()).max())


def summarize(r):
    years = len(r) / 252
    vol = r.std() * np.sqrt(252)
    return {
        "return": (1 + r).prod() ** (1 / years) - 1,
        "vol": vol,
        "sharpe": (r.mean() * 252 - RISK_FREE) / vol,
        "drawdown": max_drawdown(r),
    }


def draw_lines(df, colors, value_format):
    """Line chart with a labeled legend, a zoomed y-axis, and hover values."""
    long = df.rename_axis("Date").reset_index().melt(
        "Date", var_name="Series", value_name="Value")
    chart = (
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Value:Q", title=None, scale=alt.Scale(zero=False),
                    axis=alt.Axis(format=value_format)),
            color=alt.Color("Series:N",
                            scale=alt.Scale(domain=list(df.columns), range=colors),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("Date:T"), "Series:N",
                     alt.Tooltip("Value:Q", format=value_format)],
        )
        .properties(width="container", height=380)
    )
    st.altair_chart(chart)


try:
    returns, weights = load_data()
except FileNotFoundError:
    st.error("Could not find the files in the data folder. "
             "Run backtest.py first, and start this dashboard from the same folder.")
    st.stop()

# ---------------------------------------------------------------------------
# Header and sidebar
# ---------------------------------------------------------------------------
st.title("Model portfolio for John Doe")
st.caption("Fictional client: age 45, growth-focused, retires at 65, will not accept "
           "more than a 20% drop from a peak. Benchmark: 60% S&P 500 (VOO) / "
           "40% US bonds (BND). Weights are reset each year using only past data.")

first_day, last_day = returns.index.min().date(), returns.index.max().date()
st.sidebar.header("Controls")
picked = st.sidebar.date_input("Date range", value=(first_day, last_day),
                               min_value=first_day, max_value=last_day)
if len(picked) != 2:
    st.info("Pick an end date to continue.")
    st.stop()

start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
window = returns.loc[start:end]
if len(window) < 30:
    st.warning("Pick a longer date range (at least about six weeks).")
    st.stop()

show_benchmark = st.sidebar.checkbox("Show benchmark on charts", value=True)
st.sidebar.markdown("---")
st.sidebar.caption("Past results do not predict future returns. Taxes and trading "
                   "costs are ignored.")

port, bench = window["portfolio"], window["benchmark"]
p, b = summarize(port), summarize(bench)

# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Annual return", f"{p['return']:.1%}",
          f"{(p['return'] - b['return']) * 100:+.1f} pts vs benchmark",
          help="Average yearly growth, with compounding.")
c2.metric("Volatility", f"{p['vol']:.1%}",
          f"{(p['vol'] - b['vol']) * 100:+.1f} pts vs benchmark",
          delta_color="inverse",
          help="How much returns bounce around in a typical year. "
               "Lower means a smoother ride.")
c3.metric("Sharpe ratio", f"{p['sharpe']:.2f}", f"{p['sharpe'] - b['sharpe']:+.2f} vs benchmark",
          help="Return earned for each unit of bumpiness, after a 2% risk-free rate. "
               "Higher is better.")
c4.metric("Worst drawdown", f"-{p['drawdown']:.1%}",
          f"{(b['drawdown'] - p['drawdown']) * 100:+.1f} pts vs benchmark",
          help="The biggest fall from a peak before recovering. A peak of $100k "
               "that sinks to $80k is a 20% drawdown.")

# ---------------------------------------------------------------------------
# Stress tests (also used by the scorecard)
# ---------------------------------------------------------------------------
stress_rows, stress_wins = {}, []
for name, (a, z) in STRESS_PERIODS.items():
    sp, sb = returns["portfolio"].loc[a:z], returns["benchmark"].loc[a:z]
    if len(sp) == 0:
        continue
    stress_rows[name] = {
        "Portfolio return": (1 + sp).prod() - 1,
        "Benchmark return": (1 + sb).prod() - 1,
        "Portfolio worst drawdown": -max_drawdown(sp),
        "Benchmark worst drawdown": -max_drawdown(sb),
    }
    stress_wins.append((1 + sp).prod() > (1 + sb).prod())

# ---------------------------------------------------------------------------
# IPS scorecard
# ---------------------------------------------------------------------------
st.subheader("Did it meet John's goals?")
s1, s2, s3 = st.columns(3)

with s1:
    if p["drawdown"] <= IPS_MAX_DRAWDOWN:
        st.success(f"Drawdown limit: PASS\n\nWorst fall was {p['drawdown']:.1%} "
                   f"against a {IPS_MAX_DRAWDOWN:.0%} limit.")
    else:
        st.error(f"Drawdown limit: FAIL\n\nWorst fall was {p['drawdown']:.1%} "
                 f"against a {IPS_MAX_DRAWDOWN:.0%} limit.")
with s2:
    if p["sharpe"] >= b["sharpe"]:
        st.success(f"Beat benchmark Sharpe: PASS\n\n{p['sharpe']:.2f} vs {b['sharpe']:.2f}.")
    else:
        st.error(f"Beat benchmark Sharpe: FAIL\n\n{p['sharpe']:.2f} vs {b['sharpe']:.2f}.")
with s3:
    if stress_wins and all(stress_wins):
        st.success("Stress periods: PASS\n\nLost less than the benchmark in every "
                   "stress period tested.")
    else:
        st.error("Stress periods: FAIL\n\nDid worse than the benchmark in at least "
                 "one stress period.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["About John (IPS)", "Growth", "Drawdown", "Weights", "Stress tests",
     "What went wrong"])

with tab0:
    st.markdown("An Investment Policy Statement (IPS) is the rulebook agreed "
                "with the client before any money is invested. Everything in "
                "this dashboard is judged against it.")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
**The client**

John Doe (fictional), 45, marketing director, married with two kids.

**Objective**

Long-term growth for retirement at 65, not income.

**Time horizon**

About 20 years until retirement, then about 25 years of withdrawals.

**Return goal**

Beat the benchmark over rolling 5-year periods, aiming for roughly 6-7% a
year after inflation.

**Risk limits**

- Maximum drawdown of 20%, treated as a hard constraint. Drawdown is the fall
  from a peak to the next low point.
- Target volatility of about 10-12% a year.

**Benchmark**

60% S&P 500 (VOO) and 40% US aggregate bonds (BND).
"""
        )
    with right:
        st.markdown("**Portfolio rules the optimizer must follow**")
        rules = pd.DataFrame(
            {
                "Rule": ["Instruments", "Direction", "Single holding",
                         "Stocks (total)", "Bonds (total)", "Holdings",
                         "Rebalancing", "Taxes and trading costs"],
                "Limit": ["ETFs only, expense ratio 0.25% or less",
                          "Long only, no shorting or leverage",
                          "Maximum 20%", "Maximum 70%", "Minimum 20%",
                          "8 to 12", "Once a year", "Ignored (a limitation)"],
            }
        )
        st.dataframe(rules, hide_index=True)
        st.markdown(
            """
**How success is judged**

1. Worst drawdown stays within 20%.
2. Sharpe ratio matches or beats the benchmark.
3. Loses less than the benchmark in stress periods (the 2020 crash and 2022).
"""
        )

with tab1:
    st.markdown("**Growth of $10,000** over the selected dates.")
    growth = pd.DataFrame({"Model portfolio": 10000 * (1 + port).cumprod()})
    colors = ["#1f77b4"]
    if show_benchmark:
        growth["60/40 benchmark"] = 10000 * (1 + bench).cumprod()
        colors.append("#ff7f0e")
    draw_lines(growth, colors, "$,.0f")

with tab2:
    st.markdown("**How far below its peak** each portfolio was on every day. "
                "The red line is John's 20% limit.")
    def drawdown_series(r):
        wealth = (1 + r).cumprod()
        return wealth / wealth.cummax() - 1

    dd = pd.DataFrame({"Model portfolio": drawdown_series(port)})
    colors = ["#1f77b4"]
    if show_benchmark:
        dd["60/40 benchmark"] = drawdown_series(bench)
        colors.append("#ff7f0e")
    dd["IPS limit (-20%)"] = -IPS_MAX_DRAWDOWN
    colors.append("#d62728")
    draw_lines(dd, colors, ".0%")

with tab3:
    st.markdown("**What the portfolio held.** Once a year the optimizer resets the "
                "weights using only the previous three years of data.")
    date_choice = st.selectbox("Rebalance date", list(weights.index))
    chosen = weights.loc[date_choice]
    chosen = chosen[chosen > 0.0005].sort_values(ascending=False)
    st.bar_chart((chosen * 100).rename("Weight (%)"))
    st.markdown("**All years side by side (% of portfolio)**")
    st.bar_chart(weights * 100)
    st.dataframe((weights * 100).round(1))

with tab4:
    st.markdown("**How each portfolio did when markets fell.** These use the full "
                "backtest, whatever dates are picked on the left.")
    if stress_rows:
        table = (pd.DataFrame(stress_rows).T * 100).round(1)
        table.columns = [c + " (%)" for c in table.columns]
        st.dataframe(table)
    else:
        st.info("The backtest does not cover the stress periods.")

with tab5:
    hit_cap = (weights >= 0.199)
    voo_cap = int(hit_cap["VOO"].sum()) if "VOO" in weights else 0
    total_cells = int((weights > 0.0005).sum().sum())
    st.markdown(
        f"""
- **The optimizer kept maxing out its favorites.** VOO sat at the 20% cap in
  {voo_cap} of {len(weights)} yearly rebalances, and {int(hit_cap.sum().sum())} of the
  {total_cells} holdings it chose were pinned at the cap. Optimizers trust past
  averages too much and pile into whatever did best recently.
- **Some funds were almost never used.** Check the Weights tab: funds that looked
  weaker in the training window get zero, even though they may help in the next crisis.
- **The drawdown margin is thin.** The optimizer aimed for a smaller drawdown on its
  training data than the 20% limit, yet the real worst fall came close to the limit.
  The past is a rough guide to the future, not a promise.
- **The benchmark earned more.** This period was strong for US stocks, so holding
  less stock meant giving up return in exchange for a smoother ride.
- **Simplifications.** No taxes, no trading costs, one fixed 2% risk-free rate, and
  only about seven years of walk-forward results.
"""
    )
