"""
pages/2_Strategy_Comparison.py
Version 2 of the John Doe project: five optimization methods plus two
trend-filter strategies, side by side.

Streamlit adds this page to the sidebar of dashboard.py automatically.
Reads the files that compare_strategies.py saved in the data folder.

Run:  py -m streamlit run dashboard.py   (then pick this page in the sidebar)
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA_DIR = Path("data")
IPS_MAX_DRAWDOWN = 0.20
BENCH = "60/40 benchmark"

COLORS = {"Max-Sharpe (Version 1)": "#2a78d6", "Max-Sharpe (shrunk)": "#eb6834",
          "Minimum variance": "#1baf7a", "Risk parity": "#eda100",
          "Equal weight": "#e87ba4", "60/40 + trend filter": "#008300",
          "70/30 + trend filter": "#4a3aa7", BENCH: "#7a7974"}

ABOUT = {
    "Max-Sharpe (Version 1)": "The original optimizer. Picks the mix with the best "
    "return per unit of risk, using raw 3-year averages.",
    "Max-Sharpe (shrunk)": "Same goal, steadier inputs. Expected returns are pulled "
    "halfway toward the average fund, and the risk estimates are blended with a "
    "simpler model (Ledoit-Wolf). Like trusting a player's career stats over one hot month.",
    "Minimum variance": "Ignores return forecasts and just looks for the smoothest ride.",
    "Risk parity": "Sizes each fund so it adds a similar share of the total risk. "
    "Quiet funds (bonds) get more money, jumpy funds (stocks) get less.",
    "60/40 + trend filter": "60% VOO / 40% BND with a smoke detector. Once a month, "
    "if VOO is below its 200-day average price, the stock money moves to cash (SHV) "
    "until the trend turns back up. Checked monthly, not yearly.",
    "70/30 + trend filter": "The same monthly trend rule on a more aggressive "
    "70% VOO / 30% BND mix.",
    "Equal weight": "1/11 in every fund. No forecasting at all; the baseline any "
    "clever method should beat.",
}

# Funds colored by asset class: stocks blue, bonds green, cash gray,
# real estate orange, gold yellow.
FUND_COLORS = {"VOO": "#0d366b", "IJR": "#256abf", "VEA": "#5598e7", "VWO": "#9ec5f4",
               "BND": "#0b5e3f", "VGLT": "#1baf7a", "VCIT": "#5fd0a5",
               "SCHP": "#a6e6c9", "SHV": "#b5b4ae", "VNQ": "#eb6834",
               "IAU": "#eda100"}

st.set_page_config(page_title="Version 2: Strategy comparison", layout="wide")


@st.cache_data
def load():
    daily = pd.read_csv(DATA_DIR / "strategy_returns.csv", index_col=0, parse_dates=True)
    weights = pd.read_csv(DATA_DIR / "strategy_weights.csv")
    table = pd.read_csv(DATA_DIR / "strategy_comparison.csv", index_col=0)
    return daily, weights, table


try:
    daily, weights, table = load()
except FileNotFoundError:
    st.error("Could not find the Version 2 files. Run compare_strategies.py first, "
             "and start the dashboard from the project folder.")
    st.stop()

strategies = [c for c in daily.columns if c != BENCH]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Version 2: was the optimizer the problem?")
st.caption("Version 1 trailed the 60/40 benchmark. Round 1 re-runs the exact same "
           "test with five ways of choosing weights (same client, IPS rules, 11 ETFs, "
           "3-year lookback, yearly rebalance dates). Round 2 adds two return-seeking "
           "trend-filter strategies that are checked monthly. Every portfolio pays a "
           "0.05% trading cost on each trade.")

FEATURED = ["Max-Sharpe (shrunk)", "60/40 + trend filter", "70/30 + trend filter"]
picked = st.sidebar.multiselect(
    "Strategies to show", strategies,
    default=[s for s in FEATURED if s in strategies],
    help="Starts with the three featured strategies. Add others to compare.")
show_bench = st.sidebar.checkbox("Show 60/40 benchmark", value=True)
shown = picked + ([BENCH] if show_bench else [])
if not shown:
    st.info("Pick at least one strategy on the left.")
    st.stop()

with st.expander("What each method does"):
    for s in strategies:
        st.markdown(f"**{s}:** {ABOUT[s]}")

# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
st.subheader("Scorecard")
percent_rows = ["Annual return", "Volatility", "Max drawdown", "Avg yearly turnover",
                "COVID crash (Feb 19 - Mar 23, 2020)", "2022 (full year)",
                "Avg weight in stocks", "Avg weight in cash (SHV)"]
card = table[shown].copy().astype(object)
for row in card.index:
    for col in card.columns:
        v = table.loc[row, col]
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if row in percent_rows:
            card.loc[row, col] = f"{v:.1%}"
        elif row.startswith("Longest"):
            card.loc[row, col] = f"{v:.0f}"
        else:
            card.loc[row, col] = f"{v:.2f}"
st.dataframe(card, height=(len(card) + 1) * 35 + 3)
st.caption("Sortino is like Sharpe but only counts downside bumps. Turnover is the "
           "share of the portfolio traded in a typical year. \"Longest time "
           "underwater\" is the longest stretch spent below a previous high.")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def line_chart(df, fmt, rule=None):
    long = df.rename_axis("Date").reset_index().melt(
        "Date", var_name="Strategy", value_name="Value")
    base = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("Value:Q", title=None, scale=alt.Scale(zero=False),
                axis=alt.Axis(format=fmt)),
        color=alt.Color("Strategy:N",
                        scale=alt.Scale(domain=list(df.columns),
                                        range=[COLORS[c] for c in df.columns]),
                        legend=alt.Legend(orient="top", title=None, columns=4)),
        strokeDash=alt.condition(alt.datum.Strategy == BENCH,
                                 alt.value([6, 4]), alt.value([1, 0])),
        tooltip=[alt.Tooltip("Date:T"), "Strategy:N",
                 alt.Tooltip("Value:Q", format=fmt)],
    )
    chart = base
    if rule is not None:
        chart = chart + alt.Chart(pd.DataFrame({"y": [rule]})).mark_rule(
            color="#d03b3b", strokeDash=[2, 2]).encode(y="y:Q")
    st.altair_chart(chart.properties(width="container", height=360))


def drawdown(r):
    w = (1 + r).cumprod()
    return w / w.cummax() - 1


def drawdown_panel(name):
    """One strategy's drawdown (line) against the benchmark (gray area)."""
    df = pd.DataFrame({"Date": daily.index, "Strategy": drawdown(daily[name]).values,
                       "Benchmark": drawdown(daily[BENCH]).values})
    y = alt.Y("Benchmark:Q", title=None, axis=alt.Axis(format=".0%"),
              scale=alt.Scale(domain=[-0.25, 0]))
    area = alt.Chart(df).mark_area(color="#d9d8d2").encode(
        x=alt.X("Date:T", title=None), y=y)
    line = alt.Chart(df).mark_line(color=COLORS[name], strokeWidth=1.6).encode(
        x="Date:T", y=alt.Y("Strategy:Q", scale=alt.Scale(domain=[-0.25, 0])),
        tooltip=[alt.Tooltip("Date:T"),
                 alt.Tooltip("Strategy:Q", title=name, format=".1%"),
                 alt.Tooltip("Benchmark:Q", title=BENCH, format=".1%")])
    limit = alt.Chart(pd.DataFrame({"y": [-IPS_MAX_DRAWDOWN]})).mark_rule(
        color="#d03b3b", strokeDash=[2, 2]).encode(y="y:Q")
    worst = -float(table.loc["Max drawdown", name])
    ok = table.loc["IPS drawdown limit", name] == "PASS"
    st.markdown(f"**{name}**  \nworst {worst:.1%} · "
                f"{'within' if ok else ':red[breaks]'} the 20% limit")
    st.altair_chart((area + line + limit).properties(width="container", height=220))


tab1, tab2, tab5, tab3, tab4 = st.tabs(
    ["Growth", "Drawdown", "Risk vs return", "Weights", "What we learned"])

with tab1:
    st.markdown("**Growth of $10,000**, after trading costs. Add or remove "
                "strategies in the sidebar.")
    line_chart(10000 * (1 + daily[shown]).cumprod(), "$,.0f")

with tab2:
    st.markdown("**How far below its peak** each strategy was. The gray area is the "
                "60/40 benchmark and the red dotted line is John's 20% limit.")
    if not picked:
        st.info("Pick a strategy in the sidebar.")
    for start in range(0, len(picked), 3):
        cols = st.columns(3)
        for col, name in zip(cols, picked[start:start + 3]):
            with col:
                drawdown_panel(name)

with tab5:
    st.markdown("**Every strategy as one dot.** Higher means more return; further "
                "right means a bigger worst drop. Up and to the left is better. "
                "The pink zone breaks John's 20% limit.")
    pts = pd.DataFrame({
        "Strategy": daily.columns,
        "Worst drop": [-float(table.loc["Max drawdown", c]) for c in daily.columns],
        "Annual return": [float(table.loc["Annual return", c]) for c in daily.columns],
    })
    pts["Featured"] = pts["Strategy"].isin(FEATURED + [BENCH])
    x_scale = alt.Scale(domain=[0.13, 0.25], zero=False, nice=False)
    zone = alt.Chart(pd.DataFrame({"x": [IPS_MAX_DRAWDOWN], "x2": [0.25]})).mark_rect(
        color="#fbe9e9").encode(x=alt.X("x:Q", scale=x_scale), x2="x2:Q")
    x = alt.X("Worst drop:Q", title="Worst drop from peak", scale=x_scale,
              axis=alt.Axis(format=".0%", values=[0.14, 0.16, 0.18, 0.20, 0.22, 0.24]))
    y = alt.Y("Annual return:Q", title="Annual return",
              scale=alt.Scale(domain=[0.03, 0.115], nice=False),
              axis=alt.Axis(format=".0%", values=[0.04, 0.06, 0.08, 0.10]))
    color = alt.condition(alt.datum.Featured,
                          alt.Color("Strategy:N", legend=None,
                                    scale=alt.Scale(domain=list(COLORS),
                                                    range=list(COLORS.values()))),
                          alt.value("#9a9994"))
    dots = alt.Chart(pts).mark_circle(size=140, opacity=1, stroke="white",
                                      strokeWidth=1.5).encode(
        x=x, y=y, color=color,
        tooltip=["Strategy:N", alt.Tooltip("Annual return:Q", format=".1%"),
                 alt.Tooltip("Worst drop:Q", format=".1%")])
    # Labels to the right of each dot, except where two dots sit close together
    pts["Left"] = pts["Strategy"].isin(["Max-Sharpe (Version 1)", "Risk parity"])
    right = alt.Chart(pts[~pts["Left"]]).mark_text(
        align="left", dx=10, dy=-9, fontSize=12).encode(
        x=x, y=y, text="Strategy:N", color=color)
    left = alt.Chart(pts[pts["Left"]]).mark_text(
        align="right", dx=-10, dy=-9, fontSize=12).encode(
        x=x, y=y, text="Strategy:N", color=color)
    st.altair_chart((zone + dots + right + left).properties(width="container", height=420))

with tab3:
    choice = st.selectbox("Strategy", strategies)
    st.markdown(f"**What {choice} held at each rebalance.** {ABOUT[choice]} "
                "Blues are stocks, greens are bonds, gray is cash (SHV).")
    w = weights[weights["Strategy"] == choice].copy()
    w = w[w["Weight"] > 0.0005]
    w["rank"] = w["Ticker"].map({t: i for i, t in enumerate(FUND_COLORS)})
    bars = alt.Chart(w).mark_bar(stroke="white", strokeWidth=1).encode(
        x=alt.X("Date:N", title=None, axis=alt.Axis(labelOverlap=True)),
        y=alt.Y("Weight:Q", title=None, axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Ticker:N",
                        scale=alt.Scale(domain=list(FUND_COLORS),
                                        range=list(FUND_COLORS.values())),
                        legend=alt.Legend(orient="right", title="Fund")),
        order=alt.Order("rank:Q"),
        tooltip=["Date:N", "Ticker:N", alt.Tooltip("Weight:Q", format=".1%")],
    )
    st.altair_chart(bars.properties(width="container", height=360))
    wide = w.pivot(index="Date", columns="Ticker", values="Weight").fillna(0)
    st.dataframe((wide * 100).round(1))


def val(row, col):
    return float(table.loc[row, col])


with tab4:
    v1, v2 = "Max-Sharpe (Version 1)", "Max-Sharpe (shrunk)"
    st.markdown(
        f"""
- **Steadier inputs helped, a little.** Shrunk max-Sharpe earned about the same
  return as Version 1 ({val('Annual return', v2):.1%} vs {val('Annual return', v1):.1%})
  with a smoother ride, so its Sharpe rose from {val('Sharpe ratio', v1):.2f} to
  {val('Sharpe ratio', v2):.2f}, level with the benchmark's {val('Sharpe ratio', BENCH):.2f}.
  Its worst fall shrank from {-val('Max drawdown', v1):.1%} to {-val('Max drawdown', v2):.1%},
  it lost the least in 2022, and it traded less.
- **The benchmark broke John's rules.** 60/40 fell {-val('Max drawdown', BENCH):.1%}
  from its peak, past the 20% limit. So did equal weight. All four optimized
  Round 1 methods stayed inside it. Higher return that the client said he could not live with is not a fair win.
- **The smoothest methods were too timid.** Minimum variance and risk parity earned
  only {val('Annual return', 'Minimum variance'):.1%} and {val('Annual return', 'Risk parity'):.1%}
  a year. They are built for cautious investors, and John is growth-focused.
- **The real gap is how much stock the portfolios own.** Every Round 1 method that kept
  to the limit held roughly {val('Avg weight in stocks', 'Minimum variance'):.0%}-{val('Avg weight in stocks', v1):.0%}
  in stocks and up to {val('Avg weight in cash (SHV)', v2):.0%} in cash-like SHV, while the
  benchmark holds {val('Avg weight in stocks', BENCH):.0%} stocks. Their volatility
  ({val('Volatility', v2):.1%} for shrunk max-Sharpe) is also well under the IPS target of
  10-12%. Changing the method did not fix that, because none of the methods is told
  John wants growth.
- **Round 2: more stock plus a trend filter raised the return.** 60/40 with a monthly
  trend filter earned {val('Annual return', '60/40 + trend filter'):.1%} a year (vs
  {val('Annual return', v2):.1%} for shrunk max-Sharpe) with a Sharpe of
  {val('Sharpe ratio', '60/40 + trend filter'):.2f}, and stayed inside the limit
  ({-val('Max drawdown', '60/40 + trend filter'):.1%}). The 70/30 version earned
  {val('Annual return', '70/30 + trend filter'):.1%} but fell {-val('Max drawdown', '70/30 + trend filter'):.1%},
  just past John's 20%. The filter shone in the fast COVID crash
  ({val('COVID crash (Feb 19 - Mar 23, 2020)', '60/40 + trend filter'):.1%} vs
  {val('COVID crash (Feb 19 - Mar 23, 2020)', BENCH):.1%} for 60/40) but did not help in
  the slow 2022 decline.
- **The trade-offs.** Two very different results: shrunk max-Sharpe gives the smallest
  worst fall ({-val('Max drawdown', v2):.1%}); the trend filters give the most return
  per unit of risk. The trend filters need monthly checks (John's IPS says yearly),
  trade far more (about {val('Avg yearly turnover', '60/40 + trend filter'):.0%} of the portfolio
  a year, which could mean taxes), and were picked after testing several ideas on the
  same seven years, so they need a test on a period they have never seen.
- **Next step (Version 3):** test the trend filter on 2005-2016 (including the 2008
  crash) using older look-alike funds (SPY, AGG), to check it was not just luck.
"""
    )
