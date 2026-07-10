"""
Exchange Margin Dashboard
Exchange initial margin (ICE scanning range) vs a VaR-implied model margin,
per market (KC/CC/CT/SB), tier 1 and tier 2.

Data: ../Database/margin_scanning.parquet, prices.parquet, margin_var.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DB_DIR = Path(__file__).resolve().parent.parent / "Database"

MARKETS = ["KC", "CC", "SB", "CT"]
MARKET_NAMES = {"KC": "Arabica Coffee", "CC": "NY Cocoa", "SB": "Sugar #11", "CT": "Cotton"}

# dataviz reference palette — categorical slots (light mode)
BLUE = "#2a78d6"
AQUA = "#1baf7a"
RED = "#e34948"
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"

PLOTLY_CONFIG = {"displayModeBar": False}

st.set_page_config(page_title="Exchange Margin Dashboard", layout="wide")

st.markdown("""
<style>
  .block-container{padding-top:2.2rem;padding-bottom:2rem;max-width:1200px}
  [data-testid="stMetricValue"]{font-size:1.5rem;font-weight:600}
  [data-testid="stMetricLabel"]{font-size:.78rem;color:#898781}
  hr{margin:1.1rem 0!important;border-top:1px solid #e8e8e5!important}
  h1{font-weight:600!important;font-size:1.7rem!important}
  [data-testid="stExpander"]{border:none!important;background:#f9f9f7!important;border-radius:6px}
  [data-testid="stExpander"] summary{font-size:.85rem!important;color:#52514e!important}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data():
    scanning = pd.read_parquet(DB_DIR / "margin_scanning.parquet")
    prices = pd.read_parquet(DB_DIR / "prices.parquet")
    margin_var = pd.read_parquet(DB_DIR / "margin_var.parquet")
    margin_var["date"] = pd.to_datetime(margin_var["date"])
    prices["date"] = pd.to_datetime(prices["date"])

    # ICE republishes the full tier table every day, so a margin-change row
    # recurs in every subsequent daily CSV until the next change — collapse
    # to one row per (market, tier, effective_date).
    scanning = scanning.drop_duplicates(subset=["market", "tier", "effective_date"], keep="first")
    return scanning, prices, margin_var


def base_layout(height=340):
    return dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=SECONDARY_INK, size=11),
        height=height,
        margin=dict(l=45, r=20, t=10, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        hovermode="x unified",
    )


try:
    scanning, prices, margin_var = load_data()
except FileNotFoundError as e:
    st.error(f"Missing parquet file — run Ingest/ingest.py first. ({e})")
    st.stop()

st.title("Exchange Margin Dashboard")
st.caption("ICE initial margin (scanning range) vs a VaR-implied model margin, per market.")

with st.sidebar:
    st.header("Filters")
    market = st.selectbox("Market", MARKETS, format_func=lambda m: f"{m} — {MARKET_NAMES[m]}")
    tier = st.radio("Tier", [1, 2], horizontal=True,
                     help="Tier 1 = spot/nearby months (higher margin). Tier 2 = further-out months.")
    lookback_years = st.slider("Lookback (years)", 1, 10, 10)

    latest_date = margin_var["date"].max()
    st.divider()
    st.caption(f"Latest data: {latest_date.strftime('%d %b %Y')}")

contract = f"{market}{tier}"
cutoff = latest_date - pd.DateOffset(years=lookback_years)

mv = margin_var[(margin_var["contract"] == contract) & (margin_var["date"] >= cutoff)].copy()
mv = mv.sort_values("date")

if mv.empty:
    st.warning(f"No data for {contract} in the selected window.")
    st.stop()

latest_row = mv.iloc[-1]

st.subheader(f"{market} — {MARKET_NAMES[market]}, Tier {tier}")

# ── KPI row ──────────────────────────────────────────────────────────────────
model_val = latest_row["model_margin"]
gap = latest_row["initial_margin"] - model_val if pd.notna(model_val) else np.nan
gap_color = RED if pd.notna(gap) and gap >= 0 else BLUE

kpis = [
    ("Exchange IM (per lot)", f"${latest_row['initial_margin']:,.0f}", BLUE),
    ("Model IM (per lot)", f"${model_val:,.0f}" if pd.notna(model_val) else "n/a", AQUA),
    ("Gap (Exchange − Model)", f"${gap:,.0f}" if pd.notna(gap) else "n/a", gap_color),
    ("Flat price", f"{latest_row['flat_close']:,.2f}", MUTED),
]

card_html = '<div style="display:flex;gap:12px;margin:.4rem 0 1rem">'
for label, value, color in kpis:
    card_html += f"""
    <div style="flex:1;background:#fcfcfb;border:1px solid #e8e8e5;border-top:3px solid {color};
                border-radius:8px;padding:12px 16px">
      <div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;
                  color:#898781;margin-bottom:4px">{label}</div>
      <div style="font-size:1.35rem;font-weight:600;color:#0b0b0b">{value}</div>
    </div>"""
card_html += "</div>"
st.markdown(card_html, unsafe_allow_html=True)

st.divider()

# ── Chart 1: Exchange IM vs Model IM ────────────────────────────────────────
st.markdown("**Exchange IM vs Model IM**")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=mv["date"], y=mv["initial_margin"], mode="lines", name="Exchange IM",
    line=dict(color=BLUE, width=2, shape="hv"),
))
fig1.add_trace(go.Scatter(
    x=mv["date"], y=mv["model_margin"], mode="lines", name="Model IM (avg rv60/rv120)",
    line=dict(color=AQUA, width=2, dash="dash"),
))
fig1.update_layout(**base_layout())
fig1.update_xaxes(showgrid=False, linecolor=GRID)
fig1.update_yaxes(showgrid=True, gridcolor=GRID, title="USD per lot")
st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)

# ── Chart 2: Gap (Exchange - Model) ──────────────────────────────────────────
st.markdown("**Margin gap over time**")
mv["gap"] = mv["initial_margin"] - mv["model_margin"]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=mv["date"], y=mv["gap"].clip(lower=0), mode="lines", name="Exchange > Model",
    line=dict(width=0), fill="tozeroy", fillcolor="rgba(227,73,72,0.35)",
    hoverinfo="skip", showlegend=True,
))
fig2.add_trace(go.Scatter(
    x=mv["date"], y=mv["gap"].clip(upper=0), mode="lines", name="Model > Exchange",
    line=dict(width=0), fill="tozeroy", fillcolor="rgba(42,120,214,0.35)",
    hoverinfo="skip", showlegend=True,
))
fig2.add_hline(y=0, line_color=MUTED, line_width=1)
fig2.update_layout(**base_layout(height=260))
fig2.update_xaxes(showgrid=False, linecolor=GRID)
fig2.update_yaxes(showgrid=True, gridcolor=GRID, title="USD")
st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

# ── Chart 3: Flat price with margin change markers ──────────────────────────
st.markdown("**Price with margin-change events**")
events = scanning.copy()
events["eff_date"] = pd.to_datetime(events["effective_date"], format="%d-%b-%y", errors="coerce")
events = events[(events["market"] == market) & (events["tier"] == tier) & (events["eff_date"] >= cutoff)]

events["new_margin"] = pd.to_numeric(events["new_applied_margin_rate"], errors="coerce")
events["previous_margin"] = pd.to_numeric(events["previous_applied_margin_rate"], errors="coerce")
events["direction"] = np.where(events["new_margin"] >= events["previous_margin"], "Increase", "Decrease")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=mv["date"], y=mv["flat_close"], mode="lines", name="Flat price",
    line=dict(color=MUTED, width=1.5),
))
if not events.empty:
    ev_y = mv.set_index("date")["flat_close"].reindex(events["eff_date"], method="nearest")
    events = events.assign(flat_at_event=ev_y.values)

    up = events[events["direction"] == "Increase"]
    down = events[events["direction"] == "Decrease"]
    if not up.empty:
        fig3.add_trace(go.Scatter(
            x=up["eff_date"], y=up["flat_at_event"], mode="markers", name="Margin increase",
            marker=dict(color=RED, size=8, symbol="triangle-up"),
        ))
    if not down.empty:
        fig3.add_trace(go.Scatter(
            x=down["eff_date"], y=down["flat_at_event"], mode="markers", name="Margin decrease",
            marker=dict(color=BLUE, size=8, symbol="triangle-down"),
        ))
fig3.update_layout(**base_layout(height=260))
fig3.update_xaxes(showgrid=False, linecolor=GRID)
fig3.update_yaxes(showgrid=True, gridcolor=GRID, title="Price")
st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG)

st.divider()

# ── Table: recent margin-change events ──────────────────────────────────────
st.markdown("**Recent margin-change events**")
recent = events.sort_values("eff_date", ascending=False).head(20)[
    ["eff_date", "direction", "new_applied_margin_rate", "previous_applied_margin_rate", "percentage_change"]
].rename(columns={
    "eff_date": "Effective date",
    "direction": "Direction",
    "new_applied_margin_rate": "New margin",
    "previous_applied_margin_rate": "Previous margin",
    "percentage_change": "% change",
})
st.dataframe(recent, use_container_width=True, hide_index=True)

st.caption("Source: ICE margin scanning parameters + LSEG price/vol data.")
