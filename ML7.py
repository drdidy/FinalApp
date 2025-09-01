# app.py
# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET — FULL APP (Anchored Fan, Bias & Strategy)
# Slope fixed at ±0.216 per 30-min block; 3:00 PM CT close as anchor
# Skips maintenance hour (4–5 PM CT) and Fri 5 PM → Sun 5 PM weekend gap
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional

# ───────────────────────────────────────────────────────────────────────────────
# CORE CONFIG
# ───────────────────────────────────────────────────────────────────────────────
CT_TZ = pytz.timezone("America/Chicago")
SLOPE_PER_BLOCK = 0.216  # per 30-minute block (top +, bottom −)
RTH_START = "08:30"
RTH_END = "14:30"

# ───────────────────────────────────────────────────────────────────────────────
# PAGE & THEME (light, enterprise vibe)
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔮 SPX Prophet Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --brand: #2563eb; /* blue-600 */
  --brand-2: #10b981; /* emerald-500 */
  --surface: #ffffff;
  --muted: #f8fafc;  /* slate-50 */
  --text: #0f172a;   /* slate-900 */
  --subtext: #475569;/* slate-600 */
  --border: #e2e8f0; /* slate-200 */
  --warn: #f59e0b;   /* amber-500 */
  --danger: #ef4444; /* red-500 */
}

html, body, [class*="css"]  {
  background: var(--muted);
  color: var(--text);
}

.block-container { padding-top: 1.5rem; }

h1, h2, h3 { color: var(--text); }

.metric-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 6px 16px rgba(2,6,23,0.04);
}

.metric-title {
  font-size: 0.9rem;
  color: var(--subtext);
  margin: 0;
}

.metric-value {
  font-size: 1.8rem;
  font-weight: 700;
  margin-top: 6px;
}

.kicker {
  font-size: 0.8rem;
  color: var(--subtext);
}

.badge-open {
  color: #065f46;
  background: #d1fae5;
  border: 1px solid #99f6e4;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
}

.badge-closed {
  color: #7c2d12;
  background: #ffedd5;
  border: 1px solid #fed7aa;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
}

hr { border-top: 1px solid var(--border); }

.dataframe { background: var(--surface); border-radius: 12px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────────────────────────
def fmt_ct(dt: datetime) -> datetime:
    """Ensure timezone-aware CT."""
    if dt.tzinfo is None:
        return CT_TZ.localize(dt)
    return dt.astimezone(CT_TZ)

def between_time(df: pd.DataFrame, start_str: str, end_str: str) -> pd.DataFrame:
    return df.between_time(start_str, end_str) if not df.empty else df

def rth_slots_ct(target_date: date) -> List[datetime]:
    start_dt = fmt_ct(datetime.combine(target_date, time(8, 30)))
    end_dt = fmt_ct(datetime.combine(target_date, time(14, 30)))
    slots = []
    cur = start_dt
    while cur <= end_dt:
        slots.append(cur)
        cur += timedelta(minutes=30)
    return slots

def is_maintenance(dt: datetime) -> bool:
    """4–5 PM CT maintenance hour."""
    h, m = dt.hour, dt.minute
    return (h == 16) or (h == 17 and m == 0 and False)  # 16:00 <= t < 17:00

def in_weekend_gap(dt: datetime) -> bool:
    """
    Skip Fri 17:00 → Sun 17:00 CT.
    We will treat any half-hour mark in this gap as "do not count".
    """
    wd = dt.weekday()  # Mon=0 ... Sun=6
    # if it's Sat (5) or Sun before 17:00 (6 & hour<17) → gap
    if wd == 5:
        return True
    if wd == 6 and dt.hour < 17:
        return True
    # Friday after/equal 17:00 (4, hour>=17) → gap
    if wd == 4 and dt.hour >= 17:
        return True
    return False

def count_effective_blocks(anchor_time: datetime, target_time: datetime) -> float:
    """
    Count 30-min blocks from anchor_time → target_time,
    skipping any half-hours that fall in maintenance (4–5 PM) or weekend gap.
    """
    if target_time <= anchor_time:
        return 0.0
    t = anchor_time
    blocks = 0
    while t < target_time:
        t_next = t + timedelta(minutes=30)
        # We count this block only if the *end* time is not in forbidden windows
        # (this aligns with most intraday OHLC bar semantics).
        if not is_maintenance(t_next) and not in_weekend_gap(t_next):
            blocks += 1
        t = t_next
    return float(blocks)

def ensure_ohlc_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()
    return df

@st.cache_data(ttl=90)
def fetch_intraday(symbol: str, start_d: date, end_d: date) -> pd.DataFrame:
    """
    Robust intraday fetch, 30m, CT index, auto_adjust=False for accurate raw closes.
    Falls back to period-based fetch if start/end is empty.
    """
    def normalize(df):
        if df.empty:
            return df
        df = ensure_ohlc_cols(df)
        if df.empty:
            return df
        if df.index.tz is None:
            df.index = df.index.tz_localize("US/Eastern")
        df.index = df.index.tz_convert(CT_TZ)
        sdt = fmt_ct(datetime.combine(start_d, time(0, 0)))
        edt = fmt_ct(datetime.combine(end_d, time(23, 59)))
        return df.loc[sdt:edt]

    try:
        t = yf.Ticker(symbol)
        df = t.history(
            start=(start_d - timedelta(days=5)).strftime("%Y-%m-%d"),
            end=(end_d + timedelta(days=2)).strftime("%Y-%m-%d"),
            interval="30m",
            prepost=True,
            auto_adjust=False,
            back_adjust=False,
        )
        df = normalize(df)
        if df.empty:
            days = max(7, (end_d - start_d).days + 7)
            df2 = t.history(
                period=f"{days}d",
                interval="30m",
                prepost=True,
                auto_adjust=False,
                back_adjust=False,
            )
            df = normalize(df2)
        return df
    except Exception:
        return pd.DataFrame()

def get_prev_day_3pm_close(spx_prev: pd.DataFrame, prev_day: date) -> Optional[float]:
    """
    Get the **3:00 PM CT exact bar close** for prev_day.
    If exact 15:00 not present, use the last bar <= 15:00 within prev_day.
    """
    if spx_prev.empty:
        return None
    day_start = fmt_ct(datetime.combine(prev_day, time(0, 0)))
    day_end = fmt_ct(datetime.combine(prev_day, time(23, 59)))
    d = spx_prev.loc[day_start:day_end].copy()
    if d.empty:
        return None
    # 3:00 PM CT exact
    target = fmt_ct(datetime.combine(prev_day, time(15, 0)))
    # Find exact match
    if target in d.index:
        return float(d.loc[target, "Close"])
    # Else find last bar before/at 15:00
    prior = d.loc[:target]
    if not prior.empty:
        return float(prior.iloc[-1]["Close"])
    return None

def project_fan_from_close(close_price: float, anchor_time: datetime, target_day: date) -> pd.DataFrame:
    rows = []
    for slot in rth_slots_ct(target_day):
        blocks = count_effective_blocks(anchor_time, slot)
        top = close_price + SLOPE_PER_BLOCK * blocks
        bot = close_price - SLOPE_PER_BLOCK * blocks
        rows.append(
            {
                "Time": slot.strftime("%H:%M"),
                "Top": round(top, 2),
                "Bottom": round(bot, 2),
                "Fan_Width": round(top - bot, 2),
            }
        )
    return pd.DataFrame(rows)

def build_strategy_table(rth_prices: pd.DataFrame, fan_df: pd.DataFrame, anchor_close: float) -> pd.DataFrame:
    """
    Your simplified rules:
    - Bias: "UP" if price >= anchor_close; else "DOWN".
    - Within fan:
        - Bias UP → BUY bottom → exit at top
        - Bias DOWN → SELL top → exit at bottom
    - Above fan: SELL at top, TP2 = top - width (mirror to bottom), TP1 = n/a
    - Below fan: SELL at bottom, TP2 = bottom - width (extend), TP1 = n/a
    """
    if rth_prices.empty or fan_df.empty:
        return pd.DataFrame()

    # map time → price
    price_lu = {dt.strftime("%H:%M"): float(rth_prices.loc[dt, "Close"]) for dt in rth_prices.index}
    rows = []
    for _, row in fan_df.iterrows():
        t = row["Time"]
        if t not in price_lu:
            continue
        p = price_lu[t]
        top = row["Top"]
        bot = row["Bottom"]
        width = row["Fan_Width"]
        bias = "UP" if p >= anchor_close else "DOWN"

        direction = ""
        entry = np.nan
        tp1 = np.nan
        tp2 = np.nan
        note = ""

        if bot <= p <= top:
            if bias == "UP":
                direction = "BUY"
                entry = bot
                tp1 = top
                tp2 = top
                note = "Within fan, bias UP → buy bottom → exit top"
            else:
                direction = "SELL"
                entry = top
                tp1 = bot
                tp2 = bot
                note = "Within fan, bias DOWN → sell top → exit bottom"
        elif p > top:
            direction = "SELL"
            entry = top
            tp2 = top - width  # mirror distance
            note = "Above fan → short from Top; TP2 = Top - width"
        else:  # p < bottom
            direction = "SELL"
            entry = bot
            tp2 = bot - width  # extend distance
            note = "Below fan → short from Bottom; TP2 = Bottom - width"

        rows.append(
            {
                "Time": t,
                "Price": round(p, 2),
                "Bias": bias,
                "EntrySide": direction,
                "Entry": round(entry, 2) if not np.isnan(entry) else np.nan,
                "TP1": round(tp1, 2) if not np.isnan(tp1) else np.nan,
                "TP2": round(tp2, 2) if not np.isnan(tp2) else np.nan,
                "Top": round(top, 2),
                "Bottom": round(bot, 2),
            }
        )
    return pd.DataFrame(rows)

# ───────────────────────────────────────────────────────────────────────────────
# SIDEBAR (dates + action)
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🔧 Controls")
today_ct = datetime.now(CT_TZ).date()

prev_day = st.sidebar.date_input("Previous Trading Day", value=today_ct - timedelta(days=1))
proj_day = st.sidebar.date_input("Projection Day", value=prev_day + timedelta(days=1))
st.sidebar.caption("Anchor at **3:00 PM CT** on the previous trading day.")
st.sidebar.markdown("---")
go = st.sidebar.button("🔮 Generate Fan & Strategy", type="primary")

# ───────────────────────────────────────────────────────────────────────────────
# HEADER METRICS
# ───────────────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
now = datetime.now(CT_TZ)
with c1:
    st.markdown(
        f"""
<div class="metric-card">
  <p class="metric-title">Current Time (CT)</p>
  <div class="metric-value">🕒 {now.strftime("%H:%M:%S")}</div>
  <div class="kicker">{now.strftime("%A, %B %d, %Y")}</div>
</div>
""",
        unsafe_allow_html=True,
    )
with c2:
    is_wkday = now.weekday() < 5
    open_dt = now.replace(hour=8, minute=30, second=0, microsecond=0)
    close_dt = now.replace(hour=14, minute=30, second=0, microsecond=0)
    is_open = is_wkday and (open_dt <= now <= close_dt)
    badge = "badge-open" if is_open else "badge-closed"
    text = "Market Open" if is_open else "Closed"
    st.markdown(
        f"""
<div class="metric-card">
  <p class="metric-title">Market Status</p>
  <div class="metric-value">📊 <span class="{badge}">{text}</span></div>
  <div class="kicker">RTH: 08:30–14:30 CT • Mon–Fri</div>
</div>
""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
<div class="metric-card">
  <p class="metric-title">Slope / 30-min Block</p>
  <div class="metric-value">📐 ±{SLOPE_PER_BLOCK:.3f}</div>
  <div class="kicker">Top = +slope • Bottom = −slope</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ───────────────────────────────────────────────────────────────────────────────
# MAIN LOGIC
# ───────────────────────────────────────────────────────────────────────────────
st.subheader("SPX Close-Anchor Fan (3:00 PM CT)")

if go:
    with st.spinner("Fetching market data and building projections..."):
        # Fetch prev/proj for ^GSPC (fall back to SPY if ^GSPC empty)
        spx_prev = fetch_intraday("^GSPC", prev_day, prev_day)
        if spx_prev.empty:
            spx_prev = fetch_intraday("SPY", prev_day, prev_day)

        spx_proj = fetch_intraday("^GSPC", proj_day, proj_day)
        if spx_proj.empty:
            spx_proj = fetch_intraday("SPY", proj_day, proj_day)

        if spx_prev.empty or spx_proj.empty:
            st.error("❌ Market data connection failed for the selected dates.")
        else:
            # Get exact 3:00 PM CT previous-day close (or nearest <= 15:00)
            prev_3pm_close = get_prev_day_3pm_close(spx_prev, prev_day)
            if prev_3pm_close is None:
                st.error("Could not find a 3:00 PM CT close for the previous day.")
            else:
                anchor_time = fmt_ct(datetime.combine(prev_day, time(15, 0)))
                st.success(f"Anchor (Prev Day 3:00 PM CT) Close: **{prev_3pm_close:.2f}**")

                # Build fan for projection day
                fan_df = project_fan_from_close(prev_3pm_close, anchor_time, proj_day)

                # Strategy table requires the projection day RTH prices
                spx_proj_rth = between_time(spx_proj, RTH_START, RTH_END)
                if spx_proj_rth.empty:
                    st.error("No RTH data available for the projection day.")
                else:
                    strat_df = build_strategy_table(spx_proj_rth, fan_df, prev_3pm_close)

                    st.markdown("### 🎯 Fan Lines (Top/Bottom @ 30-min slots)")
                    st.dataframe(fan_df, use_container_width=True, hide_index=True)

                    st.markdown("### 📋 Strategy Table (Simplified)")
                    if not strat_df.empty:
                        st.dataframe(
                            strat_df[
                                ["Time", "Price", "Bias", "EntrySide", "Entry", "TP1", "TP2", "Top", "Bottom"]
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No strategy rows were generated (time alignment gap).")

else:
    st.info("Use the **sidebar** to pick dates, then click **Generate Fan & Strategy**.")

st.markdown("---")

# ───────────────────────────────────────────────────────────────────────────────
# QUICK CONNECTIVITY CHECK
# ───────────────────────────────────────────────────────────────────────────────
colA, colB = st.columns([1, 2])
with colA:
    if st.button("🔌 Test Data Connection"):
        td = fetch_intraday("^GSPC", today_ct - timedelta(days=3), today_ct)
        if td.empty:
            st.error("Couldn't fetch ^GSPC. Trying SPY…")
            td = fetch_intraday("SPY", today_ct - timedelta(days=3), today_ct)
        if not td.empty:
            st.success(f"OK — received {len(td)} bars.")
        else:
            st.error("Data fetch failed — try different dates later.")

with colB:
    st.caption("Note: We use **auto_adjust=False** to preserve *actual* closes. If ^GSPC is sparse, we fall back to **SPY** for continuity.")