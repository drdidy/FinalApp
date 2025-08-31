# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET — LIGHT UI • SPX-ONLY TABLES • ES→SPX OFFSET UNDER THE HOOD
#   - Slope per 30-min block = 0.333 (ascending +0.333, descending −0.333)
#   - Bias = position vs the *Close Fan* (Top/Bottom derived from prev ES RTH close)
#   - Maintenance hour (16:00–17:00 CT) EXCLUDED from block counts
#   - Friday→Sunday halt EXCLUDED (Sun 17:00 CT resumes); prevents Monday “wide fan”
#   - Asian anchors (Skyline/Baseline) from ES 17:00–19:30 CT window
#   - Strategy = MR when *outside* fan using High(+)/Low(−) triggers; fade when *inside* fan
#   - Signals = Touch + 1m EMA(8/21) cross confirmation (if 1m data available)
#   - UI = Light mode, icons, simplified tables
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date, time, timedelta
import pytz

# ───────────────────────────────────────────────────────────────────────────────
# CONFIG / CONSTANTS
# ───────────────────────────────────────────────────────────────────────────────
CT = pytz.timezone("America/Chicago")
UTC = pytz.UTC

SLOPE_PER_BLOCK = 0.333  # per 30-min block (your current spec)
ANCHOR_WIN_START = "17:00"  # Asian window for Skyline/Baseline (CT)
ANCHOR_WIN_END   = "19:30"
RTH_START = "08:30"
RTH_END   = "14:30"

# UI
st.set_page_config(
    page_title="🔮 SPX Prophet (Light)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light theme styling
st.markdown("""
<style>
/* Light gradient background */
.main {
  background: linear-gradient(180deg, #f8fafc 0%, #ffffff 35%, #f9fafb 100%);
}
section[data-testid="stSidebar"] {
  background: #f7f7fb;
  border-right: 1px solid #eee;
}
.metric-card {
  background: #ffffff;
  border: 1px solid #eee;
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.badge {
  display: inline-block;
  padding: 4px 10px;
  background: #eef2ff;
  color: #3730a3;
  border-radius: 999px;
  font-size: 13px;
  border: 1px solid #e0e7ff;
}
.table-note {
  font-size: 13px;
  color: #6b7280;
}
h1,h2,h3 { color: #111827; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# HELPERS — TIME / BLOCKS
# ───────────────────────────────────────────────────────────────────────────────

def to_ct(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return CT.localize(dt)
    return dt.astimezone(CT)

def rth_slots_ct(target_day: date) -> list[datetime]:
    start_dt = CT.localize(datetime.combine(target_day, datetime.strptime(RTH_START, "%H:%M").time()))
    end_dt   = CT.localize(datetime.combine(target_day, datetime.strptime(RTH_END, "%H:%M").time()))
    slots = []
    cur = start_dt
    while cur <= end_dt:
        slots.append(cur)
        cur += timedelta(minutes=30)
    return slots

def maintenance_window(dt: datetime) -> tuple[datetime, datetime]:
    day = dt.date()
    start = CT.localize(datetime.combine(day, time(16,0)))
    end   = CT.localize(datetime.combine(day, time(17,0)))
    return start, end

def is_weekend_gap(t: datetime) -> bool:
    # Treat Fri 16:00 → Sun 17:00 as halted
    wd = t.weekday()  # Mon=0..Sun=6
    if wd == 5:  # Saturday
        return True
    if wd == 6 and t.time() < time(17,0):  # Sunday before 17:00 CT
        return True
    # Friday after 16:00 considered in maintenance and halt; skip by block counter
    return False

def count_30min_blocks_excluding_maintenance_and_weekend(anchor_time: datetime, target_time: datetime) -> int:
    """Count 30-minute steps between two CT datetimes, skipping 16:00–17:00 CT daily and Fri→Sun halt."""
    if target_time < anchor_time:
        return -count_30min_blocks_excluding_maintenance_and_weekend(target_time, anchor_time)
    cur = anchor_time
    blocks = 0
    while cur < target_time:
        nxt = cur + timedelta(minutes=30)
        # Skip if in maintenance or weekend halt
        m_start, m_end = maintenance_window(cur)
        in_maint = (cur < m_end and nxt > m_start)  # overlapping any part of 16:00–17:00
        if not in_maint and not is_weekend_gap(cur):
            blocks += 1
        cur = nxt
    return blocks

# ───────────────────────────────────────────────────────────────────────────────
# DATA FETCH
# ───────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlc(symbol: str, start_d: date, end_d: date, interval: str = "30m") -> pd.DataFrame:
    """Fetch OHLCV, localize to CT; safe against empty/multiindex; inclusive on end date."""
    try:
        start = (start_d - timedelta(days=2)).strftime("%Y-%m-%d")
        end   = (end_d + timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.Ticker(symbol).history(start=start, end=end, interval=interval, auto_adjust=False, back_adjust=False, prepost=True)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        if df.index.tz is None:
            # yfinance usually returns UTC; be defensive
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(CT)
        # Keep only requested calendar days (CT)
        lo = CT.localize(datetime.combine(start_d, time(0,0)))
        hi = CT.localize(datetime.combine(end_d,   time(23,59,59)))
        df = df.loc[(df.index >= lo) & (df.index <= hi)]
        need = {"Open","High","Low","Close"}
        if not need.issubset(df.columns):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

def between_time_ct(df: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    if df.empty: return df
    return df.between_time(start_hhmm, end_hhmm)

# ───────────────────────────────────────────────────────────────────────────────
# OFFSETS / ANCHORS
# ───────────────────────────────────────────────────────────────────────────────

def rth_close_bar(df: pd.DataFrame) -> pd.Series | None:
    """Return last bar within RTH on that day, else last bar."""
    if df.empty: return None
    rth = between_time_ct(df, RTH_START, "15:00")
    if not rth.empty:
        return rth.iloc[-1]
    return df.iloc[-1]

def compute_es_spx_offset(prev_day: date) -> float | None:
    """Offset = SPX RTH close - ES RTH close (prev day)."""
    es = fetch_ohlc("ES=F", prev_day, prev_day, interval="30m")
    spx = fetch_ohlc("^GSPC", prev_day, prev_day, interval="30m")
    if es.empty or spx.empty: return None
    es_bar = rth_close_bar(es)
    spx_bar = rth_close_bar(spx)
    if es_bar is None or spx_bar is None: return None
    return float(spx_bar["Close"] - es_bar["Close"])

def prev_day_swing_ohlc(es_prev: pd.DataFrame) -> dict:
    if es_prev.empty: return {}
    day_open = es_prev.iloc[0]["Open"]
    day_high = es_prev["High"].max()
    day_low  = es_prev["Low"].min()
    day_close= es_prev.iloc[-1]["Close"]
    # times of high/low
    t_high = es_prev.index[es_prev["High"].argmax()]
    t_low  = es_prev.index[es_prev["Low"].argmin()]
    return {
        "open": (day_open, es_prev.index[0]),
        "high": (day_high, t_high),
        "low":  (day_low,  t_low),
        "close":(day_close, es_prev.index[-1]),
    }

def asian_skyline_baseline(es_prev: pd.DataFrame) -> tuple[tuple|None, tuple|None]:
    """Skyline/Baseline from Asian window (17:00–19:30 CT) of prev day."""
    if es_prev.empty: return None, None
    asian = between_time_ct(es_prev, ANCHOR_WIN_START, ANCHOR_WIN_END)
    if asian.empty: return None, None
    hi = asian["Close"].idxmax()
    lo = asian["Close"].idxmin()
    return (float(asian.loc[hi,"Close"]), hi), (float(asian.loc[lo,"Close"]), lo)

# ───────────────────────────────────────────────────────────────────────────────
# PROJECTIONS
# ───────────────────────────────────────────────────────────────────────────────

def project_line(price0: float, t0: datetime, target_times: list[datetime], slope_per_block: float) -> pd.Series:
    """Generic projection: price(t) = price0 + slope * blocks(t0→t) with maintenance+weekend skip."""
    out = []
    for tt in target_times:
        blocks = count_30min_blocks_excluding_maintenance_and_weekend(to_ct(t0), to_ct(tt))
        out.append(price0 + slope_per_block * blocks)
    return pd.Series(out, index=target_times)

def build_close_fan(prev_close_price: float, prev_close_time: datetime, target_day: date) -> pd.DataFrame:
    times = rth_slots_ct(target_day)
    top = project_line(prev_close_price, prev_close_time, times, +SLOPE_PER_BLOCK)
    bot = project_line(prev_close_price, prev_close_time, times, -SLOPE_PER_BLOCK)
    fan = pd.DataFrame({
        "Time": [t.strftime("%H:%M") for t in times],
        "Top":  top.values,
        "Bottom": bot.values,
    })
    fan["Fan_Width"] = (fan["Top"] - fan["Bottom"]).round(2)
    fan["Top"] = fan["Top"].round(2)
    fan["Bottom"] = fan["Bottom"].round(2)
    return fan

def project_anchor(price: float, t_anchor: datetime, target_day: date, sign: int) -> pd.DataFrame:
    times = rth_slots_ct(target_day)
    proj = project_line(price, t_anchor, times, sign * SLOPE_PER_BLOCK)
    return pd.DataFrame({"Time": [t.strftime("%H:%M") for t in times], "Price": proj.values})

# ───────────────────────────────────────────────────────────────────────────────
# STRATEGY & SIGNALS
# ───────────────────────────────────────────────────────────────────────────────

def simple_bias(row, last_price=None):
    if row["Last"] > row["Top"]:
        return "UP"
    if row["Last"] < row["Bottom"]:
        return "DOWN"
    return "RANGE"

def build_strategy_table(fan_df: pd.DataFrame, last_prices: pd.Series,
                         high_line: pd.DataFrame|None, low_line: pd.DataFrame|None) -> pd.DataFrame:
    """Return simplified strategy table:
       Time • Bias • Entry Trigger • TP1 • TP2 • Note
    """
    rows = []
    hl = None if high_line is None or high_line.empty else dict(zip(high_line["Time"], high_line["Price"]))
    ll = None if low_line  is None or low_line.empty  else dict(zip(low_line["Time"],  low_line["Price"]))

    for i, r in fan_df.iterrows():
        t = r["Time"]
        top, bot = r["Top"], r["Bottom"]
        mid = (top + bot)/2.0
        last = float(last_prices.get(t, mid))  # fallback to mid if not known
        bias = "UP" if last > top else "DOWN" if last < bot else "RANGE"

        if bias == "UP":
            entry = hl.get(t) if hl else top  # High(+0.333) trigger, fallback top
            tp1 = top               # mean (edge)
            tp2 = top - (top - bot) # one fan width back
            note = "Above fan → Mean-reversion short from High(+)."
        elif bias == "DOWN":
            entry = ll.get(t) if ll else bot  # Low(−0.333) trigger, fallback bottom
            tp1 = bot               # mean (edge)
            tp2 = bot + (top - bot) # one fan width up
            note = "Below fan → Mean-reversion long from Low(−)."
        else:
            # Range: fade nearer edge
            dist_top = abs(last - top)
            dist_bot = abs(last - bot)
            if dist_bot < dist_top:
                entry = bot
                # You asked: TP1 = mean (edge you called "mean" = edge), TP2 = opposite edge
                tp1 = mid
                tp2 = top
                note = "Inside fan → Fade from Bottom to Mid/Top."
            else:
                entry = top
                tp1 = mid
                tp2 = bot
                note = "Inside fan → Fade from Top to Mid/Bottom."

        rows.append({
            "Time": t,
            "Bias": bias,
            "Entry Trigger": round(entry,2),
            "TP1": round(tp1,2),
            "TP2": round(tp2,2),
            "Note": note
        })
    return pd.DataFrame(rows)

def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def detect_touch(top: float, bottom: float, bar: pd.Series, tol: float) -> str|None:
    lo, hi = bar["Low"], bar["High"]
    if (abs(hi - top) <= tol) or (lo <= top <= hi):
        return "Top"
    if (abs(lo - bottom) <= tol) or (lo <= bottom <= hi):
        return "Bottom"
    return None

def signals_table(es_30m: pd.DataFrame, fan_df: pd.DataFrame, day: date) -> pd.DataFrame:
    """Time • Touch Side • Candle Close • EMA Cross • Signal • Target
       Uses 1m ES data to confirm EMA cross if available.
    """
    if es_30m.empty or fan_df.empty:
        return pd.DataFrame()

    # 1m data for that day (may be limited by Yahoo ~7 days)
    es_1m = fetch_ohlc("ES=F", day, day, interval="1m")
    have_1m = not es_1m.empty

    rows = []
    # tolerance based on small fraction of price (or ATR-lite)
    tol = max(0.25, es_30m["Close"].median() * 0.0005)  # ~5 bps or $0.25 minimum

    # map fan edges by Time
    fan_top = dict(zip(fan_df["Time"], fan_df["Top"]))
    fan_bot = dict(zip(fan_df["Time"], fan_df["Bottom"]))

    for idx, bar in es_30m.iterrows():
        t_str = idx.strftime("%H:%M")
        if t_str not in fan_top:
            continue
        top, bot = fan_top[t_str], fan_bot[t_str]
        touch = detect_touch(top, bot, bar, tol)
        if not touch:
            continue

        ema_cross = "N/A"
        signal = "—"
        target = "—"

        if have_1m:
            # Look for EMA cross *after* the touch, within same 30-min block
            window_start = idx
            window_end = idx + timedelta(minutes=30)
            win = es_1m.loc[(es_1m.index >= window_start) & (es_1m.index <= window_end)]
            if not win.empty:
                ema8 = calc_ema(win["Close"], 8)
                ema21 = calc_ema(win["Close"], 21)
                # Find first cross after the touch
                cross_type = None
                for i in range(1, len(win)):
                    prev_bull = ema8.iloc[i-1] <= ema21.iloc[i-1]
                    curr_bull = ema8.iloc[i]   >  ema21.iloc[i]
                    prev_bear = ema8.iloc[i-1] >= ema21.iloc[i-1]
                    curr_bear = ema8.iloc[i]   <  ema21.iloc[i]
                    if prev_bull and curr_bull:
                        cross_type = "Bullish"
                        break
                    if prev_bear and curr_bear:
                        cross_type = "Bearish"
                        break

                if cross_type:
                    ema_cross = cross_type
                    if touch == "Bottom":
                        # your rule:
                        # touch from above + bullish cross → BUY
                        # touch from below + bearish cross → SELL (continuation)
                        # We infer side by close vs bottom
                        if bar["Close"] >= bot:
                            signal = "BUY"
                            target = "Top"
                        else:
                            signal = "SELL"
                            target = "Bottom"
                    else:  # Top
                        if bar["Close"] <= top:
                            signal = "SELL"
                            target = "Bottom"
                        else:
                            signal = "BUY"
                            target = "Top"

        rows.append({
            "Time": t_str,
            "Touch Side": touch,
            "Candle Close": round(float(bar["Close"]),2),
            "EMA Cross": ema_cross,
            "Signal": signal,
            "Target": target
        })

    return pd.DataFrame(rows)

# ───────────────────────────────────────────────────────────────────────────────
# UI — SIDEBAR
# ───────────────────────────────────────────────────────────────────────────────

st.sidebar.title("🔮 SPX Prophet")
st.sidebar.caption("Light mode • SPX tables • ES data under the hood")

today_ct = datetime.now(CT).date()
prev_default = today_ct - timedelta(days=1)

with st.sidebar:
    st.markdown("### 📅 Dates")
    prev_day = st.date_input("Previous Trading Day", value=prev_default, key="prev_day")
    proj_day = st.date_input("Projection Day (RTH)", value=prev_day + timedelta(days=1), key="proj_day")

    st.markdown("---")
    st.markdown("### ⚙️ Options")
    show_asian = st.toggle("Show Asian Skyline/Baseline cards (if available)", value=True)
    st.caption("Asian window: 17:00–19:30 CT on the previous day")

    st.markdown("---")
    if st.button("🚀 Generate SPX Tables", type="primary", use_container_width=True):
        st.session_state["GO"] = True

# ───────────────────────────────────────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; margin: 6px 0 14px;">
  <h1>📈 SPX Prophet — Close Fan & Strategy</h1>
  <div class="table-note">Slope per 30-min block: <b>±0.333</b> • Bias from <b>Close Fan</b> • ES→SPX offset auto</div>
</div>
""", unsafe_allow_html=True)

# Quick metrics
metric1, metric2, metric3 = st.columns(3)
with metric1:
    card = st.container(border=True)
    with card:
        st.markdown("**⏰ Current (CT)**")
        now_ct = datetime.now(CT)
        st.write(now_ct.strftime("%A %b %d, %H:%M:%S"))
with metric2:
    card = st.container(border=True)
    with card:
        is_wk = now_ct.weekday() < 5
        st.markdown("**🕒 RTH**")
        st.write("08:30–14:30 CT")
        st.write("Mon–Fri" if is_wk else "Weekend")
with metric3:
    card = st.container(border=True)
    with card:
        st.markdown("**⚡ Slope**")
        st.write("+0.333 / −0.333 per 30-min")

st.markdown("---")

# ───────────────────────────────────────────────────────────────────────────────
# CORE COMPUTATION
# ───────────────────────────────────────────────────────────────────────────────

if st.session_state.get("GO"):

    # 1) Load previous day ES (30m) for anchors and Asian skyline/baseline
    es_prev = fetch_ohlc("ES=F", prev_day, prev_day, interval="30m")
    if es_prev.empty:
        st.error("Could not fetch ES=F 30m data for the previous day. Try a different date.")
        st.stop()

    # 2) Offset ES→SPX from prev day closes
    offset = compute_es_spx_offset(prev_day)
    if offset is None:
        st.warning("Could not compute ES→SPX offset from prev day closes. Using 0.0 temporarily.")
        offset = 0.0

    # 3) Prev-day OHLC anchors (ES prices & times)
    swings = prev_day_swing_ohlc(es_prev)
    if not swings:
        st.error("Could not compute previous day swings.")
        st.stop()

    prev_close_price, prev_close_time = swings["close"]
    prev_high_price,  prev_high_time  = swings["high"]
    prev_low_price,   prev_low_time   = swings["low"]

    # 4) Asian skyline/baseline (optional)
    sky, base = asian_skyline_baseline(es_prev)

    # 5) Build the Close Fan in ES, then map to SPX with offset for display
    #    Close anchor time is the prev RTH close time (use 15:00 or last RTH bar’s index)
    close_anchor_time = to_ct(prev_close_time)
    # If the last RTH bar is not exactly 15:00, we still use its timestamp to count blocks.
    fan_es = build_close_fan(prev_close_price, close_anchor_time, proj_day)
    fan_spx = fan_es.copy()
    fan_spx["Top"]    = (fan_spx["Top"]    + offset).round(2)
    fan_spx["Bottom"] = (fan_spx["Bottom"] + offset).round(2)
    fan_spx["Fan_Width"] = (fan_spx["Top"] - fan_spx["Bottom"]).round(2)

    # 6) High(+0.333) and Low(−0.333) anchor lines in SPX terms
    high_spx = low_spx = None
    if prev_high_price and prev_high_time:
        hdf = project_anchor(prev_high_price + offset, to_ct(prev_high_time), proj_day, sign=+1)
        high_spx = hdf
    if prev_low_price and prev_low_time:
        ldf = project_anchor(prev_low_price + offset, to_ct(prev_low_time), proj_day, sign=-1)
        low_spx = ldf

    # 7) Skyline/Baseline (Asian) projections in SPX terms (cards only)
    sky_card = base_card = None
    if sky and show_asian:
        p, t = sky
        sky_card = ("Skyline (Asian +0.333)", round(p+offset, 2), to_ct(t).strftime("%H:%M"))
    if base and show_asian:
        p, t = base
        base_card = ("Baseline (Asian −0.333)", round(p+offset, 2), to_ct(t).strftime("%H:%M"))

    # 8) Last price per 30m slot (for bias context) — use ES 30m at proj day, offset to SPX view
    es_proj_30m = fetch_ohlc("ES=F", proj_day, proj_day, interval="30m")
    last_map = {}
    if not es_proj_30m.empty:
        for idx, bar in es_proj_30m.iterrows():
            last_map[idx.strftime("%H:%M")] = float(bar["Close"] + offset)
    last_series = pd.Series(last_map)

    # ───────────────────────────────────────────────────────────────────────────
    # CARDS — Prev Close (midpoint) and Anchors
    # ───────────────────────────────────────────────────────────────────────────
    cards = st.columns(4)
    with cards[0]:
        st.markdown("**📌 Prev Close (SPX-adj)**")
        st.markdown(f"<span class='badge'>{round(prev_close_price+offset,2)}</span>", unsafe_allow_html=True)

    with cards[1]:
        st.markdown("**🔺 Prev High (+0.333)**")
        st.markdown(f"<span class='badge'>{round(prev_high_price+offset,2)} @ {to_ct(prev_high_time).strftime('%H:%M')}</span>", unsafe_allow_html=True)

    with cards[2]:
        st.markdown("**🔻 Prev Low (−0.333)**")
        st.markdown(f"<span class='badge'>{round(prev_low_price+offset,2)} @ {to_ct(prev_low_time).strftime('%H:%M')}</span>", unsafe_allow_html=True)

    with cards[3]:
        if sky_card or base_card:
            s1 = f"{sky_card[0]}: {sky_card[1]} @{sky_card[2]}" if sky_card else "—"
            s2 = f"{base_card[0]}: {base_card[1]} @{base_card[2]}" if base_card else "—"
            st.markdown("**🌙 Asian Anchors**")
            st.caption(s1)
            st.caption(s2)
        else:
            st.markdown("**🌙 Asian Anchors**")
            st.caption("—")

    st.markdown("---")

    # ───────────────────────────────────────────────────────────────────────────
    # TABLE 1 — FAN & ANCHORS (SPX)
    # ───────────────────────────────────────────────────────────────────────────
    st.subheader("🧭 Fan & Anchors (SPX)")
    st.caption("Close Fan edges and width for each 30-min RTH slot.")
    fan_view = fan_spx[["Time","Top","Bottom","Fan_Width"]]
    st.dataframe(fan_view, use_container_width=True, hide_index=True)

    # ───────────────────────────────────────────────────────────────────────────
    # TABLE 2 — STRATEGY (SPX)
    # ───────────────────────────────────────────────────────────────────────────
    st.subheader("🎯 Strategy (SPX)")
    st.caption("Bias from Close Fan; MR outside fan using High(+)/Low(−) triggers; fade inside fan.")

    # Build strategy inputs: map High/Low lines by time
    high_map = low_map = None
    if high_spx is not None and not high_spx.empty:
        high_map = high_spx.copy()
    if low_spx is not None and not low_spx.empty:
        low_map = low_spx.copy()

    strat = build_strategy_table(fan_spx, last_series, high_map, low_map)
    st.dataframe(strat, use_container_width=True, hide_index=True)

    # ───────────────────────────────────────────────────────────────────────────
    # TABLE 3 — SIGNALS (Touch + EMA 1m confirm)
    # ───────────────────────────────────────────────────────────────────────────
    st.subheader("🔔 Signals (Touch + EMA 8/21 1m Confirmation)")
    st.caption("Fires when a 30-min bar touches a fan edge and a confirming 1m EMA cross appears in the same block.")

    sig_df = signals_table(es_proj_30m, fan_spx, proj_day)
    if sig_df.empty:
        st.info("No signals found (or 1-minute data not available for this day).")
    else:
        st.dataframe(sig_df, use_container_width=True, hide_index=True)

else:
    st.info("Set your dates in the sidebar and click **“🚀 Generate SPX Tables”**.")