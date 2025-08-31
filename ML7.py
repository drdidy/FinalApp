# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET — LIGHT UI • SPX-ONLY ANCHORS • WEEKEND/MAINTENANCE AWARE
#   - Slope per 30-min block = 0.333 (ascending +0.333, descending −0.333)
#   - Close fan anchored to *actual SPX RTH close & time* (not ES+offset)
#   - Prev High/Low from SPX RTH; Asian (Skyline/Baseline) from ES (offset to SPX)
#   - Maintenance hour (16:00–17:00 CT) and Fri→Sun halt excluded from blocks
#   - Signals: touch + 1m EMA(8/21) confirmation if 1m data is available
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date, time, timedelta
import pytz

CT = pytz.timezone("America/Chicago")
SLOPE_PER_BLOCK = 0.333
ANCHOR_WIN_START = "17:00"
ANCHOR_WIN_END   = "19:30"
RTH_START = "08:30"
RTH_END   = "14:30"

st.set_page_config(page_title="🔮 SPX Prophet (Light)", page_icon="📈", layout="wide")

st.markdown("""
<style>
.main {background: linear-gradient(180deg, #f8fafc 0%, #ffffff 35%, #f9fafb 100%);}
section[data-testid="stSidebar"] {background: #f7f7fb; border-right: 1px solid #eee;}
.metric-card {background:#fff;border:1px solid #eee;border-radius:14px;padding:14px 16px;box-shadow:0 2px 8px rgba(0,0,0,0.04);}
.badge{display:inline-block;padding:4px 10px;background:#eef2ff;color:#3730a3;border-radius:999px;font-size:13px;border:1px solid #e0e7ff;}
.table-note{font-size:13px;color:#6b7280;}
h1,h2,h3{color:#111827;}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# Time helpers
# ───────────────────────────────────────────────────────────────────────────────
def to_ct(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return CT.localize(dt)
    return dt.astimezone(CT)

def rth_slots_ct(day: date) -> list[datetime]:
    s = CT.localize(datetime.combine(day, datetime.strptime(RTH_START,"%H:%M").time()))
    e = CT.localize(datetime.combine(day, datetime.strptime(RTH_END,"%H:%M").time()))
    out=[]; cur=s
    while cur<=e:
        out.append(cur); cur+=timedelta(minutes=30)
    return out

def maintenance_window(dt: datetime) -> tuple[datetime, datetime]:
    d = dt.date()
    return CT.localize(datetime.combine(d, time(16,0))), CT.localize(datetime.combine(d, time(17,0)))

def is_weekend_gap(t: datetime) -> bool:
    wd = t.weekday()
    if wd==5: return True
    if wd==6 and t.time() < time(17,0): return True
    return False

def count_blocks_excl_pauses(t0: datetime, t1: datetime) -> int:
    if t1 < t0:
        return -count_blocks_excl_pauses(t1, t0)
    cur = t0; blocks=0
    while cur < t1:
        nxt = cur + timedelta(minutes=30)
        m0,m1 = maintenance_window(cur)
        if not (cur < m1 and nxt > m0) and not is_weekend_gap(cur):
            blocks += 1
        cur = nxt
    return blocks

# ───────────────────────────────────────────────────────────────────────────────
# Data fetch
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlc(symbol: str, start_d: date, end_d: date, interval="30m") -> pd.DataFrame:
    try:
        start = (start_d - timedelta(days=2)).strftime("%Y-%m-%d")
        end   = (end_d + timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.Ticker(symbol).history(start=start, end=end, interval=interval, auto_adjust=False, back_adjust=False, prepost=True)
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(CT)
        lo = CT.localize(datetime.combine(start_d, time(0,0)))
        hi = CT.localize(datetime.combine(end_d,   time(23,59,59)))
        df = df.loc[(df.index>=lo) & (df.index<=hi)]
        need = {"Open","High","Low","Close"}
        return df if need.issubset(df.columns) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def between_time_ct(df: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    if df.empty: return df
    return df.between_time(start_hhmm, end_hhmm)

# ───────────────────────────────────────────────────────────────────────────────
# SPX RTH OHLC (for exact anchors)
# ───────────────────────────────────────────────────────────────────────────────
def spx_rth_ohlc(day: date) -> dict|None:
    spx = fetch_ohlc("^GSPC", day, day, "30m")
    if spx.empty:
        # fallback to SPY if needed
        spy = fetch_ohlc("SPY", day, day, "30m")
        if spy.empty: return None
        src = spy
    else:
        src = spx
    rth = between_time_ct(src, RTH_START, RTH_END)
    if rth.empty: rth = src
    o = float(rth.iloc[0]["Open"])
    h = float(rth["High"].max()); t_h = rth["High"].idxmax()
    l = float(rth["Low"].min());  t_l = rth["Low"].idxmin()
    c = float(rth.iloc[-1]["Close"]); t_c = rth.index[-1]
    return {"open":(o, rth.index[0]), "high":(h, t_h), "low":(l, t_l), "close":(c, t_c), "source": "SPX" if not spx.empty else "SPY"}

# ES→SPX offset only for Asian conversion / fallback
def es_spx_offset(prev_day: date) -> float|None:
    es  = fetch_ohlc("ES=F", prev_day, prev_day, "30m")
    spx = fetch_ohlc("^GSPC", prev_day, prev_day, "30m")
    if es.empty or spx.empty: return None
    es_rth = between_time_ct(es, RTH_START, RTH_END)
    spx_rth = between_time_ct(spx, RTH_START, RTH_END)
    if es_rth.empty or spx_rth.empty: return None
    return float(spx_rth.iloc[-1]["Close"] - es_rth.iloc[-1]["Close"])

def asian_sky_base_es(prev_day: date) -> tuple[tuple|None, tuple|None]:
    es_prev = fetch_ohlc("ES=F", prev_day, prev_day, "30m")
    if es_prev.empty: return None, None
    asian = between_time_ct(es_prev, ANCHOR_WIN_START, ANCHOR_WIN_END)
    if asian.empty: return None, None
    hi_t = asian["Close"].idxmax(); lo_t = asian["Close"].idxmin()
    return (float(asian.loc[hi_t,"Close"]), hi_t), (float(asian.loc[lo_t,"Close"]), lo_t)

# ───────────────────────────────────────────────────────────────────────────────
# Projections
# ───────────────────────────────────────────────────────────────────────────────
def project_series(price0: float, time0: datetime, times: list[datetime], slope: float) -> pd.Series:
    vals=[]
    for tt in times:
        b = count_blocks_excl_pauses(to_ct(time0), to_ct(tt))
        vals.append(price0 + slope*b)
    return pd.Series(vals, index=times)

def build_close_fan_spx(prev_close_price: float, prev_close_time: datetime, target_day: date) -> pd.DataFrame:
    times = rth_slots_ct(target_day)
    top = project_series(prev_close_price, prev_close_time, times, +SLOPE_PER_BLOCK)
    bot = project_series(prev_close_price, prev_close_time, times, -SLOPE_PER_BLOCK)
    df = pd.DataFrame({"Time":[t.strftime("%H:%M") for t in times], "Top":top.values, "Bottom":bot.values})
    df["Fan_Width"] = (df["Top"] - df["Bottom"]).round(2)
    df["Top"] = df["Top"].round(2); df["Bottom"] = df["Bottom"].round(2)
    return df

def project_anchor_spx(price: float, t_anchor: datetime, target_day: date, sign: int) -> pd.DataFrame:
    times = rth_slots_ct(target_day)
    proj = project_series(price, t_anchor, times, sign*SLOPE_PER_BLOCK)
    col = "High_Asc" if sign>0 else "Low_Desc"
    return pd.DataFrame({"Time":[t.strftime("%H:%M") for t in times], col: proj.values})

# ───────────────────────────────────────────────────────────────────────────────
# Strategy & signals
# ───────────────────────────────────────────────────────────────────────────────
def build_strategy_table(fan_df: pd.DataFrame, last_prices: pd.Series,
                         high_line: pd.DataFrame|None, low_line: pd.DataFrame|None) -> pd.DataFrame:
    rows=[]
    hl = None if high_line is None or high_line.empty else dict(zip(high_line["Time"], high_line.iloc[:,1]))
    ll = None if low_line  is None or low_line.empty  else dict(zip(low_line["Time"],  low_line.iloc[:,1]))
    for _, r in fan_df.iterrows():
        t = r["Time"]; top=r["Top"]; bot=r["Bottom"]; width=top-bot; mid=(top+bot)/2
        last = float(last_prices.get(t, mid))
        bias = "UP" if last>top else "DOWN" if last<bot else "RANGE"
        if bias=="UP":
            entry = hl.get(t, top) if hl else top; tp1=top; tp2=top - width; note="Above fan → mean-revert short via High(+)."
        elif bias=="DOWN":
            entry = ll.get(t, bot) if ll else bot; tp1=bot; tp2=bot + width; note="Below fan → mean-revert long via Low(−)."
        else:
            # inside fan: fade toward mid/opposite edge
            if abs(last-bot) < abs(last-top):
                entry=bot; tp1=mid; tp2=top; note="Inside fan → Fade from Bottom → Mid/Top."
            else:
                entry=top; tp1=mid; tp2=bot; note="Inside fan → Fade from Top → Mid/Bottom."
        rows.append({"Time":t,"Bias":bias,"Entry Trigger":round(entry,2),"TP1":round(tp1,2),"TP2":round(tp2,2),"Note":note})
    return pd.DataFrame(rows)

def ema(series: pd.Series, span:int)->pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def detect_touch(top: float, bottom: float, bar: pd.Series, tol: float)->str|None:
    lo,hi = bar["Low"], bar["High"]
    if (abs(hi-top)<=tol) or (lo<=top<=hi): return "Top"
    if (abs(lo-bottom)<=tol) or (lo<=bottom<=hi): return "Bottom"
    return None

def signals_table(es_30m_rth: pd.DataFrame, fan_df: pd.DataFrame, day: date) -> pd.DataFrame:
    if es_30m_rth.empty or fan_df.empty: return pd.DataFrame()
    es_1m = fetch_ohlc("ES=F", day, day, "1m")
    have_1m = not es_1m.empty
    tol = max(0.25, es_30m_rth["Close"].median()*0.0005)

    top_lu = dict(zip(fan_df["Time"], fan_df["Top"]))
    bot_lu = dict(zip(fan_df["Time"], fan_df["Bottom"]))

    out=[]
    for idx, bar in es_30m_rth.iterrows():
        ts = idx.strftime("%H:%M")
        if ts not in top_lu: continue
        touch = detect_touch(top_lu[ts], bot_lu[ts], bar, tol)
        if not touch: continue

        ema_cross="N/A"; signal="—"; target="—"
        if have_1m:
            win = es_1m.loc[(es_1m.index>=idx) & (es_1m.index<=idx+timedelta(minutes=30))]
            if not win.empty:
                e8 = ema(win["Close"],8); e21 = ema(win["Close"],21)
                cross=None
                for i in range(1,len(win)):
                    if e8.iloc[i-1]<=e21.iloc[i-1] and e8.iloc[i]>e21.iloc[i]:
                        cross="Bullish"; break
                    if e8.iloc[i-1]>=e21.iloc[i-1] and e8.iloc[i]<e21.iloc[i]:
                        cross="Bearish"; break
                if cross:
                    ema_cross=cross
                    if touch=="Bottom":
                        signal = "BUY" if bar["Close"]>=bot_lu[ts] else "SELL"
                        target = "Top" if signal=="BUY" else "Bottom"
                    else:
                        signal = "SELL" if bar["Close"]<=top_lu[ts] else "BUY"
                        target = "Bottom" if signal=="SELL" else "Top"

        out.append({"Time":ts,"Touch Side":touch,"Candle Close":round(float(bar["Close"]),2),
                    "EMA Cross":ema_cross,"Signal":signal,"Target":target})
    return pd.DataFrame(out)

# ───────────────────────────────────────────────────────────────────────────────
# UI — Sidebar & Header
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🔮 SPX Prophet")
today_ct = datetime.now(CT).date()
prev_default = today_ct - timedelta(days=1)
prev_day = st.sidebar.date_input("Previous Trading Day", value=prev_default)
proj_day = st.sidebar.date_input("Projection Day (RTH)", value=prev_day+timedelta(days=1))
show_asian = st.sidebar.toggle("Show Asian Skyline/Baseline", value=True)
GO = st.sidebar.button("🚀 Generate SPX Tables", type="primary", use_container_width=True)

st.markdown("""
<div style="text-align:center; margin: 6px 0 14px;">
  <h1>📈 SPX Prophet — Close Fan & Strategy</h1>
  <div class="table-note">Fan anchored to actual <b>SPX RTH close</b>. Slope: <b>±0.333</b> per 30-min. ES used only for Asian anchors & fallback.</div>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# Main flow
# ───────────────────────────────────────────────────────────────────────────────
if GO:
    # 1) Exact SPX anchors (RTH)
    spx_prev = spx_rth_ohlc(prev_day)
    if not spx_prev:
        st.error("Could not fetch SPX (or SPY fallback) RTH data for previous day.")
        st.stop()

    spx_close_price, spx_close_time = spx_prev["close"]
    spx_high_price,  spx_high_time  = spx_prev["high"]
    spx_low_price,   spx_low_time   = spx_prev["low"]

    # 2) ES→SPX offset (for Asian & fallback)
    offset = es_spx_offset(prev_day) or 0.0

    # 3) Asian skyline/baseline → convert to SPX via offset (optional)
    sky, base = asian_sky_base_es(prev_day)
    sky_card = base_card = None
    if sky and show_asian:
        sky_card = ("Skyline (Asian +0.333)", round(sky[0]+offset,2), to_ct(sky[1]).strftime("%H:%M"))
    if base and show_asian:
        base_card = ("Baseline (Asian −0.333)", round(base[0]+offset,2), to_ct(base[1]).strftime("%H:%M"))

    # 4) Build Close fan (SPX)
    fan_spx = build_close_fan_spx(spx_close_price, to_ct(spx_close_time), proj_day)

    # 5) Project High(+0.333) & Low(−0.333) anchors (SPX)
    high_line = project_anchor_spx(spx_high_price, to_ct(spx_high_time), proj_day, +1)
    low_line  = project_anchor_spx(spx_low_price,  to_ct(spx_low_time),  proj_day, -1)

    # 6) Last price per 30m (SPX preferred; ES+offset fallback)
    spx_proj_30m = fetch_ohlc("^GSPC", proj_day, proj_day, "30m")
    last_map={}
    if not spx_proj_30m.empty:
        rth = between_time_ct(spx_proj_30m, RTH_START, RTH_END)
        for idx, bar in rth.iterrows():
            last_map[idx.strftime("%H:%M")] = float(bar["Close"])
    else:
        es_proj_30m = fetch_ohlc("ES=F", proj_day, proj_day, "30m")
        rth = between_time_ct(es_proj_30m, RTH_START, RTH_END)
        for idx, bar in rth.iterrows():
            last_map[idx.strftime("%H:%M")] = float(bar["Close"] + offset)
    last_series = pd.Series(last_map)

    # Cards
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown("**📌 Prev Close (SPX RTH)**")
        st.markdown(f"<span class='badge'>{round(spx_close_price,2)} @ {to_ct(spx_close_time).strftime('%H:%M')}</span>", unsafe_allow_html=True)
    with c2:
        st.markdown("**🔺 Prev High (SPX)**")
        st.markdown(f"<span class='badge'>{round(spx_high_price,2)} @ {to_ct(spx_high_time).strftime('%H:%M')}</span>", unsafe_allow_html=True)
    with c3:
        st.markdown("**🔻 Prev Low (SPX)**")
        st.markdown(f"<span class='badge'>{round(spx_low_price,2)} @ {to_ct(spx_low_time).strftime('%H:%M')}</span>", unsafe_allow_html=True)
    with c4:
        st.markdown("**🌙 Asian Anchors**")
        if sky_card: st.caption(f"{sky_card[0]}: {sky_card[1]} @{sky_card[2]}")
        if base_card: st.caption(f"{base_card[0]}: {base_card[1]} @{base_card[2]}")
        if not (sky_card or base_card): st.caption("—")

    st.markdown("---")

    # TABLE 1 — Fan
    st.subheader("🧭 Fan & Width (SPX)")
    st.dataframe(fan_spx[["Time","Top","Bottom","Fan_Width"]], use_container_width=True, hide_index=True)

    # TABLE 2 — Strategy
    st.subheader("🎯 Strategy (SPX)")
    strat = build_strategy_table(fan_spx, last_series, high_line, low_line)
    st.dataframe(strat, use_container_width=True, hide_index=True)

    # TABLE 3 — Signals (ES 30m + 1m confirm; display agnostic)
    st.subheader("🔔 Signals (Touch + EMA 8/21 1m Confirmation)")
    es_proj_30m = fetch_ohlc("ES=F", proj_day, proj_day, "30m")
    es_proj_30m_rth = between_time_ct(es_proj_30m, RTH_START, RTH_END) if not es_proj_30m.empty else pd.DataFrame()
    sig_df = signals_table(es_proj_30m_rth, fan_spx, proj_day)
    if sig_df.empty:
        st.info("No signals (or 1-minute data not available for this day).")
    else:
        st.dataframe(sig_df, use_container_width=True, hide_index=True)

else:
    st.info("Choose dates in the sidebar and click **🚀 Generate SPX Tables**.")

st.markdown("---")
if st.button("Test Data Connection"):
    test = fetch_ohlc("^GSPC", today_ct, today_ct, "30m")
    if not test.empty:
        st.success(f"SPX connected — {len(test)} bars today.")
    else:
        test_es = fetch_ohlc("ES=F", today_ct, today_ct, "30m")
        st.warning("SPX intraday empty; ES futures reachable." if not test_es.empty else "No intraday data returned.")