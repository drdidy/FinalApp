# app.py
# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET — Enterprise (with Bounce Forecaster & slot-based Contract Tool)
# - Anchor: SPX previous session ≤3:00 PM CT close (manual override supported)
# - Fan: ASYMMETRIC slopes per 30m (Top +0.312, Bottom −0.25) • overrideable
# - Probability Dashboard: ES→SPX offset ladder (1m→5m→30m→median(5)), booster scores
# - NEW Bounce Forecaster: input 1–2 overnight bounces (21:00–07:00 CT) → project to RTH
# - Contract Tool: slot-based P1/P2 times (prev 17:00 → proj 08:30) so post-midnight works
# - UX: forms & slot pickers prevent disruptive reruns; ⭐ 8:30 highlights
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple

# ───────────────────────────────────────────────────────────────────────────────
# CORE CONFIG
# ───────────────────────────────────────────────────────────────────────────────
CT_TZ = pytz.timezone("America/Chicago")
RTH_START = "08:30"
RTH_END   = "14:30"

TOP_SLOPE_DEFAULT    = 0.312
BOTTOM_SLOPE_DEFAULT = 0.25

STOCK_SLOPES = {
    "TSLA": 0.0285, "NVDA": 0.0860, "AAPL": 0.0155, "MSFT": 0.0541,
    "AMZN": 0.0139, "GOOGL": 0.0122, "META": 0.0674, "NFLX": 0.0230,
}

# Probability boosters (30m basis)
WEIGHTS_DEFAULT = {"ema":20, "volume":25, "wick":20, "atr":15, "tod":20, "div":0}
KEY_TOD = [(8,30), (10,0), (13,30)]
KEY_TOD_WINDOW_MIN = 7
WICK_MIN_RATIO = 0.6
ATR_LOOKBACK = 14
ATR_HIGH_PCTL = 70
ATR_LOW_PCTL  = 30
RSI_LEN = 14
RSI_WINDOW_MIN = 10  # 10×30m

# ───────────────────────────────────────────────────────────────────────────────
# PAGE & THEME
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔮 SPX Prophet Analytics (Enterprise)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --brand:#2563eb; --brand-2:#10b981; --surface:#ffffff; --muted:#f8fafc;
  --text:#0f172a; --subtext:#475569; --border:#e2e8f0; --warn:#f59e0b; --danger:#ef4444;
}
html, body, [class*="css"] { background: var(--muted); color: var(--text); }
.block-container { padding-top: 1.1rem; }
h1, h2, h3 { color: var(--text); }
.card, .metric-card {
  background: rgba(255,255,255,0.9); border: 1px solid var(--border); border-radius: 16px; padding: 16px;
  box-shadow: 0 12px 32px rgba(2,6,23,0.07); backdrop-filter: blur(8px);
}
.metric-title { font-size: .9rem; color: var(--subtext); margin: 0; }
.metric-value { font-size: 1.8rem; font-weight: 700; margin-top: 6px; }
.kicker { font-size: .8rem; color: var(--subtext); }
.badge-open { color:#065f46; background:#d1fae5; border:1px solid #99f6e4; padding:2px 8px; border-radius:999px; font-size:.8rem; font-weight:600; }
.badge-closed { color:#7c2d12; background:#ffedd5; border:1px solid #fed7aa; padding:2px 8px; border-radius:999px; font-size:.8rem; font-weight:600; }
.badge-slot { color:#1f2937; background:#e2e8f0; border:1px solid #cbd5e1; padding:2px 8px; border-radius:999px; font-size:.75rem; font-weight:600; }
hr { border-top: 1px solid var(--border); }
.dataframe { background: var(--surface); border-radius: 12px; overflow: hidden; }
.override-tag { font-size:.75rem; color:#334155; background:#e2e8f0; border:1px solid #cbd5e1; padding:2px 8px; border-radius:999px; display:inline-block; margin-top:6px; }
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────────────────────────
def fmt_ct(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return CT_TZ.localize(dt)
    return dt.astimezone(CT_TZ)

def between_time(df: pd.DataFrame, start_str: str, end_str: str) -> pd.DataFrame:
    return df.between_time(start_str, end_str) if not df.empty else df

def rth_slots_ct(target_date: date) -> List[datetime]:
    start_dt = fmt_ct(datetime.combine(target_date, time(8,30)))
    end_dt   = fmt_ct(datetime.combine(target_date, time(14,30)))
    slots = []
    cur = start_dt
    while cur <= end_dt:
        slots.append(cur)
        cur += timedelta(minutes=30)
    return slots

def gen_slots(start_dt: datetime, end_dt: datetime, step_min: int = 30) -> List[datetime]:
    start_dt = fmt_ct(start_dt); end_dt = fmt_ct(end_dt)
    out = []
    cur = start_dt
    while cur <= end_dt:
        out.append(cur)
        cur += timedelta(minutes=step_min)
    return out

def is_maintenance(dt: datetime) -> bool:
    return dt.hour == 16  # 4–5 PM CT

def in_weekend_gap(dt: datetime) -> bool:
    wd = dt.weekday()
    if wd == 5: return True
    if wd == 6 and dt.hour < 17: return True
    if wd == 4 and dt.hour >= 17: return True
    return False

def count_effective_blocks(anchor_time: datetime, target_time: datetime) -> float:
    if target_time <= anchor_time:
        return 0.0
    t = anchor_time
    blocks = 0
    while t < target_time:
        t_next = t + timedelta(minutes=30)
        if not is_maintenance(t_next) and not in_weekend_gap(t_next):
            blocks += 1
        t = t_next
    return float(blocks)

def ensure_ohlc_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    for c in ["Open","High","Low","Close"]:
        if c not in df.columns:
            return pd.DataFrame()
    return df

def normalize_to_ct(df: pd.DataFrame, start_d: date, end_d: date) -> pd.DataFrame:
    if df.empty:
        return df
    df = ensure_ohlc_cols(df)
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    df.index = df.index.tz_convert(CT_TZ)
    sdt = fmt_ct(datetime.combine(start_d, time(0,0)))
    edt = fmt_ct(datetime.combine(end_d, time(23,59)))
    return df.loc[sdt:edt]

@st.cache_data(ttl=120)
def fetch_intraday(symbol: str, start_d: date, end_d: date, interval: str) -> pd.DataFrame:
    try:
        t = yf.Ticker(symbol)
        if interval in ["1m","2m","5m","15m"]:
            days = max(1, min(7, (end_d - start_d).days + 2))
            df = t.history(period=f"{days}d", interval=interval, prepost=True,
                           auto_adjust=False, back_adjust=False)
            df = normalize_to_ct(df, start_d - timedelta(days=1), end_d + timedelta(days=1))
            sdt = fmt_ct(datetime.combine(start_d, time(0,0)))
            edt = fmt_ct(datetime.combine(end_d, time(23,59)))
            df = df.loc[sdt:edt]
        else:
            df = t.history(
                start=(start_d - timedelta(days=5)).strftime("%Y-%m-%d"),
                end=(end_d + timedelta(days=2)).strftime("%Y-%m-%d"),
                interval=interval, prepost=True, auto_adjust=False, back_adjust=False,
            )
            df = normalize_to_ct(df, start_d, end_d)
        return df
    except Exception:
        return pd.DataFrame()

def resample_to_30m_ct(min_df: pd.DataFrame) -> pd.DataFrame:
    if min_df.empty or not isinstance(min_df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    df = min_df.sort_index()
    agg = {"Open":"first","High":"max","Low":"min","Close":"last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    out = df.resample("30T", label="right", closed="right").agg(agg)
    out = out.dropna(subset=["Open","High","Low","Close"], how="any")
    return out

def get_prev_day_anchor_close_and_time(df_30m: pd.DataFrame, prev_day: date) -> Tuple[Optional[float], Optional[datetime]]:
    if df_30m.empty:
        return None, None
    day_start = fmt_ct(datetime.combine(prev_day, time(0,0)))
    day_end   = fmt_ct(datetime.combine(prev_day, time(23,59)))
    d = df_30m.loc[day_start:day_end].copy()
    if d.empty:
        return None, None
    target = fmt_ct(datetime.combine(prev_day, time(15,0)))
    if target in d.index:
        return float(d.loc[target, "Close"]), target
    prior = d.loc[:target]
    if not prior.empty:
        return float(prior.iloc[-1]["Close"]), prior.index[-1]
    return None, None

def current_spx_slopes() -> Tuple[float, float]:
    top = float(st.session_state.get("top_slope_per_block", TOP_SLOPE_DEFAULT))
    bottom = float(st.session_state.get("bottom_slope_per_block", BOTTOM_SLOPE_DEFAULT))
    return top, bottom

def project_fan_from_close(close_price: float, anchor_time: datetime, target_day: date) -> pd.DataFrame:
    top_slope, bottom_slope = current_spx_slopes()
    rows = []
    for slot in rth_slots_ct(target_day):
        blocks = count_effective_blocks(anchor_time, slot)
        top = close_price + top_slope * blocks
        bot = close_price - bottom_slope * blocks
        rows.append({"TimeDT": slot, "Time": slot.strftime("%H:%M"),
                     "Top": round(top,2), "Bottom": round(bot,2),
                     "Fan_Width": round(top-bot,2)})
    return pd.DataFrame(rows)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    return pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)

# ───────────────────────────────────────────────────────────────────────────────
# BIAS / EDGE LOGIC (for Strategy / Prob Dashboard)
# ───────────────────────────────────────────────────────────────────────────────
def compute_bias(price: float, top: float, bottom: float, tol_frac: float) -> str:
    if bottom <= price <= top:
        width = top - bottom
        center = (top + bottom)/2.0
        band = tol_frac * width
        if center - band <= price <= center + band:
            return "NO BIAS"
        dist_top = abs(top - price)
        dist_bottom = abs(price - bottom)
        return "UP" if dist_bottom < dist_top else "DOWN"
    return "NO BIAS"

def candle_class(open_, close_) -> str:
    if close_ > open_: return "Bullish"
    if close_ < open_: return "Bearish"
    return "Doji"

def touched_line(low, high, line) -> bool:
    return (low <= line <= high)

def classify_edge_touch(bar: pd.Series, top: float, bottom: float) -> Optional[Dict]:
    o = float(bar.get("Open", np.nan))
    h = float(bar.get("High", np.nan))
    l = float(bar.get("Low",  np.nan))
    c = float(bar.get("Close",np.nan))
    cls = candle_class(o, c)

    inside = (bottom <= c <= top)
    above  = (c > top)
    below  = (c < bottom)

    if touched_line(l, h, top) and cls == "Bearish":
        if inside:
            return {"edge":"Top","case":"TopTouch_BearishClose_Inside",
                    "expected":"Breakdown to Bottom → plan to BUY from Bottom",
                    "direction_hint":"DownToBottomThenBuy"}
        if above:
            return {"edge":"Top","case":"TopTouch_BearishClose_Above",
                    "expected":"Top holds as support → market buys higher",
                    "direction_hint":"BuyHigherFromTop"}

    if touched_line(l, h, bottom) and cls == "Bullish":
        if inside:
            return {"edge":"Bottom","case":"BottomTouch_BullishClose_Inside",
                    "expected":"Breakout to Top → plan to SELL from Top",
                    "direction_hint":"UpToTopThenSell"}
        if below:
            return {"edge":"Bottom","case":"BottomTouch_BullishClose_Below",
                    "expected":"Bottom fails → market drops further",
                    "direction_hint":"SellFurtherDown"}

    return None

# ───────────────────────────────────────────────────────────────────────────────
# BOOSTERS & OFFSET (used by Prob Dashboard; left intact)
# ───────────────────────────────────────────────────────────────────────────────
def compute_boosters_score_30m(df_30m: pd.DataFrame, idx_30m: pd.Timestamp,
                               expected_hint: str, weights: Dict[str,int]) -> Tuple[int, Dict[str,int]]:
    comps = {k:0 for k in ["ema","volume","wick","atr","tod","div"]}
    if df_30m.empty or idx_30m not in df_30m.index:
        return 0, comps
    upto = df_30m.loc[:idx_30m].copy()
    if upto.shape[0] < 10:
        return 0, comps

    ema8 = ema(upto["Close"], 8)
    ema21 = ema(upto["Close"], 21)
    ema_state = "Bullish" if ema8.iloc[-1] > ema21.iloc[-1] else ("Bearish" if ema8.iloc[-1] < ema21.iloc[-1] else "None")
    expected_near_term = "Up" if expected_hint in ("BuyHigherFromTop","UpToTopThenSell") else "Down"
    if (expected_near_term == "Up" and ema_state == "Bullish") or (expected_near_term == "Down" and ema_state == "Bearish"):
        comps["ema"] = weights.get("ema",0)

    if "Volume" in upto.columns and upto["Volume"].notna().any():
        vma = upto["Volume"].rolling(20).mean()
        if vma.notna().any() and vma.iloc[-1] and upto["Volume"].iloc[-1] > vma.iloc[-1] * 1.15:
            comps["volume"] = weights.get("volume",0)

    bar = upto.iloc[-1]
    o,h,l,c = float(bar["Open"]), float(bar["High"]), float(bar["Low"]), float(bar["Close"])
    body = abs(c - o) + 1e-9
    upper_wick = max(0.0, h - max(o,c))
    lower_wick = max(0.0, min(o,c) - l)
    if expected_near_term == "Up":
        if lower_wick / body >= WICK_MIN_RATIO:
            comps["wick"] = weights.get("wick",0)
    else:
        if upper_wick / body >= WICK_MIN_RATIO:
            comps["wick"] = weights.get("wick",0)

    tr = true_range(upto)
    atr = tr.rolling(ATR_LOOKBACK).mean()
    if atr.notna().sum() >= ATR_LOOKBACK:
        pct = (atr.rank(pct=True).iloc[-1]) * 100.0
        if expected_hint in ("BuyHigherFromTop","UpToTopThenSell"):
            if pct <= ATR_LOW_PCTL:
                comps["atr"] = weights.get("atr",0)
        elif expected_hint in ("SellFurtherDown","DownToBottomThenBuy"):
            if pct >= ATR_HIGH_PCTL:
                comps["atr"] = weights.get("atr",0)

    ts = fmt_ct(idx_30m.to_pydatetime())
    if any(abs((ts.hour*60 + ts.minute) - (hh*60+mm)) <= KEY_TOD_WINDOW_MIN for (hh,mm) in KEY_TOD):
        comps["tod"] = weights.get("tod",0)

    if weights.get("div",0) > 0:
        r = rsi(upto["Close"], RSI_LEN)
        if r.notna().sum() >= RSI_LEN + 2:
            window_bars = max(5, RSI_WINDOW_MIN)
            prior = upto.iloc[-window_bars:-1] if upto.shape[0] > window_bars else upto.iloc[:-1]
            if prior.shape[0] > 5:
                prior_low = prior["Close"].idxmin()
                prior_high = prior["Close"].idxmax()
                if expected_near_term == "Up":
                    if upto["Close"].iloc[-1] <= prior["Close"].min() and r.iloc[-1] > r.loc[prior_low]:
                        comps["div"] = weights.get("div",0)
                else:
                    if upto["Close"].iloc[-1] >= prior["Close"].max() and r.iloc[-1] < r.loc[prior_high]:
                        comps["div"] = weights.get("div",0)

    score = int(min(100, max(0, sum(comps.values()))))
    return score, comps

def _nearest_le_index(idx: pd.DatetimeIndex, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    s = idx[idx <= ts]
    return s[-1] if len(s) else None

def es_spx_offset_at_anchor(prev_day: date, spx_30m: pd.DataFrame) -> Optional[float]:
    spx_anchor_close, spx_anchor_time = get_prev_day_anchor_close_and_time(spx_30m, prev_day)
    if spx_anchor_close is None or spx_anchor_time is None:
        return None

    def try_sym_interval(sym: str, interval: str) -> Optional[float]:
        df = fetch_intraday(sym, prev_day, prev_day, interval)
        if df.empty or "Close" not in df.columns:
            return None
        lo = spx_anchor_time - timedelta(minutes=15)
        hi = spx_anchor_time
        window = df.loc[(df.index >= lo) & (df.index <= hi)]
        if not window.empty:
            es_close = float(window["Close"].iloc[-1])
            return es_close - spx_anchor_close
        idx = _nearest_le_index(df.index, spx_anchor_time)
        if idx is None:
            return None
        es_close = float(df.loc[idx, "Close"])
        return es_close - spx_anchor_close

    for interval in ["1m","5m"]:
        off = try_sym_interval("ES=F", interval)
        if off is not None:
            return float(off)

    es_30m = fetch_intraday("ES=F", prev_day, prev_day, "30m")
    if not es_30m.empty and "Close" in es_30m.columns:
        idx30 = _nearest_le_index(es_30m.index, spx_anchor_time)
        if idx30 is not None:
            return float(es_30m.loc[idx30, "Close"] - spx_anchor_close)

    med_vals = []
    for i in range(1, 6):
        d = prev_day - timedelta(days=i)
        spx_d = fetch_intraday("^GSPC", d, d, "30m")
        if spx_d.empty:
            spx_d = fetch_intraday("SPY", d, d, "30m")
        s_close, s_time = get_prev_day_anchor_close_and_time(spx_d, d)
        if s_close is None or s_time is None:
            continue
        es5 = fetch_intraday("ES=F", d, d, "5m")
        if not es5.empty and "Close" in es5.columns:
            lo, hi = s_time - timedelta(minutes=15), s_time
            win = es5.loc[(es5.index >= lo) & (es5.index <= hi)]
            if not win.empty:
                med_vals.append(float(win["Close"].iloc[-1] - s_close))
            else:
                idxd = _nearest_le_index(es5.index, s_time)
                if idxd is not None:
                    med_vals.append(float(es5.loc[idxd, "Close"] - s_close))
    if med_vals:
        return float(np.median(med_vals))

    return None

def fetch_overnight_minute(prev_day: date, proj_day: date) -> Tuple[pd.DataFrame, str]:
    start = fmt_ct(datetime.combine(prev_day, time(17,0)))
    end   = fmt_ct(datetime.combine(proj_day, time(8,30)))
    es_1m = fetch_intraday("ES=F", prev_day, proj_day, "1m")
    if not es_1m.empty:
        return es_1m.loc[start:end].copy(), "1m"
    es_5m = fetch_intraday("ES=F", prev_day, proj_day, "5m")
    if not es_5m.empty:
        return es_5m.loc[start:end].copy(), "5m"
    es_30m = fetch_intraday("ES=F", prev_day, proj_day, "30m")
    if not es_30m.empty:
        return es_30m.loc[start:end].copy(), "30m"
    return pd.DataFrame(), "none"

def adjust_to_spx_frame(es_df: pd.DataFrame, offset: float) -> pd.DataFrame:
    df = es_df.copy()
    for col in ["Open","High","Low","Close"]:
        if col in df:
            df[col] = df[col] - offset
    return df

def nearest_30m_index(idx_30m: pd.DatetimeIndex, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    if idx_30m.empty:
        return None
    loc_df = idx_30m[idx_30m <= ts]
    if len(loc_df) == 0:
        return None
    return loc_df[-1]

# ───────────────────────────────────────────────────────────────────────────────
# PROBABILITY DASHBOARD BUILD
# ───────────────────────────────────────────────────────────────────────────────
def build_probability_dashboard(prev_day: date, proj_day: date,
                                anchor_close: float, anchor_time: datetime,
                                tol_frac: float, weights: Dict[str,int]) -> Tuple[pd.DataFrame, pd.DataFrame, float, str]:
    fan_df = project_fan_from_close(anchor_close, anchor_time, proj_day)

    spx_prev_30m = fetch_intraday("^GSPC", prev_day, prev_day, "30m")
    if spx_prev_30m.empty:
        spx_prev_30m = fetch_intraday("SPY", prev_day, prev_day, "30m")

    off = es_spx_offset_at_anchor(prev_day, spx_prev_30m)
    if off is None:
        return pd.DataFrame(), fan_df, 0.0, "none"

    on_bars, used_interval = fetch_overnight_minute(prev_day, proj_day)
    if on_bars.empty:
        return pd.DataFrame(), fan_df, off, "none"

    on_adj = adjust_to_spx_frame(on_bars, off)
    on_adj_30m = resample_to_30m_ct(on_adj)

    detect_df = on_adj if used_interval in ("1m","5m") else on_adj_30m

    top_slope, bottom_slope = current_spx_slopes()
    rows = []
    for ts, bar in detect_df.iterrows():
        blocks = count_effective_blocks(anchor_time, ts)
        top = anchor_close + top_slope * blocks
        bottom = anchor_close - bottom_slope * blocks

        touch = classify_edge_touch(bar, top, bottom)
        if touch is None:
            continue

        idx_30m = nearest_30m_index(on_adj_30m.index, ts)
        if idx_30m is None:
            score, comps = 0, {k:0 for k in ["ema","volume","wick","atr","tod","div"]}
        else:
            score, comps = compute_boosters_score_30m(on_adj_30m, idx_30m, touch["direction_hint"], weights)

        price = float(bar["Close"])
        bias = compute_bias(price, top, bottom, tol_frac)

        rows.append({
            "TimeDT": ts, "Time": ts.strftime("%H:%M"),
            "Price": round(price,2), "Top": round(top,2), "Bottom": round(bottom,2),
            "Edge": touch["edge"], "Case": touch["case"],
            "Expectation": touch["expected"], "DirectionHint": touch["direction_hint"],
            "Bias": bias, "Score": score,
            "EMA_w": comps.get("ema",0), "Vol_w": comps.get("volume",0), "Wick_w": comps.get("wick",0),
            "ATR_w": comps.get("atr",0), "ToD_w": comps.get("tod",0), "Div_w": comps.get("div",0),
        })

    touches_df = pd.DataFrame(rows).sort_values("TimeDT").reset_index(drop=True)
    return touches_df, fan_df, off, used_interval

# ───────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Global controls
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🔧 Controls")
today_ct = datetime.now(CT_TZ).date()
prev_day = st.sidebar.date_input("Previous Trading Day", value=today_ct - timedelta(days=1))
proj_day = st.sidebar.date_input("Projection Day", value=prev_day + timedelta(days=1))
st.sidebar.caption("Anchor uses the **last SPX bar ≤ 3:00 PM CT** on the previous session (manual override available).")

st.sidebar.markdown("---")
st.sidebar.subheader("✍️ Manual Close (optional)")
use_manual_close = st.sidebar.checkbox("Enter 3:00 PM CT Close Manually", value=False)
manual_close_val = st.sidebar.number_input("Manual 3:00 PM Close", value=6400.00, step=0.01, format="%.2f",
                                           disabled=not use_manual_close,
                                           help="Overrides the fetched SPX 3:00 PM anchor close.")

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Advanced (optional)", expanded=False):
    st.caption("Adjust **asymmetric** fan slopes and the within-fan neutrality band.")
    enable_slope = st.checkbox("Enable slope override",
                               value=("top_slope_per_block" in st.session_state or "bottom_slope_per_block" in st.session_state))
    top_slope_val = st.number_input("Top slope (+ per 30m)",
                                    value=float(st.session_state.get("top_slope_per_block", TOP_SLOPE_DEFAULT)),
                                    step=0.001, format="%.3f")
    bottom_slope_val = st.number_input("Bottom slope (− per 30m)",
                                       value=float(st.session_state.get("bottom_slope_per_block", BOTTOM_SLOPE_DEFAULT)),
                                       step=0.001, format="%.3f")
    tol_frac = st.slider("Neutrality band (% of fan width)", 0, 40, 20, 1) / 100.0

    col_adv_a, col_adv_b = st.columns(2)
    with col_adv_a:
        if st.button("Apply slopes", use_container_width=True, key="apply_slope"):
            if enable_slope:
                st.session_state["top_slope_per_block"] = float(top_slope_val)
                st.session_state["bottom_slope_per_block"] = float(bottom_slope_val)
                st.success(f"Top=+{top_slope_val:.3f}  •  Bottom=−{bottom_slope_val:.3f}")
            else:
                for k in ("top_slope_per_block","bottom_slope_per_block"):
                    st.session_state.pop(k, None)
                st.info("Slope override disabled (using defaults).")
    with col_adv_b:
        if st.button("Reset slopes", use_container_width=True, key="reset_slope"):
            for k in ("top_slope_per_block","bottom_slope_per_block"):
                st.session_state.pop(k, None)
            st.success(f"Reset → Top=+{TOP_SLOPE_DEFAULT:.3f} • Bottom=−{BOTTOM_SLOPE_DEFAULT:.3f}")

st.sidebar.markdown("---")
go_spx   = st.sidebar.button("🔮 Generate / Refresh SPX Fan & Strategy", type="primary",  use_container_width=True)
run_prob = st.sidebar.button("🧠 Analyze / Refresh Probability Dashboard", type="secondary",use_container_width=True)

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
""", unsafe_allow_html=True)
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
""", unsafe_allow_html=True)
with c3:
    ts, bs = current_spx_slopes()
    st.markdown(
        f"""
<div class="metric-card">
  <p class="metric-title">SPX Slopes / 30m</p>
  <div class="metric-value">📐 Top=+{ts:.3f} • Bottom=−{bs:.3f}</div>
  <div class="kicker">Asymmetric fan</div>
  {"<div class='override-tag'>Override active</div>" if ("top_slope_per_block" in st.session_state or "bottom_slope_per_block" in st.session_state) else ""}
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ───────────────────────────────────────────────────────────────────────────────
# TABS (Signals & EMA replaced by Bounce Forecaster)
# ───────────────────────────────────────────────────────────────────────────────
tab1, tabProb, tab2, tabBounce, tab4 = st.tabs(
    ["SPX Anchors", "Probability Dashboard", "Stock Anchors", "Bounce Forecaster", "Contract Tool"]
)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 1: SPX ANCHORS                                                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.subheader("SPX Close-Anchor Fan (≤3:00 PM CT) — ⭐ 8:30 Highlight & Correct Bias")

    if go_spx:
        with st.spinner("Building SPX fan & strategy…"):
            spx_prev = fetch_intraday("^GSPC", prev_day, prev_day, "30m")
            if spx_prev.empty:
                spx_prev = fetch_intraday("SPY", prev_day, prev_day, "30m")
            if spx_prev.empty:
                st.error("❌ Previous day data missing — can’t compute the anchor.")
                st.stop()

            if use_manual_close:
                anchor_close = float(manual_close_val)
                anchor_time  = fmt_ct(datetime.combine(prev_day, time(15,0)))
            else:
                anchor_close, anchor_time = get_prev_day_anchor_close_and_time(spx_prev, prev_day)
                if anchor_close is None or anchor_time is None:
                    st.error("Could not find a ≤3:00 PM CT close for the previous day.")
                    st.stop()

            fan_df = project_fan_from_close(anchor_close, anchor_time, proj_day)

            spx_proj = fetch_intraday("^GSPC", proj_day, proj_day, "30m")
            if spx_proj.empty:
                spx_proj = fetch_intraday("SPY", proj_day, proj_day, "30m")
            spx_proj_rth = between_time(spx_proj, RTH_START, RTH_END)

            tslope, bslope = current_spx_slopes()
            rows = []
            iter_index = (spx_proj_rth.index if not spx_proj_rth.empty
                          else pd.DatetimeIndex(rth_slots_ct(proj_day)))
            for dt in iter_index:
                blocks = count_effective_blocks(anchor_time, dt)
                top = anchor_close + tslope * blocks
                bottom = anchor_close - bslope * blocks
                if not spx_proj_rth.empty and dt in spx_proj_rth.index:
                    price = float(spx_proj_rth.loc[dt, "Close"])
                    bias = compute_bias(price, top, bottom, tol_frac)
                    note = "Within/Above/Below fan per price"
                else:
                    price = np.nan
                    bias = "NO DATA"
                    note = "No RTH data yet (pre-market). Fan only."
                rows.append({
                    "Time": dt.strftime("%H:%M"),
                    "Price": (round(price,2) if not np.isnan(price) else np.nan),
                    "Bias": bias, "Top": round(top,2), "Bottom": round(bottom,2),
                    "Fan_Width": round(top-bottom,2),
                    "Slot": "⭐ 8:30" if dt.strftime("%H:%M")=="08:30" else "",
                    "Note": note
                })
            strat_df = pd.DataFrame(rows)

            st.session_state["spx_result"] = {
                "fan_df": fan_df, "strat_df": strat_df,
                "anchor_close": anchor_close, "anchor_time": anchor_time,
                "prev_day": prev_day, "proj_day": proj_day, "tol_frac": tol_frac
            }

    if "spx_result" in st.session_state:
        fan_df   = st.session_state["spx_result"]["fan_df"]
        strat_df = st.session_state["spx_result"]["strat_df"]

        st.markdown("### 🎯 Fan Lines (Top / Bottom @ 30-min)")
        st.dataframe(fan_df[["Time","Top","Bottom","Fan_Width"]], use_container_width=True, hide_index=True)

        st.markdown("### 📋 Strategy Table (Corrected Bias)")
        st.caption("Bias uses **descending anchor** proximity & neutrality band. ⭐ marks 8:30.")
        st.dataframe(
            strat_df[["Slot","Time","Price","Bias","Top","Bottom","Fan_Width","Note"]],
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Click **Generate / Refresh SPX Fan & Strategy** in the sidebar.")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 2: PROBABILITY DASHBOARD                                                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tabProb:
    st.subheader("Probability Dashboard — 1m/5m edge detection (30m last-resort) • 30m boosters")
    st.caption("ES→SPX offset ladder ensures historical robustness. Boosters use your own CT-aligned 30m resample.")

    left, right = st.columns(2)
    with left:
        enable_div = st.checkbox("Enable Oscillator Divergence (30m, lightweight)", value=False, key="prob_enable_div")
    with right:
        w_ema  = st.slider("Weight: EMA 8/21 (30m)", 0, 40, WEIGHTS_DEFAULT["ema"], 5, key="prob_w_ema")
        w_vol  = st.slider("Weight: Volume (30m)", 0, 40, WEIGHTS_DEFAULT["volume"], 5, key="prob_w_vol")
        w_wick = st.slider("Weight: Wick/Body (30m)", 0, 40, WEIGHTS_DEFAULT["wick"], 5, key="prob_w_wick")
        w_atr  = st.slider("Weight: ATR (30m)", 0, 40, WEIGHTS_DEFAULT["atr"], 5, key="prob_w_atr")
        w_tod  = st.slider("Weight: Time-of-Day", 0, 40, WEIGHTS_DEFAULT["tod"], 5, key="prob_w_tod")
        w_div  = st.slider("Weight: Divergence (if enabled)", 0, 40, 10 if enable_div else 0, 5,
                           disabled=not enable_div, key="prob_w_div")

    if run_prob:
        with st.spinner("Scoring overnight edge interactions…"):
            spx_prev = fetch_intraday("^GSPC", prev_day, prev_day, "30m")
            if spx_prev.empty:
                spx_prev = fetch_intraday("SPY", prev_day, prev_day, "30m")
            if spx_prev.empty:
                st.error("Could not fetch previous day SPX data.")
                st.stop()

            if use_manual_close:
                anchor_close = float(manual_close_val)
                anchor_time  = fmt_ct(datetime.combine(prev_day, time(15,0)))
            else:
                anchor_close, anchor_time = get_prev_day_anchor_close_and_time(spx_prev, prev_day)
                if anchor_close is None or anchor_time is None:
                    st.error("Could not find a ≤3:00 PM CT close for the previous day.")
                    st.stop()

            custom_weights = {
                "ema": st.session_state["prob_w_ema"],
                "volume": st.session_state["prob_w_vol"],
                "wick": st.session_state["prob_w_wick"],
                "atr": st.session_state["prob_w_atr"],
                "tod": st.session_state["prob_w_tod"],
                "div": (st.session_state["prob_w_div"] if st.session_state["prob_enable_div"] else 0)
            }

            touches_df, fan_df, offset_used, used_interval = build_probability_dashboard(
                prev_day, proj_day, anchor_close, anchor_time, tol_frac, custom_weights
            )

            st.session_state["prob_result"] = {
                "touches_df": touches_df, "fan_df": fan_df,
                "offset_used": float(offset_used), "used_interval": used_interval,
                "anchor_close": anchor_close, "anchor_time": anchor_time,
                "prev_day": prev_day, "proj_day": proj_day,
                "weights": custom_weights, "tol_frac": tol_frac
            }

    if "prob_result" in st.session_state:
        pr = st.session_state["prob_result"]
        touches_df = pr["touches_df"]

        cA, cB, cC = st.columns(3)
        with cA:
            st.markdown(f"<div class='metric-card'><p class='metric-title'>Anchor Close (Prev ≤3:00 PM CT)</p><div class='metric-value'>💠 {pr['anchor_close']:.2f}</div></div>", unsafe_allow_html=True)
        with cB:
            st.markdown(f"<div class='metric-card'><p class='metric-title'>Overnight Edge Touches</p><div class='metric-value'>🧩 {len(touches_df)}</div></div>", unsafe_allow_html=True)
        with cC:
            st.markdown(f"<div class='metric-card'><p class='metric-title'>ES→SPX Offset</p><div class='metric-value'>Δ {pr['offset_used']:+.2f}</div><div class='kicker'>Data: {pr['used_interval']}</div></div>", unsafe_allow_html=True)

        if pr["used_interval"] == "none":
            st.info("ES→SPX offset or overnight prints were unavailable for this date even after fallbacks. Try a nearby date.")

        st.markdown("### 📡 Overnight Edge Interactions (Scored)")
        if touches_df.empty:
            st.info("No qualifying edge touches detected for this window.")
        else:
            view_cols = ["Time","Price","Top","Bottom","Bias","Edge","Case","Expectation","Score","EMA_w","Vol_w","Wick_w","ATR_w","ToD_w","Div_w"]
            st.dataframe(touches_df[view_cols], use_container_width=True, hide_index=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 3: STOCK ANCHORS                                                        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.subheader("Stock Anchor Lines (Mon/Tue swings → two lines)")
    st.caption("Projects an ascending line from the highest swing high and a descending line from the lowest swing low (Mon+Tue combined).")

    core = list(STOCK_SLOPES.keys())
    cc1, cc2, cc3 = st.columns([1.4,1,1])
    with cc1:
        ticker = st.selectbox("Ticker", core + ["Custom…"], index=0, key="stk_ticker")
        custom_ticker = ""
        if ticker == "Custom…":
            custom_ticker = st.text_input("Custom Symbol", value="", placeholder="e.g., AMD")
    with cc2:
        monday_default = today_ct - timedelta(days=((today_ct.weekday() - 0) % 7 or 7))
        monday_date = st.date_input("Monday Date", value=monday_default)
    with cc3:
        tuesday_date = st.date_input("Tuesday Date", value=monday_date + timedelta(days=1))

    slope_mag_default = STOCK_SLOPES.get(ticker, 0.0150) if ticker != "Custom…" else 0.0150
    slope_mag = st.number_input("Slope Magnitude (per 30m)", value=float(slope_mag_default), step=0.0001, format="%.4f")

    proj_day_stock = st.date_input("Projection Day", value=tuesday_date + timedelta(days=1))
    run_stock = st.button("📈 Analyze Stock Anchors", type="primary")

    if run_stock:
        with st.spinner("Fetching and projecting…"):
            sym = custom_ticker.upper() if ticker == "Custom…" and custom_ticker else (ticker if ticker != "Custom…" else None)
            if not sym:
                st.error("Please enter a custom symbol.")
                st.stop()

            mon = fetch_intraday(sym, monday_date, monday_date, "30m")
            tue = fetch_intraday(sym, tuesday_date, tuesday_date, "30m")
            if mon.empty and tue.empty:
                st.error("No data for selected dates.")
                st.stop()

            combined = mon if tue.empty else (tue if mon.empty else pd.concat([mon, tue]).sort_index())

            if combined.empty or "High" not in combined or "Low" not in combined:
                st.error("Data missing High/Low columns.")
                st.stop()
            hi_idx = combined["High"].idxmax()
            lo_idx = combined["Low"].idxmin()
            high_price, high_time = float(combined.loc[hi_idx, "High"]), fmt_ct(hi_idx)
            low_price,  low_time  = float(combined.loc[lo_idx, "Low"]),  fmt_ct(lo_idx)

            rows = []
            for slot in rth_slots_ct(proj_day_stock):
                b_high = count_effective_blocks(high_time, slot)
                b_low  = count_effective_blocks(low_time,  slot)
                high_asc = high_price + slope_mag * b_high
                low_desc = low_price  - slope_mag * b_low
                rows.append({"Time": slot.strftime("%H:%M"),
                             "High_Asc": round(high_asc, 2),
                             "Low_Desc": round(low_desc, 2)})
            proj_df = pd.DataFrame(rows)

            cA, cB = st.columns(2)
            with cA:
                st.markdown("**Swing High (Mon/Tue):**")
                st.write(f"📈 {sym} — High: **{high_price:.2f}** at {high_time.strftime('%Y-%m-%d %H:%M CT')}")
            with cB:
                st.markdown("**Swing Low (Mon/Tue):**")
                st.write(f"📉 {sym} — Low: **{low_price:.2f}** at {low_time.strftime('%Y-%m-%d %H:%M CT')}")

            st.markdown("### 🔧 Projection (RTH)")
            st.dataframe(proj_df, use_container_width=True, hide_index=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 4: BOUNCE FORECASTER (NEW)                                              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tabBounce:
    st.subheader("Bounce Forecaster — Input Overnight Bounces (21:00–07:00 CT) → Project 8:30–14:30")
    st.caption("Use 30-minute slots. With 2 bounces we auto-fit slope; with 1 bounce you can set an assumed slope per 30m.")

    # Build slot lists (prev_day 21:00 → proj_day 07:00)
    overnight_start = fmt_ct(datetime.combine(prev_day, time(21,0)))
    overnight_end   = fmt_ct(datetime.combine(proj_day, time(7,0)))
    overnight_slots = gen_slots(overnight_start, overnight_end, 30)
    slot_labels = [dt.strftime("%Y-%m-%d %H:%M") for dt in overnight_slots]

    with st.form("bounce_form", clear_on_submit=False):
        bcount = st.radio("How many bounces are you entering?", [1, 2], index=1, horizontal=True)

        colb1, colb2 = st.columns(2)
        with colb1:
            b1_sel = st.selectbox("Bounce #1 Time (slot)", slot_labels, index=0, key="b1_sel")
            b1_price = st.number_input("Bounce #1 Price", min_value=0.01, value=6400.00, step=0.25, format="%.2f", key="b1_price")
        with colb2:
            if bcount == 2:
                b2_sel = st.selectbox("Bounce #2 Time (slot)", slot_labels, index=min(6, len(slot_labels)-1), key="b2_sel")
                b2_price = st.number_input("Bounce #2 Price", min_value=0.01, value=6412.00, step=0.25, format="%.2f", key="b2_price")

        if bcount == 1:
            assumed_slope = st.number_input("Assumed slope per 30m (optional for 1 bounce)", value=0.00, step=0.01, format="%.2f", help="If left at 0.00 the projection stays flat from Bounce #1.")

        submitted_bounce = st.form_submit_button("📈 Project RTH Bounce Line")

    if submitted_bounce:
        b1_dt = fmt_ct(datetime.strptime(b1_sel, "%Y-%m-%d %H:%M"))
        p1 = float(b1_price)
        if bcount == 2:
            b2_dt = fmt_ct(datetime.strptime(b2_sel, "%Y-%m-%d %H:%M"))
            p2 = float(b2_price)
            if b2_dt <= b1_dt:
                st.error("Bounce #2 time must be after Bounce #1 time.")
            else:
                blocks = count_effective_blocks(b1_dt, b2_dt)
                slope = (p2 - p1) / blocks if blocks > 0 else 0.0
        else:
            slope = float(assumed_slope)

        # Project across RTH
        rows = []
        for slot in rth_slots_ct(proj_day):
            b = count_effective_blocks(b1_dt, slot)
            price = p1 + slope * b
            rows.append({"Time": slot.strftime("%H:%M"),
                         "Projected_Bounce": round(price, 2),
                         "Blocks_From_First": round(b, 1)})
        out_df = pd.DataFrame(rows)
        out_df.insert(0, "Slot", out_df["Time"].apply(lambda x: "⭐ 8:30" if x=="08:30" else ""))

        mc1, mc2, mc3 = st.columns(3)
        with mc1: st.markdown(f"<div class='metric-card'><p class='metric-title'>Slope / 30m</p><div class='metric-value'>📐 {slope:+.3f}</div><div class='kicker'>Auto-fit if 2 bounces</div></div>", unsafe_allow_html=True)
        with mc2: st.markdown(f"<div class='metric-card'><p class='metric-title'>Seed Time</p><div class='metric-value'>⏱ {b1_dt.strftime('%Y-%m-%d %H:%M')}</div></div>", unsafe_allow_html=True)
        with mc3: st.markdown(f"<div class='metric-card'><p class='metric-title'>RTH Slots</p><div class='metric-value'>🧭 8:30 → 14:30</div></div>", unsafe_allow_html=True)

        st.markdown("### 🔮 Projected Bounce Line (RTH)")
        st.dataframe(out_df, use_container_width=True, hide_index=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 5: CONTRACT TOOL (slot-based)                                           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab4:
    st.subheader("Contract Tool (Point-to-Point or Δ+Θ) — Slot-based time inputs")

    # Slot pickers from prev 17:00 → proj 08:30 (covers all overnight + pre-open)
    ovn_start_all = fmt_ct(datetime.combine(prev_day, time(17,0)))
    ovn_end_all   = fmt_ct(datetime.combine(proj_day, time(8,30)))
    ovn_slots_all = gen_slots(ovn_start_all, ovn_end_all, 30)
    ovn_labels_all = [dt.strftime("%Y-%m-%d %H:%M") for dt in ovn_slots_all]

    with st.form("standalone_contract_form", clear_on_submit=False):
        mode_ct = st.selectbox("Mode", ["Two-Point Slope","Δ + Θ (model-aware)"], index=0)

        colp1, colp2 = st.columns(2)
        with colp1:
            p1_sel = st.selectbox("Point 1 Time (slot)", ovn_labels_all, index=0, key="ct_p1_sel")
            p1_price = st.number_input("Point 1 Contract Price", value=10.00, min_value=0.01, step=0.01, format="%.2f")
        with colp2:
            p2_sel = st.selectbox("Point 2 Time (slot)", ovn_labels_all, index=min(6, len(ovn_labels_all)-1), key="ct_p2_sel")
            p2_price = st.number_input("Point 2 Contract Price", value=12.00, min_value=0.01, step=0.01, format="%.2f")

        proj_day_ct = st.date_input("RTH Projection Day", value=proj_day)
        submit_ct = st.form_submit_button("🧮 Analyze Contract Projections")

    if submit_ct:
        p1_dt = fmt_ct(datetime.strptime(st.session_state["ct_p1_sel"], "%Y-%m-%d %H:%M"))
        p2_dt = fmt_ct(datetime.strptime(st.session_state["ct_p2_sel"], "%Y-%m-%d %H:%M"))

        def project_contract_two_point(p1_dt, p1_price, p2_dt, p2_price, proj_day):
            if p2_dt is not None and p2_price is not None and p2_dt > p1_dt:
                blocks = count_effective_blocks(p1_dt, p2_dt)
                slope = (p2_price - p1_price) / blocks if blocks > 0 else 0.0
            else:
                slope = 0.0
            rows = []
            for slot in rth_slots_ct(proj_day):
                b = count_effective_blocks(p1_dt, slot)
                price = p1_price + slope * b
                rows.append({"Time": slot.strftime("%H:%M"),
                             "Contract_Price": round(price, 2),
                             "Blocks": round(b, 1)})
            return pd.DataFrame(rows), slope, float(count_effective_blocks(p1_dt, fmt_ct(datetime.combine(proj_day, time(8,30)))))

        def project_contract_delta_theta(p1_dt, p1_price, underlying_slope_per_30m, delta, theta_per_day, proj_day):
            theta_per_30m = theta_per_day / 48.0
            contract_slope = delta * underlying_slope_per_30m + theta_per_30m
            rows = []
            for slot in rth_slots_ct(proj_day):
                b = count_effective_blocks(p1_dt, slot)
                price = p1_price + contract_slope * b
                rows.append({"Time": slot.strftime("%H:%M"),
                             "Contract_Price": round(price, 2),
                             "Blocks": round(b, 1)})
            return pd.DataFrame(rows), contract_slope

        if mode_ct == "Two-Point Slope":
            if p2_dt <= p1_dt:
                st.error("Point 2 must be after Point 1 for Two-Point mode.")
            else:
                proj_df, slope_ct, blocks = project_contract_two_point(p1_dt, float(p1_price), p2_dt, float(p2_price), proj_day_ct)
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1: st.markdown(f"<div class='metric-card'><p class='metric-title'>Time Span</p><div class='metric-value'>⏱ {(p2_dt-p1_dt).total_seconds()/3600:.1f}h</div></div>", unsafe_allow_html=True)
                with mc2: st.markdown(f"<div class='metric-card'><p class='metric-title'>Δ Price</p><div class='metric-value'>↕ {p2_price - p1_price:+.2f}</div></div>", unsafe_allow_html=True)
                with mc3: st.markdown(f"<div class='metric-card'><p class='metric-title'>Blocks Counted</p><div class='metric-value'>🧩 {blocks:.1f}</div></div>", unsafe_allow_html=True)
                with mc4: st.markdown(f"<div class='metric-card'><p class='metric-title'>Slope / 30m</p><div class='metric-value'>📐 {slope_ct:+.3f}</div></div>", unsafe_allow_html=True)
                proj_df.insert(0, "Slot", proj_df["Time"].apply(lambda x: "⭐ 8:30" if x=="08:30" else ""))
                st.markdown("### 📊 RTH Projection")
                st.dataframe(proj_df, use_container_width=True, hide_index=True)
        else:
            colg1, colg2, colg3 = st.columns(3)
            with colg1:
                underlying_slope = st.number_input("Underlying slope / 30m (±)", value=0.10, step=0.01, format="%.2f",
                                                   help="Approx expected SPX move per 30m (+ for up, − for down).")
            with colg2:
                delta = st.number_input("Δ (Delta)", value=0.35, step=0.01, format="%.2f")
            with colg3:
                theta_day = st.number_input("Θ per day (negative)", value=-20.00, step=0.5, format="%.2f")
            proj_df, c_slope = project_contract_delta_theta(p1_dt, float(p1_price), float(underlying_slope), float(delta), float(theta_day), proj_day_ct)
            mc1, mc2, mc3 = st.columns(3)
            with mc1: st.markdown(f"<div class='metric-card'><p class='metric-title'>Seed Time</p><div class='metric-value'>⏱ {p1_dt.strftime('%Y-%m-%d %H:%M')}</div></div>", unsafe_allow_html=True)
            with mc2: st.markdown(f"<div class='metric-card'><p class='metric-title'>Contract Slope / 30m</p><div class='metric-value'>📐 {c_slope:+.3f}</div></div>", unsafe_allow_html=True)
            with mc3: st.markdown(f"<div class='metric-card'><p class='metric-title'>Θ / 30m</p><div class='metric-value'>θ {theta_day/48.0:+.3f}</div></div>", unsafe_allow_html=True)
            proj_df.insert(0, "Slot", proj_df["Time"].apply(lambda x: "⭐ 8:30" if x=="08:30" else ""))
            st.markdown("### 📊 RTH Projection")
            st.dataframe(proj_df, use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────────────────────────────────
# FOOTER UTILITIES
# ───────────────────────────────────────────────────────────────────────────────
st.markdown("---")
colA, colB = st.columns([1, 2])
with colA:
    if st.button("🔌 Test Data Connection"):
        td = fetch_intraday("^GSPC", today_ct - timedelta(days=3), today_ct, "30m")
        if td.empty:
            td = fetch_intraday("SPY", today_ct - timedelta(days=3), today_ct, "30m")
        if not td.empty:
            st.success(f"OK — received {len(td)} bars (30m).")
        else:
            st.error("Data fetch failed — try different dates later.")
with colB:
    st.caption("Offset ladder: **1m → 5m → 30m → median(5)**. Edge detect: **1m/5m**, 30m last-resort. Forms/slots prevent reruns. ⭐ 8:30 highlighted.")