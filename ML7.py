# app.py
# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET — SPX-only Enterprise App
# Tabs: 1) SPX Anchors  2) BC Forecast (2 bounces, Entries & Exits)
#       3) Probability Board (30m-only, liquidity-weighted)  4) Plan Card
#
# Core:
# - Anchor: previous session ≤ 3:00 PM CT SPX cash close (manual override supported)
# - Fan slopes (per 30m): Top +0.312  •  Bottom −0.25  (overrideable)
# - 30m-only logic: direction-of-travel, edge interactions, bias, scoring
# - ES→SPX offset measured at anchor (under the hood; not shown in UI)
# - Overnight window: prev 17:00 → proj 08:30 CT
# - Tables are compact; advanced columns hidden behind a toggle; ⭐ 8:30 emphasis
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple

# ───────────────────────────────────────────────────────────────────────────────
# GLOBALS & CONFIG
# ───────────────────────────────────────────────────────────────────────────────
CT_TZ = pytz.timezone("America/Chicago")
RTH_START = "08:30"
RTH_END   = "14:30"

TOP_SLOPE_DEFAULT    = 0.312
BOTTOM_SLOPE_DEFAULT = 0.25

# Neutral band (inside-fan bias) default: 20% of width
NEUTRAL_BAND_DEFAULT = 0.20

# Liquidity windows (CT)
SYD_TOK = [(21,0), (21,30)]   # 9:00–9:30 PM CT
TOK_LON = [(2,0),  (2,30)]    # 2:00–2:30 AM CT
PRE_NY  = [(7,0),  (7,30)]    # 7:00–7:30 AM CT

# Liquidity weights (multipliers added to score in percent points)
W_SYD_TOK = 25
W_TOK_LON = 40
W_PRE_NY  = 20

# Probability component weights (percent points; sum can exceed 100, we clamp later)
WEIGHTS = {
    "confluence": 25,
    "structure": 20,
    "wick": 15,
    "atr": 10,
    "compression": 10,
    "gap": 10,
    "cluster": 10,
    "liquidity": 0,  # handled separately (the +25/+40/+20 bumps)
    "volume": 5,     # small final tiebreaker (if available)
}

ATR_LOOKBACK = 14
RANGE_WIN = 20     # for compression
GAP_LOOKBACK = 3   # compare vs recent average
WICK_MIN_RATIO = 0.6
TOUCH_CLUSTER_WINDOW = 6  # 6×30m = 3 hours

# ───────────────────────────────────────────────────────────────────────────────
# PAGE & THEME
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔮 SPX Prophet",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
<style>
:root { --brand:#2563eb; --brand-2:#10b981; --surface:#ffffff; --muted:#f8fafc;
        --text:#0f172a; --subtext:#475569; --border:#e2e8f0; --warn:#f59e0b; --danger:#ef4444; }
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
.override-tag { font-size:.75rem; color:#334155; background:#e2e8f0; border:1px solid #cbd5e1; padding:2px 8px; border-radius:999px; display:inline-block; margin-top:6px; }
hr { border-top: 1px solid var(--border); }
.dataframe { background: var(--surface); border-radius: 12px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────────────────────────────────────
# UTILITIES
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
    out, cur = [], start_dt
    while cur <= end_dt:
        out.append(cur); cur += timedelta(minutes=30)
    return out

def is_maintenance(dt: datetime) -> bool:
    return dt.hour == 16  # 4–5 PM CT

def in_weekend_gap(dt: datetime) -> bool:
    wd = dt.weekday()
    if wd == 5: return True
    if wd == 6 and dt.hour < 17: return True
    if wd == 4 and dt.hour >= 17: return True
    return False

def count_effective_blocks(a: datetime, b: datetime) -> float:
    if b <= a: return 0.0
    t = a; blocks = 0
    while t < b:
        t_next = t + timedelta(minutes=30)
        if not is_maintenance(t_next) and not in_weekend_gap(t_next):
            blocks += 1
        t = t_next
    return float(blocks)

def ensure_ohlc_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    for c in ["Open","High","Low","Close"]:
        if c not in df.columns:
            return pd.DataFrame()
    return df

def normalize_to_ct(df: pd.DataFrame, start_d: date, end_d: date) -> pd.DataFrame:
    if df.empty: return df
    df = ensure_ohlc_cols(df)
    if df.empty: return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    df.index = df.index.tz_convert(CT_TZ)
    sdt = fmt_ct(datetime.combine(start_d, time(0,0)))
    edt = fmt_ct(datetime.combine(end_d, time(23,59)))
    return df.loc[sdt:edt]

@st.cache_data(ttl=120, show_spinner=False)
def fetch_intraday(symbol: str, start_d: date, end_d: date, interval: str) -> pd.DataFrame:
    """Unified fetch; for >=30m we use start/end; normalize → CT."""
    try:
        t = yf.Ticker(symbol)
        if interval in ["1m","2m","5m","15m"]:
            days = max(1, min(7, (end_d - start_d).days + 2))
            df = t.history(period=f"{days}d", interval=interval, prepost=True,
                           auto_adjust=False, back_adjust=False)
            df = normalize_to_ct(df, start_d - timedelta(days=1), end_d + timedelta(days=1))
            df = df.loc[fmt_ct(datetime.combine(start_d, time(0,0))):fmt_ct(datetime.combine(end_d, time(23,59)))]
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
    """Safe 30m resample (handles missing Volume)."""
    if min_df.empty or not isinstance(min_df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    df = min_df.sort_index()
    agg = {}
    if "Open"   in df.columns: agg["Open"]   = "first"
    if "High"   in df.columns: agg["High"]   = "max"
    if "Low"    in df.columns: agg["Low"]    = "min"
    if "Close"  in df.columns: agg["Close"]  = "last"
    if "Volume" in df.columns: agg["Volume"] = "sum"
    out = df.resample("30T", label="right", closed="right").agg(agg)
    out = out.dropna(subset=[c for c in ["Open","High","Low","Close"] if c in out.columns], how="any")
    return out

def get_prev_day_anchor(spx_30m: pd.DataFrame, prev_day: date) -> Tuple[Optional[float], Optional[datetime]]:
    if spx_30m.empty: return None, None
    day_start = fmt_ct(datetime.combine(prev_day, time(0,0)))
    day_end   = fmt_ct(datetime.combine(prev_day, time(23,59)))
    d = spx_30m.loc[day_start:day_end]
    if d.empty: return None, None
    target = fmt_ct(datetime.combine(prev_day, time(15,0)))
    if target in d.index:
        return float(d.loc[target,"Close"]), target
    prior = d.loc[:target]
    if not prior.empty:
        return float(prior.iloc[-1]["Close"]), prior.index[-1]
    return None, None

def current_spx_slopes() -> Tuple[float, float]:
    top = float(st.session_state.get("top_slope_per_block", TOP_SLOPE_DEFAULT))
    bottom = float(st.session_state.get("bottom_slope_per_block", BOTTOM_SLOPE_DEFAULT))
    return top, bottom

def project_fan_from_close(close_price: float, anchor_time: datetime, target_day: date) -> pd.DataFrame:
    tslope, bslope = current_spx_slopes()
    rows = []
    for slot in rth_slots_ct(target_day):
        blocks = count_effective_blocks(anchor_time, slot)
        top = close_price + tslope * blocks
        bot = close_price - bslope * blocks
        rows.append({"TimeDT": slot, "Time": slot.strftime("%H:%M"),
                     "Top": round(top,2), "Bottom": round(bot,2), "Fan_Width": round(top-bot,2)})
    return pd.DataFrame(rows)

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    return pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)

# ───────────────────────────────────────────────────────────────────────────────
# BIAS (30m-only)
# ───────────────────────────────────────────────────────────────────────────────
def compute_bias(price: float, top: float, bottom: float, tol_frac: float) -> str:
    if price > top:
        return "UP"
    if price < bottom:
        return "DOWN"
    width = top - bottom
    center = (top + bottom)/2.0
    band = tol_frac * width
    if abs(price - center) <= band:
        return "NO BIAS"
    d_top = top - price
    d_bot = price - bottom
    return "UP" if d_bot < d_top else "DOWN"

# ───────────────────────────────────────────────────────────────────────────────
# ES→SPX OFFSET  (under the hood – not exposed in UI)
# ───────────────────────────────────────────────────────────────────────────────
def _nearest_le_index(idx: pd.DatetimeIndex, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    s = idx[idx <= ts]
    return s[-1] if len(s) else None

def es_spx_offset_at_anchor(prev_day: date, spx_30m: pd.DataFrame) -> Optional[float]:
    spx_anchor_close, spx_anchor_time = get_prev_day_anchor(spx_30m, prev_day)
    if spx_anchor_close is None or spx_anchor_time is None:
        return None

    def try_sym_interval(sym: str, interval: str) -> Optional[float]:
        df = fetch_intraday(sym, prev_day, prev_day, interval)
        if df.empty or "Close" not in df.columns:
            return None
        lo = spx_anchor_time - timedelta(minutes=30)
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

    for interval in ["1m","5m","30m"]:
        off = try_sym_interval("ES=F", interval)
        if off is not None:
            return float(off)

    # Recent median fallback using prior days (5m)
    med_vals = []
    for i in range(1, 6):
        d = prev_day - timedelta(days=i)
        spx_d = fetch_intraday("^GSPC", d, d, "30m")
        if spx_d.empty:
            spx_d = fetch_intraday("SPY", d, d, "30m")
        s_close, s_time = get_prev_day_anchor(spx_d, d)
        if s_close is None or s_time is None:
            continue
        es5 = fetch_intraday("ES=F", d, d, "5m")
        if not es5.empty and "Close" in es5.columns:
            idxd = _nearest_le_index(es5.index, s_time)
            if idxd is not None:
                med_vals.append(float(es5.loc[idxd, "Close"] - s_close))
    if med_vals:
        return float(np.median(med_vals))
    return None

def fetch_overnight(prev_day: date, proj_day: date) -> pd.DataFrame:
    """Always return 30m bars for detection/scoring. Build from 1m/5m if needed."""
    start = fmt_ct(datetime.combine(prev_day, time(17,0)))
    end   = fmt_ct(datetime.combine(proj_day, time(8,30)))
    # Try 30m directly
    es_30 = fetch_intraday("ES=F", prev_day, proj_day, "30m")
    if not es_30.empty:
        return es_30.loc[start:end].copy()
    # Fallback: build 30m by resampling finer data
    es_5 = fetch_intraday("ES=F", prev_day, proj_day, "5m")
    if not es_5.empty:
        return resample_to_30m_ct(es_5.loc[start:end].copy())
    es_1 = fetch_intraday("ES=F", prev_day, proj_day, "1m")
    if not es_1.empty:
        return resample_to_30m_ct(es_1.loc[start:end].copy())
    return pd.DataFrame()

def adjust_to_spx_frame(es_df: pd.DataFrame, offset: float) -> pd.DataFrame:
    df = es_df.copy()
    for col in ["Open","High","Low","Close"]:
        if col in df:
            df[col] = df[col] - offset
    return df

# ───────────────────────────────────────────────────────────────────────────────
# DIRECTION-OF-TRAVEL & EDGE INTERACTIONS (30m ONLY)
# ───────────────────────────────────────────────────────────────────────────────
def prior_state(prior_close: float, prior_top: float, prior_bottom: float) -> str:
    if prior_close > prior_top: return "from_above"
    if prior_close < prior_bottom: return "from_below"
    return "from_inside"

def classify_interaction_30m(prev_close: float, prev_top: float, prev_bot: float,
                             cur_bar: pd.Series, cur_top: float, cur_bot: float) -> Optional[Dict]:
    """
    Implements your rules on 30m bars:
    - From above, breaks below Top + closes inside → bearish continuation to Bottom
    - From below, breaks above Bottom + closes inside → bullish continuation to Top
    (Other cases can be added here if needed.)
    """
    state = prior_state(prev_close, prev_top, prev_bot)

    o = float(cur_bar["Open"]); h = float(cur_bar["High"]); l = float(cur_bar["Low"]); c = float(cur_bar["Close"])
    inside = (cur_bot < c < cur_top)
    # small epsilon to avoid equality issues
    eps = max(0.5, 0.02 * (cur_top - cur_bot))

    if state == "from_above":
        # traded back inside: low < top (allowing eps), and close ended inside
        if (l < cur_top + eps) and inside:
            return {"edge":"Top","case":"FromAbove_ReenterInside","expected":"Bearish continuation to Bottom","direction":"Down"}
    elif state == "from_below":
        if (h > cur_bot - eps) and inside:
            return {"edge":"Bottom","case":"FromBelow_ReenterInside","expected":"Bullish continuation to Top","direction":"Up"}

    return None

# ───────────────────────────────────────────────────────────────────────────────
# SCORING (no EMA cross; liquidity-weighted)
# ───────────────────────────────────────────────────────────────────────────────
def in_window(ts: datetime, window: List[Tuple[int,int]]) -> bool:
    hhmm = ts.hour*60 + ts.minute
    lo = window[0][0]*60 + window[0][1]
    hi = window[1][0]*60 + window[1][1]
    return lo <= hhmm <= hi

def liquidity_bump(ts: datetime) -> int:
    if in_window(ts, SYD_TOK): return W_SYD_TOK
    if in_window(ts, TOK_LON): return W_TOK_LON
    if in_window(ts, PRE_NY):  return W_PRE_NY
    return 0

def atr_percentile(df_30m: pd.DataFrame, idx: pd.Timestamp) -> float:
    upto = df_30m.loc[:idx]
    tr = true_range(upto)
    atr = tr.rolling(ATR_LOOKBACK).mean()
    if atr.notna().sum() < ATR_LOOKBACK:
        return 50.0
    pct = (atr.rank(pct=True).iloc[-1]) * 100.0
    return float(pct)

def range_compression(df_30m: pd.DataFrame, idx: pd.Timestamp) -> bool:
    upto = df_30m.loc[:idx]
    if upto.shape[0] < RANGE_WIN:
        return False
    rng = (upto["High"] - upto["Low"]).rolling(RANGE_WIN).mean()
    return bool((upto["High"].iloc[-1] - upto["Low"].iloc[-1]) <= rng.iloc[-1] * 0.85)

def gap_context(df_30m: pd.DataFrame, idx: pd.Timestamp, expected_dir: str) -> int:
    # compare last close vs open of current bar
    if idx not in df_30m.index:
        return 0
    i = df_30m.index.get_loc(idx)
    if i == 0:
        return 0
    prev_close = float(df_30m["Close"].iloc[i-1])
    cur_open   = float(df_30m["Open"].iloc[i])
    gap = cur_open - prev_close
    # if expected Up and gap up (or small gap down) → boost; mirror for Down
    if expected_dir == "Up":
        return 5 if gap >= 0 else 0
    else:
        return 5 if gap <= 0 else 0

def wick_quality(cur_bar: pd.Series, expected_dir: str) -> bool:
    o,h,l,c = float(cur_bar["Open"]), float(cur_bar["High"]), float(cur_bar["Low"]), float(cur_bar["Close"])
    body = max(1e-9, abs(c-o))
    upper = max(0.0, h - max(o,c))
    lower = max(0.0, min(o,c) - l)
    if expected_dir == "Up":
        return (lower / body) >= WICK_MIN_RATIO
    else:
        return (upper / body) >= WICK_MIN_RATIO

def touch_clustering(touches: List[pd.Timestamp], current_ts: pd.Timestamp) -> bool:
    # If a similar qualified touch occurred recently
    recent = [t for t in touches if 0 < (current_ts - t).total_seconds() <= TOUCH_CLUSTER_WINDOW*1800]
    return len(recent) > 0

def compute_score_components(df_30m: pd.DataFrame, ts: pd.Timestamp,
                             expected_dir: str, touches_recent: List[pd.Timestamp],
                             asia_hit: bool, london_hit: bool) -> Tuple[int, Dict[str,int], int]:
    # Liquidity bump
    lb = liquidity_bump(fmt_ct(ts.to_pydatetime()))
    # Confluence across sessions
    conf = WEIGHTS["confluence"] if (asia_hit and london_hit) else (WEIGHTS["confluence"]//2 if (asia_hit or london_hit) else 0)
    # Structure fit (we already qualified the interaction); give full credit
    struct = WEIGHTS["structure"]
    # Wick quality
    wick = WEIGHTS["wick"] if wick_quality(df_30m.loc[ts], expected_dir) else 0
    # ATR regime fit
    atr_pct = atr_percentile(df_30m, ts)
    if expected_dir == "Up":
        atr = WEIGHTS["atr"] if atr_pct <= 40 else 0
    else:
        atr = WEIGHTS["atr"] if atr_pct >= 60 else 0
    # Range compression
    comp = WEIGHTS["compression"] if range_compression(df_30m, ts) else 0
    # Gap context
    gap = gap_context(df_30m, ts, expected_dir)
    # Touch clustering
    cluster = WEIGHTS["cluster"] if touch_clustering(touches_recent, ts) else 0
    # Volume (small)
    vol = 0
    if "Volume" in df_30m.columns and df_30m["Volume"].notna().any():
        vma = df_30m["Volume"].rolling(20).mean()
        if vma.notna().any() and vma.loc[ts] and df_30m["Volume"].loc[ts] > vma.loc[ts]*1.15:
            vol = WEIGHTS["volume"]

    parts = {"Confluence":conf, "Structure":struct, "Wick":wick, "ATR":atr,
             "Compression":comp, "Gap":gap, "Cluster":cluster, "Volume":vol}
    score = min(100, max(0, lb + sum(parts.values())))
    return score, parts, lb

# ───────────────────────────────────────────────────────────────────────────────
# DASHBOARD BUILDS
# ───────────────────────────────────────────────────────────────────────────────
def build_probability_board(prev_day: date, proj_day: date,
                            anchor_close: float, anchor_time: datetime,
                            tol_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Returns touches_df (qualified 30m interactions), fan_df (RTH projection), offset_used
    """
    # Fan for projection day (RTH view)
    fan_df = project_fan_from_close(anchor_close, anchor_time, proj_day)

    # Previous day 30m for anchor & offset
    spx_prev_30m = fetch_intraday("^GSPC", prev_day, prev_day, "30m")
    if spx_prev_30m.empty:
        spx_prev_30m = fetch_intraday("SPY", prev_day, prev_day, "30m")

    offset = es_spx_offset_at_anchor(prev_day, spx_prev_30m)
    if offset is None:
        return pd.DataFrame(), fan_df, 0.0

    # Overnight bars as 30m
    es_on = fetch_overnight(prev_day, proj_day)
    if es_on.empty:
        return pd.DataFrame(), fan_df, float(offset)

    # Convert to SPX frame
    on_30 = adjust_to_spx_frame(es_on, offset)

    # Build 30m fan across the entire overnight period for comparisons
    # We'll create a time series of (Top/Bottom) for each 30m overnight bar
    rows = []
    for ts, bar in on_30.iterrows():
        blocks = count_effective_blocks(anchor_time, ts)
        tslope, bslope = current_spx_slopes()
        top = anchor_close + tslope * blocks
        bot = anchor_close - bslope * blocks
        rows.append((ts, bar, top, bot))
    # Scan interactions (need prior bar state)
    touches_rows = []
    asia_hits = set()    # timestamps of Asia window touches
    london_hits = set()  # timestamps of Tokyo-London window touches
    qualified_timestamps = []

    for i in range(1, len(rows)):
        prev_ts, prev_bar, prev_top, prev_bot = rows[i-1][0], rows[i-1][1], rows[i-1][2], rows[i-1][3]
        ts, bar, top, bot = rows[i][0], rows[i][1], rows[i][2], rows[i][3]

        # prior state from previous 30m close
        prev_close = float(prev_bar["Close"])
        interaction = classify_interaction_30m(prev_close, prev_top, prev_bot, bar, top, bot)
        if interaction is None:
            continue

        expected_dir = interaction["direction"]
        # Scoring
        touches_recent = qualified_timestamps[-5:]  # look back a handful
        # Session tags
        ts_ct = fmt_ct(ts.to_pydatetime())
        is_asia_overlap = in_window(ts_ct, SYD_TOK)
        is_toklon_overlap = in_window(ts_ct, TOK_LON)

        score, parts, lb = compute_score_components(on_30, ts, expected_dir, touches_recent,
                                                    asia_hit=is_asia_overlap, london_hit=is_toklon_overlap)

        qualified_timestamps.append(ts)

        if is_asia_overlap: asia_hits.add(ts)
        if is_toklon_overlap: london_hits.add(ts)

        price = float(bar["Close"])
        bias = compute_bias(price, top, bot, tol_frac)

        touches_rows.append({
            "TimeDT": ts, "Time": ts.strftime("%H:%M"),
            "Price": round(price,2), "Top": round(top,2), "Bottom": round(bot,2),
            "Bias": bias, "Edge": interaction["edge"], "Case": interaction["case"],
            "Expectation": interaction["expected"], "ExpectedDir": expected_dir,
            "Score": score, "LiquidityBonus": lb,
            "Confluence_w": parts["Confluence"], "Structure_w": parts["Structure"], "Wick_w": parts["Wick"],
            "ATR_w": parts["ATR"], "Compression_w": parts["Compression"], "Gap_w": parts["Gap"],
            "Cluster_w": parts["Cluster"], "Volume_w": parts["Volume"]
        })

    touches_df = pd.DataFrame(touches_rows).sort_values("TimeDT").reset_index(drop=True)
    return touches_df, fan_df, float(offset)

# ───────────────────────────────────────────────────────────────────────────────
# BC FORECAST (two bounces + contracts)
# ───────────────────────────────────────────────────────────────────────────────
def project_line(p1_dt, p1_price, p2_dt, p2_price, proj_day, label_proj: str):
    blocks = count_effective_blocks(p1_dt, p2_dt)
    slope = (p2_price - p1_price) / blocks if blocks > 0 else 0.0
    rows = []
    for slot in rth_slots_ct(proj_day):
        b = count_effective_blocks(p1_dt, slot)
        price = p1_price + slope * b
        rows.append({"Time": slot.strftime("%H:%M"), label_proj: round(price,2)})
    return pd.DataFrame(rows), slope

def expected_exit_time(b1_dt, h1_dt, b2_dt, h2_dt, proj_day):
    d1 = count_effective_blocks(b1_dt, h1_dt)
    d2 = count_effective_blocks(b2_dt, h2_dt)
    durations = [d for d in [d1, d2] if d > 0]
    if not durations:
        return "n/a"
    med_blocks = int(round(np.median(durations)))
    candidate = b2_dt
    for _ in range(med_blocks):
        candidate += timedelta(minutes=30)
        if is_maintenance(candidate) or in_weekend_gap(candidate):
            continue
    for slot in rth_slots_ct(proj_day):
        if slot >= candidate:
            return slot.strftime("%H:%M")
    return "n/a"

# ───────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🔧 Controls")
today_ct = datetime.now(CT_TZ).date()
prev_day = st.sidebar.date_input("Previous Trading Day", value=today_ct - timedelta(days=1))
proj_day = st.sidebar.date_input("Projection Day", value=prev_day + timedelta(days=1))
st.sidebar.caption("Anchor uses the **last SPX bar ≤ 3:00 PM CT** on the previous session (manual override available).")

st.sidebar.markdown("---")
st.sidebar.subheader("✍️ Manual Anchor (optional)")
use_manual_close = st.sidebar.checkbox("Enter 3:00 PM CT Close Manually", value=False)
manual_close_val = st.sidebar.number_input("Manual 3:00 PM Close", value=6400.00, step=0.01, format="%.2f",
                                           disabled=not use_manual_close)

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Advanced (optional)", expanded=False):
    st.caption("Adjust **asymmetric** fan slopes and within-fan neutrality band.")
    enable_slope = st.checkbox("Enable slope override",
                               value=("top_slope_per_block" in st.session_state or "bottom_slope_per_block" in st.session_state))
    top_slope_val = st.number_input("Top slope (+ per 30m)",
                                    value=float(st.session_state.get("top_slope_per_block", TOP_SLOPE_DEFAULT)),
                                    step=0.001, format="%.3f")
    bottom_slope_val = st.number_input("Bottom slope (− per 30m)",
                                       value=float(st.session_state.get("bottom_slope_per_block", BOTTOM_SLOPE_DEFAULT)),
                                       step=0.001, format="%.3f")
    tol_frac = st.slider("Neutrality band (% of fan width)", 0, 40, int(NEUTRAL_BAND_DEFAULT*100), 1) / 100.0

    colA, colB = st.columns(2)
    with colA:
        if st.button("Apply slopes", use_container_width=True, key="apply_slope"):
            if enable_slope:
                st.session_state["top_slope_per_block"] = float(top_slope_val)
                st.session_state["bottom_slope_per_block"] = float(bottom_slope_val)
                st.success(f"Top=+{top_slope_val:.3f}  •  Bottom=−{bottom_slope_val:.3f}")
            else:
                for k in ("top_slope_per_block","bottom_slope_per_block"):
                    st.session_state.pop(k, None)
                st.info("Slope override disabled (using defaults).")
    with colB:
        if st.button("Reset slopes", use_container_width=True, key="reset_slope"):
            for k in ("top_slope_per_block","bottom_slope_per_block"):
                st.session_state.pop(k, None)
            st.success(f"Reset → Top=+{TOP_SLOPE_DEFAULT:.3f} • Bottom=−{BOTTOM_SLOPE_DEFAULT:.3f}")

st.sidebar.markdown("---")
btn_anchor = st.sidebar.button("🔮 Refresh SPX Anchors", type="primary", use_container_width=True, key="btn_anchor")
btn_prob   = st.sidebar.button("🧠 Refresh Probability Board", type="secondary", use_container_width=True, key="btn_prob")

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
# TABS
# ───────────────────────────────────────────────────────────────────────────────
tabAnchors, tabBC, tabProb, tabPlan = st.tabs(
    ["SPX Anchors", "BC Forecast", "Probability Board", "Plan Card"]
)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 1: SPX ANCHORS                                                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tabAnchors:
    st.subheader("SPX Anchors — Entries & Exits from Fan (⭐ 8:30 highlight)")
    if btn_anchor:
        with st.spinner("Building anchor fan & strategy…"):
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
                anchor_close, anchor_time = get_prev_day_anchor(spx_prev, prev_day)
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
                    bar = spx_proj_rth.loc[dt]
                    price = float(bar["Close"])
                    bias = compute_bias(price, top, bottom, tol_frac)
                    note = "—"
                else:
                    price = np.nan; bias = "NO DATA"; note = "Fan only"
                rows.append({
                    "Slot": "⭐ 8:30" if dt.strftime("%H:%M")=="08:30" else "",
                    "Time": dt.strftime("%H:%M"),
                    "Price": (round(price,2) if price==price else np.nan),
                    "Bias": bias, "Top": round(top,2), "Bottom": round(bottom,2),
                    "Fan_Width": round(top-bottom,2),
                    "Note": note
                })
            strat_df = pd.DataFrame(rows)

            st.session_state["anchors"] = {
                "fan_df": fan_df, "strat_df": strat_df,
                "anchor_close": anchor_close, "anchor_time": anchor_time,
                "prev_day": prev_day, "proj_day": proj_day, "tol_frac": tol_frac
            }

    if "anchors" in st.session_state:
        fan_df   = st.session_state["anchors"]["fan_df"]
        strat_df = st.session_state["anchors"]["strat_df"]

        st.markdown("### 🎯 Fan Lines (Top / Bottom @ 30-min)")
        st.dataframe(fan_df[["Time","Top","Bottom","Fan_Width"]], use_container_width=True, hide_index=True)

        st.markdown("### 📋 Strategy Table")
        st.caption("Bias uses within-fan proximity with neutrality band; ⭐ highlights 8:30.")
        st.dataframe(
            strat_df[["Slot","Time","Price","Bias","Top","Bottom","Fan_Width","Note"]],
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Use **Refresh SPX Anchors** in the sidebar.")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 2: BC FORECAST                                                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tabBC:
    st.subheader("BC Forecast — Bounce + Contract Forecast (Asia/Europe → NY 8:30–14:30)")
    st.caption("Requires **exactly 2 SPX bounces** (times + prices). For each contract, provide prices at both bounces and highs after each bounce (with times).")

    # Overnight entry slots (7PM prev → 7AM proj)
    asia_start = fmt_ct(datetime.combine(prev_day, time(19,0)))
    euro_end   = fmt_ct(datetime.combine(proj_day, time(7,0)))
    session_slots = []
    cur = asia_start
    while cur <= euro_end:
        session_slots.append(cur)
        cur += timedelta(minutes=30)
    slot_labels = [dt.strftime("%Y-%m-%d %H:%M") for dt in session_slots]

    with st.form("bc_form_v3", clear_on_submit=False):
        st.markdown("**Underlying bounces (exactly two):**")
        c1, c2 = st.columns(2)
        with c1:
            b1_sel = st.selectbox("Bounce #1 Time (slot)", slot_labels, index=0, key="bc_b1_sel")
            b1_spx = st.number_input("Bounce #1 SPX Price", value=6500.00, step=0.25, format="%.2f", key="bc_b1_spx")
        with c2:
            b2_sel = st.selectbox("Bounce #2 Time (slot)", slot_labels, index=min(6, len(slot_labels)-1), key="bc_b2_sel")
            b2_spx = st.number_input("Bounce #2 SPX Price", value=6512.00, step=0.25, format="%.2f", key="bc_b2_spx")

        st.markdown("---")
        st.markdown("**Contract A (required)**")
        ca_sym = st.text_input("Contract A Label", value="6525c", key="bc_ca_sym")
        ca_b1_price = st.number_input("A: Price at Bounce #1", value=10.00, step=0.05, format="%.2f", key="bc_ca_b1_price")
        ca_b2_price = st.number_input("A: Price at Bounce #2", value=12.50, step=0.05, format="%.2f", key="bc_ca_b2_price")
        ca_h1_time  = st.selectbox("A: High after Bounce #1 — Time", slot_labels, index=min(2, len(slot_labels)-1), key="bc_ca_h1_time")
        ca_h1_price = st.number_input("A: High after Bounce #1 — Price", value=14.00, step=0.05, format="%.2f", key="bc_ca_h1_price")
        ca_h2_time  = st.selectbox("A: High after Bounce #2 — Time", slot_labels, index=min(8, len(slot_labels)-1), key="bc_ca_h2_time")
        ca_h2_price = st.number_input("A: High after Bounce #2 — Price", value=16.00, step=0.05, format="%.2f", key="bc_ca_h2_price")

        st.markdown("---")
        st.markdown("**Contract B (optional)**")
        cb_enable = st.checkbox("Add Contract B", value=False, key="bc_cb_enable")
        if cb_enable:
            cb_sym = st.text_input("Contract B Label", value="6515c", key="bc_cb_sym")
            cb_b1_price = st.number_input("B: Price at Bounce #1", value=9.50, step=0.05, format="%.2f", key="bc_cb_b1_price")
            cb_b2_price = st.number_input("B: Price at Bounce #2", value=11.80, step=0.05, format="%.2f", key="bc_cb_b2_price")
            cb_h1_time  = st.selectbox("B: High after Bounce #1 — Time", slot_labels, index=min(3, len(slot_labels)-1), key="bc_cb_h1_time")
            cb_h1_price = st.number_input("B: High after Bounce #1 — Price", value=13.30, step=0.05, format="%.2f", key="bc_cb_h1_price")
            cb_h2_time  = st.selectbox("B: High after Bounce #2 — Time", slot_labels, index=min(9, len(slot_labels)-1), key="bc_cb_h2_time")
            cb_h2_price = st.number_input("B: High after Bounce #2 — Price", value=15.10, step=0.05, format="%.2f", key="bc_cb_h2_price")

        submit_bc = st.form_submit_button("📈 Project NY Session (8:30–14:30)")

    if submit_bc:
        try:
            b1_dt = fmt_ct(datetime.strptime(st.session_state["bc_b1_sel"], "%Y-%m-%d %H:%M"))
            b2_dt = fmt_ct(datetime.strptime(st.session_state["bc_b2_sel"], "%Y-%m-%d %H:%M"))
            if b2_dt <= b1_dt:
                st.error("Bounce #2 must occur after Bounce #1.")
            else:
                # Underlying slope from bounces
                blocks_u = count_effective_blocks(b1_dt, b2_dt)
                u_slope = (float(b2_spx) - float(b1_spx)) / blocks_u if blocks_u > 0 else 0.0

                # SPX projection from bounces
                rows_u = []
                for slot in rth_slots_ct(proj_day):
                    b = count_effective_blocks(b1_dt, slot)
                    price = float(b1_spx) + u_slope * b
                    rows_u.append({"Time": slot.strftime("%H:%M"), "SPX_Projected": round(price,2)})
                spx_proj_df = pd.DataFrame(rows_u)
                spx_proj_df.insert(0, "Slot", spx_proj_df["Time"].apply(lambda x: "⭐ 8:30" if x=="08:30" else ""))

                # Contract A lines
                ca_entry_df, ca_entry_slope = project_line(b1_dt, float(ca_b1_price), b2_dt, float(ca_b2_price), proj_day, f"{st.session_state['bc_ca_sym']}_Entry")
                ca_h1_dt = fmt_ct(datetime.strptime(st.session_state["bc_ca_h1_time"], "%Y-%m-%d %H:%M"))
                ca_h2_dt = fmt_ct(datetime.strptime(st.session_state["bc_ca_h2_time"], "%Y-%m-%d %H:%M"))
                ca_exit_df, ca_exit_slope = project_line(ca_h1_dt, float(ca_h1_price), ca_h2_dt, float(ca_h2_price), proj_day, f"{st.session_state['bc_ca_sym']}_Exit")

                # Contract B (optional)
                cb_entry_df = cb_exit_df = None
                cb_entry_slope = cb_exit_slope = 0.0
                if cb_enable:
                    cb_entry_df, cb_entry_slope = project_line(b1_dt, float(cb_b1_price), b2_dt, float(cb_b2_price), proj_day, f"{st.session_state['bc_cb_sym']}_Entry")
                    cb_h1_dt = fmt_ct(datetime.strptime(st.session_state["bc_cb_h1_time"], "%Y-%m-%d %H:%M"))
                    cb_h2_dt = fmt_ct(datetime.strptime(st.session_state["bc_cb_h2_time"], "%Y-%m-%d %H:%M"))
                    cb_exit_df, cb_exit_slope = project_line(cb_h1_dt, float(cb_h1_price), cb_h2_dt, float(cb_h2_price), proj_day, f"{st.session_state['bc_cb_sym']}_Exit")

                # Merge & spreads
                out = spx_proj_df.merge(ca_entry_df, on="Time", how="left").merge(ca_exit_df, on="Time", how="left")
                ca = st.session_state["bc_ca_sym"]
                out[f"{ca}_Spread"] = out[f"{ca}_Exit"] - out[f"{ca}_Entry"]
                if cb_enable and cb_entry_df is not None and cb_exit_df is not None:
                    cb = st.session_state["bc_cb_sym"]
                    out = out.merge(cb_entry_df, on="Time", how="left").merge(cb_exit_df, on="Time", how="left")
                    out[f"{cb}_Spread"] = out[f"{cb}_Exit"] - out[f"{cb}_Entry"]

                # Expected exits
                ca_expected = expected_exit_time(b1_dt, ca_h1_dt, b2_dt, ca_h2_dt, proj_day)
                cb_expected = expected_exit_time(b1_dt, cb_h1_dt, b2_dt, cb_h2_dt, proj_day) if cb_enable else None

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                with m1: st.markdown(f"<div class='metric-card'><p class='metric-title'>Underlying Slope /30m</p><div class='metric-value'>📐 {u_slope:+.3f}</div><div class='kicker'>From 2 bounces</div></div>", unsafe_allow_html=True)
                with m2: st.markdown(f"<div class='metric-card'><p class='metric-title'>{ca} Entry Slope /30m</p><div class='metric-value'>📈 {ca_entry_slope:+.3f}</div><div class='kicker'>Exit slope {ca_exit_slope:+.3f} • Expected exit ≈ {ca_expected}</div></div>", unsafe_allow_html=True)
                with m3:
                    if cb_enable:
                        cb = st.session_state["bc_cb_sym"]
                        st.markdown(f"<div class='metric-card'><p class='metric-title'>{cb} Entry Slope /30m</p><div class='metric-value'>📈 {cb_entry_slope:+.3f}</div><div class='kicker'>Exit slope {cb_exit_slope:+.3f} • Expected exit ≈ {cb_expected}</div></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='metric-card'><p class='metric-title'>Contracts</p><div class='metric-value'>1</div></div>", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"<div class='metric-card'><p class='metric-title'>BC Forecast</p><div class='metric-value'>⭐ 8:30 highlighted</div><div class='kicker'>Spread = Exit − Entry</div></div>", unsafe_allow_html=True)

                st.markdown("### 🔮 NY Session Projection (SPX + Contract Entry/Exit Lines)")
                st.dataframe(out, use_container_width=True, hide_index=True)

                st.session_state["bc_result"] = {
                    "table": out, "u_slope": u_slope,
                    "ca_sym": ca, "cb_sym": (st.session_state["bc_cb_sym"] if cb_enable else None),
                    "ca_expected": ca_expected, "cb_expected": cb_expected
                }
        except Exception as e:
            st.error(f"BC Forecast error: {e}")

    if "bc_result" not in st.session_state:
        st.info("Fill the form and click **Project NY Session**.")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 3: PROBABILITY BOARD (30m-only)                                         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tabProb:
    st.subheader("Probability Board — Overnight Edge Confidence (30m-only, liquidity-weighted)")
    show_adv = st.checkbox("Show advanced component columns", value=False, key="prob_show_adv")

    if btn_prob:
        with st.spinner("Evaluating qualified 30m interactions…"):
            # Anchor
            spx_prev = fetch_intraday("^GSPC", prev_day, prev_day, "30m")
            if spx_prev.empty:
                spx_prev = fetch_intraday("SPY", prev_day, prev_day, "30m")
            if spx_prev.empty:
                st.error("Could not fetch previous day SPX 30m data.")
                st.stop()

            if use_manual_close:
                anchor_close = float(manual_close_val)
                anchor_time  = fmt_ct(datetime.combine(prev_day, time(15,0)))
            else:
                anchor_close, anchor_time = get_prev_day_anchor(spx_prev, prev_day)
                if anchor_close is None or anchor_time is None:
                    st.error("Could not find a ≤3:00 PM CT close for the previous day.")
                    st.stop()

            touches_df, fan_df, offset_used = build_probability_board(
                prev_day, proj_day, anchor_close, anchor_time, st.session_state.get("tol_frac", NEUTRAL_BAND_DEFAULT)
            )

            st.session_state["prob_result"] = {
                "touches_df": touches_df, "fan_df": fan_df,
                "offset_used": float(offset_used),
                "anchor_close": anchor_close, "anchor_time": anchor_time,
                "prev_day": prev_day, "proj_day": proj_day,
                "tol_frac": st.session_state.get("tol_frac", NEUTRAL_BAND_DEFAULT)
            }

    if "prob_result" in st.session_state:
        pr = st.session_state["prob_result"]
        touches_df = pr["touches_df"]

        cA, cB, cC = st.columns(3)
        with cA:
            st.markdown(f"<div class='metric-card'><p class='metric-title'>Anchor Close (Prev ≤3:00 PM)</p><div class='metric-value'>💠 {pr['anchor_close']:.2f}</div></div>", unsafe_allow_html=True)
        with cB:
            st.markdown(f"<div class='metric-card'><p class='metric-title'>Qualified Interactions</p><div class='metric-value'>🧩 {len(touches_df)}</div></div>", unsafe_allow_html=True)
        with cC:
            st.markdown(f"<div class='metric-card'><p class='metric-title'>ES→SPX Offset</p><div class='metric-value'>Δ {pr['offset_used']:+.2f}</div><div class='kicker'>Applied under the hood</div></div>", unsafe_allow_html=True)

        st.markdown("### 📡 Overnight 30m Edge Interactions (Scored)")
        if touches_df.empty:
            st.info("No qualified edge interactions detected for this window.")
        else:
            base_cols = ["Time","Price","Top","Bottom","Bias","Edge","Case","Expectation","ExpectedDir","Score","LiquidityBonus"]
            adv_cols  = ["Confluence_w","Structure_w","Wick_w","ATR_w","Compression_w","Gap_w","Cluster_w","Volume_w"]
            cols = base_cols + (adv_cols if show_adv else [])
            st.dataframe(touches_df[cols], use_container_width=True, hide_index=True)
    else:
        st.info("Use **Refresh Probability Board** in the sidebar.")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 4: PLAN CARD                                                            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tabPlan:
    st.subheader("Plan Card — 8:25 Session Prep")

    ready = ("anchors" in st.session_state) and ("prob_result" in st.session_state)
    if not ready:
        st.info("Generate **SPX Anchors** and **Probability Board** first. (BC Forecast optional but recommended.)")
    else:
        an = st.session_state["anchors"]
        pr = st.session_state["prob_result"]
        bc = st.session_state.get("bc_result", None)

        m1, m2, m3, m4 = st.columns(4)
        with m1: st.markdown(f"<div class='metric-card'><p class='metric-title'>Anchor Close</p><div class='metric-value'>💠 {an['anchor_close']:.2f}</div><div class='kicker'>Prev ≤ 3:00 PM CT</div></div>", unsafe_allow_html=True)
        with m2: 
            w830 = an['fan_df'].loc[an['fan_df']['Time']=='08:30','Fan_Width']
            fan_w = float(w830.iloc[0]) if not w830.empty else np.nan
            st.markdown(f"<div class='metric-card'><p class='metric-title'>Fan Width @ 8:30</p><div class='metric-value'>🧭 {fan_w:.2f}</div></div>", unsafe_allow_html=True)
        with m3: st.markdown(f"<div class='metric-card'><p class='metric-title'>Offset</p><div class='metric-value'>Δ {pr['offset_used']:+.2f}</div><div class='kicker'>Overnight basis</div></div>", unsafe_allow_html=True)
        with m4:
            tdf = pr["touches_df"]
            top3 = int(np.mean(sorted(tdf["Score"].tolist(), reverse=True)[:3])) if not tdf.empty else 0
            st.markdown(f"<div class='metric-card'><p class='metric-title'>Readiness</p><div class='metric-value'>🔥 {top3}</div><div class='kicker'>Top-3 score avg</div></div>", unsafe_allow_html=True)

        st.markdown("---")
        colL, colR = st.columns(2)
        with colL:
            st.markdown("### 🎯 Primary Setup (from Anchors)")
            srow = an["strat_df"][an["strat_df"]["Time"]=="08:30"]
            if not srow.empty:
                srow = srow.iloc[0]
                st.write(f"- **8:30 Bias:** {srow['Bias']}")
                st.write(f"- **8:30 Fan:** Top {srow['Top']:.2f} / Bottom {srow['Bottom']:.2f} (width {srow['Fan_Width']:.2f})")
                st.write(f"- **Edge Note:** {srow['Note']}")
            else:
                st.write("- 8:30 row not available; see Strategy Table.")

            st.markdown("### 🧠 Probability Notes (overnight)")
            tdf = pr["touches_df"]
            if not tdf.empty:
                for _, r in tdf.sort_values("Score", ascending=False).head(3).iterrows():
                    st.write(f"- {r['Time']}: **{r['Edge']}** {r['Case']} → *{r['Expectation']}* (Score {r['Score']}, +{r['LiquidityBonus']} liquidity)")
            else:
                st.write("- No scored interactions.")

        with colR:
            st.markdown("### 💼 Trade Plan (guide)")
            if bc and "table" in bc:
                t = bc["table"]
                row830 = t[t["Time"]=="08:30"].head(1)
                if not row830.empty:
                    st.write(f"- **SPX @ 8:30:** {float(row830['SPX_Projected']):.2f}")
                    ca = bc["ca_sym"]; cb = bc.get("cb_sym")
                    if f"{ca}_Entry" in row830:
                        st.write(f"- **{ca} Entry @ 8:30:** {float(row830[f'{ca}_Entry']):.2f}")
                    if f"{ca}_Exit" in row830:
                        st.write(f"- **{ca} ExitRef @ 8:30:** {float(row830[f'{ca}_Exit']):.2f}")
                    if cb:
                        if f"{cb}_Entry" in row830:
                            st.write(f"- **{cb} Entry @ 8:30:** {float(row830[f'{cb}_Entry']):.2f}")
                        if f"{cb}_Exit" in row830:
                            st.write(f"- **{cb} ExitRef @ 8:30:** {float(row830[f'{cb}_Exit']):.2f}")
                    # Expected exit chips
                    if bc.get("ca_expected") and bc["ca_expected"] != "n/a":
                        st.write(f"- **{ca} expected exit ≈ {bc['ca_expected']}**")
                    if cb and bc.get("cb_expected") and bc["cb_expected"] != "n/a":
                        st.write(f"- **{cb} expected exit ≈ {bc['cb_expected']}**")
            else:
                st.write("- Use BC Forecast to pre-compute contract lines and exits.")

# ───────────────────────────────────────────────────────────────────────────────
# FOOTER
# ───────────────────────────────────────────────────────────────────────────────
st.markdown("---")
colF1, colF2 = st.columns([1,2])
with colF1:
    if st.button("🔌 Test Data Connection"):
        td = fetch_intraday("^GSPC", today_ct - timedelta(days=3), today_ct, "30m")
        if td.empty:
            td = fetch_intraday("SPY", today_ct - timedelta(days=3), today_ct, "30m")
        if not td.empty:
            st.success(f"OK — received {len(td)} bars (30m).")
        else:
            st.error("Data fetch failed — try different dates.")
with colF2:
    st.caption("SPX Prophet • 30m-only interactions • Fan Top +0.312 / Bottom −0.25 per 30m • Liquidity-weighted Probability Board • ⭐ 8:30 focus")