# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET — PART 1: ES/SPX DATA FOUNDATION & OFFSET (auto-run, robust)
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import yfinance as yf
import pytz
from datetime import datetime, date, time, timedelta
from typing import Tuple, Optional

# ───────────────────────────────────────────────────────────────────────────────
# 🌎 CORE CONFIG
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SPX Prophet — Part 1", page_icon="📈", layout="wide")

CT_TZ = pytz.timezone("America/Chicago")
RTH_START = "08:30"  # CT
RTH_END   = "15:00"  # CT (use 15:00 for overlap calc)
ES_SYMBOL   = "ES=F"     # E-mini S&P 500
SPX_SYMBOL  = "^GSPC"    # Index (intraday flaky) → fallback to SPY
SPX_FALLBACK = "SPY"

# ───────────────────────────────────────────────────────────────────────────────
# 🔁 HELPERS
# ───────────────────────────────────────────────────────────────────────────────
def _infer_period_interval(days: int) -> Tuple[str, str]:
    if days <= 7:   return "7d", "30m"
    if days <= 30:  return "30d", "30m"
    return "60d", "30m"

def _to_ct(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # yfinance intraday is tz-aware UTC; occasionally naive
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.tz_convert(CT_TZ)

def _between_ct(df: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df.between_time(start_hhmm, end_hhmm)

def _validate_ohlc(df: pd.DataFrame) -> bool:
    if df.empty: return False
    need = {"Open","High","Low","Close"}
    if not need.issubset(df.columns): return False
    bad = (
        (df["High"] < df["Low"]) |
        (df["High"] < df["Open"]) | (df["High"] < df["Close"]) |
        (df["Low"]  > df["Open"]) | (df["Low"]  > df["Close"]) |
        (df["Close"] <= 0) | (df["High"] <= 0)
    )
    return not bad.any()

# ───────────────────────────────────────────────────────────────────────────────
# 📥 ROBUST FETCHERS (INTRADAY)
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_intraday(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    days = (end_date - start_date).days + 1
    period, interval = _infer_period_interval(days)

    def _dl(sym: str) -> pd.DataFrame:
        df = yf.download(
            tickers=sym,
            period=period,
            interval=interval,
            auto_adjust=False,
            back_adjust=False,
            threads=False,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df

    df = _dl(symbol)
    if df.empty:
        df = _dl(symbol)  # retry once

    # Fallback for ^GSPC intraday
    if df.empty and symbol.upper() == "^GSPC":
        df = _dl(SPX_FALLBACK)

    if df.empty:
        return df

    df = _to_ct(df)
    start_dt = CT_TZ.localize(datetime.combine(start_date, time(0, 0)))
    end_dt   = CT_TZ.localize(datetime.combine(end_date, time(23, 59)))
    df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
    return df

# ───────────────────────────────────────────────────────────────────────────────
# 📏 OFFSET CALCULATION
# ───────────────────────────────────────────────────────────────────────────────
def last_close_in_overlap(es_df: pd.DataFrame, spx_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    es_rth  = _between_ct(es_df,  RTH_START, RTH_END)
    spx_rth = _between_ct(spx_df, RTH_START, RTH_END)

    if not es_rth.empty and not spx_rth.empty:
        return float(es_rth["Close"].iloc[-1]), float(spx_rth["Close"].iloc[-1])

    es_last  = float(es_df["Close"].iloc[-1])  if not es_df.empty  else None
    spx_last = float(spx_df["Close"].iloc[-1]) if not spx_df.empty else None
    return es_last, spx_last

def calc_es_to_spx_offset(es_df: pd.DataFrame, spx_df: pd.DataFrame) -> Optional[float]:
    es_close, spx_close = last_close_in_overlap(es_df, spx_df)
    if es_close is None or spx_close is None:
        return None
    return round(spx_close - es_close, 1)  # positive if SPX > ES

# ───────────────────────────────────────────────────────────────────────────────
# 🧭 UI — CONTROLS
# ───────────────────────────────────────────────────────────────────────────────
st.title("🔮 SPX Prophet — Part 1: ES/SPX Data & Offset")

with st.sidebar:
    st.header("Settings")
    default_end = datetime.now(CT_TZ).date()
    default_start = default_end - timedelta(days=5)
    start_date = st.date_input("Start date (CT)", default_start)
    end_date   = st.date_input("End date (CT)", default_end)
    if start_date > end_date:
        st.error("Start date must be ≤ end date.")

    es_symbol  = st.text_input("ES symbol", ES_SYMBOL)
    spx_symbol = st.text_input("SPX symbol (uses SPY if ^GSPC empty)", SPX_SYMBOL)
    autorun    = st.checkbox("Auto-run on load", value=True)
    run        = st.button("Fetch & Compute Offset", type="primary")

# ───────────────────────────────────────────────────────────────────────────────
# 📟 STATUS STRIP
# ───────────────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    now_ct = datetime.now(CT_TZ)
    st.metric("Current Time (CT)", now_ct.strftime("%H:%M:%S"), now_ct.strftime("%a %b %d"))

with col2:
    is_weekday = now_ct.weekday() < 5
    open_ct  = now_ct.replace(hour=8,  minute=30, second=0, microsecond=0)
    close_ct = now_ct.replace(hour=15, minute=0,  second=0, microsecond=0)
    status = "MARKET OPEN" if (is_weekday and open_ct <= now_ct <= close_ct) else ("WEEKEND" if not is_weekday else "MARKET CLOSED")
    st.metric("Market Status", status, "RTH 08:30–15:00 CT")

offset_placeholder = st.empty()

# ───────────────────────────────────────────────────────────────────────────────
# 🚀 FETCH & SHOW (auto-run or button)
# ───────────────────────────────────────────────────────────────────────────────
should_run = autorun or run

if should_run:
    with st.spinner("Fetching intraday data…"):
        es_df  = fetch_intraday(es_symbol.strip(),  start_date, end_date)
        spx_df = fetch_intraday(spx_symbol.strip(), start_date, end_date)

    cols = st.columns(2)
    with cols[0]:
        if es_df.empty:
            st.error(f"❌ No ES data for {es_symbol} in selected range.")
        else:
            st.success(f"✅ ES bars: {len(es_df)} | Index tz: {es_df.index.tz}")
            st.dataframe(_between_ct(es_df, RTH_START, RTH_END).tail(10))

    with cols[1]:
        if spx_df.empty:
            st.error(f"❌ No SPX data for {spx_symbol} (SPY used if you entered ^GSPC and Yahoo returned none).")
        else:
            # Display name (show when fallback happened)
            label = spx_symbol
            if spx_symbol.strip().upper() == "^GSPC" and "^GSPC" not in getattr(spx_df, "attrs", {}):
                label = f"{spx_symbol} / {SPX_FALLBACK}"
            st.success(f"✅ {label} bars: {len(spx_df)} | Index tz: {spx_df.index.tz}")
            st.dataframe(_between_ct(spx_df, RTH_START, RTH_END).tail(10))

    # Compute offset
    if not es_df.empty and not spx_df.empty:
        offset = calc_es_to_spx_offset(es_df, spx_df)
        if offset is not None:
            offset_placeholder.metric("🔄 ES→SPX Offset (last in overlap)", f"{offset:+.1f}")
        else:
            offset_placeholder.warning("Offset unavailable (no overlapping bars).")
    else:
        offset_placeholder.info("Awaiting valid ES & SPX data to compute offset.")

else:
    st.info("Enable **Auto-run on load** or click **Fetch & Compute Offset**.")
