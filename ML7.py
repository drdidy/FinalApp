# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET - PART 1: ES/SPX DATA FOUNDATION & OFFSET 📊  (fully functional)
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────────────────────────────────────
# 🌎 CORE CONFIG
# ───────────────────────────────────────────────────────────────────────────────
CT_TZ = pytz.timezone('America/Chicago')
UTC_TZ = pytz.UTC

RTH_START = "08:30"  # CT
RTH_END   = "15:00"  # CT (use 15:00 for overlap calc; your UI can show 14:30)
ES_SYMBOL   = "ES=F"     # CME E-mini S&P 500
SPX_SYMBOL  = "^GSPC"    # S&P 500 index (fallback to SPY)

# ───────────────────────────────────────────────────────────────────────────────
# 🎨 STREAMLIT PAGE
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔮 SPX Prophet — Part 1 (ES/SPX Offset)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0b1020 100%); color: #eef2ff; }
    .metric-container { 
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.05));
        padding: 1.0rem; border-radius: 14px; border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 6px 22px rgba(0,0,0,0.25);
    }
    .metric-container h3 { margin: 0 0 .4rem 0; font-weight: 600; }
    .stDataFrame { background: rgba(255,255,255,0.95); border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# 🔁 HELPERS
# ───────────────────────────────────────────────────────────────────────────────
def _infer_period_and_interval(days: int) -> Tuple[str, str]:
    """
    For intraday download with yfinance, pick valid (period, interval).
    30m supports up to ~60d reliably.
    """
    if days <= 7:
        return "7d", "30m"
    elif days <= 30:
        return "30d", "30m"
    else:
        return "60d", "30m"  # longest typical for 30m

def _to_ct_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if df.index.tz is None:
        # sometimes naive intraday comes back; assume UTC then convert
        df.index = df.index.tz_localize("UTC")
    return df.tz_convert(CT_TZ)

def validate_ohlc_data(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    for col in ['Open','High','Low','Close']:
        if col not in df.columns:
            return False
    bad = (
        (df['High'] < df['Low']) |
        (df['High'] < df['Open']) | (df['High'] < df['Close']) |
        (df['Low']  > df['Open']) | (df['Low']  > df['Close']) |
        (df['Close'] <= 0) | (df['High'] <= 0)
    )
    return not bad.any()

def between_ct(df: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    if df.empty:
        return df
    # Data should be CT indexed; between_time uses local time of index
    return df.between_time(start_hhmm, end_hhmm)

# ───────────────────────────────────────────────────────────────────────────────
# 📥 ROBUST FETCHERS (INTRADAY)
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_intraday(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Robust intraday fetch using period/interval. Retries once.
    For ^GSPC (index), if empty, fall back to SPY.
    """
    days = (end_date - start_date).days + 1
    period, interval = _infer_period_and_interval(days)

    # Indexes have no 'prepost' concept; let yfinance decide (prepost not used with download)
    def _dl(sym: str) -> pd.DataFrame:
        df = yf.download(
            tickers=sym,
            period=period,
            interval=interval,
            auto_adjust=False,
            back_adjust=False,
            threads=False,
            progress=False,
            timeout=25,
        )
        # Normalize columns if MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df

    df = _dl(symbol)
    if df.empty:
        df = _dl(symbol)  # retry once

    if df.empty and symbol.upper() == "^GSPC":
        df = _dl("SPY")  # fallback for intraday if ^GSPC is flaky

    if df.empty:
        return df

    df = _to_ct_index(df)

    # Clip to requested calendar dates after download
    start_dt = CT_TZ.localize(datetime.combine(start_date, time(0, 0)))
    end_dt   = CT_TZ.localize(datetime.combine(end_date, time(23, 59)))
    df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]

    # Basic sanity
    if not validate_ohlc_data(df):
        # Still return; caller can decide to warn
        pass
    return df

# ───────────────────────────────────────────────────────────────────────────────
# 📏 OFFSET CALCULATION
# ───────────────────────────────────────────────────────────────────────────────
def last_close_in_overlap(es_df: pd.DataFrame, spx_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Use CT RTH overlap (08:30–15:00). Take last available bar within that window.
    If either side is empty in that window, fall back to last available Close.
    """
    es_rth  = between_ct(es_df,  RTH_START, RTH_END)
    spx_rth = between_ct(spx_df, RTH_START, RTH_END)

    if not es_rth.empty and not spx_rth.empty:
        return float(es_rth['Close'].iloc[-1]), float(spx_rth['Close'].iloc[-1])

    es_last  = float(es_df['Close'].iloc[-1])  if not es_df.empty  else None
    spx_last = float(spx_df['Close'].iloc[-1]) if not spx_df.empty else None
    return es_last, spx_last

def calc_es_to_spx_offset(es_df: pd.DataFrame, spx_df: pd.DataFrame) -> Optional[float]:
    """
    ES→SPX offset = SPX_close - ES_close (positive if SPX > ES).
    """
    es_close, spx_close = last_close_in_overlap(es_df, spx_df)
    if es_close is None or spx_close is None:
        return None
    return round(spx_close - es_close, 1)

# ───────────────────────────────────────────────────────────────────────────────
# 🧭 TIME / MARKET STATUS
# ───────────────────────────────────────────────────────────────────────────────
def market_status_ct(now_ct: datetime) -> Tuple[str, str]:
    is_weekday = now_ct.weekday() < 5
    open_ct  = now_ct.replace(hour=8,  minute=30, second=0, microsecond=0)
    close_ct = now_ct.replace(hour=15, minute=0,  second=0, microsecond=0)
    within   = open_ct <= now_ct <= close_ct
    if not is_weekday:
        return "WEEKEND", "#ff6b6b"
    return ("MARKET OPEN", "#00ff88") if within else ("MARKET CLOSED", "#ffbb33")

# ─────────────
