# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - CLEAN VERSION
# PART 1: FOUNDATION & STRATEGY (NO EXTERNAL DEPENDENCIES)
# ═══════════════════════════════════════════════════════════════════════════════

import json
import base64
from datetime import datetime, date, time, timedelta
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SPXForecastStrategy:
    """Core SPX forecasting strategy based on time-block calculations and slope projections."""
    
    def __init__(self):
        # Default slopes for each asset
        self.base_slopes = {
            "SPX_HIGH": -0.2792, "SPX_CLOSE": -0.2792, "SPX_LOW": -0.2792,
            "TSLA": -0.1508, "NVDA": -0.0485, "AAPL": -0.0750,
            "MSFT": -0.17, "AMZN": -0.03, "GOOGL": -0.07,
            "META": -0.035, "NFLX": -0.23,
        }
        
        # Current slopes (can be modified)
        self.slopes = self.base_slopes.copy()
        
        # SPX operates 8:30-14:30, others 7:30-14:30
        self.spx_start_time = time(8, 30)
        self.general_start_time = time(7, 30)
    
    def generate_time_slots(self, start_time: time = None) -> List[str]:
        """Generate 30-minute time slots for forecasting."""
        if start_time is None:
            start_time = self.general_start_time
            
        base = datetime(2025, 1, 1, start_time.hour, start_time.minute)
        slots = []
        
        # Calculate number of slots (15 total, minus 2 if SPX starts at 8:30)
        num_slots = 15 - (2 if start_time.hour == 8 and start_time.minute == 30 else 0)
        
        for i in range(num_slots):
            slot_time = base + timedelta(minutes=30 * i)
            slots.append(slot_time.strftime("%H:%M"))
            
        return slots
    
    def calculate_spx_blocks(self, anchor_time: datetime, target_time: datetime) -> int:
        """Calculate time blocks for SPX (skips 4:00 PM hour)."""
        blocks = 0
        current = anchor_time
        
        while current < target_time:
            if current.hour != 16:  # Skip 4:00 PM hour
                blocks += 1
            current += timedelta(minutes=30)
            
        return blocks
    
    def calculate_stock_blocks(self, anchor_time: datetime, target_time: datetime) -> int:
        """Calculate time blocks for regular stocks (simple 30-min intervals)."""
        time_diff = target_time - anchor_time
        return max(0, int(time_diff.total_seconds() // 1800))  # 1800 seconds = 30 minutes
    
    def project_price(self, base_price: float, slope: float, blocks: int) -> float:
        """Core price projection formula."""
        return base_price + (slope * blocks)
    
    def generate_forecast_table(self, base_price: float, slope: float, anchor_time: datetime, 
                              forecast_date: date, is_spx: bool = True, fan_mode: bool = False) -> pd.DataFrame:
        """Generate a forecast table for given parameters."""
        
        # Get appropriate time slots
        start_time = self.spx_start_time if is_spx else self.general_start_time
        slots = self.generate_time_slots(start_time)
        
        rows = []
        for slot in slots:
            hour, minute = map(int, slot.split(":"))
            target_time = datetime.combine(forecast_date, time(hour, minute))
            
            # Calculate blocks based on asset type
            if is_spx:
                blocks = self.calculate_spx_blocks(anchor_time, target_time)
            else:
                blocks = self.calculate_stock_blocks(anchor_time, target_time)
            
            # Generate projection
            if fan_mode:
                # Fan mode: entry and exit prices
                entry_price = self.project_price(base_price, slope, blocks)
                exit_price = self.project_price(base_price, -slope, blocks)
                rows.append({
                    "Time": slot,
                    "Entry": round(entry_price, 2),
                    "Exit": round(exit_price, 2)
                })
            else:
                # Regular mode: single projected price
                projected_price = self.project_price(base_price, slope, blocks)
                rows.append({
                    "Time": slot,
                    "Projected": round(projected_price, 2)
                })
        
        return pd.DataFrame(rows)
    
    def spx_forecast(self, high_price: float, high_time: time, close_price: float, close_time: time,
                    low_price: float, low_time: time, forecast_date: date) -> Dict[str, pd.DataFrame]:
        """Generate SPX forecast with all three anchor points."""
        
        # Create anchor datetimes (previous day)
        prev_day = forecast_date - timedelta(days=1)
        high_anchor = datetime.combine(prev_day, high_time)
        close_anchor = datetime.combine(prev_day, close_time)
        low_anchor = datetime.combine(prev_day, low_time)
        
        forecasts = {}
        
        # Generate forecasts for each anchor
        for anchor_type, price, anchor_time, slope_key in [
            ("High", high_price, high_anchor, "SPX_HIGH"),
            ("Close", close_price, close_anchor, "SPX_CLOSE"),
            ("Low", low_price, low_anchor, "SPX_LOW")
        ]:
            forecasts[anchor_type] = self.generate_forecast_table(
                price, self.slopes[slope_key], anchor_time, forecast_date, 
                is_spx=True, fan_mode=True
            )
        
        return forecasts
    
    def contract_line_forecast(self, low1_price: float, low1_time: time, low2_price: float, 
                             low2_time: time, forecast_date: date) -> Tuple[pd.DataFrame, Dict]:
        """Generate contract line forecast using two-point interpolation."""
        
        # Create anchor datetime
        anchor_time = datetime.combine(forecast_date, low1_time)
        
        # Calculate slope between the two points
        low2_datetime = datetime.combine(forecast_date, low2_time)
        blocks_between = self.calculate_spx_blocks(anchor_time, low2_datetime)
        
        if blocks_between == 0:
            slope = 0
        else:
            slope = (low2_price - low1_price) / blocks_between
        
        # Generate forecast table
        forecast_table = self.generate_forecast_table(
            low1_price, slope, anchor_time, forecast_date, 
            is_spx=False, fan_mode=False
        )
        
        # Return table and contract parameters for lookup
        contract_params = {
            "anchor_time": anchor_time,
            "slope": slope,
            "base_price": low1_price
        }
        
        return forecast_table, contract_params
    
    def lookup_contract_price(self, contract_params: Dict, lookup_time: time, forecast_date: date) -> float:
        """Look up contract price at any specific time."""
        if not contract_params:
            return 0.0
            
        target_time = datetime.combine(forecast_date, lookup_time)
        blocks = self.calculate_spx_blocks(contract_params["anchor_time"], target_time)
        
        return self.project_price(contract_params["base_price"], contract_params["slope"], blocks)
    
    def stock_forecast(self, ticker: str, low_price: float, low_time: time,
                      high_price: float, high_time: time, forecast_date: date) -> Dict[str, pd.DataFrame]:
        """Generate stock forecast with low and high anchors."""
        
        if ticker not in self.slopes:
            raise ValueError(f"Unknown ticker: {ticker}")
        
        # Create anchor datetimes
        low_anchor = datetime.combine(forecast_date, low_time)
        high_anchor = datetime.combine(forecast_date, high_time)
        
        forecasts = {
            "Low": self.generate_forecast_table(
                low_price, self.slopes[ticker], low_anchor, forecast_date,
                is_spx=False, fan_mode=True
            ),
            "High": self.generate_forecast_table(
                high_price, self.slopes[ticker], high_anchor, forecast_date,
                is_spx=False, fan_mode=True
            )
        }
        
        return forecasts
    
    def update_slope(self, asset: str, new_slope: float):
        """Update slope for a specific asset."""
        if asset in self.slopes:
            self.slopes[asset] = new_slope
    
    def reset_slopes(self):
        """Reset all slopes to default values."""
        self.slopes = self.base_slopes.copy()
    
    def get_available_tickers(self) -> List[str]:
        """Get list of available stock tickers."""
        return [k for k in self.slopes.keys() if not k.startswith("SPX_")]

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION & INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SPX Prophet",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "SPX Prophet - Advanced Financial Forecasting Tool"}
)

# Initialize strategy
@st.cache_resource
def get_strategy():
    return SPXForecastStrategy()

strategy = get_strategy()

# Session state initialization
if 'current_forecasts' not in st.session_state:
    st.session_state.current_forecasts = {}
if 'contract_params' not in st.session_state:
    st.session_state.contract_params = {}
if 'contract_table' not in st.session_state:
    st.session_state.contract_table = pd.DataFrame()
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "SPX"

# ═══════════════════════════════════════════════════════════════════════════════
# STYLING & CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.main {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

.hero-container {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    backdrop-filter: blur(10px);
    padding: 2rem;
    margin: 1rem 0;
    text-align: center;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
}

.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    font-size: 1.2rem;
    color: rgba(255, 255, 255, 0.7);
    margin-bottom: 1rem;
}

.metric-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
    height: 100%;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15);
    border-color: rgba(255, 255, 255, 0.2);
}

.input-container {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    margin: 1rem 0;
}

/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Animation */
.fade-in {
    animation: fadeIn 0.6s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_metric_card(icon: str, title: str, value: str, subtitle: str = ""):
    """Create a beautiful metric card"""
    return f"""
    <div class="metric-card fade-in">
        <div style="font-size: 2.5rem; margin-bottom: 1rem;">{icon}</div>
        <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">{value}</div>
        <div style="font-size: 0.9rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.5px;">{title}</div>
        {f'<div style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.6;">{subtitle}</div>' if subtitle else ''}
    </div>
    """

def display_forecast_table(df: pd.DataFrame, title: str):
    """Display a forecast table with nice formatting"""
    st.subheader(title)
    
    # Format the dataframe
    if not df.empty:
        display_df = df.copy()
        
        # Format price columns
        for col in ['Entry', 'Exit', 'Projected']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {title} Data",
            data=csv,
            file_name=f"{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No data to display")

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER SECTION
# ═══════════════════════════════════════════════════════════════════════════════

def render_hero_section():
    """Render the hero section"""
    current_time = datetime.now().strftime("%H:%M:%S")
    market_status = "🟢 Market Open" if 9 <= datetime.now().hour <= 16 else "🔴 Market Closed"
    
    st.markdown(f"""
    <div class="hero-container">
        <h1 class="hero-title">🔮 SPX Prophet</h1>
        <p class="hero-subtitle">Advanced Financial Forecasting with Time-Based Projections</p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
            <div style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 8px;">
                <strong>⏰ {current_time}</strong>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 8px;">
                <strong>{market_status}</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

render_hero_section()

# Success message
st.sidebar.success("✅ SPX Prophet Loaded Successfully!")





# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: NAVIGATION & SPX FORECASTING PAGE
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🧭 Navigation")

# Page selection
page_options = {
    "SPX": "🧭 SPX Forecast",
    "Contract": "📈 Contract Line", 
    "TSLA": "🚗 Tesla",
    "NVDA": "🧠 NVIDIA",
    "AAPL": "🍎 Apple",
    "MSFT": "🪟 Microsoft",
    "AMZN": "📦 Amazon",
    "GOOGL": "🔍 Google",
    "META": "📘 Meta",
    "NFLX": "📺 Netflix"
}

selected_page = st.sidebar.selectbox(
    "Select Page",
    options=list(page_options.keys()),
    format_func=lambda x: page_options[x],
    index=0
)

st.session_state.selected_page = selected_page

# ═══════════════════════════════════════════════════════════════════════════════
# SLOPE MANAGEMENT IN SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 📐 Slope Management")

with st.sidebar.expander("🔧 Adjust Slopes", expanded=False):
    st.markdown("**SPX Slopes:**")
    
    # SPX slopes
    for spx_key in ["SPX_HIGH", "SPX_CLOSE", "SPX_LOW"]:
        new_slope = st.slider(
            spx_key.replace("SPX_", ""),
            min_value=-1.0,
            max_value=1.0,
            value=strategy.slopes[spx_key],
            step=0.0001,
            format="%.4f",
            key=f"slope_{spx_key}"
        )
        strategy.slopes[spx_key] = new_slope
    
    st.markdown("**Stock Slopes:**")
    
    # Stock slopes
    for ticker in strategy.get_available_tickers():
        new_slope = st.slider(
            ticker,
            min_value=-1.0,
            max_value=1.0,
            value=strategy.slopes[ticker],
            step=0.0001,
            format="%.4f",
            key=f"slope_{ticker}"
        )
        strategy.slopes[ticker] = new_slope
    
    if st.button("🔄 Reset All Slopes", key="reset_all_slopes"):
        strategy.reset_slopes()
        st.success("✅ All slopes reset to defaults")
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# SPX FORECASTING PAGE
# ═══════════════════════════════════════════════════════════════════════════════

if selected_page == "SPX":
    st.markdown("## 🧭 SPX Forecasting Dashboard")
    st.caption("Generate forecasts using High, Close, and Low anchor points from the previous day")
    
    # Input section
    st.markdown("### 📊 Forecast Parameters")
    
    # Date selection
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_date = st.date_input(
            "📅 Forecast Date",
            value=date.today() + timedelta(days=1),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=30),
            key="spx_forecast_date"
        )
    
    with col2:
        weekday_name = forecast_date.strftime("%A")
        st.info(f"📅 **{weekday_name}** - {forecast_date.strftime('%B %d, %Y')}")
    
    st.markdown("---")
    
    # Anchor points input
    st.markdown("### 🎯 Previous Day Anchor Points")
    
    # Create three columns for anchors
    high_col, close_col, low_col = st.columns(3)
    
    with high_col:
        st.markdown("**🟢 High Anchor**")
        high_price = st.number_input(
            "High Price",
            value=6185.8,
            min_value=0.0,
            step=0.1,
            format="%.2f",
            key="spx_high_price"
        )
        high_time = st.time_input(
            "High Time",
            value=time(11, 30),
            key="spx_high_time"
        )
    
    with close_col:
        st.markdown("**🔵 Close Anchor**")
        close_price = st.number_input(
            "Close Price",
            value=6170.2,
            min_value=0.0,
            step=0.1,
            format="%.2f",
            key="spx_close_price"
        )
        close_time = st.time_input(
            "Close Time",
            value=time(15, 0),
            key="spx_close_time"
        )
    
    with low_col:
        st.markdown("**🔴 Low Anchor**")
        low_price = st.number_input(
            "Low Price",
            value=6130.4,
            min_value=0.0,
            step=0.1,
            format="%.2f",
            key="spx_low_price"
        )
        low_time = st.time_input(
            "Low Time",
            value=time(13, 30),
            key="spx_low_time"
        )
    
    # Quick metrics
    if high_price > 0 and low_price > 0 and close_price > 0:
        price_range = high_price - low_price
        range_percentage = (price_range / close_price) * 100
        
        st.markdown("### 📈 Quick Metrics")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown(
                create_metric_card("📊", "Range", f"${price_range:.2f}", f"{range_percentage:.1f}% of close"),
                unsafe_allow_html=True
            )
        
        with metric_col2:
            volatility = "Low" if range_percentage < 2 else "High" if range_percentage > 5 else "Normal"
            st.markdown(
                create_metric_card("⚡", "Volatility", volatility, f"{range_percentage:.1f}% range"),
                unsafe_allow_html=True
            )
        
        with metric_col3:
            midpoint = (high_price + low_price) / 2
            close_position = "Above Mid" if close_price > midpoint else "Below Mid"
            st.markdown(
                create_metric_card("🎯", "Close Position", close_position, f"Mid: ${midpoint:.2f}"),
                unsafe_allow_html=True
            )
    
    st.markdown("---")
    
    # Generate forecast button
    if st.button("🚀 Generate SPX Forecast", key="generate_spx_forecast", type="primary"):
        with st.spinner("🔮 Generating SPX forecasts..."):
            try:
                # Generate forecasts
                forecasts = strategy.spx_forecast(
                    high_price, high_time, close_price, close_time,
                    low_price, low_time, forecast_date
                )
                
                # Store in session state
                st.session_state.current_forecasts = forecasts
                st.session_state.forecast_metadata = {
                    "date": forecast_date,
                    "high_price": high_price,
                    "close_price": close_price,
                    "low_price": low_price,
                    "generated_at": datetime.now()
                }
                
                st.success("✅ SPX forecasts generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error generating forecasts: {str(e)}")
    
    # Display results
    if st.session_state.current_forecasts:
        st.markdown("## 📊 SPX Forecast Results")
        
        forecasts = st.session_state.current_forecasts
        metadata = st.session_state.get('forecast_metadata', {})
        
        # Summary metrics
        total_forecasts = len(forecasts)
        forecast_date_str = str(metadata.get('date', 'N/A'))
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown(
                create_metric_card("📅", "Forecast Date", forecast_date_str, "Target date"),
                unsafe_allow_html=True
            )
        
        with summary_col2:
            st.markdown(
                create_metric_card("🔢", "Anchor Points", str(total_forecasts), "Generated forecasts"),
                unsafe_allow_html=True
            )
        
        # Create tabs for each anchor
        if len(forecasts) >= 3:
            high_tab, close_tab, low_tab = st.tabs(["🟢 High Anchor", "🔵 Close Anchor", "🔴 Low Anchor"])
            
            with high_tab:
                if "High" in forecasts:
                    display_forecast_table(forecasts["High"], "High Anchor Forecast")
            
            with close_tab:
                if "Close" in forecasts:
                    display_forecast_table(forecasts["Close"], "Close Anchor Forecast")
            
            with low_tab:
                if "Low" in forecasts:
                    display_forecast_table(forecasts["Low"], "Low Anchor Forecast")
        
        # Export all data
        st.markdown("### 📤 Export Options")
        
        if st.button("📥 Download All SPX Data", key="download_all_spx"):
            # Combine all forecasts into one file
            all_data = []
            for anchor_name, forecast_df in forecasts.items():
                df_copy = forecast_df.copy()
                df_copy['Anchor'] = anchor_name
                all_data.append(df_copy)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                csv = combined_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Combined SPX Forecasts",
                    data=csv,
                    file_name=f"spx_all_forecasts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_combined_spx"
                )
    
    else:
        st.info("👆 Enter your anchor prices and times, then click 'Generate SPX Forecast' to see results.")







# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: CONTRACT LINE PAGE
# ═══════════════════════════════════════════════════════════════════════════════

if selected_page == "Contract":
    st.markdown("## 📈 Contract Line Forecasting")
    st.caption("Create forecasts using two-point interpolation between contract price levels")
    
    # Contract line explanation
    with st.expander("ℹ️ How Contract Line Works", expanded=False):
        st.markdown("""
        **Contract Line Forecasting** uses two price points at different times to create a trend line:
        
        1. **Low-1**: Your first reference point (price + time)
        2. **Low-2**: Your second reference point (price + time)  
        3. **Slope Calculation**: The system calculates the rate of change between these points
        4. **Projection**: Extends this trend across all time slots for the forecast day
        
        This method is particularly useful for options trading and intraday momentum strategies.
        """)
    
    # Input section
    st.markdown("### ⚙️ Contract Parameters")
    
    # Date input
    contract_date = st.date_input(
        "📅 Contract Date",
        value=date.today() + timedelta(days=1),
        min_value=date.today(),
        max_value=date.today() + timedelta(days=30),
        key="contract_date"
    )
    
    st.markdown("---")
    
    # Two-column layout for input
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown("**📍 Low-1 Reference Point**")
        
        low1_price = st.number_input(
            "💰 Low-1 Price",
            value=10.0,
            min_value=0.0,
            step=0.01,
            format="%.2f",
            key="low1_price",
            help="First reference price point"
        )
        
        low1_time = st.time_input(
            "⏰ Low-1 Time",
            value=time(2, 0),
            step=300,  # 5-minute steps
            key="low1_time",
            help="Time for first reference point"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with input_col2:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown("**📍 Low-2 Reference Point**")
        
        low2_price = st.number_input(
            "💰 Low-2 Price",
            value=12.0,
            min_value=0.0,
            step=0.01,
            format="%.2f",
            key="low2_price",
            help="Second reference price point"
        )
        
        low2_time = st.time_input(
            "⏰ Low-2 Time",
            value=time(3, 30),
            step=300,  # 5-minute steps
            key="low2_time",
            help="Time for second reference point"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Validation and analytics
    if low2_time <= low1_time:
        st.error("⚠️ Low-2 time must be after Low-1 time")
    else:
        # Calculate metrics
        time_minutes = (datetime.combine(contract_date, low2_time) - 
                       datetime.combine(contract_date, low1_time)).total_seconds() / 60
        
        price_change = low2_price - low1_price
        price_change_pct = (price_change / low1_price) * 100 if low1_price > 0 else 0
        hourly_rate = (price_change / time_minutes) * 60 if time_minutes > 0 else 0
        
        # Display analytics
        st.markdown("### 📊 Contract Analytics")
        
        analytics_col1, analytics_col2, analytics_col3 = st.columns(3)
        
        with analytics_col1:
            st.markdown(
                create_metric_card(
                    "💰", "Price Change", f"${price_change:+.2f}", 
                    f"{price_change_pct:+.1f}%"
                ),
                unsafe_allow_html=True
            )
        
        with analytics_col2:
            st.markdown(
                create_metric_card(
                    "⚡", "Hourly Rate", f"${hourly_rate:+.2f}/hr", 
                    f"{time_minutes:.0f} min span"
                ),
                unsafe_allow_html=True
            )
        
        with analytics_col3:
            trend_emoji = "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"
            trend_text = "Bullish" if price_change > 0 else "Bearish" if price_change < 0 else "Flat"
            st.markdown(
                create_metric_card(
                    trend_emoji, "Trend", trend_text, 
                    "Direction indicator"
                ),
                unsafe_allow_html=True
            )
    
    st.markdown("---")
    
    # Generate contract forecast button
    if st.button("🎯 Generate Contract Line Forecast", key="generate_contract", type="primary"):
        
        if low2_time <= low1_time:
            st.error("❌ Please ensure Low-2 time is after Low-1 time")
        else:
            with st.spinner("📈 Generating contract line forecast..."):
                try:
                    # Generate contract forecast
                    contract_table, contract_params = strategy.contract_line_forecast(
                        low1_price, low1_time, low2_price, low2_time, contract_date
                    )
                    
                    # Store in session state
                    st.session_state.contract_params = contract_params
                    st.session_state.contract_table = contract_table
                    st.session_state.contract_metadata = {
                        "date": contract_date,
                        "low1_price": low1_price,
                        "low1_time": low1_time,
                        "low2_price": low2_price,
                        "low2_time": low2_time,
                        "slope": contract_params.get("slope", 0)
                    }
                    
                    st.success("✅ Contract line forecast generated!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating contract forecast: {str(e)}")
    
    # Display contract results
    if not st.session_state.contract_table.empty:
        st.markdown("## 📊 Contract Line Results")
        
        contract_df = st.session_state.contract_table
        metadata = st.session_state.get("contract_metadata", {})
        
        # Results summary
        if not contract_df.empty:
            min_price = contract_df['Projected'].min()
            max_price = contract_df['Projected'].max()
            price_range = max_price - min_price
            
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.markdown(
                    create_metric_card("📉", "Min Price", f"${min_price:.2f}", "Lowest projection"),
                    unsafe_allow_html=True
                )
            
            with result_col2:
                st.markdown(
                    create_metric_card("📈", "Max Price", f"${max_price:.2f}", "Highest projection"),
                    unsafe_allow_html=True
                )
            
            with result_col3:
                st.markdown(
                    create_metric_card("📏", "Range", f"${price_range:.2f}", "Total spread"),
                    unsafe_allow_html=True
                )
        
        # Display the table
        display_forecast_table(contract_df, "Contract Line Projection")
    
    # Real-time lookup section
    st.markdown("---")
    st.markdown("### 🔍 Real-Time Price Lookup")
    st.caption("Get instant price projections for any time using your contract line")
    
    if not st.session_state.contract_params:
        st.warning("⚠️ Generate a contract line forecast first to use the lookup system")
    else:
        lookup_col1, lookup_col2 = st.columns([1, 2])
        
        with lookup_col1:
            lookup_time = st.time_input(
                "🕐 Lookup Time",
                value=time(9, 25),
                step=300,
                key="lookup_time_input",
                help="Enter any time to get projected price"
            )
            
            if st.button("🔍 Lookup Price", key="lookup_button"):
                try:
                    contract_date = st.session_state.contract_metadata.get("date", date.today())
                    lookup_price = strategy.lookup_contract_price(
                        st.session_state.contract_params, 
                        lookup_time, 
                        contract_date
                    )
                    
                    st.session_state.last_lookup_result = {
                        "time": lookup_time,
                        "price": lookup_price
                    }
                    
                except Exception as e:
                    st.error(f"❌ Lookup error: {str(e)}")
        
        with lookup_col2:
            # Display lookup result
            if hasattr(st.session_state, 'last_lookup_result'):
                result = st.session_state.last_lookup_result
                
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #6366f1, #8b5cf6); 
                               padding: 1rem; border-radius: 12px; text-align: center; color: white;">
                        <h3 style="margin: 0; font-size: 1.2rem;">💰 Projected Price</h3>
                        <h2 style="margin: 0.5rem 0; font-size: 2rem; font-weight: bold;">
                            ${result["price"]:.2f}
                        </h2>
                        <p style="margin: 0; opacity: 0.8;">
                            @ {result["time"].strftime('%H:%M')}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Additional analytics
                base_price = st.session_state.contract_metadata.get("low1_price", 0)
                if base_price > 0:
                    price_change = result["price"] - base_price
                    change_percent = (price_change / base_price) * 100
                    
                    change_col1, change_col2 = st.columns(2)
                    with change_col1:
                        st.metric("📊 Price Change", f"${price_change:+.2f}")
                    with change_col2:
                        st.metric("📈 Percentage", f"{change_percent:+.1f}%")
    
    # Batch lookup feature
    st.markdown("---")
    st.markdown("**⚡ Batch Lookup**")
    st.caption("Enter multiple times separated by commas (e.g., 09:30, 10:00, 14:30)")
    
    batch_times_input = st.text_input(
        "Times (HH:MM format)",
        placeholder="09:30, 10:00, 11:30, 14:00",
        key="batch_lookup_input"
    )
    
    if st.button("🔍 Batch Lookup", key="batch_lookup_button") and batch_times_input and st.session_state.contract_params:
        try:
            # Parse times
            time_strings = [t.strip() for t in batch_times_input.split(',')]
            lookup_results = []
            
            contract_date = st.session_state.contract_metadata.get("date", date.today())
            
            for time_str in time_strings:
                try:
                    hour, minute = map(int, time_str.split(':'))
                    lookup_time_obj = time(hour, minute)
                    
                    price = strategy.lookup_contract_price(
                        st.session_state.contract_params,
                        lookup_time_obj,
                        contract_date
                    )
                    
                    lookup_results.append({
                        'Time': time_str,
                        'Projected Price': f"${price:.2f}",
                        'Price (Raw)': price
                    })
                    
                except ValueError:
                    st.warning(f"⚠️ Invalid time format: {time_str}")
            
            if lookup_results:
                # Display results table
                results_df = pd.DataFrame(lookup_results)
                
                # Add change calculations
                if len(results_df) > 1:
                    results_df['Change'] = results_df['Price (Raw)'].diff()
                    results_df['Change'] = results_df['Change'].apply(
                        lambda x: f"${x:+.2f}" if pd.notna(x) else "-"
                    )
                
                # Style and display
                display_columns = ['Time', 'Projected Price']
                if 'Change' in results_df.columns:
                    display_columns.append('Change')
                
                st.dataframe(
                    results_df[display_columns],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download batch results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Batch Results",
                    data=csv,
                    file_name=f"batch_lookup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_batch"
                )
                
        except Exception as e:
            st.error(f"❌ Batch lookup error: {str(e)}")




# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: INDIVIDUAL STOCK PAGES
# ═══════════════════════════════════════════════════════════════════════════════

# Stock information dictionary
stock_info = {
    "TSLA": {"name": "Tesla", "icon": "🚗", "sector": "Automotive"},
    "NVDA": {"name": "NVIDIA", "icon": "🧠", "sector": "Semiconductors"},
    "AAPL": {"name": "Apple", "icon": "🍎", "sector": "Technology"},
    "MSFT": {"name": "Microsoft", "icon": "🪟", "sector": "Technology"},
    "AMZN": {"name": "Amazon", "icon": "📦", "sector": "E-commerce"},
    "GOOGL": {"name": "Google", "icon": "🔍", "sector": "Technology"},
    "META": {"name": "Meta", "icon": "📘", "sector": "Social Media"},
    "NFLX": {"name": "Netflix", "icon": "📺", "sector": "Streaming"}
}

# Handle stock pages
if selected_page in stock_info:
    ticker = selected_page
    info = stock_info[ticker]
    
    # Page header
    st.markdown(f"## {info['icon']} {info['name']} ({ticker}) Analysis")
    st.caption(f"Individual stock forecasting for {info['name']} in the {info['sector']} sector")
    
    # Current slope display
    current_slope = strategy.slopes.get(ticker, 0)
    slope_color = "🟢" if current_slope > 0 else "🔴" if current_slope < 0 else "🟡"
    
    st.info(f"**Current Slope:** {slope_color} {current_slope:.4f}")
    
    # Analysis parameters
    st.markdown("### ⚙️ Analysis Parameters")
    
    # Date and basic settings
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        analysis_date = st.date_input(
            "📅 Analysis Date",
            value=date.today() + timedelta(days=1),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=30),
            key=f"{ticker}_analysis_date"
        )
    
    with analysis_col2:
        st.info(f"📅 **{analysis_date.strftime('%A')}** - {analysis_date.strftime('%B %d, %Y')}")
    
    st.markdown("---")
    
    # Stock data entry
    st.markdown("### 📋 Previous Day Data Entry")
    
    # Two-column layout for low and high anchors
    low_col, high_col = st.columns(2)
    
    with low_col:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown("**📉 Low Anchor**")
        
        low_price = st.number_input(
            "Previous Day Low Price",
            value=0.0,
            min_value=0.0,
            step=0.01,
            format="%.2f",
            key=f"{ticker}_low_price",
            help=f"Enter {ticker}'s previous day low price"
        )
        
        low_time = st.time_input(
            "Low Time",
            value=time(7, 30),
            key=f"{ticker}_low_time",
            help="Time when the low occurred"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with high_col:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown("**📈 High Anchor**")
        
        high_price = st.number_input(
            "Previous Day High Price",
            value=0.0,
            min_value=0.0,
            step=0.01,
            format="%.2f",
            key=f"{ticker}_high_price",
            help=f"Enter {ticker}'s previous day high price"
        )
        
        high_time = st.time_input(
            "High Time",
            value=time(7, 30),
            key=f"{ticker}_high_time",
            help="Time when the high occurred"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Validation and quick metrics
    if low_price > 0 and high_price > 0:
        if high_price <= low_price:
            st.error("⚠️ High price must be greater than low price")
        else:
            # Calculate quick metrics
            price_range = high_price - low_price
            range_percentage = (price_range / low_price) * 100
            midpoint = (high_price + low_price) / 2
            
            st.markdown("### 📊 Quick Analysis")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.markdown(
                    create_metric_card("💰", "Range", f"${price_range:.2f}", f"{range_percentage:.1f}%"),
                    unsafe_allow_html=True
                )
            
            with metric_col2:
                volatility = "Low" if range_percentage < 3 else "High" if range_percentage > 8 else "Normal"
                st.markdown(
                    create_metric_card("⚡", "Volatility", volatility, f"{range_percentage:.1f}% range"),
                    unsafe_allow_html=True
                )
            
            with metric_col3:
                st.markdown(
                    create_metric_card("🎯", "Midpoint", f"${midpoint:.2f}", "Range center"),
                    unsafe_allow_html=True
                )
    
    st.markdown("---")
    
    # Generate analysis button
    generate_key = f"generate_{ticker}_analysis"
    
    if st.button(f"🚀 Generate {ticker} Analysis", key=generate_key, type="primary"):
        
        # Validation
        if low_price <= 0 or high_price <= 0:
            st.error("❌ Please enter valid prices for both low and high anchors")
        elif high_price <= low_price:
            st.error("❌ High price must be greater than low price")
        else:
            with st.spinner(f"📊 Analyzing {ticker}..."):
                try:
                    # Generate stock forecast
                    forecast = strategy.stock_forecast(
                        ticker,
                        low_price, low_time,
                        high_price, high_time,
                        analysis_date
                    )
                    
                    # Store results in session state with ticker-specific key
                    forecast_key = f"{ticker}_forecasts"
                    metadata_key = f"{ticker}_metadata"
                    
                    st.session_state[forecast_key] = forecast
                    st.session_state[metadata_key] = {
                        "date": analysis_date,
                        "low_price": low_price,
                        "low_time": low_time,
                        "high_price": high_price,
                        "high_time": high_time,
                        "generated_at": datetime.now()
                    }
                    
                    st.success(f"✅ {ticker} analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis error: {str(e)}")
    
    # Display results
    forecast_key = f"{ticker}_forecasts"
    metadata_key = f"{ticker}_metadata"
    
    if forecast_key in st.session_state:
        st.markdown(f"## 📊 {ticker} Analysis Results")
        
        forecast_data = st.session_state[forecast_key]
        metadata = st.session_state.get(metadata_key, {})
        
        # Results summary
        analysis_date_str = str(metadata.get('date', 'N/A'))
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown(
                create_metric_card("📅", "Analysis Date", analysis_date_str, "Target date"),
                unsafe_allow_html=True
            )
        
        with summary_col2:
            anchors_count = len(forecast_data)
            st.markdown(
                create_metric_card("🔢", "Anchor Points", str(anchors_count), "Generated forecasts"),
                unsafe_allow_html=True
            )
        
        # Create tabs for Low and High anchors
        if "Low" in forecast_data and "High" in forecast_data:
            low_tab, high_tab, summary_tab = st.tabs(["📉 Low Anchor", "📈 High Anchor", "📋 Summary"])
            
            with low_tab:
                low_df = forecast_data["Low"]
                display_forecast_table(low_df, f"{ticker} Low Anchor Forecast")
                
                # Low anchor insights
                if not low_df.empty and 'Entry' in low_df.columns:
                    entry_range = low_df['Entry'].max() - low_df['Entry'].min()
                    exit_range = low_df['Exit'].max() - low_df['Exit'].min()
                    avg_spread = (low_df['Entry'] - low_df['Exit']).mean()
                    
                    insight_col1, insight_col2, insight_col3 = st.columns(3)
                    
                    with insight_col1:
                        st.metric("📊 Entry Range", f"${entry_range:.2f}")
                    with insight_col2:
                        st.metric("📊 Exit Range", f"${exit_range:.2f}")
                    with insight_col3:
                        st.metric("💰 Avg Spread", f"${avg_spread:.2f}")
            
            with high_tab:
                high_df = forecast_data["High"]
                display_forecast_table(high_df, f"{ticker} High Anchor Forecast")
                
                # High anchor insights
                if not high_df.empty and 'Entry' in high_df.columns:
                    entry_range = high_df['Entry'].max() - high_df['Entry'].min()
                    exit_range = high_df['Exit'].max() - high_df['Exit'].min()
                    avg_spread = (high_df['Entry'] - high_df['Exit']).mean()
                    
                    insight_col1, insight_col2, insight_col3 = st.columns(3)
                    
                    with insight_col1:
                        st.metric("📊 Entry Range", f"${entry_range:.2f}")
                    with insight_col2:
                        st.metric("📊 Exit Range", f"${exit_range:.2f}")
                    with insight_col3:
                        st.metric("💰 Avg Spread", f"${avg_spread:.2f}")
            
            with summary_tab:
                # Generate summary insights
                low_df = forecast_data.get("Low", pd.DataFrame())
                high_df = forecast_data.get("High", pd.DataFrame())
                
                if not low_df.empty and not high_df.empty:
                    st.markdown("### 📊 Performance Comparison")
                    
                    # Calculate key metrics
                    low_volatility = (low_df['Entry'].max() - low_df['Entry'].min()) / low_df['Entry'].mean() * 100
                    high_volatility = (high_df['Entry'].max() - high_df['Entry'].min()) / high_df['Entry'].mean() * 100
                    
                    low_max_spread = (low_df['Entry'] - low_df['Exit']).max()
                    high_max_spread = (high_df['Entry'] - high_df['Exit']).max()
                    
                    # Display comparison
                    comparison_col1, comparison_col2 = st.columns(2)
                    
                    with comparison_col1:
                        st.markdown("**📊 Volatility Analysis**")
                        st.metric("Low Anchor Volatility", f"{low_volatility:.1f}%")
                        st.metric("High Anchor Volatility", f"{high_volatility:.1f}%")
                        
                        # Risk assessment
                        avg_volatility = (low_volatility + high_volatility) / 2
                        risk_level = "Low" if avg_volatility < 5 else "High" if avg_volatility > 15 else "Medium"
                        risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                        
                        st.info(f"**Risk Level:** {risk_emoji[risk_level]} {risk_level}")
                    
                    with comparison_col2:
                        st.markdown("**💰 Profit Potential**")
                        st.metric("Low Anchor Max", f"${low_max_spread:.2f}")
                        st.metric("High Anchor Max", f"${high_max_spread:.2f}")
                        
                        better_anchor = "Low" if low_max_spread > high_max_spread else "High"
                        st.success(f"**Best Anchor:** {better_anchor}")
                    
                    # Overall recommendation
                    st.markdown("### 🎯 Trading Recommendation")
                    
                    if risk_level == "Low" and max(low_max_spread, high_max_spread) > 5:
                        recommendation = "🟢 **FAVORABLE** - Low risk with good profit potential"
                    elif risk_level == "High" and max(low_max_spread, high_max_spread) > 10:
                        recommendation = "🟡 **MODERATE** - High risk but high reward potential"
                    elif risk_level == "Low":
                        recommendation = "🔵 **CONSERVATIVE** - Low risk, modest returns"
                    else:
                        recommendation = "🔴 **CAUTION** - High risk, assess carefully"
                    
                    st.markdown(recommendation)
        
        # Export options
        st.markdown("### 📤 Export Options")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if "Low" in forecast_data:
                csv_low = forecast_data["Low"].to_csv(index=False)
                st.download_button(
                    f"📉 {ticker} Low Data",
                    csv_low,
                    f"{ticker}_low_{datetime.now().strftime('%Y%m%d')}.csv",
                    key=f"download_{ticker}_low"
                )
        
        with export_col2:
            if "High" in forecast_data:
                csv_high = forecast_data["High"].to_csv(index=False)
                st.download_button(
                    f"📈 {ticker} High Data",
                    csv_high,
                    f"{ticker}_high_{datetime.now().strftime('%Y%m%d')}.csv",
                    key=f"download_{ticker}_high"
                )
    
    else:
        st.info(f"👆 Enter {ticker}'s previous day high and low prices, then click 'Generate {ticker} Analysis' to see results.")







# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: SETTINGS & SLOPE MANAGEMENT (SIDEBAR ADDITIONS)
# ═══════════════════════════════════════════════════════════════════════════════

# Additional sidebar features
st.sidebar.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# PRESET MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 💾 Preset Management")

with st.sidebar.expander("📋 Save/Load Presets", expanded=False):
    # Initialize presets in session state if not exists
    if "presets" not in st.session_state:
        st.session_state.presets = {}
    
    # Save preset
    st.markdown("**💾 Save Current Slopes**")
    preset_name = st.text_input(
        "Preset Name",
        placeholder="My Strategy",
        key="preset_name_input"
    )
    
    if st.button("💾 Save Preset", key="save_preset") and preset_name:
        st.session_state.presets[preset_name] = strategy.slopes.copy()
        st.success(f"✅ Saved preset: {preset_name}")
    
    # Load preset
    if st.session_state.presets:
        st.markdown("**📂 Load Preset**")
        selected_preset = st.selectbox(
            "Available Presets",
            list(st.session_state.presets.keys()),
            key="preset_selector"
        )
        
        preset_col1, preset_col2 = st.columns(2)
        
        with preset_col1:
            if st.button("📂 Load", key="load_preset"):
                for asset, slope in st.session_state.presets[selected_preset].items():
                    strategy.update_slope(asset, slope)
                st.success(f"✅ Loaded: {selected_preset}")
                st.rerun()
        
        with preset_col2:
            if st.button("🗑️ Delete", key="delete_preset"):
                del st.session_state.presets[selected_preset]
                st.success(f"✅ Deleted: {selected_preset}")
                st.rerun()
    else:
        st.info("No presets saved yet")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT/IMPORT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 📤 Configuration")

with st.sidebar.expander("💼 Export/Import", expanded=False):
    # Export configuration
    st.markdown("**📤 Export Settings**")
    if st.button("📤 Export Config", key="export_config"):
        config_data = {
            "slopes": strategy.slopes,
            "presets": st.session_state.get("presets", {}),
            "exported_at": datetime.now().isoformat(),
            "version": "2.0"
        }
        
        config_json = json.dumps(config_data, indent=2)
        st.download_button(
            "📥 Download Config",
            config_json,
            f"spx_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="download_config"
        )
    
    # Import configuration
    st.markdown("**📂 Import Settings**")
    uploaded_config = st.file_uploader(
        "Upload Config File",
        type=['json'],
        key="import_config"
    )
    
    if uploaded_config:
        try:
            config_data = json.loads(uploaded_config.read())
            
            if "slopes" in config_data:
                for asset, slope in config_data["slopes"].items():
                    if asset in strategy.slopes:
                        strategy.update_slope(asset, slope)
            
            if "presets" in config_data:
                if "presets" not in st.session_state:
                    st.session_state.presets = {}
                st.session_state.presets.update(config_data["presets"])
            
            st.success("✅ Configuration imported successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Import error: {str(e)}")

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION INFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## ℹ️ Session Info")

with st.sidebar.expander("📊 Current Session", expanded=False):
    # Count active forecasts
    spx_forecasts = len(st.session_state.get('current_forecasts', {}))
    contract_active = not st.session_state.get('contract_table', pd.DataFrame()).empty
    
    stock_forecasts = 0
    for ticker in strategy.get_available_tickers():
        if f"{ticker}_forecasts" in st.session_state:
            stock_forecasts += 1
    
    st.markdown(f"""
    **📊 Active Forecasts:**
    - SPX Anchors: {spx_forecasts}
    - Contract Line: {'✅' if contract_active else '❌'}
    - Stock Analysis: {stock_forecasts}
    
    **⏰ Session Time:** {datetime.now().strftime('%H:%M:%S')}
    
    **📅 Current Page:** {st.session_state.selected_page}
    """)
    
    # Clear session data
    if st.button("🧹 Clear All Data", key="clear_session"):
        # Clear all forecast data
        keys_to_clear = ['current_forecasts', 'contract_table', 'contract_params']
        
        # Clear stock-specific data
        for ticker in strategy.get_available_tickers():
            keys_to_clear.extend([f"{ticker}_forecasts", f"{ticker}_metadata"])
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("✅ All forecast data cleared!")
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# HELP & DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## ❓ Help")

with st.sidebar.expander("📖 Quick Guide", expanded=False):
    st.markdown("""
    **🧭 SPX Forecasting:**
    1. Select "SPX Forecast" page
    2. Enter High, Close, Low prices from previous day
    3. Set corresponding times
    4. Generate forecast to see projections
    
    **📈 Contract Line:**
    1. Go to "Contract Line" page  
    2. Set Low-1 and Low-2 reference points
    3. Generate forecast for interpolated prices
    4. Use lookup for specific time queries
    
    **📊 Stock Analysis:**
    1. Select any stock page (TSLA, NVDA, etc.)
    2. Enter previous day High and Low prices
    3. Set anchor times
    4. Generate analysis for entry/exit projections
    
    **📐 Slope Management:**
    Use the "Adjust Slopes" section to fine-tune forecasting parameters for each asset.
    """)

with st.sidebar.expander("⚙️ Tips & Tricks", expanded=False):
    st.markdown("""
    **💡 Best Practices:**
    - Use accurate previous day data for better forecasts
    - Save slope configurations as presets for different market conditions
    - Compare multiple anchor points for validation
    - Export data for external analysis
    
    **🎯 Slope Tuning:**
    - Negative slopes: Price decreases over time
    - Positive slopes: Price increases over time  
    - Magnitude: How steep the price change
    
    **📊 Reading Results:**
    - Entry: Suggested entry price
    - Exit: Suggested exit price
    - Spread: Difference between entry/exit
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER INFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="text-align: center; opacity: 0.7; font-size: 0.8rem;">
        <strong>SPX Prophet v2.0</strong><br>
        Session: {datetime.now().strftime('%H:%M:%S')}<br>
        Page: {st.session_state.selected_page}
    </div>
    """,
    unsafe_allow_html=True
)







# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: FOOTER & COMPLETION
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

# Add spacing before footer
st.markdown("<br><br>", unsafe_allow_html=True)

# Application footer
st.markdown("---")

# Footer with statistics and info
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("### 📊 Session Statistics")
    
    # Calculate session stats
    total_forecasts = 0
    
    # Count SPX forecasts
    if st.session_state.get('current_forecasts'):
        total_forecasts += len(st.session_state.current_forecasts)
    
    # Count contract line
    if not st.session_state.get('contract_table', pd.DataFrame()).empty:
        total_forecasts += 1
    
    # Count stock forecasts
    stock_count = 0
    for ticker in strategy.get_available_tickers():
        if f"{ticker}_forecasts" in st.session_state:
            stock_count += 1
    
    total_forecasts += stock_count
    
    st.markdown(f"""
    - **Total Forecasts Generated:** {total_forecasts}
    - **Active Stock Analysis:** {stock_count}
    - **Current Page:** {st.session_state.selected_page}
    - **Session Duration:** {datetime.now().strftime('%H:%M:%S')}
    """)

with footer_col2:
    st.markdown("### 🎯 Quick Actions")
    
    quick_col1, quick_col2 = st.columns(2)
    
    with quick_col1:
        if st.button("🔄 Reset Slopes", key="footer_reset_slopes"):
            strategy.reset_slopes()
            st.success("✅ Slopes reset!")
            st.rerun()
        
        if st.button("📊 SPX Page", key="footer_spx"):
            st.session_state.selected_page = "SPX"
            st.rerun()
    
    with quick_col2:
        if st.button("📈 Contract Page", key="footer_contract"):
            st.session_state.selected_page = "Contract"  
            st.rerun()
        
        if st.button("🧹 Clear Data", key="footer_clear"):
            # Clear main forecast data
            keys_to_clear = ['current_forecasts', 'contract_table', 'contract_params']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ Data cleared!")
            st.rerun()

with footer_col3:
    st.markdown("### ℹ️ About")
    
    st.markdown("""
    **SPX Prophet v2.0**
    
    Advanced financial forecasting tool using time-based projections and slope analysis.
    
    **Features:**
    - 🧭 SPX three-anchor forecasting
    - 📈 Contract line interpolation  
    - 📊 Individual stock analysis
    - 📐 Customizable slope parameters
    - 💾 Preset management
    - 📤 Data export capabilities
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL DISCLAIMER & LEGAL
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")

# Disclaimer
st.markdown("""
<div style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); 
           border-radius: 16px; padding: 1rem; margin: 1rem 0; text-align: center;">
    <h4 style="color: #f59e0b; margin-top: 0;">⚠️ Important Disclaimer</h4>
    <p style="margin-bottom: 0; opacity: 0.8; font-size: 0.9rem;">
        This tool is for educational and analysis purposes only. Past performance does not guarantee future results. 
        Always conduct your own research and risk management before making any trading decisions. 
        The creators are not responsible for any financial losses incurred from using this tool.
    </p>
</div>
""", unsafe_allow_html=True)

# Final footer
st.markdown(
    f"""
    <div style="text-align: center; opacity: 0.6; font-size: 0.8rem; margin-top: 2rem;">
        SPX Prophet v2.0 • Built with Streamlit • 
        Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • 
        Page: {st.session_state.selected_page}
    </div>
    """,
    unsafe_allow_html=True
)

# ═══════════════════════════════════════════════════════════════════════════════
# SUCCESS MESSAGE (DEVELOPMENT ONLY - REMOVE IN PRODUCTION)
# ═══════════════════════════════════════════════════════════════════════════════

# This message confirms the app loaded successfully
# Remove this section when deploying to production
if st.sidebar.button("🎉 App Status", key="app_status"):
    st.sidebar.success("✅ SPX Prophet fully loaded!")
    st.sidebar.info(f"Current slopes: {len(strategy.slopes)} assets configured")
    st.sidebar.info(f"Available tickers: {len(strategy.get_available_tickers())} stocks")

# End of application
# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION COMPLETE - ALL 6 PARTS LOADED
# ═══════════════════════════════════════════════════════════════════════════════