from integrations.deriv_ws_client import (
    get_account_profit_percent,
    get_total_trades,
    get_win_rate,
    get_strategy_impacts,
    get_latest_tick,
    run_in_background,  # <-- Add this line
)
import streamlit as st
from streamlit_autorefresh import st_autorefresh

def get_real_time_metrics():
    return {
        "profit_percent": get_account_profit_percent(),
        "total_trades": get_total_trades(),
        "win_rate": get_win_rate(),
        "strategy_impacts": get_strategy_impacts(),
    }

if "ws_started" not in st.session_state:
    run_in_background()
    st.session_state["ws_started"] = True

st_autorefresh(interval=2000, key="datarefresh_realtime")

with st.spinner("Loading real-time metrics..."):
    metrics = get_real_time_metrics()

tick = get_latest_tick()
if tick:
    st.write(tick)
else:
    st.info("Waiting for real-time data...")