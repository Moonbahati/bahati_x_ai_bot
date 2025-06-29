

from integrations.deriv_ws_client import (
    get_account_profit_percent,
    get_total_trades,
    get_win_rate,
    get_strategy_impacts,
    get_latest_tick,
    run_in_background,
)

def get_real_time_metrics():
    return {
        "profit_percent": get_account_profit_percent(),
        "total_trades": get_total_trades(),
        "win_rate": get_win_rate(),
        "strategy_impacts": get_strategy_impacts(),
    }

def launch_real_time_dashboard():
    import streamlit as st
    try:
        from streamlit_autorefresh import st_autorefresh
    except ImportError:
        def st_autorefresh(*args: object, **kwargs: object) -> None:
            pass  # fallback: do nothing if not available

    if "ws_started" not in st.session_state:
        run_in_background()
        st.session_state["ws_started"] = True

    st.title("AI Trading Bot Real-Time Dashboard")
    st_autorefresh(interval=2000, key="datarefresh_realtime")

    with st.spinner("Loading real-time metrics..."):
        metrics = get_real_time_metrics()

    st.metric("Profit %", f"{metrics['profit_percent']:.2f}%")
    st.metric("Total Trades", metrics['total_trades'])
    st.metric("Win Rate", f"{metrics['win_rate']:.2f}%")

    if metrics["strategy_impacts"]:
        st.subheader("Strategy Impacts")
        st.write(metrics["strategy_impacts"])

    tick = get_latest_tick()
    st.subheader("Latest Tick Data")
    if tick:
        st.json(tick)
    else:
        st.info("Waiting for real-time data...")

if __name__ == "__main__":
    launch_real_time_dashboard()