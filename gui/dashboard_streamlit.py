import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
from core.strategy_evolver import StrategyEvolver
from core.federated_agent import FederatedAgent
from engine.market_analyzer import analyze_market_conditions
from engine.dna_profiler import summarize_dna_profiles
from ai.self_evolver import get_self_evolution_status
from ai.fraud_detection_ai import fetch_recent_fraud_flags
from security.quantum_resistant_encryption import verify_encryption_integrity
from simulator.real_time_tester import get_real_time_metrics
from simulator.post_launch_lab import post_launch_feedback_summary
from logs.logger_manager import load_security_logs
from gui.voice_output import speak
from integrations.deriv_ws_client import run_in_background, get_latest_tick, get_balance
from streamlit_autorefresh import st_autorefresh
import json

# Set up Streamlit Page
st.set_page_config(page_title="Legendary AI Bot Dashboard", layout="wide")

# Sidebar Config
st.sidebar.title(" Control Panel")
selected_view = st.sidebar.radio("Select View", [
    " Performance Overview",
    " Strategy Intelligence",
    " Security Status",
    " AI Evolution & Insights",
    " Alerts & Notifications"
])

# Voice Activation Trigger
voice_enabled = st.sidebar.toggle("🎤 Enable Voice Feedback", value=False)

def say(msg):
    if voice_enabled:
        speak(msg)

# Header
st.markdown("##  BAHATI_X_LEGENDARY_AI BOT")
st.markdown("Trading Agent")

# Section: Performance Overview
if selected_view == " Performance Overview":
    if "ws_started" not in st.session_state:
        run_in_background()
        st.session_state["ws_started"] = True
    st_autorefresh(interval=2000, key="datarefresh_dashboard")

    # 1. Real-Time Performance Metrics (top)
    st.markdown("###  Real-Time Performance Metrics")
    try:
        # --- REAL-TIME BALANCE ---
        balance = get_balance()
        st.markdown(f"#### 💰 Real-Time Account Balance: ${balance:,.2f}")
        say(f"Current account balance is {balance:,.2f} dollars")
        metrics = get_real_time_metrics()
        profit_percent = metrics.get("profit_percent", 0.0)
        total_trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0.0)

        col1, col2, col3 = st.columns(3)
        if profit_percent is None:
            col1.info("Profit % data is not available.")
        else:
            col1.metric("Profit %", f"{profit_percent:.2f}%")
        if total_trades is None:
            col2.info("Total Trades data is not available.")
        else:
            col2.metric("Total Trades", total_trades)
        if win_rate is None:
            col3.info("Winning Rate data is not available.")
        else:
            col3.metric("Winning Rate", f"{win_rate:.1f}%")
        say("Real-time performance loaded")
    except Exception as e:
        st.error(f"Error loading metrics: {e}")

    # 2. Strategy Impact Visualization (graph)
    st.markdown("### Strategy Impact Visualization")
    strategy_impacts = metrics.get("strategy_impacts", [])
    df = pd.DataFrame(strategy_impacts)

    if df.empty or not all(col in df.columns for col in ["strategy", "impact"]):
        st.info("Strategy impact data is not available.")
    else:
        fig = px.bar(df, x="strategy", y="impact", color="strategy", title="Strategy Impact Analysis")
        st.plotly_chart(fig, use_container_width=True)

    # 3. Real-Time Deriv Data (bottom)
    st.markdown("### 🟢 Real-Time Deriv Data")
    tick = get_latest_tick()
    if tick:
        st.write(tick)
    else:
        st.info("Waiting for real-time data...")

# Section: Strategy Intelligence
elif selected_view == " Strategy Intelligence":
    st.markdown("###  Strategy Genetic Overview")
    evolver = StrategyEvolver(base_dna={})
    st.json(vars(evolver))
    say("Displaying current strategy overview")
    st.markdown("###  DNA Profile Summary")
    st.write(summarize_dna_profiles())
    st.markdown("###  Federated Learning Insights")
    agent = FederatedAgent()
    st.write(vars(agent))

# Section: Security Status
elif selected_view == " Security Status":
    st.markdown("###  Encryption Integrity Check")
    status = verify_encryption_integrity({})
    if status:
        st.success("Encryption is secure and quantum-resistant.")
    else:
        st.error("Encryption breach detected!")

    st.markdown("###  Recent Fraud Detection Flags")
    fraud_flags = fetch_recent_fraud_flags()
    if fraud_flags:
        st.warning(f" Fraud Alerts: {len(fraud_flags)}")
        st.table(pd.DataFrame(fraud_flags))
    else:
        st.success("No fraud patterns detected in the last 24h.")

    st.markdown("### 🧾 Security Logs")
    logs = load_security_logs()
    if logs:
        st.json(logs[-5:])
    else:
        st.info("No logs found.")

# Section: AI Evolution & Insights
elif selected_view == " AI Evolution & Insights":
    st.markdown("###  AI Self-Evolution Tracker")
    evolution_data = get_self_evolution_status()
    st.json(evolution_data)
    st.markdown("### 🔍 Post-Launch Feedback Loop")
    feedback_summary = post_launch_feedback_summary()
    st.write(feedback_summary)
    st.markdown("###  Intelligence Summary")
    st.info("AI is in adaptive learning mode. Monitoring strategy-response alignment.")

# Section: Alerts & Notifications
elif selected_view == " Alerts & Notifications":
    st.markdown("###  Critical System Notifications")
    alerts = []
    metrics = get_real_time_metrics()
    status = verify_encryption_integrity({})
    fraud_flags = fetch_recent_fraud_flags()
    if metrics.get('profit_percent', 0) < 0:
        alerts.append(" System is operating at a loss!")
    if not status:
        alerts.append(" Encryption status compromised!")
    if fraud_flags:
        alerts.append(f" {len(fraud_flags)} suspicious activities detected!")
    if alerts:
        for alert in alerts:
            st.error(alert)
            say(alert)
    else:
        st.success(" All systems are optimal.")

# Footer
st.markdown("---")
st.caption(" Powered by Bahati_Legendary_AI_Engine™ — Quantum-Secure. Self-Healing. Federated.")

def launch_dashboard():
    import streamlit.web.bootstrap
    import os
    script_path = os.path.abspath(__file__)
    streamlit.web.bootstrap.run(script_path, '', [], {})