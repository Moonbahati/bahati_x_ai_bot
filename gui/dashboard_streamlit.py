# gui/dashboard_streamlit.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import time
from datetime import datetime
from core.strategy_evolver import StrategyEvolver
from core.federated_agent import FederatedAgent
from engine.market_analyzer import analyze_market_conditions
from engine.dna_profiler import summarize_dna_profiles
from ai.self_evolver import get_self_evolution_status
from ai.fraud_detection_ai import fetch_recent_fraud_flags
from security.quantum_resistant_encryption import verify_encryption_integrity
from simulator.real_time_tester import get_real_time_metrics
from simulator.post_launch_lab import post_launch_feedback_summary
from logs.logger_manager import load_security_logs  # Ensure this exists
from gui.voice_output import speak  # Optional for voice output if enabled

# Set up Streamlit Page
st.set_page_config(page_title="Legendary AI Bot Dashboard", layout="wide")

# Sidebar Config
st.sidebar.title("🧠 UltraBot Control Panel")
selected_view = st.sidebar.radio("Select View", [
    "📊 Performance Overview",
    "🧬 Strategy Intelligence",
    "🛡️ Security Status",
    "🎯 AI Evolution & Insights",
    "📣 Alerts & Notifications"
])

# Voice Activation Trigger
voice_enabled = st.sidebar.toggle("🎤 Enable Voice Feedback", value=False)

def say(msg):
    if voice_enabled:
        speak(msg)

# Header
st.markdown("## 🧠 BAHATI_X_LEGENDARY_AI BOT")
st.markdown("Trading Agent")

# Section: Performance Overview
if selected_view == "📊 Performance Overview":
    st.markdown("### 📊 Real-Time Performance Metrics")
    metrics = get_real_time_metrics()

    profit_percent = metrics.get('profit_percent', 0.0)
    total_trades = metrics.get('total_trades', 0)
    win_rate = metrics.get('win_rate', 0.0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Profit %", f"{profit_percent:.2f}%")
    col2.metric("Total Trades", total_trades)
    col3.metric("Winning Rate", f"{win_rate:.1f}%")
    say("Real-time performance loaded")

    st.markdown("### 📈 Strategy Impact Visualization")
    df = pd.DataFrame(metrics['strategy_impacts'])
    fig = px.bar(df, x="strategy", y="impact", color="strategy", title="Strategy Impact Analysis")
    st.plotly_chart(fig, use_container_width=True)

# Section: Strategy Intelligence
elif selected_view == "🧬 Strategy Intelligence":
    st.markdown("### 🧬 Strategy Genetic Overview")
    evolver = StrategyEvolver(base_dna={})
    st.json(vars(evolver))  # Shows all attributes of the object
    say("Displaying current strategy overview")

    st.markdown("### 🧠 Federated Learning Insights")
    agent = FederatedAgent()
    st.write(vars(agent))  # or use the correct method/property if you know it

    st.markdown("### 🧪 DNA Profile Summary")
    st.code(summarize_dna_profiles(), language='json')

# Section: Security Status
elif selected_view == "🛡️ Security Status":
    st.markdown("### 🛡️ Quantum Security Status")
    status = verify_encryption_integrity({})  # Pass test or real data
    st.success("Encryption is secure and quantum-resistant.") if status else st.error("Encryption breach detected!")

    st.markdown("### 🕵️ Recent Fraud Detection Flags")
    fraud_flags = fetch_recent_fraud_flags()
    if fraud_flags:
        st.warning(f"⚠️ Fraud Alerts: {len(fraud_flags)}")
        st.table(pd.DataFrame(fraud_flags))
    else:
        st.success("No fraud patterns detected in the last 24h.")

    st.markdown("### 🧾 Security Logs")
    logs = load_security_logs()
    if logs:
        st.json(logs[-5:])  # show last 5 entries
    else:
        st.info("No logs found.")

# Section: AI Evolution
elif selected_view == "🎯 AI Evolution & Insights":
    st.markdown("### 🔁 AI Self-Evolution Tracker")
    evolution_data = get_self_evolution_status()
    st.json(evolution_data)

    st.markdown("### 🔍 Post-Launch Feedback Loop")
    feedback_summary = post_launch_feedback_summary()
    st.write(feedback_summary)

    st.markdown("### 🧠 Intelligence Summary")
    st.info("AI is in adaptive learning mode. Monitoring strategy-response alignment.")

# Section: Alerts
elif selected_view == "📣 Alerts & Notifications":
    st.markdown("### 📣 Critical System Notifications")
    alerts = []
    metrics = get_real_time_metrics()
    status = verify_encryption_integrity({})
    fraud_flags = fetch_recent_fraud_flags()

    if metrics['profit_percent'] < 0:
        alerts.append("⚠️ System is operating at a loss!")
    if not status:
        alerts.append("🛡️ Encryption status compromised!")
    if fraud_flags:
        alerts.append(f"🚨 {len(fraud_flags)} suspicious activities detected!")

    if alerts:
        for alert in alerts:
            st.error(alert)
            say(alert)
    else:
        st.success("✅ All systems are optimal.")

# Footer
st.markdown("---")
st.caption("🧠 Powered by Bahati_Legendary_AI_Engine™ — Quantum-Secure. Self-Healing. Federated.")

def launch_dashboard():
    # This function is a placeholder to allow import from main.py
    # Streamlit apps are usually run with `streamlit run`, not as a function.
    pass
