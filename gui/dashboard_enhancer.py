# gui/dashboard_enhancer.py

import streamlit as st
import pandas as pd
from ai.threat_prediction_ai import forecast_threats
from ai.fraud_detection_ai import fetch_recent_fraud_flags
from engine.market_analyzer import analyze_market_conditions
from core.strategy_evolver import get_current_strategy
from datetime import datetime


class DashboardEnhancer:
    def __init__(self):
        self.colors = {
            "normal": "#1f77b4",
            "alert": "#ff4b4b",
            "positive": "#2ca02c",
            "info": "#17becf"
        }

    def display_banner(self):
        st.markdown(
            "<h2 style='text-align: center; color: #00d4ff;'>🚀 Hybrid Intelligence Dashboard Enhancer</h2>",
            unsafe_allow_html=True
        )

    def show_market_status(self):
        market = analyze_market_conditions()
        st.subheader("📊 Market Analyzer")
        st.info(f"**Trend:** {market['trend']} | **Volatility:** {market['volatility']} | **Sentiment:** {market.get('sentiment', 'Neutral')}")

    def show_strategy_summary(self):
        strategy = get_current_strategy()
        st.subheader("🧠 Strategy Engine")
        st.success(f"**Active Strategy:** {strategy.get('summary', 'No strategy')} | **AI Version:** {strategy.get('version', 'vX')}")

    def show_threat_prediction(self):
        threat = forecast_threats()
        st.subheader("🛡️ Threat Prediction")
        if "none" in threat.lower():
            st.markdown(f"<span style='color:{self.colors['positive']}'>✅ No critical threats detected.</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:{self.colors['alert']}'>⚠️ {threat}</span>", unsafe_allow_html=True)

    def show_fraud_flags(self):
        flags = fetch_recent_fraud_flags()
        st.subheader("💸 Fraud Detection AI")
        if not flags:
            st.success("No fraudulent activity detected.")
        else:
            st.error(f"⚠️ {len(flags)} suspicious activities flagged.")
            df = pd.DataFrame(flags)
            st.dataframe(df)

    def show_system_time(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.sidebar.markdown(f"🕒 **System Time:** `{now}`")

    def inject_theme_switch(self):
        theme = st.sidebar.radio("Choose Theme", ["💡 Light Mode", "🌙 Dark Mode"])
        st.session_state['theme'] = theme

    def show_status_badges(self):
        st.sidebar.markdown("### 🔍 System Badges")
        st.sidebar.success("AI Health ✅")
        st.sidebar.info("Quantum Security Active 🧬")
        st.sidebar.warning("Latency Monitor: Stable ⏱️")

    def render_all(self):
        self.display_banner()
        self.inject_theme_switch()
        self.show_system_time()
        self.show_market_status()
        self.show_strategy_summary()
        self.show_threat_prediction()
        self.show_fraud_flags()
        self.show_status_badges()


# Standalone demo (optional)
if __name__ == "__main__":
    st.set_page_config(page_title="Enhanced AI Dashboard", layout="wide")
    enhancer = DashboardEnhancer()
    enhancer.render_all()
