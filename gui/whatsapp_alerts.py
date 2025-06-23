# gui/whatsapp_alerts.py

import os
import time
from twilio.rest import Client
from engine.market_analyzer import analyze_market_conditions
from core.strategy_evolver import get_current_strategy
from ai.fraud_detection_ai import fetch_recent_fraud_flags
from ai.self_evolver import get_self_evolution_status
from ai.threat_prediction_ai import forecast_threats
from simulator.real_time_tester import get_real_time_metrics
from security.quantum_resistant_encryption import verify_encryption_integrity

# Twilio WhatsApp Configs (Environment Variables recommended)
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
FROM_NUMBER = "whatsapp:+14155238886"  # Twilio Sandbox default
TO_NUMBER = os.getenv("WHATSAPP_TO_NUMBER")  # e.g., "whatsapp:+2547XXXXXXX"

client = Client(TWILIO_SID, TWILIO_AUTH)

# Send WhatsApp Message
def send_whatsapp_message(message):
    try:
        message = client.messages.create(
            from_=FROM_NUMBER,
            body=message,
            to=TO_NUMBER
        )
        print(f"✅ WhatsApp alert sent: SID {message.sid}")
    except Exception as e:
        print(f"❌ Error sending WhatsApp message: {e}")

# Construct and Dispatch Alerts
def dispatch_alerts():
    messages = []

    # 1. Strategy Update
    strategy = get_current_strategy()
    messages.append(f"📈 Strategy Update:\n{strategy.get('summary', 'No summary')}")

    # 2. Market Conditions
    market = analyze_market_conditions()
    messages.append(f"📊 Market Snapshot:\nVolatility: {market['volatility']}\nTrend: {market['trend']}")

    # 3. Real-Time Trading Metrics
    metrics = get_real_time_metrics()
    messages.append(f"📉 Live Metrics:\nProfit: {metrics['profit_percent']}%\nWin Rate: {metrics['win_rate']}%")

    # 4. Fraud Detection
    frauds = fetch_recent_fraud_flags()
    if frauds:
        messages.append(f"⚠️ Fraud Alerts:\n{len(frauds)} suspicious entries flagged.")
    else:
        messages.append("✅ No fraud detected.")

    # 5. Self-Evolution State
    evolver = get_self_evolution_status()
    messages.append(f"🧠 AI Evolution:\n{evolver['status']}")

    # 6. Threat Forecasting
    threats = forecast_threats()
    messages.append(f"🔮 Threat Forecast:\n{threats}")

    # 7. Encryption Integrity Check
    secure = verify_encryption_integrity()
    enc_msg = "🔐 Encryption Secure ✅" if secure else "🛑 Quantum Risk: Encryption Breach Detected!"
    messages.append(enc_msg)

    # Combine & Send
    full_alert = "\n\n".join(messages)
    send_whatsapp_message(full_alert)

# Scheduler or Manual Trigger
if __name__ == "__main__":
    print("📡 Sending hybrid AI WhatsApp alert...")
    dispatch_alerts()
