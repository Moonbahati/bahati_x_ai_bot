# gui/telegram_alerts.py

import os
import asyncio
import aiohttp
from engine.market_analyzer import analyze_market_conditions
from core.strategy_evolver import get_current_strategy
from ai.fraud_detection_ai import fetch_recent_fraud_flags
from ai.threat_prediction_ai import forecast_threats
from simulator.real_time_tester import get_real_time_metrics
from ai.self_evolver import get_self_evolution_status
from security.quantum_resistant_encryption import verify_encryption_integrity

# Bot Credentials (secure via .env)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # Format: 123456789:ABCDEF...
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")      # Your Telegram User/Group ID

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

async def send_telegram_message(message: str, parse_mode: str = "Markdown"):
    url = f"{BASE_URL}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            if resp.status == 200:
                print("✅ Telegram alert sent.")
            else:
                print(f"❌ Failed to send alert. Status code: {resp.status}")
                print(await resp.text())

async def dispatch_telegram_alerts():
    # Strategy Details
    strategy = get_current_strategy()
    strategy_msg = f"*📈 Strategy:* `{strategy.get('summary', 'No info available')}`"

    # Market Analysis
    market = analyze_market_conditions()
    market_msg = (
        f"*📊 Market Update:*\n"
        f"`Volatility:` {market['volatility']}\n"
        f"`Trend:` {market['trend']}"
    )

    # Real-Time Metrics
    metrics = get_real_time_metrics()
    metrics_msg = (
        f"*📉 Live Metrics:*\n"
        f"`Profit:` {metrics['profit_percent']}%\n"
        f"`Win Rate:` {metrics['win_rate']}%"
    )

    # Fraud Report
    frauds = fetch_recent_fraud_flags()
    fraud_msg = (
        f"*⚠️ Fraud Detection:*\n"
        f"{len(frauds)} anomalies flagged." if frauds else "*✅ No fraud detected.*"
    )

    # AI Self-Evolution
    ai_status = get_self_evolution_status()
    ai_msg = f"*🧠 AI Status:*\n`{ai_status['status']}`"

    # Threat Forecast
    threats = forecast_threats()
    threat_msg = f"*🔮 Threat Forecast:*\n`{threats}`"

    # Encryption Check
    enc_ok = verify_encryption_integrity()
    encryption_msg = (
        "*🔐 Encryption:* Secure ✅"
        if enc_ok else
        "*🛑 Encryption Risk:* Breach suspected!"
    )

    # Combine All Messages
    full_alert = "\n\n".join([
        strategy_msg,
        market_msg,
        metrics_msg,
        fraud_msg,
        ai_msg,
        threat_msg,
        encryption_msg
    ])

    # Dispatch
    await send_telegram_message(full_alert)

# Manual trigger
if __name__ == "__main__":
    print("🚀 Dispatching Telegram hybrid alert...")
    asyncio.run(dispatch_telegram_alerts())
