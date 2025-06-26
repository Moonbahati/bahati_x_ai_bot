# gui/voice_output.py

import asyncio
import os
from ai.self_evolver import get_self_evolution_status
from ai.threat_prediction_ai import forecast_threats
from ai.fraud_detection_ai import fetch_recent_fraud_flags
from engine.market_analyzer import analyze_market_conditions
from core.strategy_evolver import get_current_strategy

try:
    import edge_tts  # High-quality async TTS
except ImportError:
    edge_tts = None

import pyttsx3

engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()


def voice_notify(message):
    print(f"[VOICE NOTIFY]: {message}")


class VoiceOutputManager:
    def __init__(self):
        self.voice = "en-US-JennyNeural"
        self.rate = "+5%"
        self.pitch = "+0%"
        self.fallback_engine = None

        if not edge_tts:
            self.fallback_engine = pyttsx3.init()
            self.fallback_engine.setProperty("rate", 160)

    async def speak(self, text: str):
        if edge_tts:
            communicate = edge_tts.Communicate(text, self.voice)
            communicate.tts_speed = self.rate
            communicate.tts_pitch = self.pitch
            await communicate.stream()
        elif self.fallback_engine:
            self.fallback_engine.say(text)
            self.fallback_engine.runAndWait()
        else:
            print(f"[VOICE OUTPUT]: {text}")

    async def narrate_system_status(self):
        strategy = get_current_strategy().get("summary", "Unknown strategy")
        market = analyze_market_conditions()
        threats = forecast_threats()
        frauds = fetch_recent_fraud_flags()
        ai_status = get_self_evolution_status().get("status", "stable")

        lines = []

        lines.append("System hybrid voice report initialized.")
        lines.append(f"Strategy in operation: {strategy}.")
        lines.append(f"Market trend is {market['trend']} with {market['volatility']} volatility.")

        if threats and "none" not in threats.lower():
            lines.append(f"⚠️ Alert: Threat prediction module warns of: {threats}")
        else:
            lines.append("No immediate threats predicted.")

        if frauds:
            lines.append(f"{len(frauds)} potential fraud anomalies have been detected.")
        else:
            lines.append("Fraud detection reports a clean system.")

        lines.append(f"AI self-evolution status: {ai_status}.")

        # Speak all lines one by one
        for line in lines:
            await self.speak(line)
            await asyncio.sleep(0.5)  # Space out speech

    async def whisper_alert(self, text: str):
        # Subtle voice tone for sensitive data
        self.voice = "en-US-AnaNeural"
        self.pitch = "-10%"
        self.rate = "-5%"
        await self.speak(text)
        # Reset voice settings
        self.voice = "en-US-JennyNeural"
        self.pitch = "+0%"
        self.rate = "+5%"


# Run Demo
if __name__ == "__main__":
    print("🔊 Generating AI voice system report...")
    voice_manager = VoiceOutputManager()
    asyncio.run(voice_manager.narrate_system_status())
