# gui/ai_voice_chat.py

import speech_recognition as sr
import pyttsx3
import time
from ai.intent_recognizer import recognize_intent
from ai.chatgpt_connector import ask_gpt
from core.strategy_evolver import get_current_strategy
from engine.market_analyzer import analyze_market_conditions
from ai.fraud_detection_ai import fetch_recent_fraud_flags
from simulator.real_time_tester import get_real_time_metrics
from ai.self_evolver import get_self_evolution_status
from security.quantum_resistant_encryption import verify_encryption_integrity

# Initialize Text-to-Speech Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 165)

# Initialize Speech Recognizer
recognizer = sr.Recognizer()
try:
    mic = sr.Microphone()
except OSError:
    mic = None
    print("⚠️ No default input device available. Voice features will be disabled.")

# Speak Function
def speak(text):
    print("🤖 AI Bot says:", text)
    tts_engine.say(text)
    tts_engine.runAndWait()

# Listen Function
def listen():
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("🎤 Listening for command...")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        print(f"🧠 You said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that.")
        return ""
    except sr.RequestError:
        speak("Speech recognition service is unavailable.")
        return ""

# Command Handler
def handle_command(command):
    intent = recognize_intent(command)

    if intent == "get_strategy":
        strategy = get_current_strategy()
        speak("Here is the current trading strategy.")
        return str(strategy)

    elif intent == "market_analysis":
        data = analyze_market_conditions()
        speak("Market analysis shows current conditions.")
        return str(data)

    elif intent == "fraud_status":
        fraud = fetch_recent_fraud_flags()
        if fraud:
            speak(f"{len(fraud)} suspicious activities detected.")
            return str(fraud)
        else:
            speak("No fraud activities found recently.")
            return "Clean status."

    elif intent == "real_time_metrics":
        metrics = get_real_time_metrics()
        speak("Real-time trading metrics are as follows.")
        return f"Profit: {metrics['profit_percent']}%, Win Rate: {metrics['win_rate']}%"

    elif intent == "evolution_status":
        status = get_self_evolution_status()
        speak("The AI is currently evolving based on new patterns.")
        return str(status)

    elif intent == "encryption_check":
        secure = verify_encryption_integrity()
        if secure:
            speak("Encryption is secure and quantum-resistant.")
            return "Status: Secure"
        else:
            speak("Warning: Encryption may be compromised!")
            return "Status: Breached"

    elif intent == "general_chat":
        speak("Let me think...")
        response = ask_gpt(command)
        speak(response)
        return response

    else:
        speak("Command not recognized. Please try again.")
        return "Unknown command."

# Voice Chat Loop
def voice_chat_loop():
    speak("Hello, I am your Legendary AI Assistant. How can I help you today?")
    while True:
        command = listen()
        if not command:
            continue
        if "exit" in command or "quit" in command:
            speak("Shutting down. Have a great day!")
            break
        response = handle_command(command)
        print("📡 Response:\n", response)
        print("-" * 60)
        time.sleep(1)

def launch_voice_ai():
    if mic is None:
        print("Voice features are disabled (no microphone detected).")
        return
    voice_chat_loop()

if __name__ == "__main__":
    launch_voice_ai()
