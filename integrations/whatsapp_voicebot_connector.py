# integrations/whatsapp_voicebot_connector.py

import os
import requests
import base64
import json
from datetime import datetime
from ai.ai_voice_chat import AIVoiceChat
from ai.intent_recognizer import IntentRecognizer
from security.voiceprint_auth import VoiceprintAuth

class WhatsAppVoiceBotConnector:
    def __init__(self, whatsapp_api_url: str, auth_token: str, ai_voice: AIVoiceChat):
        self.api_url = whatsapp_api_url
        self.auth_token = auth_token
        self.voice_engine = ai_voice
        self.intent_engine = IntentRecognizer()
        self.voice_auth = VoiceprintAuth()
        self.headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }

    def _send_text_message(self, to: str, message: str):
        payload = {
            "to": to,
            "type": "text",
            "text": {
                "body": message
            }
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.ok:
            print(f"✅ Message sent to {to}")
        else:
            print(f"❌ Failed to send message: {response.status_code} | {response.text}")

    def _send_voice_message(self, to: str, audio_base64: str):
        payload = {
            "to": to,
            "type": "audio",
            "audio": {
                "base64": audio_base64
            }
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.ok:
            print(f"✅ Voice message sent to {to}")
        else:
            print(f"❌ Failed to send voice message: {response.status_code} | {response.text}")

    def process_incoming_voice(self, user_id: str, audio_file_path: str):
        if not self.voice_auth.authenticate(audio_file_path, user_id):
            self._send_text_message(user_id, "🚫 Voice authentication failed. Please try again.")
            return

        recognized_text = self.voice_engine.transcribe(audio_file_path)
        if not recognized_text:
            self._send_text_message(user_id, "⚠️ Could not understand your message.")
            return

        intent = self.intent_engine.recognize(recognized_text)
        response = self.voice_engine.respond(intent, recognized_text)

        if response["type"] == "text":
            self._send_text_message(user_id, response["data"])
        elif response["type"] == "audio":
            audio_base64 = base64.b64encode(response["data"]).decode("utf-8")
            self._send_voice_message(user_id, audio_base64)

    def broadcast_announcement(self, user_list: list, announcement_text: str):
        for user in user_list:
            self._send_text_message(user, f"📢 ANNOUNCEMENT: {announcement_text}")


# Example usage
if __name__ == "__main__":
    whatsapp_url = os.getenv("WHATSAPP_API_URL", "https://api.chat-api.com/instanceXXXX/message")
    token = os.getenv("WHATSAPP_AUTH_TOKEN", "your-token")
    
    ai_voice_system = AIVoiceChat()
    connector = WhatsAppVoiceBotConnector(whatsapp_url, token, ai_voice_system)

    print("📡 Simulating incoming audio...")
    test_user_id = "user-0001"
    test_audio_path = "samples/user-0001_voice_message.wav"
    connector.process_incoming_voice(test_user_id, test_audio_path)
