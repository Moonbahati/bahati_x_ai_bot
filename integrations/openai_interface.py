# integrations/openai_interface.py

import os
import openai
import time
import json
from integrations.api_token_manager import APITokenManager
from typing import Optional, Dict

# Load OpenAI API Key securely
openai.api_key = os.getenv("OPENAI_API_KEY", "your-fallback-api-key")

# Ultra Secure Token Layer
token_manager = APITokenManager()

class OpenAIInterface:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token_cache: Dict[str, str] = {}

    def _verify_client(self, client_token: str) -> bool:
        return token_manager.validate_token(client_token)

    def chat(self, prompt: str, client_token: str, user_id: Optional[str] = None) -> dict:
        if not self._verify_client(client_token):
            return {"error": "Invalid or expired token. Access denied."}

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an advanced AI trading assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                user=user_id or "anonymous"
            )

            reply = response.choices[0].message.content.strip()
            return {
                "status": "success",
                "reply": reply,
                "timestamp": time.time()
            }

        except openai.error.OpenAIError as e:
            return {
                "status": "error",
                "message": f"OpenAI API failed: {str(e)}"
            }

    def summarize(self, long_text: str, client_token: str) -> dict:
        summary_prompt = f"Summarize the following in simple terms:\n\n{long_text}"
        return self.chat(summary_prompt, client_token)

    def extract_insights(self, raw_data: str, client_token: str) -> dict:
        insight_prompt = f"Analyze this raw financial data and give insights:\n\n{raw_data}"
        return self.chat(insight_prompt, client_token)

    def generate_strategic_report(self, topic: str, client_token: str) -> dict:
        strategy_prompt = (
            f"Generate a professional strategic report about '{topic}' with actionable insights, "
            f"risks, predictions, and optimal responses."
        )
        return self.chat(strategy_prompt, client_token)

# Testing logic (can be safely removed in production)
if __name__ == "__main__":
    dummy_token = token_manager.generate_token("test_user_001")
    ai = OpenAIInterface()

    result = ai.chat("What is the future of AI in trading?", dummy_token)
    print("🤖 AI Response:\n", json.dumps(result, indent=4))
