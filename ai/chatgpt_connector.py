import openai
import logging
from datetime import datetime
from core.emotion_manager import EmotionManager
from ai.auto_feedback_loop import feedback_evaluator

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegendaryChatGPTConnector")

class ChatGPTConnector:
    def __init__(self, api_key, model="gpt-4o", max_context_tokens=4096, stream_mode=True):
        openai.api_key = api_key
        self.model = model
        self.stream_mode = stream_mode
        self.context = []
        self.max_context_tokens = max_context_tokens
        self.emotion_manager = EmotionManager()
        self.fallback_enabled = True

    def _prune_context(self):
        while self._estimate_token_usage() > self.max_context_tokens:
            if self.context:
                self.context.pop(0)

    def _estimate_token_usage(self):
        return sum(len(msg["content"].split()) for msg in self.context) * 1.5

    def _append_to_context(self, role, content):
        self.context.append({"role": role, "content": content})
        self._prune_context()

    def chat(self, user_input):
        logger.info(f"🗨️ User: {user_input}")
        self._append_to_context("user", user_input)

        try:
            if self.stream_mode:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=self.context,
                    stream=True
                )
                output = ""
                for chunk in response:
                    delta = chunk['choices'][0]['delta'].get('content', '')
                    print(delta, end='', flush=True)
                    output += delta
            else:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=self.context
                )
                output = response['choices'][0]['message']['content']
                print(output)

            self._append_to_context("assistant", output)
            self._analyze_feedback(output)
            return output

        except Exception as e:
            logger.error(f"🚨 ChatGPT API Error: {e}")
            if self.fallback_enabled:
                return self._offline_fallback(user_input)
            return "⚠️ Unable to process your request at the moment."

    def _offline_fallback(self, user_input):
        # Basic fallback logic (non-AI)
        logger.warning("🧭 Activating fallback response engine.")
        fallback_response = f"I'm currently offline. You said: '{user_input}'. Let's talk soon!"
        self._append_to_context("assistant", fallback_response)
        return fallback_response

    def _analyze_feedback(self, response):
        # Use AI feedback loop
        feedback_score = feedback_evaluator(response)
        emotion_adjustment = "reward" if feedback_score > 0.6 else "penalty"
        self.emotion_manager.inject_feedback(emotion_adjustment)

        logger.info(f"🔁 Feedback: {feedback_score:.2f} | Emotion → {emotion_adjustment}")

    def get_context_summary(self):
        summary = {
            "total_messages": len(self.context),
            "current_dominant_emotion": self.emotion_manager.get_emotional_summary()["dominant_emotion"],
            "last_exchange": self.context[-1] if self.context else {}
        }
        return summary

def ask_gpt(prompt):
    # ...your code to call GPT...
    pass
