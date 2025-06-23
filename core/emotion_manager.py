import logging
import numpy as np
from collections import deque
from sklearn.preprocessing import normalize

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegendaryEmotionManager")

class EmotionManager:
    def __init__(self, buffer_size=50, decay_rate=0.9, volatility_threshold=0.7):
        self.emotion_buffer = deque(maxlen=buffer_size)
        self.current_state = {"happy": 0.25, "neutral": 0.5, "angry": 0.25}
        self.decay_rate = decay_rate
        self.volatility_threshold = volatility_threshold
        self.state_history = []
        self.last_reaction = None

    def update_emotion(self, input_signal):
        """
        input_signal: Dict of emotions with confidence e.g., {"happy": 0.6, "neutral": 0.3, "angry": 0.1}
        """
        normalized_signal = self._normalize_emotion_signal(input_signal)
        self._decay_current_state()
        for emotion, val in normalized_signal.items():
            self.current_state[emotion] = self.current_state.get(emotion, 0) + val

        self._normalize_current_state()
        self.emotion_buffer.append(self.current_state.copy())
        self.state_history.append(self.current_state.copy())

        logger.info(f"🧠 Updated Emotion State: {self.current_state}")
        return self.current_state

    def _normalize_emotion_signal(self, signal):
        emotions = list(signal.keys())
        values = np.array([signal[e] for e in emotions])
        normed = normalize([values], norm='l1')[0]
        return dict(zip(emotions, normed))

    def _normalize_current_state(self):
        values = np.array(list(self.current_state.values()))
        normed = normalize([values], norm='l1')[0]
        for i, key in enumerate(self.current_state.keys()):
            self.current_state[key] = normed[i]

    def _decay_current_state(self):
        for emotion in self.current_state:
            self.current_state[emotion] *= self.decay_rate

    def predict_reaction(self):
        """
        Predict system reaction based on dominant emotion state.
        """
        dominant_emotion = max(self.current_state, key=self.current_state.get)
        reaction_map = {
            "happy": "reinforce_positive_loop",
            "neutral": "maintain_state",
            "angry": "trigger_adaptive_response"
        }
        reaction = reaction_map.get(dominant_emotion, "maintain_state")
        self.last_reaction = reaction
        logger.info(f"⚡ Predicted Reaction: {reaction}")
        return reaction

    def detect_volatility(self):
        """
        Measures emotional volatility over the buffer period.
        """
        if len(self.emotion_buffer) < 2:
            return False

        diffs = []
        for i in range(1, len(self.emotion_buffer)):
            prev = np.array(list(self.emotion_buffer[i-1].values()))
            curr = np.array(list(self.emotion_buffer[i].values()))
            diffs.append(np.linalg.norm(curr - prev))

        volatility = np.mean(diffs)
        is_volatile = volatility > self.volatility_threshold
        logger.info(f"📊 Emotional Volatility: {volatility:.4f} | Threshold: {self.volatility_threshold} | Volatile: {is_volatile}")
        return is_volatile

    def inject_feedback(self, feedback_type):
        """
        Inject external feedback into the emotional state.
        feedback_type: "reward", "penalty", or "neutral"
        """
        impact_map = {
            "reward": {"happy": 0.3, "neutral": 0.1, "angry": -0.2},
            "penalty": {"happy": -0.2, "neutral": -0.1, "angry": 0.3},
            "neutral": {"neutral": 0.05}
        }

        adjustments = impact_map.get(feedback_type, {})
        for emotion, change in adjustments.items():
            self.current_state[emotion] = max(0.0, self.current_state.get(emotion, 0) + change)

        self._normalize_current_state()
        logger.info(f"🧭 Feedback injected: {feedback_type} → New State: {self.current_state}")
        return self.current_state

    def get_emotional_summary(self):
        dominant = max(self.current_state, key=self.current_state.get)
        volatility = self.detect_volatility()
        return {
            "dominant_emotion": dominant,
            "volatility": volatility,
            "last_reaction": self.last_reaction
        }

def analyze_emotion(text):
    # Your emotion analysis logic here
    return "neutral"

def get_risk_emotion_score(*args, **kwargs):
    # TODO: Implement or proxy to emotion analysis logic if needed
    return 0.0
