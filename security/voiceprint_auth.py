import os
import hashlib
import base64
import logging
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.mixture import GaussianMixture
from ai.self_evolver import ai_self_optimize
from ai.fraud_detection_ai import detect_voice_fraud
from ai.pattern_memory import learn_voice_pattern

logger = logging.getLogger("VoiceprintAuth")
logging.basicConfig(level=logging.INFO)

class VoiceprintAuth:
    def __init__(self):
        self.user_voice_db = {}
        self.voice_model = GaussianMixture(n_components=2, covariance_type='full')
        self.threshold = 0.15  # Lower = stricter

    def _extract_features(self, voice_signal):
        """
        Simulate voice feature extraction. Replace with real MFCC in production.
        """
        np.random.seed(hash(voice_signal) % 10000)
        return np.random.rand(128)  # Simulated 128-dim feature vector

    def _hash_voice(self, features):
        """
        Quantum-safe voiceprint hash using SHA3-512.
        """
        feature_str = ",".join([f"{x:.8f}" for x in features])
        return hashlib.sha3_512(feature_str.encode()).hexdigest()

    def register_user(self, username, voice_sample):
        features = self._extract_features(voice_sample)
        voice_hash = self._hash_voice(features)
        self.user_voice_db[username] = {
            "features": features,
            "hash": voice_hash,
        }
        learn_voice_pattern(username, features)
        logger.info(f"[✅] Voiceprint registered for {username}.")

    def authenticate_user(self, username, voice_attempt):
        if username not in self.user_voice_db:
            logger.warning(f"[❌] Unknown user: {username}")
            return False

        stored = self.user_voice_db[username]
        attempt_features = self._extract_features(voice_attempt)
        score = 1 - cosine(stored["features"], attempt_features)

        fraud_score = detect_voice_fraud(username, attempt_features)

        if score >= 1 - self.threshold and fraud_score < 0.4:
            ai_self_optimize("voice_auth", username, score)
            logger.info(f"[🔐] Auth success: Voiceprint match for {username}")
            return True
        else:
            logger.warning(f"[🚫] Auth failed: Voice mismatch or fraud risk ({fraud_score:.2f})")
            return False

    def export_voiceprint(self, username):
        """
        Export hashed voiceprint (e.g., for federated identity use).
        """
        if username not in self.user_voice_db:
            return None
        return {
            "username": username,
            "voice_hash": self.user_voice_db[username]["hash"],
        }

    def delete_user(self, username):
        if username in self.user_voice_db:
            del self.user_voice_db[username]
            logger.info(f"[🗑️] Voiceprint deleted for {username}")
            return True
        return False

# Singleton instance for global system use
voice_authenticator = VoiceprintAuth()

# Demo usage
if __name__ == "__main__":
    voice_authenticator.register_user("bahati", "my_unique_voice_sample_789")
    auth_result = voice_authenticator.authenticate_user("bahati", "my_unique_voice_sample_789")
    print("Authentication passed?" if auth_result else "Authentication failed.")
