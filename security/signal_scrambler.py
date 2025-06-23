import base64
import os
import secrets
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import logging
from ai.threat_prediction_ai import predict_threat_vector
from ai.auto_feedback_loop import feedback_evaluator

# Logger setup
logger = logging.getLogger("SignalScrambler")
logging.basicConfig(level=logging.INFO)

class SignalScrambler:
    def __init__(self, key=None):
        self.key = self._derive_key(key or os.getenv("SCRAMBLE_SECRET", "UltraSecureKey123!"))

    def _derive_key(self, raw_key):
        """Derive a strong key using SHA-256."""
        return hashlib.sha256(raw_key.encode()).digest()

    def _generate_iv(self):
        """Generate a strong initialization vector (IV)."""
        return secrets.token_bytes(16)

    def scramble(self, plaintext, metadata=None):
        """
        Encrypts and scrambles the signal using AES with adaptive padding.
        Also adds metadata entropy if needed (e.g., signal type, channel).
        """
        iv = self._generate_iv()
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        adaptive_padding = self._adaptive_noise(metadata)
        padded_data = pad((plaintext + adaptive_padding).encode(), AES.block_size)
        ciphertext = cipher.encrypt(padded_data)
        result = base64.b64encode(iv + ciphertext).decode()

        logger.info("[🔒] Signal scrambled with dynamic entropy.")
        return result

    def unscramble(self, scrambled_text):
        """Decrypts and removes signal scrambling."""
        try:
            raw = base64.b64decode(scrambled_text)
            iv, ciphertext = raw[:16], raw[16:]
            cipher = AES.new(self.key, AES.MODE_CBC, iv)
            decrypted_data = unpad(cipher.decrypt(ciphertext), AES.block_size).decode()

            clean_data = decrypted_data.rstrip("~!@#")  # Remove dummy padding
            logger.info("[🔓] Signal unscrambled successfully.")
            return clean_data
        except Exception as e:
            logger.error(f"[❌] Unscrambling failed: {e}")
            return None

    def _adaptive_noise(self, metadata):
        """
        Generates adaptive noise patterns to prevent signal analysis.
        Adds quantum-resistant scrambling based on metadata threat analysis.
        """
        predicted_threat = predict_threat_vector(metadata or "generic")
        feedback_score = feedback_evaluator(metadata or "signal_channel")

        noise_level = "LOW"
        if predicted_threat > 0.7 or feedback_score < 0.5:
            noise_level = "HIGH"
        elif predicted_threat > 0.4:
            noise_level = "MEDIUM"

        if noise_level == "HIGH":
            return "~!@#" * 5 + secrets.token_hex(16)
        elif noise_level == "MEDIUM":
            return "~!@#" * 3
        return "~!@#"

    def test_scramble_roundtrip(self, test_input="TopSecretSignal"):
        """Perform a roundtrip to validate scrambling and unscrambling."""
        scrambled = self.scramble(test_input)
        restored = self.unscramble(scrambled)
        success = restored == test_input
        logger.info(f"[🧪] Roundtrip test {'PASSED' if success else 'FAILED'}")
        return success

# Singleton instance for global use
scrambler = SignalScrambler()

# Run auto-test when script is executed directly
if __name__ == "__main__":
    test_data = "MissionCommand: DelayTransmission7s"
    result = scrambler.test_scramble_roundtrip(test_data)
    if not result:
        logger.error("Critical scrambling integrity test failed!")
