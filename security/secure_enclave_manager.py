import os
import json
import base64
import logging
import secrets
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend
from ai.threat_prediction_ai import evaluate_enclave_threats
from ai.self_evolver import self_tune_enclave_security

logger = logging.getLogger("SecureEnclave")
logging.basicConfig(level=logging.INFO)

class SecureEnclave:
    def __init__(self, enclave_id):
        self.enclave_id = enclave_id
        self.active = False
        self.secret_key = self._generate_key()
        self.nonce_length = 12  # Standard for AES-GCM
        logger.info(f"[🛡️] Enclave {enclave_id} initialized with quantum-safe key.")

    def _generate_key(self):
        entropy = secrets.token_bytes(64)
        hkdf = HKDF(
            algorithm=hashes.SHA3_256(),
            length=32,
            salt=None,
            info=b"secure-enclave-key",
            backend=default_backend()
        )
        return hkdf.derive(entropy)

    def start(self):
        self.active = True
        logger.info(f"[✅] Enclave {self.enclave_id} is now active.")

    def stop(self):
        self.active = False
        logger.warning(f"[⛔] Enclave {self.enclave_id} has been shut down.")

    def encrypt_data(self, plain_data):
        if not self.active:
            raise RuntimeError("Enclave must be active to encrypt.")

        aesgcm = AESGCM(self.secret_key)
        nonce = secrets.token_bytes(self.nonce_length)
        encrypted = aesgcm.encrypt(nonce, plain_data.encode(), None)
        encrypted_payload = {
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(encrypted).decode()
        }
        return json.dumps(encrypted_payload)

    def decrypt_data(self, encrypted_json):
        if not self.active:
            raise RuntimeError("Enclave must be active to decrypt.")

        payload = json.loads(encrypted_json)
        nonce = base64.b64decode(payload["nonce"])
        ciphertext = base64.b64decode(payload["ciphertext"])

        aesgcm = AESGCM(self.secret_key)
        decrypted = aesgcm.decrypt(nonce, ciphertext, None)
        return decrypted.decode()

    def perform_security_check(self, input_metadata):
        risk_score = evaluate_enclave_threats(input_metadata)
        self_tune_enclave_security(self.enclave_id, risk_score)

        if risk_score > 0.75:
            logger.critical(f"[🚨] High threat level detected in enclave {self.enclave_id}.")
            self.stop()
        elif risk_score > 0.5:
            logger.warning(f"[⚠️] Medium threat level in enclave {self.enclave_id}. Reinforcing key...")
            self.secret_key = self._generate_key()
        else:
            logger.info(f"[🧠] Enclave {self.enclave_id} passed security check with low risk.")

    def export_public_manifest(self):
        """
        Export public enclave status info for monitoring.
        """
        return {
            "enclave_id": self.enclave_id,
            "status": "active" if self.active else "inactive",
            "security_level": "ultra_advanced",
            "nonce_length": self.nonce_length
        }

# Example usage
if __name__ == "__main__":
    enclave = SecureEnclave("grayluck-core")
    enclave.start()

    secret = "Bahati's quantum vault secret"
    encrypted = enclave.encrypt_data(secret)
    print(f"🔐 Encrypted: {encrypted}")

    decrypted = enclave.decrypt_data(encrypted)
    print(f"🔓 Decrypted: {decrypted}")

    enclave.perform_security_check({"source": "external_probe", "entropy": 0.92})
