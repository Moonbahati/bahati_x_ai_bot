# integrations/api_token_manager.py

import os
import time
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta
from typing import Optional

# Ultra-Secure Quantum-Resistant Token Manager
class APITokenManager:
    def __init__(self, secret_key_env="AI_BOT_SECRET_KEY"):
        self.secret_key = os.getenv(secret_key_env, self._generate_secret_key())
        self.tokens = {}
        self.expiry_minutes = 60  # configurable token lifespan

    def _generate_secret_key(self) -> str:
        key = secrets.token_urlsafe(64)
        print(f"[SECURITY] New secret key generated: {key}")
        return key

    def generate_token(self, client_id: str) -> str:
        payload = {
            "client_id": client_id,
            "timestamp": time.time(),
            "expires_at": (datetime.utcnow() + timedelta(minutes=self.expiry_minutes)).timestamp()
        }
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(self.secret_key.encode(), payload_str.encode(), hashlib.sha512).hexdigest()
        token = f"{payload_str}.{signature}"
        self.tokens[client_id] = {"token": token, "expires_at": payload["expires_at"]}
        return token

    def validate_token(self, token: str) -> bool:
        try:
            payload_str, signature = token.rsplit('.', 1)
            expected_signature = hmac.new(self.secret_key.encode(), payload_str.encode(), hashlib.sha512).hexdigest()
            if not hmac.compare_digest(signature, expected_signature):
                print("[SECURITY] Signature mismatch.")
                return False

            payload = json.loads(payload_str)
            if datetime.utcnow().timestamp() > payload["expires_at"]:
                print("[SECURITY] Token expired.")
                return False

            return True
        except Exception as e:
            print(f"[ERROR] Token validation failed: {e}")
            return False

    def invalidate_token(self, client_id: str) -> bool:
        if client_id in self.tokens:
            del self.tokens[client_id]
            return True
        return False

    def refresh_token(self, client_id: str) -> Optional[str]:
        if client_id in self.tokens:
            return self.generate_token(client_id)
        return None

    def list_active_tokens(self) -> dict:
        now = datetime.utcnow().timestamp()
        return {
            cid: tdata for cid, tdata in self.tokens.items()
            if tdata["expires_at"] > now
        }


# Example usage
if __name__ == "__main__":
    manager = APITokenManager()
    test_client_id = "alpha_node_123"

    token = manager.generate_token(test_client_id)
    print(f"🔐 Generated Token:\n{token}\n")

    is_valid = manager.validate_token(token)
    print(f"✅ Token valid? {is_valid}")

    refreshed = manager.refresh_token(test_client_id)
    print(f"♻️ Refreshed Token:\n{refreshed}")
