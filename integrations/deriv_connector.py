# integrations/deriv_connector.py

import os
import json
import requests
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import websocket

class DerivConnector:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.endpoint = "wss://ws.deriv.com/websockets/v3"
        self.session = None
        self.trade_log = []
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

    def _send_request(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            response = requests.post("https://api.deriv.com/api/v1", json=payload, headers=self.headers)
            if response.ok:
                return response.json()
            else:
                print("⚠️ API Error:", response.status_code, response.text)
        except Exception as e:
            print(f"❌ Connection error: {e}")
        return None

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        payload = {
            "authorize": self.api_token
        }
        return self._send_request(payload)

    def get_asset_index(self) -> Optional[Dict[str, Any]]:
        payload = {
            "asset_index": 1
        }
        return self._send_request(payload)

    def place_trade(self, symbol: str, duration: int, amount: float, contract_type: str = "CALL") -> Optional[Dict[str, Any]]:
        payload = {
            "buy": 1,
            "price": amount,
            "parameters": {
                "amount": amount,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": duration,
                "duration_unit": "m",
                "symbol": symbol
            }
        }
        result = self._send_request(payload)
        if result:
            self._log_trade(result)
        return result

    def _log_trade(self, trade_result: Dict[str, Any]):
        timestamp = datetime.utcnow().isoformat()
        entry = {
            "timestamp": timestamp,
            "trade": trade_result
        }
        self.trade_log.append(entry)
        with open("logs/trade_log.csv", "a") as log_file:
            log_file.write(f"{timestamp},{json.dumps(trade_result)}\n")

    def monitor_balance(self, interval_sec: int = 10):
        def loop():
            while True:
                info = self.get_account_info()
                if info:
                    balance = info.get("balance", {}).get("balance")
                    print(f"[{datetime.utcnow()}] 💰 Balance: ${balance}")
                threading.Event().wait(interval_sec)
        threading.Thread(target=loop, daemon=True).start()

    def on_message(self, ws, message):
        data = json.loads(message)
        print("Received:", data)
        # Here you can update your bot's state or Streamlit dashboard

    def on_open(self, ws):
        # Authorize
        ws.send(json.dumps({"authorize": self.api_token}))
        # Subscribe to ticks for R_50 (change symbol as needed)
        ws.send(json.dumps({"ticks": "R_50"}))

    def run_ws(self):
        ws = websocket.WebSocketApp(
            "wss://ws.deriv.com/websockets/v3",
            on_open=self.on_open,
            on_message=self.on_message
        )
        ws.run_forever()


# Sample usage (for testing)
if __name__ == "__main__":
    TEST_TOKEN = os.getenv("HtInIuKmiplAINX", "your-test-token")
    deriv = DerivConnector(api_token=TEST_TOKEN)

    print("🔍 Fetching account info...")
    account_info = deriv.get_account_info()
    print(json.dumps(account_info, indent=2))

    print("📊 Getting asset index...")
    assets = deriv.get_asset_index()
    print(json.dumps(assets, indent=2))

    print("🚀 Placing sample trade...")
    result = deriv.place_trade(symbol="R_50", duration=1, amount=1.0)
    print(json.dumps(result, indent=2))

    threading.Thread(target=deriv.run_ws).start()
