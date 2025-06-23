# engine/tick_collector.py
import json
import time
import threading
import requests
import websocket
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UltraTickCollector")

class TickCollector:
    def __init__(self, api_token, symbol="R_100", use_websocket=True):
        self.api_token = api_token
        self.symbol = symbol
        self.use_websocket = use_websocket
        self.ws_url = "wss://ws.deriv.com/websockets/v3"
        self.rest_url = f"https://api.deriv.com/api/ticks/{symbol}"
        self.connected = False
        self.ws = None
        self.callback = None

    def _fetch_tick_rest(self):
        """Fallback to REST API."""
        headers = {"Authorization": f"Bearer {self.api_token}"}
        try:
            response = requests.get(self.rest_url, headers=headers)
            if response.ok:
                return response.json()
            else:
                logger.error("❌ REST tick fetch failed.")
        except Exception as e:
            logger.error(f"🔥 REST tick fetch error: {e}")
        return None

    def _on_message(self, ws, message):
        data = json.loads(message)
        if "tick" in data:
            tick_data = data["tick"]
            logger.debug(f"📥 Live Tick: {tick_data}")
            if self.callback:
                self.callback(tick_data)

    def _on_error(self, ws, error):
        logger.error(f"⚠️ WebSocket Error: {error}")
        self.connected = False

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"🔌 WebSocket Closed: {close_msg}")
        self.connected = False

    def _on_open(self, ws):
        logger.info("🔗 WebSocket connection established.")
        self.connected = True
        payload = {
            "ticks": self.symbol,
            "subscribe": 1
        }
        ws.send(json.dumps(payload))

    def _start_websocket(self):
        """Start WebSocket tick stream."""
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        ws_thread.start()

    def start_stream(self, callback, fallback_interval=2):
        """
        Start collecting tick data using WebSocket with REST fallback.
        """
        self.callback = callback
        logger.info("🚀 Starting Hybrid Tick Stream...")

        if self.use_websocket:
            self._start_websocket()
            time.sleep(3)
            if not self.connected:
                logger.warning("⚠️ WebSocket failed. Falling back to REST polling.")
                self.use_websocket = False

        if not self.use_websocket:
            while True:
                tick = self._fetch_tick_rest()
                if tick:
                    logger.debug("📡 Tick received via REST")
                    if self.callback:
                        self.callback(tick)
                time.sleep(fallback_interval)

    def stop(self):
        if self.ws:
            logger.info("🛑 Stopping WebSocket stream.")
            self.ws.close()

def fetch_tick_data(api_token, symbol="R_100"):
    collector = TickCollector(api_token=api_token, symbol=symbol, use_websocket=False)
    # Implement logic to fetch a single tick or batch of ticks here
    # For now, just return a placeholder
    return {"price": 100.0, "symbol": symbol, "timestamp": time.time()}

