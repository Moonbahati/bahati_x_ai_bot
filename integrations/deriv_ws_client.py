import websocket
import json
import socket
import threading
import logging
import time

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
# --- SECURITY: Load API key from environment variable ---
DERIV_TOKEN = os.environ.get("DERIV_TOKEN")
if not DERIV_TOKEN:
    raise RuntimeError("DERIV_TOKEN environment variable not set. Please set it before running the bot.")

def on_message(ws, message):
    data = json.loads(message)
    if data.get("msg_type") == "tick":
        tick = data["tick"]
        save_latest_tick(tick)
        logger.info(f"Latest tick: {tick}")

def on_open(ws):
    ws.send(json.dumps({"authorize": DERIV_TOKEN}))
    ws.send(json.dumps({"ticks": "R_100"}))

def on_error(ws, error):
    logger.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.info(f"WebSocket closed: {close_status_code} {close_msg}")

def run_ws():
    ws = websocket.WebSocketApp(
        "wss://ws.derivws.com/websockets/v3?app_id=1089",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

def run_in_background(on_tick_callback=None):
    def _on_message(ws, message):
        data = json.loads(message)
        if data.get("msg_type") == "tick":
            tick = data["tick"]
            save_latest_tick(tick)
            if on_tick_callback:
                on_tick_callback(tick)
            logger.info(f"Latest tick in background: {tick}")

    ws = websocket.WebSocketApp(
        "wss://ws.derivws.com/websockets/v3?app_id=1089",
        on_open=on_open,
        on_message=_on_message,
        on_error=on_error,
        on_close=on_close
    )
    threading.Thread(target=ws.run_forever, daemon=True).start()

def get_latest_tick():
    try:
        with open("/tmp/latest_tick.json") as f:
            return json.load(f)
    except Exception:
        return None

def save_latest_tick(tick):
    with open("/tmp/latest_tick.json", "w") as f:
        json.dump(tick, f)

def get_balance():
    ws = websocket.create_connection("wss://ws.derivws.com/websockets/v3?app_id=1089")
    ws.send(json.dumps({"authorize": DERIV_TOKEN}))
    response = ws.recv()
    data = json.loads(response)
    ws.close()
    print("DEBUG: Balance response:", data)
    # Check for error in response
    if "error" in data:
        logger.error(f"Error in balance response: {data['error']}")
        return 0.0  # or handle as appropriate for your app
    # Check for balance in authorize response
    if "authorize" in data and "balance" in data["authorize"]:
        return data["authorize"]["balance"]
    # If you want to support the balance endpoint as well, add:
    if "balance" in data and "balance" in data["balance"]:
        return data["balance"]["balance"]
    logger.error(f"Balance key not found in response: {data}")
    return 0.0  # or handle as appropriate for your app

def get_account_profit_percent():
    starting_balance = 9500.00  # Set your actual starting balance
    current_balance = get_balance()
    if starting_balance == 0:
        return 0.0
    return ((current_balance - starting_balance) / starting_balance) * 100

def get_total_trades():
    ws = websocket.create_connection("wss://ws.derivws.com/websockets/v3?app_id=1089")
    ws.send(json.dumps({"authorize": DERIV_TOKEN}))
    ws.recv()  # Receive authorize response
    ws.send(json.dumps({"statement": 1, "limit": 100}))
    response = ws.recv()
    ws.close()
    data = json.loads(response)
    print("DEBUG: Statement response:", data)
    if "statement" in data and "transactions" in data["statement"]:
        trades = [tx for tx in data["statement"]["transactions"] if tx.get("action_type") in ("buy", "sell")]
        return len(trades)
    else:
        print("No statement in response, possible error:", data.get("error"))
        return 0

def get_win_rate():
    ws = websocket.create_connection("wss://ws.derivws.com/websockets/v3?app_id=1089")
    ws.send(json.dumps({"authorize": DERIV_TOKEN}))
    ws.recv()  # Receive authorize response
    ws.send(json.dumps({"statement": 1, "limit": 100}))
    response = ws.recv()
    ws.close()
    data = json.loads(response)
    print("DEBUG: Statement response:", data)
    if "statement" in data and "transactions" in data["statement"]:
        trades = [tx for tx in data["statement"]["transactions"] if tx.get("action_type") in ("buy", "sell")]
        if not trades:
            return 0.0
        wins = [tx for tx in trades if float(tx.get("amount", 0)) > 0]
        return (len(wins) / len(trades)) * 100
    else:
        print("No statement in response, possible error:", data.get("error"))
        return 0.0

def get_strategy_impacts():
    # Analyze logs/trade_log.csv for strategy impact summary
    import os
    import pandas as pd
    log_path = "logs/trade_log.csv"
    if not os.path.exists(log_path):
        return []
    try:
        df = pd.read_csv(log_path, header=None, names=["timestamp", "trade", "response"])
        # For demo: count trades by contract_type (CALL/PUT)
        def parse_trade(x):
            try:
                return json.loads(x) if isinstance(x, str) else {}
            except Exception:
                return {}
        df["trade_json"] = df["trade"].apply(parse_trade)
        def extract_strategy(t):
            try:
                return t.get("parameters", {}).get("contract_type", "UNKNOWN")
            except Exception:
                return "UNKNOWN"
        df["strategy"] = df["trade_json"].apply(extract_strategy)
        impact = df["strategy"].value_counts().reset_index()
        impact.columns = ["strategy", "impact"]
        return impact.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error in get_strategy_impacts: {e}")
        return []

__all__ = [
    "run_in_background",
    "get_latest_tick",
    "save_latest_tick",
    "get_balance",
    "get_account_profit_percent",
    "get_total_trades",
    "get_win_rate",
    "get_strategy_impacts",
]

if __name__ == "__main__":
    try:
        print("Google IP:", socket.gethostbyname('www.google.com'))
        print("Deriv DNS Info:", socket.getaddrinfo('ws.derivws.com', 443))
    except socket.gaierror as e:
        print("DNS resolution error:", e)

    print("Balance:", get_balance())
    print("Profit %:", get_account_profit_percent())
    print("Total trades:", get_total_trades())
    print("Win rate:", get_win_rate())
    print("Strategy impacts:", get_strategy_impacts())