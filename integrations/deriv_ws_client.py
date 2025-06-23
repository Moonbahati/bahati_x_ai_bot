import websocket
import json
import socket

DERIV_TOKEN = "HtInIuKmiplAINX"  # Replace with your real Deriv demo token

def on_message(ws, message):
    print("on_message called")
    data = json.loads(message)
    print("Received:", json.dumps(data, indent=2))

def on_open(ws):
    print("Connection opened!")
    ws.send(json.dumps({"authorize": DERIV_TOKEN}))
    ws.send(json.dumps({"ticks": "R_100"}))

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed:", close_status_code, close_msg)

def run_ws():
    ws = websocket.WebSocketApp(
        "wss://ws.derivws.com/websockets/v3?app_id=1089",  # You can replace 1089 with your own app_id
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

if __name__ == "__main__":
    try:
        print("Google IP:", socket.gethostbyname('www.google.com'))
        print("Deriv DNS Info:", socket.getaddrinfo('ws.derivws.com', 443))
    except socket.gaierror as e:
        print("DNS resolution error:", e)

    run_ws()