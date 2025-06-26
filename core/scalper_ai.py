import time
import logging
import numpy as np
import pandas as pd
from collections import deque
from integrations.deriv_connector import DerivConnector, deriv_connector  # Make sure this is your instance
from integrations.deriv_ws_client import run_in_background, get_latest_tick
from gui.whatsapp_alerts import send_whatsapp_message
import pickle
import csv
import json
from engine.risk_guardian import RiskGuardian

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegendaryUltraScalperAI")

DERIV_TOKEN = "HtInIuKmiplAINX"  # Replace with your real token or load from env
deriv_connector = DerivConnector(DERIV_TOKEN)
risk_guardian = RiskGuardian()

class ScalperAI:
    def __init__(self, history_len=20):
        self.weights = np.random.randn(11)  # Now 11 features
        self.learning_rate = 0.0001
        self.last_volatility = None
        self.last_strategy_report = None
        self.price_history = deque(maxlen=history_len)
        self.ema = None
        self.rsi_period = 14

    def extract_features(self, tick):
        price = tick.get('quote', 0)
        self.price_history.append(price)

        sma = np.mean(self.price_history) if len(self.price_history) > 1 else price
        if self.ema is None:
            self.ema = price
        else:
            alpha = 2 / (len(self.price_history) + 1)
            self.ema = alpha * price + (1 - alpha) * self.ema
        rsi = self.compute_rsi()
        volatility = np.std(self.price_history) if len(self.price_history) > 1 else 0
        self.last_volatility = volatility
        moving_average = sma
        upper_band, lower_band = self.compute_bollinger_bands()
        momentum = self.compute_momentum()
        macd, macd_signal = self.compute_macd()

        # --- PROPER NORMALIZATION ---
        # Use the first price as a base for normalization
        base_price = self.price_history[0] if len(self.price_history) > 0 else 1000

        features = np.array([
            (price - base_price) / base_price,             # Relative price change
            volatility / base_price,                       # Relative volatility
            (moving_average - base_price) / base_price,    # Relative SMA
            (self.ema - base_price) / base_price,          # Relative EMA
            rsi / 100,                                     # 0-1
            len(self.price_history) / 100,                 # 0-1 or small
            (upper_band - base_price) / base_price,        # Relative upper band
            (lower_band - base_price) / base_price,        # Relative lower band
            momentum / base_price,                          # Relative momentum
            macd,                                           # MACD value
            macd_signal                                     # MACD signal line value
        ])

        # Check for NaN/Inf and replace with 0
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print("WARNING: NaN or Inf in features!", features)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def compute_rsi(self):
        if len(self.price_history) < self.rsi_period + 1:
            return 50  # Neutral if not enough data
        prices = np.array(self.price_history)
        deltas = np.diff(prices)
        ups = deltas.clip(min=0)
        downs = -deltas.clip(max=0)
        avg_gain = np.mean(ups[-self.rsi_period:])
        avg_loss = np.mean(downs[-self.rsi_period:])
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def compute_bollinger_bands(self, n=20, k=2):
        if len(self.price_history) < n:
            return 0, 0  # Not enough data
        prices = np.array(self.price_history)[-n:]
        sma = np.mean(prices)
        std = np.std(prices)
        upper = sma + k * std
        lower = sma - k * std
        return upper, lower

    def compute_macd(self, short=12, long=26, signal=9):
        prices = np.array(self.price_history)
        if len(prices) < long:
            return 0, 0
        ema_short = pd.Series(prices).ewm(span=short, adjust=False).mean().iloc[-1]
        ema_long = pd.Series(prices).ewm(span=long, adjust=False).mean().iloc[-1]
        macd = ema_short - ema_long
        signal_line = pd.Series(prices).ewm(span=signal, adjust=False).mean().iloc[-1]
        return macd, signal_line

    def compute_momentum(self, n=10):
        if len(self.price_history) < n + 1:
            return 0
        prices = np.array(self.price_history)
        return prices[-1] - prices[-n-1]

    def predict(self, features):
        score = np.dot(self.weights, features)
        if score > 0.5:
            logger.info("Signal: BUY")
            return "buy"
        elif score < -0.5:
            logger.info("Signal: SELL")
            return "sell"
        else:
            logger.info("Signal: HOLD")
            return "hold"

    def learn(self, features, label):
        label_map = {"buy": 1, "sell": -1, "hold": 0}
        if isinstance(label, str):
            label = label_map.get(label, 0)
        prediction = np.dot(self.weights, features)
        error = label - prediction
        self.weights += self.learning_rate * error * features
        self.weights = np.clip(self.weights, -1000, 1000)  # Clipping weights
        logger.info(f"Learning step: label={label}, error={error:.4f}, new_weights={self.weights}")
        outcome = "win" if (label == 1 and prediction > 0) or (label == -1 and prediction < 0) else "loss"
        self.track_change(features, label, outcome, self.weights)

        # WhatsApp notification for every trade outcome
        send_whatsapp_message(
            f"LegendaryUltraScalperAI Trade Alert:\n"
            f"Outcome: {outcome.upper()}\n"
            f"Label: {label}\n"
            f"Prediction: {prediction:.4f}\n"
            f"Weights (preview): {self.weights[:3]}"
        )

    def track_change(self, features, label, outcome, weights):
        # You can expand this to log to a file or database
        print(f"[TRACK] features={features}, label={label}, outcome={outcome}, weights={weights}")

    @property
    def dna(self):
        return self.weights.tolist()

    def run_live(self, interval=1):
        logger.info("🚀 Starting Legendary Ultra ScalperAI in LIVE mode.")
        run_in_background()
        print("Starting real-time trading loop...")
        while True:
            logger.info("🔄 Loop tick: waiting for new tick...")
            tick = get_latest_tick()
            if tick:
                features = self.extract_features(tick)
                decision = self.predict(features)
                amount = 1.0  # Define your trade amount
                if decision in ["buy", "sell"]:
                    # Execute trade using the deriv_connector instance
                    if decision == "buy":
                        if risk_guardian.is_trade_allowed(amount):
                            deriv_connector.place_trade_over_r_100(amount=amount, duration=1)
                        else:
                            logger.warning("Trade blocked by risk management.")
                    elif decision == "sell":
                        if risk_guardian.is_trade_allowed(amount):
                            deriv_connector.place_trade_under_r_100(amount=amount, duration=1)
                        else:
                            logger.warning("Trade blocked by risk management.")

                    last_trade = deriv_connector.trade_log[-1]
                    response = last_trade["response"]
                    print(f"Trade executed: {decision} at {tick['quote']}")
                    # Parse response as needed for learning and WhatsApp alerts
                    # Suppose result["profit"] > 0 means win, < 0 means loss
                    try:
                        result = json.loads(response)
                    except Exception:
                        result = {}

                    if result.get("profit", 0) > 0:
                        label = 1 if decision == "buy" else -1
                    else:
                        label = -1 if decision == "buy" else 1

                    if "outcome" in result:
                        label = 1 if result["outcome"] == "win" else -1
                    self.learn(features, label)
                    self.log_trade(tick, decision, response, self.weights)
                    send_whatsapp_message(f"Trade executed: {decision} at {tick['quote']}")
            else:
                print("Waiting for tick...")
            time.sleep(interval)
            if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)):
                print("WARNING: NaN or Inf in weights! Resetting weights.")
                self.weights = np.random.randn(len(self.weights))

    def save_weights(self, path="scalper_weights.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.weights, f)

    def load_weights(self, path="scalper_weights.pkl"):
        try:
            with open(path, "rb") as f:
                self.weights = pickle.load(f)
        except FileNotFoundError:
            pass  # Start fresh if no weights file

    def hybrid_decision(self, features):
        votes = []
        votes.append(self.predict(features))
        if len(self.price_history) > 10:
            sma_short = np.mean(list(self.price_history)[-5:])
            sma_long = np.mean(list(self.price_history)[-10:])
            if sma_short > sma_long:
                votes.append("buy")
            elif sma_short < sma_long:
                votes.append("sell")
            else:
                votes.append("hold")
        rsi = features[4] * 100
        if rsi > 70:
            votes.append("sell")
        elif rsi < 30:
            votes.append("buy")
        else:
            votes.append("hold")
        return max(set(votes), key=votes.count)

    def log_trade(self, tick, decision, result, weights):
        with open("trade_log.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([tick['epoch'], tick['quote'], decision, result, list(weights)])

if __name__ == "__main__":
    scalper = ScalperAI()
    scalper.load_weights()
    scalper.run_live(interval=1)  # You can adjust the interval as needed
    scalper.save_weights()