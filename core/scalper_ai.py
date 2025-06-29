import time
import logging
import numpy as np
import pandas as pd
from collections import deque
from integrations.deriv_connector import DerivConnector, deriv_connector  # Make sure this is your instance
from integrations.deriv_ws_client import run_in_background, get_latest_tick, get_balance


import pickle
import csv
import json
import os
from engine.risk_guardian import RiskGuardian
from cryptography.fernet import Fernet

# --- SECURITY: Encryption key for sensitive files ---
BOT_SECRET_KEY = os.environ.get("BOT_SECRET_KEY")
if not BOT_SECRET_KEY:
    raise RuntimeError("BOT_SECRET_KEY environment variable not set. Please set it before running the bot.")
fernet = Fernet(BOT_SECRET_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegendaryUltraScalperAI")

# --- SECURITY: Load API key from environment variable ---
DERIV_TOKEN = os.environ.get("DERIV_TOKEN")
if not DERIV_TOKEN:
    raise RuntimeError("DERIV_TOKEN environment variable not set. Please set it before running the bot.")
deriv_connector = DerivConnector(DERIV_TOKEN)
risk_guardian = RiskGuardian()

class ScalperAI:
    # --- ADVANCED RISK MANAGEMENT CONFIG ---
    MAX_DAILY_LOSS = 200.0  # Example: stop trading after this loss
    MAX_CONSECUTIVE_LOSSES = 5
    CONFIDENCE_THRESHOLD = 0.6  # Only trade if consensus/confidence is above this

    def reset_risk_counters(self):
        self.daily_loss = 0.0
        self.consecutive_losses = 0
        self.last_trade_day = time.strftime('%Y-%m-%d')

    def get_strategy_impacts(self, log_path: str = "trade_log.csv") -> list[dict]:
        """
        Analyze trade_log.csv and summarize trade outcomes by strategy for dashboard analytics.
        Returns a list of dicts: [{"strategy": str, "trades": int, "wins": int, "losses": int, "win_rate": float, "total_profit": float}]
        """
        import os
        import csv
        import json
        from typing import Any
        if not os.path.exists(log_path):
            return []
        strategy_stats = {}
        try:
            # --- Decrypt and read log file ---
            with open(log_path, "rb") as f:
                encrypted = f.read()
                if encrypted:
                    decrypted = fernet.decrypt(encrypted)
                    lines = decrypted.decode().splitlines()
                    reader = csv.reader(lines)
                else:
                    reader = []
                for row in reader:
                    if len(row) < 4:
                        continue
                    # row: [epoch, quote, decision, result, weights]
                    decision: str = row[2]
                    result_str: str = row[3]
                    # Try to parse result as JSON, fallback to string
                    try:
                        result = json.loads(result_str)
                    except Exception:
                        result = {}
                    # Strategy: use decision as a proxy (or add more logic if available)
                    strategy: str = decision
                    if strategy not in strategy_stats:
                        strategy_stats[strategy] = {"trades": 0, "wins": 0, "losses": 0, "total_profit": 0.0}
                    strategy_stats[strategy]["trades"] += 1
                    # Determine win/loss
                    outcome = result.get("outcome")
                    profit = result.get("profit", 0.0)
                    try:
                        profit_f: float = float(profit)
                    except Exception:
                        profit_f = 0.0
                    if outcome == "win" or profit_f > 0:
                        strategy_stats[strategy]["wins"] += 1
                    elif outcome == "loss" or profit_f < 0:
                        strategy_stats[strategy]["losses"] += 1
                    # Sum profit
                    strategy_stats[strategy]["total_profit"] += profit_f
            # Format for dashboard
            impacts = []
            for strategy, stats in strategy_stats.items():
                trades: int = int(stats["trades"])
                wins: int = int(stats["wins"])
                losses: int = int(stats["losses"])
                total_profit: float = float(stats["total_profit"])
                win_rate: float = wins / trades if trades > 0 else 0.0
                impacts.append({
                    "strategy": strategy,
                    "trades": trades,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": round(win_rate, 3),
                    "total_profit": round(total_profit, 2)
                })
            return impacts
        except Exception as e:
            print(f"Error in get_strategy_impacts: {e}")
            return []
    def __init__(self, history_len=20):
        self.reset_risk_counters()
        self.weights = np.random.randn(11)  # Now 11 features
        self.learning_rate = 0.0001
        self.last_volatility = None
        self.last_strategy_report = None
        self.price_history = deque(maxlen=history_len)
        self.ema = None
        self.rsi_period = 14
        from engine.stake_manager import StakeManager
        self.stake_manager = StakeManager()
        self.stake_strategy = "dynamic"  # or 'martingale', etc.
        self.last_stake = 5.0
        self.last_result = 0.0
        # Self-Evolver AI modules
        from ai.self_evolver import SelfEvolver
        self.self_evolver = SelfEvolver()
        self.current_asset = "R_100"  # Default asset, can be changed by evolver
        self.strategy_pool = ["dynamic", "martingale", "fixed"]
        self.performance_history = []
        # --- Fail-Safe Mechanisms ---
        self.api_fail_count = 0
        self.api_fail_threshold = 3
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 30  # seconds
        self.api_keys = [DERIV_TOKEN]  # Add more keys as needed
        self.api_key_index = 0
        self.crash_recovery_attempts = 0
        self.max_recovery_attempts = 5

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
        # WhatsApp notification removed (module not found)

    def track_change(self, features, label, outcome, weights):
        # You can expand this to log to a file or database
        print(f"[TRACK] features={features}, label={label}, outcome={outcome}, weights={weights}")

    @property
    def dna(self):
        return self.weights.tolist()

    def rotate_api_key(self):
        self.api_key_index = (self.api_key_index + 1) % len(self.api_keys)
        new_key = self.api_keys[self.api_key_index]
        global deriv_connector
        deriv_connector = DerivConnector(new_key)
        logger.warning(f"[API] Rotated API key. Now using key index {self.api_key_index}.")

    def heartbeat_check(self):
        now = time.time()
        if now - self.last_heartbeat > self.heartbeat_interval:
            try:
                # Simple heartbeat: check balance or ping API
                _ = get_balance()
                self.last_heartbeat = now
                logger.info("[HEARTBEAT] Bot healthy.")
                return True
            except Exception as e:
                logger.error(f"[HEARTBEAT] Failed: {e}")
                return False
        return True

    def auto_reconnect(self):
        global deriv_connector
        try:
            deriv_connector.reconnect()
            logger.info("[FAILSAFE] Auto-reconnected to API.")
        except Exception as e:
            logger.error(f"[FAILSAFE] Auto-reconnect failed: {e}")

    def auto_recover(self):
        self.crash_recovery_attempts += 1
        if self.crash_recovery_attempts > self.max_recovery_attempts:
            logger.critical("[FAILSAFE] Max auto-recovery attempts reached. Manual intervention required.")
            raise RuntimeError("Max auto-recovery attempts reached.")
        logger.warning(f"[FAILSAFE] Attempting auto-recovery (attempt {self.crash_recovery_attempts})...")
        time.sleep(5)
        self.run_live()

    def run_live(self, interval=1):
        from engine.dna_profiler import classify_market_behavior
        from engine.market_analyzer import MarketAnalyzer
        from engine.chaos_filter import ChaosFilter
        from ai.rl_trainer import RLTrainer
        from ai.pattern_memory import PatternMemory
        from ai.threat_prediction_ai import ThreatPredictor
        from ai.intent_recognizer import IntentRecognizerAI
        from ai.auto_feedback_loop import AutoFeedbackLoop

        logger.info("🚀 Starting Legendary Ultra ScalperAI in LIVE mode.")
        run_in_background()
        print("Starting real-time trading loop...")
        market_analyzer = MarketAnalyzer()
        chaos_filter = ChaosFilter()
        rl_trainer = RLTrainer(state_size=11, action_size=3)
        pattern_memory = PatternMemory()
        threat_predictor = ThreatPredictor()
        intent_ai = IntentRecognizerAI()
        feedback_loop = AutoFeedbackLoop()
        predictions_history = []
        targets_history = []
        while True:
            # --- Heartbeat check ---
            if not self.heartbeat_check():
                self.api_fail_count += 1
                if self.api_fail_count >= self.api_fail_threshold:
                    logger.warning("[FAILSAFE] API fail threshold reached. Rotating API key and reconnecting...")
                    self.rotate_api_key()
                    self.auto_reconnect()
                    self.api_fail_count = 0
                time.sleep(5)
                continue
            else:
                self.api_fail_count = 0
            try:
                today = time.strftime('%Y-%m-%d')
                if today != self.last_trade_day:
                    self.reset_risk_counters()
                logger.info("🔄 Loop tick: waiting for new tick...")
                tick = get_latest_tick()
                if tick:
                    features = self.extract_features(tick)
                    # --- DNA PROFILER ---
                    dna_profile = classify_market_behavior(list(self.price_history))
                    # --- MARKET ANALYZER ---
                    tick_data = [{"quote": p} for p in list(self.price_history)]
                    market_state = market_analyzer.analyze_ticks(tick_data)
                    # --- CHAOS FILTER ---
                    is_chaotic, chaos_info = chaos_filter.is_market_chaotic(list(self.price_history))
                    # --- RL SIGNAL ---
                    rl_action_idx = rl_trainer.act(features)
                    rl_action = ["hold", "buy", "sell"][rl_action_idx]
                    # --- PATTERN MEMORY ---
                    pattern_memory.store_pattern(features)
                    pattern_volatility = pattern_memory.analyze_volatility_behavior(features)
                    # --- THREAT PREDICTION ---
                    threat_result = threat_predictor.analyze(features)
                    # --- INTENT RECOGNITION ---
                    intent_result = intent_ai.predict_intent(str(features.tolist()))
                    intent_confidence = intent_result.get("confidence", 0)
                    # --- SELF-EVOLVER: ADAPTIVE STRATEGY, STAKE, ASSET ---
                    # Collect performance for evolver
                    if hasattr(self, 'performance_history'):
                        perf_snapshot = {
                            "profit": getattr(self, 'daily_loss', 0.0),
                            "consecutive_losses": getattr(self, 'consecutive_losses', 0),
                            "confidence": intent_confidence,
                            "tick": tick,
                            "features": features.tolist(),
                            "strategy": self.stake_strategy,
                            "asset": self.current_asset
                        }
                        self.performance_history.append(perf_snapshot)
                        # Evolve every 20 trades or on poor performance
                        if len(self.performance_history) > 20 or self.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
                            evolver_result = self.self_evolver.evolve(
                                self.performance_history[-20:],
                                available_strategies=self.strategy_pool,
                                current_strategy=self.stake_strategy,
                                current_asset=self.current_asset,
                                weights=self.weights.tolist(),
                            )
                            # Update bot config if evolver suggests
                            if evolver_result:
                                if 'stake_strategy' in evolver_result:
                                    self.stake_strategy = evolver_result['stake_strategy']
                                    logger.info(f"[Self-Evolver] Switched strategy to {self.stake_strategy}")
                                if 'asset' in evolver_result:
                                    self.current_asset = evolver_result['asset']
                                    logger.info(f"[Self-Evolver] Switched asset to {self.current_asset}")
                                if 'weights' in evolver_result:
                                    self.weights = np.array(evolver_result['weights'])
                                    logger.info(f"[Self-Evolver] Updated weights via evolution.")

                    # --- STRATEGY BLENDING & CONFIDENCE ---
                    votes = [self.predict(features), self.hybrid_decision(features), rl_action]
                    analyzer_action = market_analyzer.recommend_action()
                    if analyzer_action in ["buy", "sell"]:
                        votes.append(analyzer_action)
                    # Confidence: % of votes for the majority
                    from collections import Counter
                    vote_counts = Counter(votes)
                    decision, count = vote_counts.most_common(1)[0]
                    confidence = count / len(votes)

                    # --- ADVANCED RISK & AI FILTERS ---
                    veto = False
                    veto_reasons = []
                    if is_chaotic:
                        veto = True
                        veto_reasons.append(f"Market is chaotic: {chaos_info}")
                    if market_state.get("chaotic") or analyzer_action == "AVOID_TRADE":
                        veto = True
                        veto_reasons.append("MarketAnalyzer recommends to avoid trade.")
                    if self.daily_loss <= -self.MAX_DAILY_LOSS:
                        veto = True
                        veto_reasons.append("Max daily loss reached.")
                    if self.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
                        veto = True
                        veto_reasons.append("Max consecutive losses reached.")
                    if confidence < self.CONFIDENCE_THRESHOLD:
                        veto = True
                        veto_reasons.append(f"Low consensus confidence: {confidence:.2f}")
                    if intent_confidence < 0.45:
                        veto = True
                        veto_reasons.append(f"Low intent confidence: {intent_confidence:.2f}")
                    if threat_result.get("threat_detected", False):
                        veto = True
                        veto_reasons.append(f"Threat detected (score: {threat_result.get('threat_score', 0):.2f})")
                    if pattern_volatility == "🔥 High Volatility":
                        veto = True
                        veto_reasons.append("High volatility detected.")

                    if veto:
                        logger.warning(f"Trade vetoed: {' | '.join(veto_reasons)}")
                        print(f"[VETO] Skipping trade. Reasons: {' | '.join(veto_reasons)}")
                        time.sleep(interval)
                        continue

                    # --- SMART STAKE MANAGEMENT ---
                    amount = self.stake_manager.smart_stake_decision(
                        self.stake_strategy,
                        self.last_stake,
                        self.last_result,
                        signal_strength=confidence,
                        recent_prices=list(self.price_history),
                        trade_intent=decision
                    )
                    self.last_stake = amount
                    if decision in ["buy", "sell"]:
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
                        try:
                            result = json.loads(response)
                        except Exception:
                            result = {}
                        # Update last_result for stake strategy
                        profit = result.get("profit", 0)
                        if profit > 0:
                            self.last_result = profit
                            label = 1 if decision == "buy" else -1
                            self.consecutive_losses = 0
                        else:
                            self.last_result = -self.last_stake
                            label = -1 if decision == "buy" else 1
                            self.consecutive_losses += 1
                        if "outcome" in result:
                            label = 1 if result["outcome"] == "win" else -1
                            if result["outcome"] == "win":
                                self.consecutive_losses = 0
                            else:
                                self.consecutive_losses += 1
                        # Track daily loss
                        self.daily_loss += profit if isinstance(profit, (int, float)) else 0.0
                        self.learn(features, label)
                        self.log_trade(tick, decision, response, self.weights)
                        # --- FEEDBACK LOOP ---
                        predictions_history.append(confidence)
                        targets_history.append(1 if profit > 0 else 0)
                        if len(predictions_history) > 10:
                            feedback_loop.apply_feedback(self, predictions_history[-10:], targets_history[-10:])
                        # --- FETCH AND PRINT BALANCE AFTER TRADE ---
                        balance = get_balance()
                        print(f"[Balance Update] Account balance after trade: ${balance:,.2f}")
                else:
                    print("Waiting for tick...")
                time.sleep(interval)
                if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)):
                    print("WARNING: NaN or Inf in weights! Resetting weights.")
                    self.weights = np.random.randn(len(self.weights))
            except Exception as e:
                logger.error(f"[FAILSAFE] Exception in main loop: {e}")
                self.auto_recover()

    def save_weights(self, path="scalper_weights.pkl"):
        # --- Encrypt model weights ---
        data = pickle.dumps(self.weights)
        encrypted = fernet.encrypt(data)
        with open(path, "wb") as f:
            f.write(encrypted)

    def load_weights(self, path="scalper_weights.pkl"):
        try:
            with open(path, "rb") as f:
                encrypted = f.read()
                if encrypted:
                    data = fernet.decrypt(encrypted)
                    self.weights = pickle.loads(data)
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
        log_path = "trade_log.csv"
        # Read, decrypt, and append
        rows = []
        if os.path.exists(log_path):
            with open(log_path, "rb") as f:
                encrypted = f.read()
                if encrypted:
                    decrypted = fernet.decrypt(encrypted)
                    lines = decrypted.decode().splitlines()
                    reader = csv.reader(lines)
                    rows = list(reader)
        # Append new row
        rows.append([tick['epoch'], tick['quote'], decision, result, list(weights)])
        # Encrypt and write
        output = []
        for row in rows:
            output.append(",".join(map(str, row)))
        encrypted = fernet.encrypt("\n".join(output).encode())
        with open(log_path, "wb") as f:
            f.write(encrypted)

if __name__ == "__main__":
    scalper = ScalperAI()
    scalper.load_weights()
    scalper.run_live(interval=1)  # You can adjust the interval as needed
    scalper.save_weights()