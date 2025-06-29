import logging
import numpy as np
from datetime import datetime
from ai.intent_recognizer import recognize_intent
from ai.auto_feedback_loop import adjust_risk_profile
from engine.policy_enforcer import enforce_stake_limits
from core.emotion_manager import get_risk_emotion_score

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegendaryStakeManager")

class StakeManager:
    def martingale_stake(self, last_stake: float, last_result: float) -> float:
        """
        Martingale: Double stake after a loss, reset after a win.
        """
        if last_result < 0:
            return float(min(last_stake * 2, self.capital * self.max_exposure))
        else:
            return float(self.min_stake)

    def anti_martingale_stake(self, last_stake: float, last_result: float) -> float:
        """
        Anti-Martingale: Increase stake after a win, reset after a loss.
        """
        if last_result > 0:
            return float(min(last_stake * 2, self.capital * self.max_exposure))
        else:
            return float(self.min_stake)

    def soros_stake(self, last_stake: float, last_result: float, profit_stack: int = 2) -> float:
        """
        Soros/Compound: Stack profits for a set number of rounds, then reset.
        """
        if not hasattr(self, '_soros_counter'):
            self._soros_counter = 0
        if not hasattr(self, '_soros_base'):
            self._soros_base = self.min_stake
        if last_result > 0 and self._soros_counter < profit_stack:
            self._soros_counter += 1
            return float(min(last_stake + last_result, self.capital * self.max_exposure))
        else:
            self._soros_counter = 0
            return float(self.min_stake)

    def smart_stake_decision(self, strategy: str, last_stake: float, last_result: float, **kwargs) -> float:
        """
        Smart trade management: choose stake based on selected strategy.
        strategy: 'dynamic', 'martingale', 'anti-martingale', 'soros'
        """
        if strategy == 'martingale':
            return self.martingale_stake(last_stake, last_result)
        elif strategy == 'anti-martingale':
            return self.anti_martingale_stake(last_stake, last_result)
        elif strategy == 'soros':
            return self.soros_stake(last_stake, last_result, kwargs.get('profit_stack', 2))
        else:
            # Default to dynamic AI-based stake
            return float(self.compute_stake(kwargs.get('signal_strength', 1.0), kwargs.get('recent_prices', [self.capital]), kwargs.get('trade_intent', 'neutral')))
    def __init__(self, initial_capital=10000.0, max_exposure=0.3, min_stake=5.0,
                 emotion_factor=True, volatility_weight=0.2, memory_weight=0.3):
        self.capital = initial_capital
        self.max_exposure = max_exposure
        self.min_stake = min_stake
        self.volatility_weight = volatility_weight
        self.memory_weight = memory_weight
        self.trade_history = []
        self.emotion_factor = emotion_factor

    def _calculate_volatility(self, recent_prices):
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns)
        logger.debug(f"Volatility: {volatility:.6f}")
        return volatility

    def _calculate_risk_emotion_modifier(self):
        if not self.emotion_factor:
            return 1.0
        emotion_score = get_risk_emotion_score()
        modifier = 1.0 - (emotion_score * 0.5)
        logger.debug(f"Emotion Risk Modifier: {modifier:.2f}")
        return modifier

    def _compute_memory_bias(self):
        if not self.trade_history:
            return 1.0
        recent_trades = self.trade_history[-5:]
        wins = sum(1 for t in recent_trades if t['profit'] > 0)
        loss_bias = (5 - wins) / 5
        memory_adjustment = 1.0 - (self.memory_weight * loss_bias)
        logger.debug(f"Memory Adjustment: {memory_adjustment:.2f}")
        return memory_adjustment

    def compute_stake(self, signal_strength, recent_prices, trade_intent="neutral"):
        """
        Determines how much capital to stake on a trade.
        - Uses AI intent + market volatility + emotion + trade memory
        """
        volatility = self._calculate_volatility(recent_prices)
        base_stake = self.capital * self.max_exposure * signal_strength
        base_stake *= (1 - self.volatility_weight * volatility)

        # AI-Informed modifiers
        intent_multiplier = recognize_intent(trade_intent)
        emotion_modifier = self._calculate_risk_emotion_modifier()
        memory_modifier = self._compute_memory_bias()

        raw_stake = base_stake * intent_multiplier * emotion_modifier * memory_modifier
        stake = max(self.min_stake, min(raw_stake, self.capital * self.max_exposure))

        logger.info(f"💰 Stake calculated: {stake:.2f} | Intent: {trade_intent}, Volatility: {volatility:.4f}")
        return stake

    def execute_trade(self, stake_amount, asset_symbol):
        """
        Simulates a trade and logs stake usage.
        """
        logger.info(f"🚀 Executing trade on {asset_symbol} with stake: ${stake_amount:.2f}")
        simulated_profit = simulate_market_impact(stake_amount, asset_symbol)
        self.capital += simulated_profit - stake_amount
        self.trade_history.append({
            "timestamp": datetime.now(),
            "asset": asset_symbol,
            "stake": stake_amount,
            "profit": simulated_profit - stake_amount
        })
        adjust_risk_profile(self.trade_history[-1])
        logger.info(f"📊 Post-trade capital: ${self.capital:.2f} | Profit: {simulated_profit - stake_amount:.2f}")

    def auto_stake_decision(self, signal_strength, prices, symbol, intent):
        stake = self.compute_stake(signal_strength, prices, trade_intent=intent)

        # Enforce policy from the AI governor
        if enforce_stake_limits(stake, symbol):
            self.execute_trade(stake, symbol)
        else:
            logger.warning(f"🚫 Stake rejected by policy: {stake:.2f} on {symbol}")

    def emergency_throttle(self):
        """
        Triggered when anomalies or breaches occur.
        """
        self.max_exposure *= 0.5
        logger.warning(f"🛡️ EMERGENCY THROTTLE ACTIVATED! New max exposure: {self.max_exposure:.2f}")

    def some_function(self, *args, **kwargs):
        from engine.scalper_ai import evaluate_micro_trade_opportunity
        # ...use it here...

def get_current_risk_profile(*args, **kwargs):
    # TODO: Implement risk profile logic
    return {"risk_level": 0.05}

def simulate_market_impact(stake_amount, asset_symbol):
    """
    Dummy market impact simulation for testing stake strategies.
    Returns a simulated profit or loss (randomized for now).
    """
    import random
    # Simulate win/loss: 50% win, 50% loss, win returns 95% of stake, loss loses all stake
    if random.random() < 0.5:
        return stake_amount * 1.95  # win: get stake back + 95% profit
    else:
        return 0  # loss: lose all stake
