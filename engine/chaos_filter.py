# chaos_filter.py
"""
Legend-Level Chaos Filter Module
Detects highly volatile or manipulated market states using entropy,
volatility indices, and time-series noise metrics.

Created by Bahati Lusabe
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(filename='logs/security_audit_log.json', level=logging.INFO)

# === CONFIGURATION ===
WINDOW_SIZE = 50              # Number of ticks to consider
VOLATILITY_THRESHOLD = 0.012  # Custom tuned threshold for high volatility
ENTROPY_THRESHOLD = 2.4       # Tuned for Deriv synthetic index behavior

class ChaosFilter:
    # === CORE FUNCTION ===
    def is_market_chaotic(self, price_ticks):
        """
        Determines if the market is in a chaotic state.

        Args:
            price_ticks (List[float]): Recent price points from tick_collector.py

        Returns:
            (bool, dict): True if chaotic, and detailed analytics dict
        """

        if len(price_ticks) < WINDOW_SIZE:
            return False, {"reason": "Not enough data points"}

        series = np.array(price_ticks[-WINDOW_SIZE:])
        returns = np.diff(series) / series[:-1]  # percent changes

        # === 1. Volatility Calculation ===
        volatility = np.std(returns)

        # === 2. Entropy Calculation ===
        counts, bins = np.histogram(returns, bins=10)
        probs = counts / np.sum(counts)
        entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])

        # === 3. Chaos Flagging ===
        is_chaotic = volatility > VOLATILITY_THRESHOLD or entropy > ENTROPY_THRESHOLD

        # === 4. Logging if chaotic ===
        if is_chaotic:
            chaos_info = {
                "timestamp": str(datetime.utcnow()),
                "volatility": round(volatility, 5),
                "entropy": round(entropy, 5),
                "trigger": "volatility" if volatility > VOLATILITY_THRESHOLD else "entropy",
            }
            logging.info(f"[CHAOS DETECTED] {chaos_info}")
            return True, chaos_info

        return False, {
            "volatility": round(volatility, 5),
            "entropy": round(entropy, 5),
            "chaotic": False
        }

    # === OPTIONAL: Chaos Analyzer for Visualization or Strategy Layer ===
    def chaos_score(self, price_ticks):
        """
        Returns a score (0-1) for how chaotic the market is.
        Combines volatility and entropy into a scaled metric.
        """
        _, data = self.is_market_chaotic(price_ticks)
        if isinstance(data, dict) and "entropy" in data and "volatility" in data:
            norm_vol = min(data["volatility"] / VOLATILITY_THRESHOLD, 2.0)
            norm_ent = min(data["entropy"] / ENTROPY_THRESHOLD, 2.0)
            return round((norm_vol + norm_ent) / 4.0, 3)  # scaled to [0,1]
        return 0.0

    def is_chaotic(self, price_ticks):
        chaotic, _ = self.is_market_chaotic(price_ticks)
        return chaotic

    def filter(self):
        # TODO: Implement chaos filtering logic
        pass
