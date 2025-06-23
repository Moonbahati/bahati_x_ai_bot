import numpy as np
import pandas as pd
import logging
from engine.dna_profiler import generate_market_dna
from engine.chaos_filter import ChaosFilter
from core.strategy_evolver import recommend_strategy
from datetime import datetime
from scipy.stats import kurtosis, skew

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HybridUltraMarketAnalyzer")


class MarketAnalyzer:
    def __init__(self, adaptive_memory=True, noise_filter_threshold=0.0005):
        self.dna_signature = None
        self.market_state = {}
        self.noise_filter_threshold = noise_filter_threshold
        self.adaptive_memory = adaptive_memory
        self.historical_snapshots = []

    def _apply_noise_filter(self, df):
        df['price_change'] = df['quote'].diff()
        df = df[df['price_change'].abs() > self.noise_filter_threshold]
        return df

    def _extract_features(self, df):
        df['momentum'] = df['quote'].diff()
        df['rolling_mean'] = df['quote'].rolling(window=5).mean()
        df['rolling_std'] = df['quote'].rolling(window=5).std()

        volatility = df['momentum'].std()
        trend_strength = df['rolling_mean'].diff().mean()
        return volatility, trend_strength

    def _enhance_dna_with_quant_signals(self, df):
        kurt = kurtosis(df['quote'])
        skewness = skew(df['quote'])
        signal_time = datetime.utcnow().isoformat()

        return {
            "kurtosis": kurt,
            "skew": skewness,
            "timestamp": signal_time
        }

    def analyze_ticks(self, tick_data):
        df = pd.DataFrame(tick_data)

        if df.empty or 'quote' not in df:
            logger.warning("⚠️ MarketAnalyzer received invalid tick data.")
            return {}

        df = self._apply_noise_filter(df)
        if df.empty:
            logger.warning("📉 Noise filter eliminated all data.")
            return {"status": "NO_SIGNAL"}

        volatility, trend_strength = self._extract_features(df)
        dna = generate_market_dna(df)
        chaos_filter = ChaosFilter()
        chaos = chaos_filter.is_chaotic(df['quote'])
        quant_enhancement = self._enhance_dna_with_quant_signals(df)

        self.market_state = {
            "volatility": volatility,
            "trend_strength": trend_strength,
            "chaotic": chaos,
            "dna": dna,
            "enhanced": quant_enhancement,
            "snapshot_time": quant_enhancement["timestamp"]
        }

        if self.adaptive_memory:
            self.historical_snapshots.append(self.market_state)

        logger.info(f"📊 MarketState: {self.market_state}")
        return self.market_state

    def recommend_action(self):
        if not self.market_state:
            return "WAIT_FOR_SIGNAL"

        if self.market_state.get("chaotic"):
            return "AVOID_TRADE"

        strategy = recommend_strategy(
            dna=self.market_state.get("dna"),
            volatility=self.market_state.get("volatility"),
            trend=self.market_state.get("trend_strength")
        )

        logger.info(f"🧠 Recommended Strategy: {strategy}")
        return strategy

    def last_snapshot(self):
        return self.historical_snapshots[-1] if self.historical_snapshots else None

    def clear_memory(self):
        self.historical_snapshots.clear()


def analyze_market_conditions(*args, **kwargs):
    # TODO: Implement market condition analysis logic
    return {}
