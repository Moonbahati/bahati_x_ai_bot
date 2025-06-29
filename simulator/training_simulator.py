
# simulator/training_simulator.py
# --- Add start_training_simulation public function ---
def start_training_simulation(epochs=30):
    simulator = TrainingSimulator(epochs=epochs)
    simulator.simulate()
    return simulator.get_log()

import json
import logging
import random
import time
from core.digit_predictor.ensemble_voter import evaluate_strategy
from engine.dna_profiler import check_dna_uniqueness
from engine.adaptive_segmentation import segment_behavior
from ai.auto_feedback_loop import feedback_evaluator
from ai.threat_prediction_ai import predict_threat  # Fixed import name
from ai.pattern_memory import recall_patterns
from ai.self_evolver import evolve_self, track_change

from integrations.deriv_ws_client import run_in_background
from core.scalper_ai import ScalperAI
from engine.market_analyzer import analyze_market_conditions
from engine.stake_manager import StakeManager
from engine.policy_enforcer import enforce_stake_limits
from engine.dna_profiler import compute_trend_strength, compute_fractal_dimension, compute_entropy, normalize_fingerprint, compute_fingerprint, classify_market_behavior
from logs.logger_manager import log_trade
from engine.risk_guardian import RiskGuardian

logger = logging.getLogger("UltraTrainingSimulator")
logging.basicConfig(level=logging.INFO)

scalper_ai = ScalperAI()
stake_manager = StakeManager()
risk_guardian = RiskGuardian()

class TrainingSimulator:
    def __init__(self, epochs=50, mutation_factor=0.03, historical_data=None):
        self.epochs = epochs
        self.mutation_factor = mutation_factor
        self.historical_data = historical_data or self._generate_synthetic_data()
        self.training_log = []

    def _generate_synthetic_data(self, num_points=1000):
        return [
            {
                "price": random.uniform(1.0, 100.0),
                "volume": random.randint(100, 10000),
                "volatility": random.uniform(0.1, 0.5),
                "sentiment": random.choice(["bullish", "bearish", "neutral"]),
            }
            for _ in range(num_points)
        ]

    def _run_single_cycle(self, data_point):
        strategy_vector = [random.uniform(0, 1) for _ in range(10)]

        base_score = evaluate_strategy(strategy_vector)
        dna_score = check_dna_uniqueness(strategy_vector)
        feedback = feedback_evaluator(strategy_vector)
        threat_prediction = predict_threat(data_point)
        segmentation = segment_behavior(data_point)
        pattern_response = recall_patterns(strategy_vector)


        intelligence_vector = evolve_self(strategy_vector)

        final_score = (
            0.25 * base_score +
            0.15 * dna_score +
            0.20 * feedback +
            0.15 * (1 - threat_prediction.get("risk_score", 0)) +
            0.10 * segmentation.get("confidence", 0) +
            0.10 * pattern_response.get("resonance", 0.5)
        )

        return {
            "strategy_vector": strategy_vector,
            "intelligence_vector": intelligence_vector,
            "final_score": round(final_score, 4),
            "threat_level": threat_prediction.get("risk_label", "unknown"),
            "segment": segmentation.get("label", "n/a")
        }

    def simulate(self):
        logger.info("🚀 Starting Training Simulation")
        for epoch in range(1, self.epochs + 1):
            data_point = random.choice(self.historical_data)
            result = self._run_single_cycle(data_point)
            self.training_log.append(result)

            if epoch % 5 == 0:
                logger.info(f"📈 Epoch {epoch}: Score={result['final_score']} | Threat={result['threat_level']} | Segment={result['segment']}")

        logger.info("✅ Simulation Complete")

    def get_log(self):
        return self.training_log

    def save_metrics(metrics, filename="metrics.json"):
        with open(filename, "w") as f:
            json.dump(metrics, f)

# Optional standalone run
if __name__ == "__main__":
    simulator = TrainingSimulator(epochs=30)
    simulator.simulate()
    log = simulator.get_log()
    print(f"🧠 Final Intelligence Snapshot (Last):\n{log[-1]}")

def process_tick(tick):
    price = tick['quote']
    features = [price]

    # 1. Predict
    prediction = scalper_ai.predict(features)

    # 2. Analyze market volatility
    volatility = analyze_market_conditions(features)

    # 3. Manage position sizing
    position_size = stake_manager.calculate_position_size(prediction, volatility)

    # 4. Enforce policy/security
    enforce_stake_limits(prediction, position_size)

    # 5. Analyze strategy behavior
    # analyze_strategy_behavior is missing; use a composite analysis instead
    dna = scalper_ai.dna
    # Example: compute trend, fractal, entropy, and classification
    trend = compute_trend_strength(dna)
    fractal = compute_fractal_dimension(dna)
    entropy = compute_entropy(dna)
    classification = classify_market_behavior(dna)
    strategy_report = {
        "trend_strength": trend,
        "fractal_dimension": fractal,
        "entropy": entropy,
        "classification": classification
    }

    # 6. Learn from the prediction
    scalper_ai.learn(features, prediction)

    # 7. Log the trade
    log_trade({
        "price": price,
        "features": features,
        "prediction": prediction,
        "position_size": position_size,
        "volatility": volatility,
        "strategy_report": strategy_report
    })

    # Track the change for self-evolution
    # You may need to define 'outcome' based on your logic
    outcome = "win"  # or "loss" or your own logic
    track_change(features, prediction, outcome, scalper_ai.weights)

    # --- Save metrics to file for dashboard ---
    metrics = {
        "profit_percent": 0,  # Replace with your real calculation
        "total_trades": 0,    # Replace with your real calculation
        "win_rate": 0,        # Replace with your real calculation
        "strategy_impacts": [],  # Replace with your real calculation
        "last_prediction": prediction,
        "last_price": price,
        "last_position_size": position_size,
        "last_volatility": volatility,
        "last_strategy_report": strategy_report,
    }
    save_metrics(metrics)

    print(f"[LIVE] Price: {price} | Prediction: {prediction} | Position: {position_size} | Volatility: {volatility}")

if __name__ == "__main__":
    print("🚀 Starting Real-Time Training Simulator...")
    run_in_background(on_tick_callback=process_tick)
    import time
    while True:
        time.sleep(5)
