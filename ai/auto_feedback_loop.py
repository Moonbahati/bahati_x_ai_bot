import numpy as np
import logging
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegendaryAutoFeedbackLoop")

class AutoFeedbackLoop:
    def __init__(self, learning_rate=0.05, feedback_mode="dynamic", decay=0.98):
        self.learning_rate = learning_rate
        self.feedback_mode = feedback_mode  # ["static", "dynamic"]
        self.decay = decay
        self.history = []
        self.global_bias = 0.0
        self.last_feedback_score = None
        self.stability_score = 0.0

    def evaluate_performance(self, predictions, targets):
        predictions = np.array(predictions)
        targets = np.array(targets)

        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        stability = np.std(predictions - targets)

        composite_score = 1 / (1 + mse + mae + stability + abs(self.global_bias))

        self.last_feedback_score = round(composite_score, 5)
        self.history.append({
            "timestamp": time.time(),
            "mse": mse,
            "mae": mae,
            "stability": stability,
            "bias": self.global_bias,
            "feedback_score": composite_score
        })

        logger.info(f"🧠 Feedback Score: {composite_score:.5f} | MSE: {mse:.4f}, MAE: {mae:.4f}, Bias: {self.global_bias:.4f}")
        return composite_score

    def tune_bias(self, predictions, targets):
        predictions = np.array(predictions)
        targets = np.array(targets)
        error = np.mean(predictions - targets)

        self.global_bias -= self.learning_rate * error
        self.global_bias *= self.decay

    def apply_feedback(self, model, predictions, targets):
        score = self.evaluate_performance(predictions, targets)
        self.tune_bias(predictions, targets)

        if self.feedback_mode == "dynamic":
            adjustment_factor = min(1.0, max(0.0, score))
            model.learning_rate *= (1 + self.learning_rate * (adjustment_factor - 0.5))
            model.learning_rate = max(1e-5, min(model.learning_rate, 1.0))

            logger.info(f"🎯 Adjusted model learning rate to {model.learning_rate:.5f}")

        return score

    def get_latest_feedback(self):
        if self.history:
            return self.history[-1]
        return {}

    def is_converging(self, recent=5, threshold=0.0005):
        if len(self.history) < recent:
            return False

        recent_scores = [h["feedback_score"] for h in self.history[-recent:]]
        diffs = [abs(recent_scores[i] - recent_scores[i-1]) for i in range(1, len(recent_scores))]
        avg_change = np.mean(diffs)

        logger.info(f"🔍 Avg feedback delta over last {recent}: {avg_change:.6f}")
        return avg_change < threshold

    def explain_feedback(self):
        if not self.history:
            return "⚠️ No feedback history available."

        latest = self.history[-1]
        return {
            "🧪 Mean Squared Error": round(latest["mse"], 4),
            "📏 Mean Absolute Error": round(latest["mae"], 4),
            "📉 Model Bias": round(latest["bias"], 4),
            "💡 Feedback Score": round(latest["feedback_score"], 5)
        }

def feedback_evaluator(data):
    # Example placeholder logic
    return "feedback processed"

def adjust_risk_profile(*args, **kwargs):
    # TODO: Implement or proxy to feedback logic if needed
    pass
