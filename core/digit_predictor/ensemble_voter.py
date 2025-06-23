import numpy as np
import logging

from core.digit_predictor.logistic_model import LegendaryLogisticModel
from core.digit_predictor.lstm_model import LegendaryLSTM
from core.digit_predictor.random_forest import LegendaryRandomForest

from ai.fraud_detection_ai import detect_fraud_pattern
from ai.auto_feedback_loop import feedback_evaluator
from engine.dna_profiler import check_dna_uniqueness
from core.emotion_manager import analyze_emotion
from ai.pattern_memory import PatternMemory

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegendaryUltraEnsemble")

class LegendaryEnsembleVoter:
    def __init__(self):
        self.logistic = LegendaryLogisticModel()
        self.lstm = LegendaryLSTM()
        self.rf = LegendaryRandomForest()

        # Tuned weights (auto-tunable in future)
        self.weights = {
            "logistic": 0.30,
            "lstm": 0.35,
            "rf": 0.35
        }

        self.memory = PatternMemory()

    def train_all(self, X, y):
        logger.info("📚 Training all base models...")
        self.logistic.train(X, y)
        self.lstm.train(X, y)
        self.rf.train(X, y)

    def _confidence_weight(self, individual):
        dna = check_dna_uniqueness(individual)
        feedback = feedback_evaluator(individual)
        emotion_score = analyze_emotion(individual)

        fraud_penalty = 0.5 if detect_fraud_pattern(individual) else 0.0

        confidence = (dna * 0.4) + (feedback * 0.3) + (emotion_score * 0.3) - fraud_penalty
        return max(0, confidence)

    def predict(self, X):
        X = np.nan_to_num(np.array(X))

        preds_log = self.logistic.predict(X)
        preds_lstm = self.lstm.predict(X)
        preds_rf = self.rf.predict(X)

        final_preds = []

        for i in range(len(X)):
            individual = X[i].reshape(1, -1)

            votes = {
                "logistic": preds_log[i],
                "lstm": preds_lstm[i],
                "rf": preds_rf[i]
            }

            vote_weight = {
                model: self.weights[model] * self._confidence_weight(individual)
                for model in votes
            }

            score_map = {}
            for model, pred in votes.items():
                score_map[pred] = score_map.get(pred, 0) + vote_weight[model]

            final_decision = max(score_map, key=score_map.get)
            self.memory.store_pattern(individual, final_decision)
            final_preds.append(final_decision)

        return np.array(final_preds)

    def evaluate_strategy(self, data):
        return self.predict(data)

def evaluate_strategy(data):
    voter = LegendaryEnsembleVoter()
    return voter.evaluate_strategy(data)
