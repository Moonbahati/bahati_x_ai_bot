import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import logging
import joblib

from ai.fraud_detection_ai import detect_fraud_pattern
from ai.auto_feedback_loop import feedback_evaluator
from engine.dna_profiler import check_dna_uniqueness

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegendaryUltraRF")

class LegendaryRandomForest:
    def __init__(self, n_estimators=150, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.pipeline = self._build_pipeline()
        logger.info("🌲 Legendary Ultra Random Forest model initialized.")

    def _build_pipeline(self):
        base_rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        boosted = AdaBoostClassifier(base_rf, n_estimators=10)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("boosted_rf", boosted)
        ])

    def preprocess(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Simple sanitization: replace NaNs and clip extreme values
        X = np.nan_to_num(X)
        X = np.clip(X, -9999, 9999)
        return X

    def train(self, X, y, test_size=0.2):
        logger.info("🔍 Preprocessing & Splitting Dataset")
        X = self.preprocess(X)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=self.random_state)

        logger.info("🚀 Training the hybrid RandomForest + AdaBoost model")
        self.pipeline.fit(X_train, y_train)

        predictions = self.pipeline.predict(X_val)
        acc = accuracy_score(y_val, predictions)
        logger.info(f"✅ Validation Accuracy: {acc:.4f}")
        logger.info(classification_report(y_val, predictions))

    def predict(self, X):
        X = self.preprocess(X)

        base_prediction = self.pipeline.predict(X)
        feedback_mod = np.array([feedback_evaluator(x) for x in X])
        fraud_penalty = np.array([0.0 if not detect_fraud_pattern(x) else 1 for x in X])
        dna_score = np.array([check_dna_uniqueness(x) for x in X])

        enhanced_prediction = []
        for i, pred in enumerate(base_prediction):
            score = pred
            # Decision correction based on fraud, uniqueness, and feedback
            if fraud_penalty[i] > 0:
                score = 0  # force reject
            elif dna_score[i] < 0.5:
                score = 0  # low uniqueness, possible mimic
            elif feedback_mod[i] < 0.3:
                score = 0  # poor performance memory
            enhanced_prediction.append(score)
        
        return np.array(enhanced_prediction)

    def evaluate(self, X, y_true):
        preds = self.predict(X)
        acc = accuracy_score(y_true, preds)
        logger.info("📊 Evaluation Metrics")
        logger.info(classification_report(y_true, preds))
        return acc

    def save(self, path="legendary_rf_model.pkl"):
        joblib.dump(self.pipeline, path)
        logger.info(f"💾 Model saved at: {path}")

    def load(self, path="legendary_rf_model.pkl"):
        self.pipeline = joblib.load(path)
        logger.info(f"📥 Model loaded from: {path}")
