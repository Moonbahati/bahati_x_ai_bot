import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegendaryFraudDetectionAI")

class FraudDetectionAI:
    def __init__(self, contamination=0.05, n_estimators=100, use_lof=True):
        self.contamination = contamination
        self.use_lof = use_lof
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
        self.lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
        self.trained = False
        self.last_auc = None

    def fit(self, X):
        X = np.array(X)
        self.model.fit(X)
        if self.use_lof:
            self.lof.fit(X)
        self.trained = True
        logger.info(f"🔐 FraudDetectionAI trained on {len(X)} samples.")

    def detect(self, X):
        if not self.trained:
            logger.warning("⚠️ FraudDetectionAI is not trained yet.")
            return []

        X = np.array(X)
        iso_scores = -self.model.decision_function(X)
        lof_scores = -self.lof.decision_function(X) if self.use_lof else np.zeros_like(iso_scores)

        combined_scores = 0.6 * iso_scores + 0.4 * lof_scores
        threshold = np.percentile(combined_scores, 100 * (1 - self.contamination))

        result = [score > threshold for score in combined_scores]
        return result

    def evaluate(self, X, y_true):
        """Evaluate model on labeled fraud/non-fraud data"""
        if not self.trained:
            return None
        X = np.array(X)
        scores = -self.model.decision_function(X)
        auc = roc_auc_score(y_true, scores)
        self.last_auc = auc
        logger.info(f"📊 FraudDetectionAI AUC = {auc:.4f}")
        return auc

    def explain_detection(self, X, index):
        """Explain why a pattern was flagged (simplified SHAP-style logic)"""
        if not self.trained:
            return "Model not trained."

        pattern = np.array(X[index])
        feature_contributions = pattern - np.mean(X, axis=0)
        importance = np.abs(feature_contributions)
        explanation = {
            f"feature_{i}": round(score, 4)
            for i, score in enumerate(importance)
        }
        return explanation

    def adapt_to_drift(self, new_data, drift_threshold=0.15):
        """Re-train model if data drift is detected"""
        if not self.trained:
            return self.fit(new_data)

        old_mean = self.model.decision_function(new_data).mean()
        self.model.fit(new_data)
        new_mean = self.model.decision_function(new_data).mean()

        drift_score = abs(old_mean - new_mean)
        if drift_score > drift_threshold:
            logger.info("⚠️ Drift detected! Re-training FraudDetectionAI.")
            self.fit(new_data)
        else:
            logger.info("✅ No significant drift detected.")

    def is_fraudulent_pattern(self, pattern):
        """Single-pattern fraud check for fast inference mode"""
        if not self.trained:
            logger.warning("⚠️ FraudDetectionAI is not trained yet.")
            return False

        pattern = np.array(pattern).reshape(1, -1)
        score = -self.model.decision_function(pattern)[0]
        threshold = np.percentile(-self.model.decision_function([p for p, _ in self.model.estimators_samples_]), 100 * (1 - self.contamination))
        return score > threshold

def detect_fraud_pattern(data):
    # Example placeholder logic
    return False

def fetch_recent_fraud_flags():
    # TODO: Implement logic to fetch recent fraud flags
    return []
