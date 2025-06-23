import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import logging

from ai.fraud_detection_ai import detect_fraud_pattern
from ai.auto_feedback_loop import feedback_evaluator
from engine.dna_profiler import check_dna_uniqueness

# Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegendaryUltraLogistic")

class LegendaryLogisticModel:
    def __init__(self, penalty='l2', C=1.0, max_iter=1000, solver='lbfgs', random_state=42):
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self.pipeline = self._build_pipeline()
        logger.info("🚀 Legendary Ultra Logistic Regression initialized")

    def _build_pipeline(self):
        model = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=self.random_state
        )
        return Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', model)
        ])

    def preprocess(self, X):
        X = np.nan_to_num(np.array(X))
        X = np.clip(X, -9999, 9999)
        return X

    def train(self, X, y, test_size=0.2):
        logger.info("🧪 Splitting and training dataset")
        X = self.preprocess(X)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        self.pipeline.fit(X_train, y_train)

        preds = self.pipeline.predict(X_val)
        acc = accuracy_score(y_val, preds)
        logger.info(f"✅ Validation Accuracy: {acc:.4f}")
        logger.info(classification_report(y_val, preds))

    def predict(self, X):
        X = self.preprocess(X)
        base_preds = self.pipeline.predict(X)
        enhanced_preds = []

        for i, ind in enumerate(X):
            fraud_flag = detect_fraud_pattern(ind)
            uniqueness_score = check_dna_uniqueness(ind)
            feedback = feedback_evaluator(ind)

            # Hybrid Enhancement Layer
            if fraud_flag or uniqueness_score < 0.3 or feedback < 0.4:
                enhanced_preds.append(0)  # reject
            else:
                enhanced_preds.append(base_preds[i])

        return np.array(enhanced_preds)

    def evaluate(self, X, y_true):
        preds = self.predict(X)
        acc = accuracy_score(y_true, preds)
        logger.info("📊 Evaluation Report:")
        logger.info(classification_report(y_true, preds))
        return acc

    def explain_decision(self, input_data):
        input_data = self.preprocess(input_data)
        proba = self.pipeline.predict_proba(input_data)
        coef = self.pipeline.named_steps['logistic'].coef_
        explanation = input_data @ coef.T
        return explanation, proba

    def save(self, path="legendary_logistic_model.pkl"):
        joblib.dump(self.pipeline, path)
        logger.info(f"💾 Model saved to: {path}")

    def load(self, path="legendary_logistic_model.pkl"):
        self.pipeline = joblib.load(path)
        logger.info(f"📥 Model loaded from: {path}")
