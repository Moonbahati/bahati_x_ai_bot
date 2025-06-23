import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntentRecognizerAI")

class IntentRecognizerAI:
    def __init__(self):
        logger.info("🚀 Initializing Hybrid Legendary Intent Recognizer AI...")

        # Load transformer model
        model_name = "microsoft/deberta-v3-small"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, return_all_scores=True)

        # Intent labels and encoder
        self.intent_labels = [
            "BuySignal", "SellSignal", "QueryPortfolio", "MarketCondition",
            "RiskAssessment", "WithdrawFunds", "InjectCapital", "Alert"
        ]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.intent_labels)

        logger.info("🎯 Intent categories loaded and model initialized.")

    def predict_intent(self, message: str) -> dict:
        logger.debug(f"🔍 Analyzing intent for message: {message}")
        predictions = self.classifier(message)[0]

        intents = {self.intent_labels[i]: round(pred["score"], 4) for i, pred in enumerate(predictions)}
        ranked = sorted(intents.items(), key=lambda x: x[1], reverse=True)

        top_intent, confidence = ranked[0]
        logger.info(f"✅ Detected Intent: {top_intent} ({confidence:.4f})")
        return {
            "top_intent": top_intent,
            "confidence": confidence,
            "all_intents": ranked
        }

    def feedback_loop(self, message: str, true_label: str):
        logger.info(f"🔁 Feedback received. Message: '{message}' | True Intent: {true_label}")
        return {"status": "logged", "true_label_encoded": self.label_encoder.transform([true_label])[0]}

    def detect_anomalous_intent(self, message: str) -> bool:
        intent_data = self.predict_intent(message)
        low_conf = intent_data["confidence"] < 0.45
        ambiguous = abs(intent_data["confidence"] - intent_data["all_intents"][1][1]) < 0.1

        if low_conf or ambiguous:
            logger.warning("⚠️ Suspicious or ambiguous intent detected!")
            return True
        return False

    def escalate_if_threat(self, message: str) -> str:
        if self.detect_anomalous_intent(message):
            logger.critical(f"🚨 Escalation Triggered for Message: {message}")
            return "escalate_security_protocol"
        return "intent_normal"

# 🔧 Global Singleton Instance
_intent_ai = IntentRecognizerAI()

# ✅ FIXED: Add this function to resolve import errors
def analyze_intent(input_text: str) -> float:
    """
    Simple callable function for modules expecting a function named 'analyze_intent'
    Returns confidence score for top predicted intent.
    """
    result = _intent_ai.predict_intent(input_text)
    return result["confidence"]

# Optional utility wrappers
def assess_trade_intent(message: str) -> bool:
    result = _intent_ai.predict_intent(message)
    return result["top_intent"] in ["BuySignal", "SellSignal"]

def recognize_intent(message: str) -> str:
    return _intent_ai.predict_intent(message)["top_intent"]

def recognize_trading_intent(message: str) -> dict:
    return _intent_ai.predict_intent(message)
