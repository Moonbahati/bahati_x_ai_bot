import logging
import random
import numpy as np
from core.emotion_manager import EmotionManager
from ai.intent_recognizer import IntentRecognizerAI
from ai.fraud_detection_ai import detect_fraud_pattern
from engine.dna_profiler import check_dna_uniqueness
from ai.rl_trainer import RLTrainer

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NegotiationEngine")

class NegotiationEngine:
    def __init__(self):
        logger.info("⚖️ Initializing Legendary Ultra Negotiation Engine...")
        self.emotion_ai = EmotionManager()
        self.intent_ai = IntentRecognizerAI()
        self.rl_agent = RLTrader()  # Reinforcement learning-based agent
        self.negotiation_memory = []

        # Negotiation personality archetypes
        self.archetypes = ["Aggressive", "Analytical", "Empathetic", "Tactical", "Neutral"]

    def _assign_archetype(self, dna_signature):
        score = check_dna_uniqueness(dna_signature)
        if score > 0.8:
            return "Analytical"
        elif score > 0.6:
            return "Empathetic"
        elif score > 0.4:
            return "Tactical"
        else:
            return random.choice(self.archetypes)

    def _simulate_concession_curve(self, tension, emotion):
        base_curve = np.tanh(tension / 2)
        if emotion in ['angry', 'anxious']:
            return max(0.2, base_curve * 0.75)
        elif emotion in ['happy', 'calm']:
            return min(0.95, base_curve * 1.2)
        return base_curve

    def negotiate(self, message: str, agent_dna=None) -> dict:
        logger.info(f"🗨️ Incoming negotiation message: '{message}'")

        # Intent & Emotion
        intent_data = self.intent_ai.predict_intent(message)
        emotion = self.emotion_ai.analyze_emotion(message)
        intent = intent_data['top_intent']

        # Fraud filter
        if detect_fraud_pattern(message):
            logger.critical("⚠️ Fraudulent negotiation attempt detected!")
            return {"status": "rejected", "reason": "fraud_detected"}

        # Assign archetype
        archetype = self._assign_archetype(agent_dna or message)
        logger.info(f"🧬 Assigned Personality Archetype: {archetype}")

        # RL-based decision (score represents strategy confidence)
        strategy_score = self.rl_agent.predict_strategy_score(intent=intent, emotion=emotion)
        tension = random.uniform(0.1, 1.0) * (1.0 - strategy_score)
        concession = self._simulate_concession_curve(tension, emotion)

        # Log and return outcome
        decision = {
            "intent": intent,
            "emotion": emotion,
            "archetype": archetype,
            "strategy_score": round(strategy_score, 4),
            "concession_ratio": round(concession, 3),
            "tactical_recommendation": self._tactical_suggestion(intent, archetype),
            "status": "negotiated"
        }

        self.negotiation_memory.append(decision)
        logger.info(f"🤝 Negotiation outcome: {decision}")
        return decision

    def _tactical_suggestion(self, intent, archetype):
        suggestions = {
            "BuySignal": {
                "Aggressive": "Counter with premium tier offer.",
                "Empathetic": "Explain long-term partnership benefits.",
                "Analytical": "Offer data-based justification for price."
            },
            "WithdrawFunds": {
                "Tactical": "Delay via verification request.",
                "Empathetic": "Offer loyalty incentives.",
                "Neutral": "Acknowledge and confirm processing."
            }
        }
        return suggestions.get(intent, {}).get(archetype, "Proceed with adaptive default.")

    def summarize_session(self):
        return {
            "total_negotiations": len(self.negotiation_memory),
            "last_outcome": self.negotiation_memory[-1] if self.negotiation_memory else None
        }

# THIS FUNCTION MUST BE OUTSIDE THE CLASS!
def adjust_strategy_dynamically(*args, **kwargs):
    # TODO: Implement your dynamic strategy adjustment logic here
    pass
