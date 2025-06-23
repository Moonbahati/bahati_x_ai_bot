import logging
import random
import threading
import time
from uuid import uuid4
from collections import defaultdict
from copy import deepcopy

from ai.intent_recognizer import recognize_intent
from ai.fraud_detection_ai import detect_fraud_pattern
from ai.threat_prediction_ai import predict_market_threat
from ai.auto_feedback_loop import feedback_evaluator
from ai.negotiation_engine import adjust_strategy_dynamically
from engine.dna_profiler import check_dna_uniqueness
from core.strategy_evolver import StrategyEvolver
from engine.risk_guardian import RiskGuardian
from core.emotion_manager import analyze_emotion

# Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegendaryUltraFederatedAgent")


class FederatedAgent:
    def __init__(self, agent_id=None, strategy_dna=None, peers=None):
        self.agent_id = agent_id or f"agent-{uuid4().hex[:8]}"
        self.strategy_dna = strategy_dna or self._generate_initial_dna()
        self.peers = peers if peers else []
        self.knowledge_base = defaultdict(list)
        self.performance_score = 0
        self.last_update = time.time()

    def _generate_initial_dna(self):
        return {
            "risk_threshold": random.uniform(0.1, 0.5),
            "aggressiveness": random.uniform(0.1, 1.0),
            "reaction_delay": random.uniform(0.01, 0.5),
            "data_fusion_bias": random.uniform(0.1, 0.9),
            "reputation_score": 1.0,
            "trust_index": random.uniform(0.2, 1.0)
        }

    def analyze_market_context(self):
        signals = {
            "intent": recognize_intent(list(self.strategy_dna.values())),
            "emotion": analyze_emotion(list(self.strategy_dna.values())),
            "fraud": detect_fraud_pattern(list(self.strategy_dna.values())),
            "threat": predict_market_threat(list(self.strategy_dna.values())),
            "feedback": feedback_evaluator(list(self.strategy_dna.values())),
            "dna_unique": check_dna_uniqueness(list(self.strategy_dna.values()))
        }
        return signals

    def evaluate_strategy(self):
        signals = self.analyze_market_context()
        penalty = 0.3 if signals["fraud"] else 0.0
        score = (
            (signals["intent"] * 0.2) +
            (signals["emotion"] * 0.2) +
            (signals["feedback"] * 0.3) -
            (signals["threat"] * 0.2) -
            penalty
        ) * signals["dna_unique"]

        self.performance_score = max(0, score)
        return self.performance_score

    def share_knowledge(self):
        bundle = {
            "agent_id": self.agent_id,
            "score": self.performance_score,
            "dna": self.strategy_dna,
            "timestamp": time.time()
        }
        for peer in self.peers:
            peer.receive_knowledge(bundle)

    def receive_knowledge(self, bundle):
        agent_id = bundle["agent_id"]
        if agent_id == self.agent_id:
            return
        self.knowledge_base[agent_id].append(bundle)

    def collaborate(self):
        self.evaluate_strategy()
        self.share_knowledge()

    def adapt_strategy(self):
        if not self.knowledge_base:
            return
        all_dnas = [bundle["dna"] for bundles in self.knowledge_base.values() for bundle in bundles]
        if all_dnas:
            merged_dna = self._merge_peer_strategies(all_dnas)
            adjust_strategy_dynamically(merged_dna, self.performance_score)
            self.strategy_dna = merged_dna

    def _merge_peer_strategies(self, dnas):
        merged = deepcopy(self.strategy_dna)
        for key in merged:
            peer_vals = [dna.get(key, merged[key]) for dna in dnas]
            merged[key] = sum(peer_vals) / len(peer_vals)
        return merged

    def run_cycle(self, cycles=10, delay=2):
        for _ in range(cycles):
            self.collaborate()
            self.adapt_strategy()
            logger.info(f"[{self.agent_id}] Score: {self.performance_score:.4f}")
            time.sleep(delay)

    def launch_async(self, cycles=10, delay=2):
        thread = threading.Thread(target=self.run_cycle, args=(cycles, delay), daemon=True)
        thread.start()

