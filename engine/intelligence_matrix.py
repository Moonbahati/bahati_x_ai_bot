import threading
import random

class IntelligenceMatrix:
    def __init__(self):
        self.registry = {}
        self.lock = threading.Lock()

    def register_ai_agent(self, name, callback):
        with self.lock:
            self.registry[name] = callback

    def deregister_ai_agent(self, name):
        with self.lock:
            if name in self.registry:
                del self.registry[name]

    def execute(self, context_data):
        results = {}
        with self.lock:
            for name, agent in self.registry.items():
                try:
                    result = agent(context_data)
                    results[name] = result
                except Exception as e:
                    results[name] = {"error": str(e)}
        return results

    def route_decision(self, strategy_scores):
        # Advanced routing logic could be placed here
        sorted_scores = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        if sorted_scores:
            best_agent = sorted_scores[0][0]
            return best_agent
        return None
