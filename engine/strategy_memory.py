import json
import os
import threading

class StrategyMemory:
    def __init__(self, storage_path='data/strategy_memory.json'):
        self.storage_path = storage_path
        self.lock = threading.Lock()
        self.memory = {}
        self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    self.memory = json.load(f)
            except Exception:
                self.memory = {}
        else:
            self.memory = {}

    def save_strategy(self, strategy_id, metrics):
        with self.lock:
            self.memory[strategy_id] = metrics
            self._persist()

    def get_strategy(self, strategy_id):
        return self.memory.get(strategy_id, None)

    def get_all_strategies(self):
        return self.memory

    def remove_strategy(self, strategy_id):
        with self.lock:
            if strategy_id in self.memory:
                del self.memory[strategy_id]
                self._persist()

    def _persist(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.memory, f, indent=4)
