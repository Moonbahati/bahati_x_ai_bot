import hashlib
import numpy as np
import logging
from collections import deque, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegendaryPatternMemory")

class PatternMemory:
    def __init__(self, memory_limit=5000, similarity_threshold=0.92, cluster_count=10):
        self.memory_limit = memory_limit
        self.similarity_threshold = similarity_threshold
        self.patterns = deque(maxlen=self.memory_limit)
        self.cluster_count = cluster_count
        self.pattern_clusters = defaultdict(list)

    def _hash_pattern(self, pattern):
        """Generate a hash ID from pattern for quick lookup"""
        pattern_str = str(np.round(pattern, decimals=4).tolist())
        return hashlib.sha256(pattern_str.encode()).hexdigest()

    def _vectorize(self, pattern):
        """Convert pattern to a numpy vector"""
        return np.array(pattern).reshape(1, -1)

    def store_pattern(self, pattern):
        """Store pattern if it's not too similar to existing ones"""
        if self.is_unique(pattern):
            pattern_id = self._hash_pattern(pattern)
            self.patterns.append((pattern_id, pattern))
            logger.info(f"🧠 New unique pattern stored. Total: {len(self.patterns)}")
            return True
        return False

    def is_unique(self, pattern):
        """Check if pattern is unique based on cosine similarity"""
        if not self.patterns:
            return True

        pattern_vector = self._vectorize(pattern)

        for _, existing in self.patterns:
            existing_vector = self._vectorize(existing)
            similarity = cosine_similarity(pattern_vector, existing_vector)[0][0]
            if similarity >= self.similarity_threshold:
                return False
        return True

    def get_similar_patterns(self, pattern, top_n=5):
        """Retrieve top-N most similar patterns from memory"""
        pattern_vector = self._vectorize(pattern)
        similarities = []

        for pid, existing in self.patterns:
            existing_vector = self._vectorize(existing)
            score = cosine_similarity(pattern_vector, existing_vector)[0][0]
            similarities.append((pid, score, existing))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

    def cluster_patterns(self):
        """Apply clustering on stored patterns for pattern archetypes"""
        if len(self.patterns) < self.cluster_count:
            logger.warning("⚠️ Not enough patterns for clustering.")
            return {}

        vectors = [self._vectorize(p)[0] for _, p in self.patterns]
        kmeans = KMeans(n_clusters=self.cluster_count, n_init='auto', random_state=42)
        labels = kmeans.fit_predict(vectors)

        self.pattern_clusters.clear()
        for label, (_, pattern) in zip(labels, self.patterns):
            self.pattern_clusters[label].append(pattern)

        logger.info(f"🧬 {len(self.pattern_clusters)} pattern clusters formed.")
        return self.pattern_clusters

    def analyze_volatility_behavior(self, pattern):
        """Advanced volatility-based behavioral tagging (Ultra Enhancement)"""
        std_dev = np.std(pattern)
        mean_val = np.mean(pattern)

        if std_dev > mean_val * 0.8:
            return "🔥 High Volatility"
        elif std_dev < mean_val * 0.2:
            return "❄️ Low Volatility"
        else:
            return "⚖️ Moderate Volatility"

    def decay_memory(self, factor=0.9):
        """Decays memory to forget the oldest patterns based on a factor"""
        new_limit = int(self.memory_limit * factor)
        if len(self.patterns) > new_limit:
            removed = len(self.patterns) - new_limit
            for _ in range(removed):
                self.patterns.popleft()
            logger.info(f"🧽 Memory decayed. Removed {removed} old patterns.")

def recall_patterns(*args, **kwargs):
    # TODO: Implement pattern recall logic
    return {"resonance": 0.5, "label": "none"}

def reinforce_patterns(patterns, feedback=None):
    # TODO: Implement logic to reinforce patterns based on feedback
    return {"status": "Pattern reinforcement not implemented yet."}
