import logging
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Valid imports
from ai.threat_prediction_ai import detect_insider_behavior
from engine.policy_enforcer import enforce_zero_trust_policy

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegendaryAdaptiveSegmentation")


class AdaptiveSegmenter:
    def __init__(self, eps=0.6, min_samples=4, auto_learn=False):
        self.eps = eps
        self.min_samples = min_samples
        self.auto_learn = auto_learn
        self.model = None
        self.last_segments = None

    def extract_behavioral_features(self, raw_logs):
        """
        Simplified internal version of behavioral feature extractor.
        Converts raw logs into feature vectors (e.g., for clustering).
        """
        logger.info("🔍 Internally extracting behavioral features...")
        features = []
        for log in raw_logs:
            features.append([
                log.get("login_count", 0),
                log.get("avg_session_time", 0),
                log.get("suspicious_attempts", 0),
                log.get("data_transferred_MB", 0),
                log.get("commands_executed", 0)
            ])
        return np.array(features)

    def preprocess(self, data):
        """Scale and prepare data for clustering."""
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    def segment_network(self, raw_packet_logs):
        """Applies DBSCAN to segment user behavior."""
        logger.info("📊 Preprocessing and clustering...")
        X = self.extract_behavioral_features(raw_packet_logs)
        X_scaled = self.preprocess(X)
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X_scaled)
        labels = self.model.labels_
        self.last_segments = labels

        segment_count = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"🧱 Segments identified: {segment_count}")
        return labels

    def label_segment(self, label):
        """Converts numeric label into human-readable name."""
        return f"Segment_{label}" if label != -1 else "Noise"

    def enforce_policies(self, logs, segment_labels):
        """Applies zero-trust security policy per user segment."""
        for i, label in enumerate(segment_labels):
            if label == -1:
                continue  # Skip noise or outliers
            node_profile = logs[i]
            threat_score = detect_insider_behavior(node_profile)
            segment_name = self.label_segment(label)
            logger.info(f"🚨 [{segment_name}] Threat Score: {threat_score:.2f}")
            enforce_zero_trust_policy(node_profile, segment_name, threat_score)

    def adaptive_segmentation(self, logs):
        """Full segmentation lifecycle with optional feedback learning."""
        try:
            logger.info("🚦 Running adaptive segmentation...")
            labels = self.segment_network(logs)
            self.enforce_policies(logs, labels)

            if self.auto_learn:
                logger.info("🤖 Auto-learning is enabled. (No-op placeholder)")

        except Exception as e:
            logger.error(f"🔥 Segmentation Failure: {e}")
            raise

    def get_latest_segments(self):
        return self.last_segments

def segment_behavior(*args, **kwargs):
    # TODO: Implement segmentation logic
    return None

