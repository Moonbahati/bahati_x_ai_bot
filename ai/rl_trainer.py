import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
from collections import deque
from ai.auto_feedback_loop import feedback_evaluator
from ai.pattern_memory import PatternMemory
from core.digit_predictor.ensemble_voter import evaluate_strategy
from engine.dna_profiler import check_dna_uniqueness
from ai.fraud_detection_ai import detect_fraud_pattern

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RLTrainerUltra")

# Neural Network for Q-learning
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.model(x)

class RLTrainer:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=64,
                 memory_size=5000):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.pattern_memory = PatternMemory()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values[0]).item()

    def replay(self):
        """Train Q-network from replay buffer"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            state_t = torch.FloatTensor(state)
            next_state_t = torch.FloatTensor(next_state)

            target = reward
            if not done:
                target += self.gamma * torch.max(self.q_network(next_state_t)).item()

            current_q = self.q_network(state_t)[action]
            loss = self.loss_fn(current_q, torch.tensor(target))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Decay epsilon after each training iteration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def evaluate_action_reward(self, strategy):
        """Advanced feedback and fraud-aware scoring"""
        score = evaluate_strategy(strategy)
        feedback = feedback_evaluator(strategy)
        dna_score = check_dna_uniqueness(strategy)
        fraud_penalty = 0.5 if detect_fraud_pattern(strategy) else 0

        # Combined weighted scoring
        return (score * 0.4) + (feedback * 0.3) + (dna_score * 0.3) - fraud_penalty

    def train_episode(self, environment):
        """Run one training episode"""
        state = environment.reset()
        total_reward = 0

        for t in range(100):  # Max steps per episode
            action = self.act(state)
            next_state, reward, done, info = environment.step(action)

            # Legendary layer: reward augmentation from external AI modules
            enhanced_reward = reward + self.evaluate_action_reward(info.get("strategy", []))
            total_reward += enhanced_reward

            self.remember(state, action, enhanced_reward, next_state, done)
            state = next_state

            if done:
                break

        self.replay()
        logger.info(f"🎯 Episode completed | Total Reward: {total_reward:.2f}")
        return total_reward
