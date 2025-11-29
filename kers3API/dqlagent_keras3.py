#
# DQL Agent with Keras 3 API and PyTorch Backend
# Adapted from dqlagent_pytorch.py
#
# (c) Dr. Yves J. Hilpisch
# Reinforcement Learning for Finance
# Keras 3 adaptation for PSC
#

import os
import random
import warnings
import numpy as np
from collections import deque

warnings.simplefilter('ignore')
os.environ['PYTHONHASHSEED'] = '0'

# Configure Keras to use PyTorch backend
os.environ['KERAS_BACKEND'] = 'torch'

import keras
from keras import layers, models, optimizers


class DQLAgent:
    """
    Deep Q-Learning Agent using Keras 3 API with PyTorch backend.

    Production-ready agent that works with any Gym-style environment.
    """

    def __init__(self, symbol, feature, n_features, env, hu=24, lr=0.001):
        """
        Initialize DQL Agent.

        Args:
            symbol: Trading symbol (e.g., 'EUR=', 'AAPL.O')
            feature: Feature to use ('r' for returns, symbol for prices)
            n_features: Number of features in state (input dimension)
            env: Gym-style environment with reset(), step(), action_space
            hu: Hidden units in neural network layers
            lr: Learning rate for optimizer
        """
        self.epsilon = 1.0
        self.epsilon_decay = 0.9975
        self.epsilon_min = 0.1
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.gamma = 0.5
        self.trewards = []
        self.max_treward = -np.inf
        self.n_features = n_features
        self.env = env
        self.episodes = 0

        # Create Keras model
        self.model = self._create_model(hu, lr)

    def _create_model(self, hu, lr):
        """Create neural network using Keras 3 Sequential API"""
        model = models.Sequential([
            layers.Dense(hu, activation='relu', input_shape=(self.n_features,)),
            layers.Dense(hu, activation='relu'),
            layers.Dense(self.env.action_space.n, activation='linear')
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss='mse'
        )

        return model

    def _reshape(self, state):
        """Ensure state is in correct shape for neural network"""
        state = state.flatten()
        return np.reshape(state, [1, len(state)])

    def act(self, state):
        """
        Epsilon-greedy action selection.

        Args:
            state: Current environment state

        Returns:
            action: Integer action to take
        """
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        # Reshape if needed
        if state.ndim == 1:
            state = np.reshape(state, [1, self.n_features])

        # Predict Q-values
        q_values = self.model.predict(state, verbose=0)
        return int(np.argmax(q_values[0]))

    def replay(self):
        """
        Experience replay - learn from batch of past experiences.
        Uses Keras 3 fit API for training.
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)

        # Extract batch components
        states = np.vstack([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        next_states = np.vstack([e[2] for e in batch])
        rewards = np.array([e[3] for e in batch], dtype=np.float32)
        dones = np.array([e[4] for e in batch], dtype=bool)

        # Predict Q-values for current and next states
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        # Calculate target Q-values using Bellman equation
        target_q_values = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])

        # Train model
        self.model.fit(states, target_q_values, epochs=1, verbose=0, batch_size=self.batch_size)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, episodes):
        """
        Train the agent over multiple episodes.

        Args:
            episodes: Number of episodes to train
        """
        for e in range(1, episodes + 1):
            self.episodes += 1
            state, _ = self.env.reset()
            state = self._reshape(state)
            treward = 0

            for f in range(1, 5000):
                self.f = f
                action = self.act(state)
                next_state, reward, done, trunc, _ = self.env.step(action)
                treward += reward
                next_state = self._reshape(next_state)

                # Store experience
                self.memory.append((state, action, next_state, reward, done))
                state = next_state

                if done:
                    self.trewards.append(treward)
                    self.max_treward = max(self.max_treward, treward)
                    templ = f'episode={self.episodes:4d} | '
                    templ += f'treward={treward:7.3f} | max={self.max_treward:7.3f}'
                    print(templ, end='\r')
                    break

            # Experience replay after episode
            if len(self.memory) > self.batch_size:
                self.replay()

        print()

    def test(self, episodes, min_accuracy=0.0, min_performance=0.0, verbose=True, full=True):
        """
        Test the trained agent.

        Args:
            episodes: Number of test episodes
            min_accuracy: Minimum accuracy threshold (overrides environment setting)
            min_performance: Minimum performance threshold
            verbose: Print results
            full: Print full output vs compact
        """
        # Backup and set environment thresholds
        ma = getattr(self.env, 'min_accuracy', None)
        if hasattr(self.env, 'min_accuracy'):
            self.env.min_accuracy = min_accuracy

        mp = None
        if hasattr(self.env, 'min_performance'):
            mp = self.env.min_performance
            self.env.min_performance = min_performance
            self.performances = []

        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            state = self._reshape(state)

            for f in range(1, 5001):
                action = self.act(state)
                state, reward, done, trunc, _ = self.env.step(action)
                state = self._reshape(state)

                if done:
                    templ = f'total reward={f:4d} | accuracy={self.env.accuracy:.3f}'
                    if hasattr(self.env, 'min_performance'):
                        self.performances.append(self.env.performance)
                        templ += f' | performance={self.env.performance:.3f}'
                    if verbose:
                        if full:
                            print(templ)
                        else:
                            print(templ, end='\r')
                    break

        # Restore environment thresholds
        if hasattr(self.env, 'min_accuracy') and ma is not None:
            self.env.min_accuracy = ma
        if mp is not None:
            self.env.min_performance = mp

        print()
