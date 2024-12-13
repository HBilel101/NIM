import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random

class Nim:
    def __init__(self, initial=[1, 3, 5, 7]):
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player):
        return 0 if player == 1 else 1

    def switch_player(self):
        self.player = Nim.other_player(self.player)

    def move(self, action):
        pile, count = action
        if self.winner is not None:
            raise Exception("Game already won")
        if pile < 0 or pile >= len(self.piles) or count < 1 or count > self.piles[pile]:
            raise Exception("Invalid move")
        self.piles[pile] -= count
        self.switch_player()
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class PolicyGradientAI:
    def __init__(self, learning_rate=0.01, gamma=0.99):
        self.gamma = gamma
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _build_model(self):
        
        model = tf.keras.Sequential([
            layers.Input(shape=(4,)),  # Input: piles state
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(16, activation='softmax')  # Output: probabilities for 16 possible actions
        ])
        return model

    def choose_action(self, state, epsilon=0.1):
        state_input = np.array([state])
        probabilities = self.model(state_input).numpy().flatten()

    # Convert state to available actions
        available_actions = list(Nim.available_actions(state))
        action_probs = np.zeros(len(available_actions))

    # Map probabilities to available actions
        for idx, action in enumerate(available_actions):
            pile, count = action
            action_index = pile * max(state) + (count - 1)
            if action_index < len(probabilities):
                 action_probs[idx] = probabilities[action_index]

    # Normalize probabilities if sum > 0, otherwise use uniform distribution
        if np.sum(action_probs) > 0:
            action_probs /= np.sum(action_probs)
        else:
            action_probs = np.ones(len(action_probs)) / len(action_probs)

        # Choose action based on epsilon-greedy
        if random.random() < epsilon:
            return random.choice(available_actions)
        else:
            chosen_index = np.random.choice(len(available_actions), p=action_probs)
            return available_actions[chosen_index]


    def train(self, episodes):
        cumulative_rewards = []
        for episode in range(episodes):
            game = Nim()
            state_history, action_history, rewards_history = [], [], []
            cumulative_reward = 0

            while game.winner is None:
                state = game.piles
                epsilon = max(0.1, 0.99**episode)
                action = self.choose_action(state,epsilon=epsilon)
                state_history.append(state)
                action_history.append(action)

                game.move(action)

                if game.winner is not None:
                    reward = 1 if game.winner == 0 else -1
                    rewards_history.append(reward)
                    cumulative_reward += reward
                else:
                    rewards_history.append(0)

            G = np.zeros_like(rewards_history, dtype=np.float32)
            cumulative = 0
            for t in reversed(range(len(rewards_history))):
                cumulative = rewards_history[t] + self.gamma * cumulative
                G[t] = cumulative

            self._update_policy(state_history, action_history, G)
            cumulative_rewards.append(cumulative_reward)

    def plot_training_performance(self, cumulative_rewards):
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_rewards, label="Cumulative Rewards")
        plt.xlabel("Games")
        plt.ylabel("Cumulative Reward")
        plt.title("AI Training Performance")
        plt.legend()
        plt.show()
    def _update_policy(self, states, actions, rewards_to_go):
        with tf.GradientTape() as tape:
            loss = 0
            for state, action, reward in zip(states, actions, rewards_to_go):
                state_input = np.array([state])
                action_probs = self.model(state_input)

            # Map action to index
                action_index = action[0] * 4 + (action[1] - 1)

            # Ensure the action index is within bounds (0-15)
                action_index = min(max(action_index, 0), 15)

            # Compute loss
                loss -= tf.math.log(action_probs[0, action_index]) * reward

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
