import math
import random
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Définition de l'Environnement de Jeu
class Nim:
    def __init__(self, initial=[ 8, 10, 1, 3, 15, 5, 7]):
        self.piles = initial.copy()  # Initialise les piles d'objets
        self.player = 0  # Le joueur actuel (0 ou 1)
        self.winner = None  # Indique le gagnant une fois la partie terminée

    @classmethod
    def available_actions(cls, piles):
        # Retourne les actions valides sous forme de (pile, nombre d'objets à retirer)
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player):
        # Change le joueur actuel (0 devient 1 et vice-versa)
        return 0 if player == 1 else 1

    def switch_player(self):
        # Alterne entre les joueurs
        self.player = Nim.other_player(self.player)

    def move(self, action):
        # Applique une action et met à jour l'état du jeu
        pile, count = action
        if self.winner is not None:
            raise Exception("Partie déjà gagnée")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Pile invalide")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Nombre d'objets invalide")
        self.piles[pile] -= count
        self.switch_player()
        # Vérifie si toutes les piles sont vides (fin de partie)
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player

# Définition du Réseau de Neurones
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64) 
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Agent IA utilisant DQN
class NimAI:
    def __init__(self, state_size, action_size, alpha=0.001, gamma=0.9, epsilon=0.1, buffer_size=10000, batch_size=32, target_update=10):
        self.state_size = state_size   # nombre de piles
        self.action_size = action_size # Nombre total d'actions possibles
        self.alpha = alpha             # learning rate
        self.gamma = gamma             # Facteur d'actualisation pour les récompenses futures
        self.epsilon = epsilon         # Contrôle le niveau d'exploration
        self.batch_size = batch_size   # Taille des échantillons utilisés pour l'entraînement
        self.target_update = target_update # Fréquence de mise à jour du réseau cible

        # Modèle principal et réseau cible
        self.model = QNetwork(state_size, action_size)  # Réseau principal
        self.target_model = QNetwork(state_size, action_size)  # Réseau cible
        self.target_model.load_state_dict(self.model.state_dict())  # Synchronisation initiale des poids
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)  # Optimiseur pour entraîner le modèle

        # Replay buffer pour stocker les expériences
        self.memory = deque(maxlen=buffer_size)
        self.steps = 0  # Compteur de pas pour suivre les mises à jour

    def remember(self, state, action, reward, next_state, done):
        # Ajoute une expérience au buffer
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon=True):
        # Choisit une action en fonction de la stratégie ε-greedy
        if epsilon and random.random() < self.epsilon:
            return random.randrange(self.action_size)  # Exploration aléatoire
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convertit l'état en tenseur
            q_values = self.model(state_tensor)  # Prédit les valeurs Q pour toutes les actions
            return torch.argmax(q_values).item()  # Sélectionne l'action avec la plus grande valeur Q

    def replay(self):
        # Entraîne le réseau principal en utilisant un échantillon du buffer
        if len(self.memory) < self.batch_size:
            return  # Attendre que le buffer ait assez d'échantillons

        # Sélection aléatoire d'expériences
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Conversion des données en tenseurs
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Calcul des valeurs Q pour les actions actuelles et suivantes
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))  # Cible Q

        # Calcul de la perte et rétropropagation
        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Mise à jour du réseau cible
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

def train(n_games):
    ai = NimAI(state_size=8, action_size=160)  # Initialisation de l'IA avec les dimensions d'état et d'action
    rewards = []  # Suivi des récompenses

    for episode in range(n_games):
        game = Nim()  # Nouvelle partie
        total_reward = 0
        state = game.piles.copy()

        while game.winner is None:
            action = ai.choose_action(state)  # Sélection d'une action
            pile, count = action // 4, action % 4 + 1  # Conversion de l'action en (pile, nombre)
            if (pile, count) not in Nim.available_actions(game.piles):
                reward = -1  # Récompense négative pour une action invalide
                done = True
            else:
                game.move((pile, count))
                next_state = game.piles.copy()
                reward = 1 if game.winner == ai.player else -1  # Récompense si l'IA gagne
                done = game.winner is not None
                ai.remember(state, action, reward, next_state, done)  # Ajout de l'expérience
                total_reward += reward
                state = next_state
            ai.replay()  # Entraînement de l'IA
        rewards.append(total_reward)  # Suivi des récompenses cumulées

    return ai, rewards

def plot_eval(rewards):
    # Affiche les récompenses cumulées au fil des parties
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Récompense cumulée")
    plt.xlabel("Jeux joués")
    plt.ylabel("Récompense cumulée")
    plt.title("Récompense cumulée au cours de l'entraînement")
    plt.legend()
    plt.show()

def play(ai):
    # Permet à un humain de jouer contre l'IA
    game = Nim()
    while game.winner is None:
        print(f"\nPiles: {game.piles}")
        if game.player == 0:
            pile = int(input("Choisissez une pile: "))
            count = int(input("Choisissez combien retirer: "))
        else:
            state = game.piles.copy()
            action = ai.choose_action(state, epsilon=False)  # L'IA choisit une action optimale
            pile, count = action // 4, action % 4 + 1
            print(f"L'IA retire {count} de la pile {pile}")
        game.move((pile, count))
    print("Fin de la partie!")
    print("Gagnant:", "Humain" if game.winner == 0 else "IA")



