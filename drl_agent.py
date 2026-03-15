import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.state_dim = config['drl']['state_dim']
        self.action_dim = config['drl']['action_dim']
        self.hidden_dim = config['drl']['hidden_dim']
        
        # Q-Network
        self.q_net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        
        # Target Network
        self.target_net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer - FIXED: use 'learning_rate' from config
        lr = config['drl'].get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # Replay Memory
        memory_size = config['training'].get('memory_size', 10000)
        self.memory = deque(maxlen=memory_size)
        
        # Epsilon for exploration
        self.epsilon = config['drl']['epsilon_start']
        self.epsilon_min = config['drl']['epsilon_min']
        self.epsilon_decay = config['drl']['epsilon_decay']

    def act(self, state, epsilon=None):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: numpy array of shape (state_dim,)
            epsilon: exploration rate (if None, use self.epsilon)
        
        Returns:
            action: integer action index
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim), None
        
        # Convert state to tensor
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            q_values = self.q_net(state)
            action = q_values.argmax(dim=1).item()
            return action, q_values.squeeze().cpu().numpy()

    def remember(self, state, action, reward, next_state, done):
        """Store transition in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """Train the Q-network using a batch from replay memory"""
        batch_size = self.config['training']['batch_size']
        
        if len(self.memory) < batch_size:
            return None  # Not enough samples yet
        
        # Sample random batch
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q = self.q_net(states).gather(1, actions).squeeze()
        
        # Target Q values using target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            # If done, target is just reward. Otherwise reward + gamma * next_q
            targets = rewards + (1 - dones) * self.config['training']['gamma'] * next_q
        
        # Compute loss (Huber loss for stability)
        loss = nn.SmoothL1Loss()(current_q, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Debug: Gradient norm
        total_norm = 0.0
        for p in self.q_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip gradients (increased to 10.0)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Store for debug
        self.last_q_values = current_q.detach().mean().item()
        self.last_grad_norm = total_norm
        
        return loss.item()

    def update_target(self):
        """Update target network with Q-network weights"""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save_model(self, path):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory_size': len(self.memory)
        }, path)
        print(f"[INFO] Model saved to {path}")

    def load_model(self, path):
        """Load model checkpoint"""
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.q_net.load_state_dict(checkpoint['q_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            print(f"[INFO] Model loaded from {path}")
            return True
        else:
            print(f"[WARN] Model file {path} not found")
            return False