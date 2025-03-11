import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List
from copy import deepcopy

class DDPGActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.network(state)

class DDPGCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([state, action], dim=1))

class DDPG:
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.005):
        self.actor = DDPGActor(state_dim, action_dim, max_action)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = DDPGCritic(state_dim, action_dim)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = gamma
        self.tau = tau
        
    def select_action(self, state: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.actor(state).numpy()
        
    def update(self, state: torch.Tensor, action: torch.Tensor,
              reward: float, next_state: torch.Tensor,
              done: bool) -> Tuple[float, float]:
        
        # Update critic
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.gamma * target_Q
            
        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), 
                                     self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
            
        for param, target_param in zip(self.actor.parameters(), 
                                     self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
            
        return actor_loss.item(), critic_loss.item() 