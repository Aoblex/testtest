import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict, Any
from copy import deepcopy

from .base import BaseAgent

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

class DDPG(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu"
    ):
        self.max_action = max_action
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        super().__init__(state_dim, action_dim, device)
        
    def init_networks(self) -> None:
        self.actor = DDPGActor(
            self.state_dim, 
            self.action_dim, 
            self.max_action
        ).to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        
        self.critic = DDPGCritic(
            self.state_dim, 
            self.action_dim
        ).to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
    def select_action(self, state: torch.Tensor) -> Tuple[np.ndarray, None]:
        state = state.to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()
        return action, None
        
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = torch.tensor(batch['rewards']).to(self.device)
        next_states = batch['next_states'].to(self.device)
        terminated = torch.tensor(batch['terminated']).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - terminated) * self.gamma * target_Q
            
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
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
            
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        } 