import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict, Any

from .networks import ActorCriticNetwork
from .base import BaseAgent
from utils.metrics import compute_returns, compute_advantage
from utils.transform import normalize

class PPO(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cpu"
    ):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        super().__init__(state_dim, action_dim, device)
        
    def init_networks(self) -> None:
        self.network = ActorCriticNetwork(
            self.state_dim, 
            self.action_dim
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        state = state.to(self.device)
        logits, value = self.network(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value
        
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards']
        # Unpack and move each tensor in the lists to device
        old_log_probs = [log_prob.to(self.device) for log_prob in batch['action_infos'][0]]
        values = [value.to(self.device) for value in batch['action_infos'][1]]
        next_value = values[-1] if not batch['terminated'][-1] else 0.0
        
        # Calculate advantages using GAE
        advantages = compute_advantage(rewards, values, next_value, self.gamma, 0.95)
        advantages = normalize(advantages)
        
        # Calculate returns
        returns = compute_returns(rewards, self.gamma, device=self.device)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # PPO epochs
        for _ in range(10):
            logits, current_values = self.network(states)
            dist = Categorical(logits=logits)
            current_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Calculate ratios and surrogate losses
            ratios = torch.exp(current_log_probs - torch.stack(old_log_probs))
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            # Calculate losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - current_values.squeeze()).pow(2).mean()
            
            # Combined loss
            loss = (policy_loss + 
                   self.value_coef * value_loss - 
                   self.entropy_coef * entropy)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            
        num_epochs = 10
        return {
            'policy_loss': total_policy_loss / num_epochs,
            'value_loss': total_value_loss / num_epochs,
            'entropy': total_entropy / num_epochs
        } 