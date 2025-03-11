import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple

from models.networks import ActorCriticNetwork
from utils.metrics import compute_gae
from utils.transform import normalize

class PPO:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, epsilon: float = 0.2, 
                 value_coef: float = 0.5, entropy_coef: float = 0.01):
        self.network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        logits, value = self.network(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value
        
    def update(self, states: torch.Tensor, actions: torch.Tensor,
              old_log_probs: List[torch.Tensor], rewards: List[float],
              values: List[torch.Tensor], next_value: torch.Tensor) -> Tuple[float, float]:
        
        # Calculate advantages using GAE
        advantages = compute_gae(rewards, values, next_value, self.gamma, 0.95)
        advantages = normalize(advantages)
        
        # Calculate returns
        returns = advantages + torch.tensor(values)
        
        for _ in range(10):  # PPO epochs
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
            
        return policy_loss.item(), value_loss.item() 