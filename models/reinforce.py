import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple

from models.networks import ActorNetwork
from utils.metrics import compute_gae

class REINFORCE:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, gamma: float = 0.99):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.gamma = gamma
        
    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
        
    def update(self, rewards: List[float], log_probs: List[torch.Tensor]) -> float:
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        policy_loss = torch.stack(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item() 