import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict, Any

from .networks import ActorNetwork
from .base import BaseAgent
from utils.metrics import compute_returns
from utils.transform import normalize

class REINFORCE(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu"
    ):
        self.lr = lr
        self.gamma = gamma
        super().__init__(state_dim, action_dim, device)
        
    def init_networks(self) -> None:
        self.actor = ActorNetwork(
            self.state_dim, 
            self.action_dim
        ).to(self.device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        
    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        state = state.to(self.device)
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
        
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        rewards = batch['rewards']
        log_probs = [log_prob.to(self.device) for log_prob in batch['action_infos'][0]]
        
        returns = compute_returns(rewards, self.gamma, device=self.device)
        returns = normalize(returns)
        
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        policy_loss = torch.stack(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item()
        } 