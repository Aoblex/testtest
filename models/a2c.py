import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict, Any

from .networks import ActorCriticNetwork
from .base import BaseAgent
from utils.transform import normalize
from utils.collector import TrajectoryCollector

class A2C(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cpu"
    ):
        self.lr = lr  # Save before super().__init__
        self.gamma = gamma
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
        log_probs = [log_prob.to(self.device) for log_prob in batch['action_infos'][0]]
        
        # Calculate returns and advantages using collector
        returns = TrajectoryCollector.compute_returns(
            batch,
            gamma=self.gamma
        ).to(self.device)
        
        advantages = TrajectoryCollector.compute_advantages(
            batch,
            gamma=self.gamma,
            gae_lambda=0.95
        ).to(self.device)
        advantages = normalize(advantages)
        
        # Forward pass
        logits, current_values = self.network(states)
        dist = Categorical(logits=logits)
        current_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Calculate losses
        policy_loss = -(current_log_probs * advantages).mean()
        value_loss = 0.5 * (returns - current_values.squeeze()).pow(2).mean()
        
        # Combined loss
        loss = (policy_loss + 
               self.value_coef * value_loss - 
               self.entropy_coef * entropy)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }
        
    def train(self) -> None:
        self.network.train()
        
    def eval(self) -> None:
        self.network.eval() 