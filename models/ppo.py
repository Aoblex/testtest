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

class PPO(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-5,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        num_minibatches: int = 8,
        device: str = "cpu"
    ):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.num_minibatches = num_minibatches
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
        old_log_probs = torch.stack([log_prob.to(self.device)
                                     for log_prob in batch['action_infos'][0]]).to(self.device)
        
        # Calculate advantages and returns using collector
        advantages = TrajectoryCollector.compute_advantages(
            batch,
            gamma=self.gamma,
            gae_lambda=0.95
        ).to(self.device)
        advantages = normalize(advantages)
        
        returns = TrajectoryCollector.compute_returns(
            batch,
            gamma=self.gamma
        ).to(self.device)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # PPO minibatches
        b_indices = np.random.permutation(len(states))
        batch_size = len(states) // self.num_minibatches
        mse_loss = torch.nn.MSELoss()
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            mb_indices = b_indices[start:end]
            mb_states = states[mb_indices]
            mb_actions = actions[mb_indices]
            mb_old_log_probs = old_log_probs[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_returns = returns[mb_indices]
            
            logits, current_values = self.network(mb_states)
            dist = Categorical(logits=logits)
            current_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()
            
            # Calculate ratios and surrogate losses
            mb_ratios = torch.exp(current_log_probs - mb_old_log_probs)
            surr1 = mb_ratios * mb_advantages
            surr2 = torch.clamp(mb_ratios, 1 - self.epsilon, 1 + self.epsilon) * mb_advantages
                
            # Calculate losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = mse_loss(current_values.squeeze(), mb_returns)
            
            # Combined loss
            loss = (policy_loss + 
                   self.value_coef * value_loss - 
                   self.entropy_coef * entropy)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            
        return {
            'policy_loss': total_policy_loss / self.num_minibatches,
            'value_loss': total_value_loss / self.num_minibatches,
            'entropy': total_entropy / self.num_minibatches
        } 