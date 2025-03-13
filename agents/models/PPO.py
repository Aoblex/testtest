import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np
from torch.distributions import Categorical
from .BaseModel import BaseModel

class PPO(BaseModel):
    """Proximal Policy Optimization (PPO) Algorithm"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        **kwargs,
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space

            kwargs:
                hidden_dim: Hidden dimension of networks
                lr: Learning rate
                device: Device to run on
        """
        super().__init__(state_dim, action_dim, **kwargs)
        
        # Networks
        hidden_dim = kwargs.pop("hidden_dim", 256)
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            action_logits: Action logits of shape (batch_size, action_dim)
            value: Value prediction of shape (batch_size, 1)
        """
        features = self.features(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def select_action(
        self,
        state: torch.Tensor | np.ndarray,
        action: torch.Tensor | np.ndarray | None = None,
        requires_grad: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Select an action given the current state.
        
        Args:
            state: Current state tensor
            action: Selected action
            requires_grad: Whether to require gradients
        Returns:
            action: Selected action
            info: Dictionary containing log_prob and value
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        else:
            state = state.to(self.device)

        with torch.set_grad_enabled(requires_grad):
            logits, value = self(state)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            else:
                if isinstance(action, np.ndarray):
                    action = torch.from_numpy(action).long()
                else:
                    action = action
            log_prob = dist.log_prob(action)
            
        return action, {'log_prob': log_prob, 'value': value, 'entropy': dist.entropy()}
