import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np
from torch.distributions import Categorical
from .BaseModel import BaseModel

class REINFORCE(BaseModel):
    """REINFORCE Algorithm"""

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
        """
        super().__init__(state_dim, action_dim)

        # Policy network
        hidden_dim = kwargs.pop("hidden_dim", 256)
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            action_logits: Action logits of shape (batch_size, action_dim)
        """
        action_logits = self.policy(state)
        return action_logits
    
    def select_action(
        self,
        state: torch.Tensor | np.ndarray,
        requires_grad: bool = False,
    ) -> Tuple[float, Dict[str, torch.Tensor]]:
        """Select an action given the current state.
        
        Args:
            state: Current state tensor
            requires_grad: Whether to require gradients
        Returns:
            action: Selected action
            info: Dictionary containing log_prob and value
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        with torch.set_grad_enabled(requires_grad):
            logits = self(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), {'log_prob': log_prob}