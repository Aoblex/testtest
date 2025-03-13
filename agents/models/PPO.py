import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from typing import Dict, Tuple
import numpy as np
from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.multi_discrete import MultiDiscrete
from gymnasium.spaces.multi_binary import MultiBinary

from .BaseModel import BaseModel

class PPO(BaseModel):
    """Proximal Policy Optimization (PPO) Algorithm"""

    def __init__(
        self,
        observation_space: Box | Discrete | MultiDiscrete | MultiBinary,
        action_space: Box | Discrete | MultiDiscrete | MultiBinary,
        **kwargs,
    ):
        """
        Args:
            observation_space: The space of the observation.
            action_space: The space of the action.

            kwargs:
                hidden_dim: Hidden dimension of networks
                lr: Learning rate
                device: Device to run on
        """
        super().__init__(observation_space, action_space, **kwargs)

        assert isinstance(observation_space, Box), \
               f"Observation space must be a Box, got {type(observation_space)}"
        assert isinstance(action_space, Discrete) \
               or isinstance(action_space, Box), \
               f"Action space must be a Discrete or Box, got {type(action_space)}"
        
        # Networks
        hidden_dim = kwargs.pop("hidden_dim", 256)
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(self.observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor Network
        if isinstance(action_space, Discrete):  
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, self.action_dim)
            )
        elif isinstance(action_space, Box):
            self.actor_mean = nn.Sequential(
                nn.Linear(hidden_dim, self.action_dim),
                nn.Tanh(),
            )
            self.actor_std = nn.Sequential(
                nn.Linear(hidden_dim, self.action_dim),
                nn.Softplus(),
            )
        
        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            observation: Observation tensor of shape (batch_size, observation_dim)
            
        Returns:
            action_logits: Action logits of shape (batch_size, action_dim)
            value: Value prediction of shape (batch_size, 1)
        """
        features = self.features(observation)
        value = self.critic(features)
        if isinstance(self.action_space, Discrete):
            action_logits = self.actor(features)
            return action_logits, value
        elif isinstance(self.action_space, Box):
            action_mean = self.actor_mean(features)
            action_std = self.actor_std(features)
            return action_mean, action_std, value
        else:
            raise ValueError(f"Unsupported action space type: {type(self.action_space)}")
    
    def select_action(
        self,
        observation: torch.Tensor | np.ndarray,
        action: torch.Tensor | np.ndarray | None = None,
        requires_grad: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Select an action given the current state.
        
        Args:
            observation: Current observation tensor
            action: Selected action
            requires_grad: Whether to require gradients
        Returns:
            action: Selected action
            info: Dictionary containing log_prob and value
        """
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float().to(self.device)
        else:
            observation = observation.to(self.device)
        
        if len(observation.shape) > len(self.observation_space.shape): # The input is batched
            observation = observation.reshape(observation.shape[0], -1)

        with torch.set_grad_enabled(requires_grad):

            # When action space is Discrete
            if isinstance(self.action_space, Discrete):
                action_logits, value = self(observation)
                dist = Categorical(logits=action_logits)
                if action is None:
                    action = dist.sample()
                else:
                    if isinstance(action, np.ndarray):
                        action = torch.from_numpy(action)
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

            # When action space is Box
            elif isinstance(self.action_space, Box):
                action_mean, action_std, value = self(observation)
                dist = Normal(action_mean, action_std)
                
                if action is None:
                    # Clip actions to valid range
                    action = torch.clamp(
                        dist.sample(),
                        torch.tensor(self.action_space.low, device=self.device),
                        torch.tensor(self.action_space.high, device=self.device),
                    )
                else:
                    if isinstance(action, np.ndarray):
                        action = torch.from_numpy(action)

                # Assume actions are independent and the sum of log_prob is the log_prob of the action
                log_prob = torch.sum(dist.log_prob(action))
                entropy = torch.mean(dist.entropy())
            else:
                raise ValueError(f"Unsupported action space type: {type(self.action_space)}")
            
        return action, \
            {
                'log_prob': log_prob,
                'value': value,
                'entropy': entropy,
            }
