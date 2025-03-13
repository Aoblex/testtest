import torch
import numpy as np
from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.multi_discrete import MultiDiscrete
from gymnasium.spaces.multi_binary import MultiBinary

import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(
        self,
        observation_space: Box | Discrete | MultiDiscrete | MultiBinary,
        action_space: Box | Discrete | MultiDiscrete | MultiBinary,
        **kwargs,
    ) -> None:
        """
        Args:
            observation_space: The space of the observation.
            action_space: The space of the action.

            kwargs:
                device: The device to run the model on.
        """
        super().__init__()
        # Spaces
        self.observation_space = observation_space
        self.action_space = action_space

        # Shapes
        self.observation_dim = self._get_observation_dim()
        self.action_dim = self._get_action_dim()

        # Device
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    def _get_observation_dim(self,):
        if isinstance(self.observation_space, Box):
            return np.prod(self.observation_space.shape)
        elif isinstance(self.observation_space, Discrete):
            return self.observation_space.n
        elif isinstance(self.observation_space, MultiDiscrete):
            return np.sum(self.observation_space.nvec)
        elif isinstance(self.observation_space, MultiBinary):
            return self.observation_space.n
        else:
            raise ValueError(f"Unsupported observation space type: {type(self.observation_space)}")
    
    def _get_action_dim(self,):
        if isinstance(self.action_space, Box):
            return np.prod(self.action_space.shape)
        elif isinstance(self.action_space, Discrete):
            return self.action_space.n
        elif isinstance(self.action_space, MultiDiscrete):
            return np.sum(self.action_space.nvec)
        elif isinstance(self.action_space, MultiBinary):
            return self.action_space.n
        else:
            raise ValueError(f"Unsupported action space type: {type(self.action_space)}")
   
    def forward(self,):
        """Forward pass through the network."""
        raise NotImplementedError("This method should be implemented by the subclass.")
    
    def select_action(self,):
        """Select an action given the current state."""
        raise NotImplementedError("This method should be implemented by the subclass.")