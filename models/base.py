from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple, Any, Dict, List, Optional
import gymnasium as gym

class BaseAgent(ABC):
    """Base class for all RL agents"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        **kwargs
    ):
        """Initialize the base agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Device to run the agent on ("cpu" or "cuda")
            **kwargs: Additional arguments for specific implementations
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # Training mode flag
        self._training = True
        
        # Initialize networks (to be implemented by subclasses)
        self.init_networks()
        
    @abstractmethod
    def init_networks(self) -> None:
        """Initialize neural networks.
        
        This method should be implemented by subclasses to set up
        their specific network architectures.
        """
        pass
        
    @abstractmethod
    def select_action(self, state: torch.Tensor) -> Tuple[Any, ...]:
        """Select an action given the current state.
        
        Args:
            state: Current state tensor
            
        Returns:
            Tuple containing at minimum (action, action_info)
            where action_info contains necessary information for updating
        """
        pass
        
    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update the agent's parameters using the provided batch of experience.
        
        Args:
            batch: Dictionary containing experience data
            
        Returns:
            Dictionary of loss metrics
        """
        pass
    
    def train(self) -> None:
        """Set the agent to training mode."""
        self._training = True
        for module in self.modules():
            if isinstance(module, nn.Module):
                module.train()
    
    def eval(self) -> None:
        """Set the agent to evaluation mode."""
        self._training = False
        for module in self.modules():
            if isinstance(module, nn.Module):
                module.eval()
                
    def modules(self) -> List[nn.Module]:
        """Get all neural network modules in the agent.
        
        Returns:
            List of all nn.Module instances in the agent
        """
        return [module for module in self.__dict__.values() 
                if isinstance(module, nn.Module)]
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict of all networks.
        
        Returns:
            Dictionary containing state of all networks
        """
        return {
            name: module.state_dict()
            for name, module in self.__dict__.items()
            if isinstance(module, nn.Module)
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict into networks.
        
        Args:
            state_dict: Dictionary containing network states
        """
        for name, module in self.__dict__.items():
            if isinstance(module, nn.Module) and name in state_dict:
                module.load_state_dict(state_dict[name])
                
    def to(self, device: torch.device) -> 'BaseAgent':
        """Move all networks to specified device.
        
        Args:
            device: Device to move networks to
            
        Returns:
            Self for chaining
        """
        self.device = device
        for module in self.modules():
            module.to(device)
        return self 