from abc import ABC, abstractmethod
import torch
from typing import Tuple, Any, Dict, List

class BaseAgent(ABC):
    """Base class for all RL agents"""
    
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
        """Set the agent to training mode"""
        pass
    
    def eval(self) -> None:
        """Set the agent to evaluation mode"""
        pass 