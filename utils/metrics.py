import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional

def compute_returns(rewards: List[float], gamma: float, device: str = 'cpu') -> torch.Tensor:
    """Compute discounted returns.
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
    
    Returns:
        Tensor of discounted returns
    """
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, device=device)

def compute_advantage(rewards: List[float], values: List[torch.Tensor], 
                     next_value: float, gamma: float, lambda_: float) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        next_value: Value estimate for next state
        gamma: Discount factor
        lambda_: GAE parameter
        
    Returns:
        Tensor of advantage estimates
    """
    device = values[0].device if isinstance(values[0], torch.Tensor) else 'cpu'
    advantages = []
    gae = 0
    
    for r, v in zip(reversed(rewards), reversed(values)):
        delta = r + gamma * next_value - v.item()
        gae = delta + gamma * lambda_ * gae
        advantages.append(gae)
        next_value = v.item()
        
    advantages = advantages[::-1]
    return torch.tensor(advantages, device=device)

def compute_td_error(rewards: List[float], values: List[float], 
                    next_values: List[float], gamma: float) -> torch.Tensor:
    """Compute TD error.
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        next_values: List of next state value estimates
        gamma: Discount factor
        
    Returns:
        Tensor of TD errors
    """
    td_errors = []
    for r, v, nv in zip(rewards, values, next_values):
        td_error = r + gamma * nv - v
        td_errors.append(td_error)
    return torch.tensor(td_errors)