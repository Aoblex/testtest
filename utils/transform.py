import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Dict

def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to zero mean and unit variance"""
    return (x - x.mean()) / (x.std() + 1e-8) 