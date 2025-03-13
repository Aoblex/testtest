import torch
import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        **kwargs,
    ) -> None:
        """
        Args:
            state_dim: The dimension of the state space.
            action_dim: The dimension of the action space.

            kwargs:
                device: The device to run the model on.
        """
        super().__init__()

        # Shapes
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Device
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self,):
        """Forward pass through the network."""
        raise NotImplementedError("This method should be implemented by the subclass.")
    
    def select_action(self,):
        """Select an action given the current state."""
        raise NotImplementedError("This method should be implemented by the subclass.")