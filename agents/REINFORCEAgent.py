import gymnasium as gym
import torch

from .BaseAgent import BaseAgent
from .models import REINFORCE

class REINFORCEAgent(BaseAgent):

    def __init__(
        self,
        env_id: str | gym.envs.registration.EnvSpec,
        **kwargs,
    ) -> None:
        """Initialize the REINFORCE agent."""
        super().__init__(env_id, **kwargs)

        # Configure the model.
        self.model = REINFORCE(
            observation_space=self.observation_space,
            action_space=self.action_space,
            **kwargs,
        ).to(self.device)

        # Parameters for loss functions.
        self.gamma = kwargs.pop("gamma", 0.99)
        self.normalize = kwargs.pop("normalize", True)

        # Parameters for training.
        self.num_epochs = kwargs.pop("num_epochs", 100)
        self.num_steps = kwargs.pop("num_steps", 100)

        # Configure the saving options and the optimizer.
        self._set_model_settings()
    
    def update(
        self,
    ) -> None:
        """Update the model."""

        discounted_returns = self.get_discounted_returns(
            gamma=self.gamma,
            normalize=self.normalize,
        ).to(self.device)

        action_info = self.buffer.get_action_info_list()
        log_probs = torch.stack([info["log_prob"] for info in action_info])

        # Compute the loss.
        loss = -torch.mean(log_probs * discounted_returns)

        # Backpropagate the loss.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(
        self,
        plot: bool = True,
        window_size: int = 10,
    ) -> None:
        """Train the model."""

        epoch_returns = []
        for epoch in range(self.num_epochs):
            # Sample trajectories whose total number of steps is `num_steps`.
            self.rollout(self.num_steps, requires_grad=True)

            # Update the model with the trajectories stored in the buffer.
            self.update()

            # Get the mean episode return.
            epoch_return = self.get_mean_episode_return()
            print(f"Epoch {epoch + 1}/{self.num_epochs} - "
                  f"Epoch Return: {epoch_return:.2f}")
            epoch_returns.append(epoch_return)

            # Remember to clear the buffer after updating the model.
            self.buffer.clear()

        if plot:
            self.plot_results(epoch_returns, window_size=window_size)