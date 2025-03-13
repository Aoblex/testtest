import gymnasium as gym
import torch
from typing import Tuple

from .BaseAgent import BaseAgent
from .models import A2C

class A2CAgent(BaseAgent):

    def __init__(
        self,
        env_id: str | gym.envs.registration.EnvSpec,
        **kwargs,
    ) -> None:
        """Initialize the A2C agent."""
        super().__init__(env_id, **kwargs)
        
        # Configure the model.
        self.model = A2C(
            state_dim=self.observation_space.shape[0],
            action_dim=self.action_space.n,
            **kwargs,
        ).to(self.device)

        # Parameters loss functions
        self.gamma = kwargs.pop("gamma", 0.99)
        self.gae_lambda = kwargs.pop("gae_lambda", 0.98)
        self.normalize = kwargs.pop("normalize", True)

        # Parameters for training.
        self.num_epochs = kwargs.pop("num_epochs", 100)
        self.num_steps = kwargs.pop("num_steps", 1000)

        # Save settings.
        self._set_model_settings()

    def update(
        self,
    ) -> Tuple[float, float]:
        """Update the model."""

        # Get log_probs from action_info.
        action_info = self.buffer.get_action_info_list()
        log_probs = torch.stack([
            info["log_prob"] for info in action_info
        ]).to(self.device)

        # Compute GAE.
        advantages = self.get_advantages(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            normalize=self.normalize,
        ).to(self.device)

        # Compute losses.
        policy_loss = -torch.mean(advantages * log_probs)
        value_loss = advantages.pow(2).mean()

        # Compute total loss.
        total_loss = policy_loss + value_loss

        # Update the model.
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Return the policy loss and value loss.
        return policy_loss.item(), value_loss.item()
        
    def train(
        self,
        plot: bool = True,
        window_size: int = 10,
    ) -> None:
        """Train the agent."""
        epoch_returns = []
        
        for epoch in range(self.num_epochs):
            self.rollout(self.num_steps, requires_grad=True)
            policy_loss, value_loss = self.update()

            epoch_return = self.get_mean_episode_return()
            print(f"Epoch {epoch + 1}/{self.num_epochs} - "
                  f"Return: {epoch_return:.2f}, "
                  f"Policy Loss: {policy_loss:.2f}, Value Loss: {value_loss:.2f}")
            
            if (epoch + 1) % self.model_save_interval == 0:
                self.save_model(epoch + 1)
            
            epoch_returns.append(epoch_return)

            # Clear buffer after update.
            self.buffer.clear()
        
        if plot:
            self.plot_results(epoch_returns, window_size)
