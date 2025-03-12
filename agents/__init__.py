import gymnasium as gym
import numpy as np
import torch
from typing import Tuple, List, Optional
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from .utils import ReplayBuffer
from .models import (
    PPO,
    REINFORCE,
    A2C
)

class REINFORCEAgent:

    def __init__(
        self,
        env_id: str | gym.envs.registration.EnvSpec,
        **kwargs,
    ) -> None:
        """Initialize the REINFORCE agent."""

        # Configure the environment.
        self.env = gym.make(env_id)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        if not isinstance(self.observation_space, gym.spaces.Box):
            raise ValueError("Observation space must be a Box.")

        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise ValueError("Action space must be a Discrete.")

        # Configure the model.
        self.device = kwargs.pop(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = REINFORCE(
            state_dim=self.observation_space.shape[0],
            action_dim=self.action_space.n,
            device=self.device,
        ).to(self.device)

        # Initialize the buffer.
        self.buffer = ReplayBuffer()

        # Training parameters.
        self.gamma = kwargs.pop("gamma", 0.99)
        self.num_epochs = kwargs.pop("num_epochs", 4)
        self.num_steps = kwargs.pop("num_steps", 1024)

        # Save settings.
        self.model_save_interval = kwargs.pop("model_save_interval", 100)
        self.model_save_dir = kwargs.pop(
            "model_save_dir",
            Path("results") / self.env.unwrapped.spec.id / \
                self.model.__class__.__name__ / \
                datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

    def rollout(
        self,
        num_steps: int
    ) -> ReplayBuffer:
        """Rollout the agent for a given number of steps.
        During rollout, the gradient should be enabled,
        because we need to compute the gradient of the policy.
        
        Args:
            num_steps: Number of steps to rollout the agent for.
            
        Returns:
            buffer: The replay buffer containing the rollout.
        """
        state, info = self.env.reset()
        terminated = False
        truncated = False

        for _ in range(num_steps):
            # We need to require gradients here to 
            # get the computation graph for log_prob.
            action, action_info = self.model.select_action(
                state,
                requires_grad=True
            )
            next_state, reward, terminated, truncated, info = self.env.step(action)

            self.buffer.push(
                next_state,
                reward,
                terminated,
                truncated,
                info,
                state,
                action,
                action_info
            )

            if terminated or truncated:
                state, info = self.env.reset()
            else:
                state = next_state
        
        return self.buffer

    def get_mean_episode_return(
        self,
    ) -> float:
        """Get the mean episode return of the agent."""
        episodes_returns = []
        current_episode_returns = []
        for reward, mask in zip(self.buffer.get_reward_list(),
                                self.buffer.get_mask_list()):
            current_episode_returns.append(reward)
            if mask == 0:
                episodes_returns.append(current_episode_returns)
                current_episode_returns = []
        # Remember to append the last episode.
        episodes_returns.append(current_episode_returns)

        return np.mean([sum(returns) for returns in episodes_returns])
   
    def get_returns(
        self,
        gamma: float,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Get the returns of the agent. Usually represented as G_t in the literature."""
        rewards = torch.tensor(self.buffer.get_reward_list())
        masks = torch.tensor(self.buffer.get_mask_list())
        returns = torch.zeros_like(rewards)

        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * masks[t]
            returns[t] = running_return

        if normalize:
            returns = (returns - returns.mean()) / returns.std()

        return returns

    def update(
        self,
    ) -> None:
        """Update the model in agent."""
        returns = self.get_returns(gamma=self.gamma, normalize=True).to(self.device)
        action_info = self.buffer.get_action_info_list()
        log_probs = torch.stack([info["log_prob"] for info in action_info]).to(self.device)

        loss = -torch.mean(returns * log_probs)
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

    def save_model(
        self,
        epoch: int,
    ) -> None:
        """Save the model."""
        torch.save(
            self.model.state_dict(),
            self.model_save_dir / f"REINFORCE_epoch_{epoch}.pt"
        )

    def load_model(
        self,
        epoch: int,
    ) -> None:
        """Load the model."""
        self.model.load_state_dict(
            torch.load(self.model_save_dir / f"REINFORCE_epoch_{epoch}.pt")
        )

    def plot_results(
        self,
        epoch_returns: List[float],
        window_size: int = 10,
    ) -> None:
        """Plot the smoothed results.
        Args:
            epoch_returns: List of returns for each epoch.
            window_size: Size of the window to smooth the results.
        """
        # Plot the smoothed returns.
        smoothed_returns = np.convolve(epoch_returns, np.ones(window_size) / window_size, mode="valid")
        plt.plot(smoothed_returns, label="Smoothed Return")
        plt.xlabel("Epoch")
        plt.ylabel("Return")
        plt.title(f"Smoothed Return (Window Size: {window_size})")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(self.model_save_dir / "results.png")
        plt.close()

    def train(
        self,
        num_epochs: Optional[int] = None,
        num_steps: Optional[int] = None,
        plot: bool = True,
        window_size: int = 10,
    ) -> None:

        if num_epochs is None:
            num_epochs = self.num_epochs

        if num_steps is None:
            num_steps = self.num_steps

        epoch_returns = []
        for epoch in range(num_epochs):
            self.rollout(num_steps)
            self.update()

            epoch_return = self.get_mean_episode_return()
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Epoch Return: {epoch_return:.2f}")

            if (epoch + 1) % self.model_save_interval == 0:
                self.save_model(epoch)

            epoch_returns.append(epoch_return)

            # Remember to clear the buffer after updating the model.
            self.buffer.clear()

        if plot:
            self.plot_results(epoch_returns, window_size=window_size)

class A2CAgent:

    def __init__(
        self,
        env_id: str | gym.envs.registration.EnvSpec,
        **kwargs,
    ) -> None:
        """Initialize the A2C agent."""

        # Configure the environment.
        self.env = gym.make(env_id)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        if not isinstance(self.observation_space, gym.spaces.Box):
            raise ValueError("Observation space must be a Box.")

        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise ValueError("Action space must be a Discrete.")

        # Configure the model.
        self.device = kwargs.pop(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = A2C(
            state_dim=self.observation_space.shape[0],
            action_dim=self.action_space.n,
            device=self.device,
        ).to(self.device)

        # Initialize the buffer.
        self.buffer = ReplayBuffer()

        # Training parameters.
        self.num_epochs = kwargs.pop("num_epochs", 100)
        self.num_steps = kwargs.pop("num_steps", 1000)
        self.gae_lambda = kwargs.pop("gae_lambda", 0.95) # GAE parameters.
        self.gamma = kwargs.pop("gamma", 0.99) # Return discount factor.

        # Save settings
        self.model_save_interval = kwargs.pop("model_save_interval", 100)
        self.model_save_dir = kwargs.pop(
            "model_save_dir",
            Path("results") / self.env.unwrapped.spec.id / \
                self.model.__class__.__name__ / \
                datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

    def rollout(
        self,
        num_steps: int
    ) -> ReplayBuffer:
        """Rollout the agent for a given number of steps.
        
        Args:
            num_steps: Number of steps to rollout the agent for.
            
        Returns:
            buffer: The replay buffer containing the rollout.
        """
        state, info = self.env.reset()
        terminated = False
        truncated = False

        for _ in range(num_steps):
            action, action_info = self.model.select_action(
                state,
                requires_grad=True
            )
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            self.buffer.push(
                next_state,
                reward,
                terminated,
                truncated,
                info,
                state,
                action,
                action_info
            )

            if terminated or truncated:
                state, info = self.env.reset()
            else:
                state = next_state

        return self.buffer

    def get_mean_episode_return(
        self,
    ) -> float:
        """Get the mean episode return of the agent."""
        episodes_returns = []
        current_episode_returns = []
        for reward, mask in zip(self.buffer.get_reward_list(),
                                self.buffer.get_mask_list()):
            current_episode_returns.append(reward)
            if mask == 0:
                episodes_returns.append(current_episode_returns)
                current_episode_returns = []
        
        # Remember to append the last episode.
        episodes_returns.append(current_episode_returns)
    
        return np.mean([sum(returns) for returns in episodes_returns])

    def get_td_errors_and_advantages(
        self,
        gamma: Optional[float]= None,
        gae_lambda: Optional[float]= None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if gamma is None:
            gamma = self.gamma

        if gae_lambda is None:
            gae_lambda = self.gae_lambda

        td_errors = torch.zeros(self.buffer.buffer_size)
        advantages = torch.zeros(self.buffer.buffer_size)

        rewards = torch.tensor(self.buffer.get_reward_list()).to(self.device)
        masks = torch.tensor(self.buffer.get_mask_list()).to(self.device)
        values = torch.stack([info["value"] for info in self.buffer.get_action_info_list()]).to(self.device)

        running_gae = 0
        for t in reversed(range(self.buffer.buffer_size - 1)):
            td_errors[t] = rewards[t] + gamma * values[t + 1] * masks[t] - values[t]
            advantages[t] = td_errors[t] + gamma * gae_lambda * masks[t] * running_gae
            running_gae = advantages[t] + gamma * gae_lambda * masks[t] * running_gae

        return td_errors, advantages

    def update(
        self,
    ) -> None:
        """Update the model using A2C algorithm."""
        # Get data from buffer
        action_info = self.buffer.get_action_info_list()
        
        # Extract log probabilities and values
        log_probs = torch.stack([info["log_prob"] for info in action_info]).to(self.device)
        
        # Compute GAE and returns
        td_errors, advantages = self.get_td_errors_and_advantages(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        td_errors = td_errors.to(self.device)
        advantages = advantages.to(self.device)
        
        # Compute losses
        policy_loss = -(advantages.detach() * log_probs).mean()
        value_loss = advantages.pow(2).mean()
        
        # Combined loss (negative because we want to maximize)
        total_loss = policy_loss + value_loss
        
        # Update model
        self.model.optimizer.zero_grad()
        total_loss.backward()
        self.model.optimizer.step()

    def save_model(
        self,
        epoch: int,
    ) -> None:
        """Save the model."""
        torch.save(
            self.model.state_dict(),
            self.model_save_dir / f"A2C_epoch_{epoch}.pt"
        )

    def load_model(
        self,
        epoch: int,
    ) -> None:
        """Load the model."""
        self.model.load_state_dict(
            torch.load(self.model_save_dir / f"A2C_epoch_{epoch}.pt")
        )

    def plot_results(
        self,
        epoch_returns: List[float],
    ) -> None:
        """Plot training results."""
        plt.plot(epoch_returns, label="Return")
        plt.xlabel("Epoch")
        plt.ylabel("Return")
        plt.title("Return")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(self.model_save_dir / "results.png")
        plt.close()

    def train(
        self,
        num_epochs: Optional[int] = None,
        num_steps: Optional[int] = None,
        plot: bool = True,
    ) -> None:
        """Train the agent."""
        if num_epochs is None:
            num_epochs = self.num_epochs

        if num_steps is None:
            num_steps = self.num_steps

        epoch_returns = []
        for epoch in range(num_epochs):
            self.rollout(num_steps)
            self.update()

            epoch_return, epoch_length = self.get_mean_episode_return_and_length()
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Return: {epoch_return:.2f}, Length: {epoch_length:.2f}")

            if (epoch + 1) % self.model_save_interval == 0:
                self.save_model(epoch)

            epoch_returns.append(epoch_return)

            # Clear buffer after update
            self.buffer.clear()

        if plot:
            self.plot_results(epoch_returns)

__all__ = [
    "PPO", "REINFORCE", "A2C"
]