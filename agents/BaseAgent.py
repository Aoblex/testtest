import gymnasium as gym
import numpy as np
import torch
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from .models import BaseModel
from .utils import ReplayBuffer

class BaseAgent:

    def __init__(
        self,
        env_id: str | gym.envs.registration.EnvSpec,
        **kwargs,
    ) -> None:
        """Initialize the base agent.
        After init, the following attributes should be set:
            - self.env: The environment.
            - self.observation_space: The observation space.
            - self.action_space: The action space.
            - self.model: Defaults to None.
            - self.device: The device to run the model on.
            - self.buffer: The replay buffer.
            - self.kwargs: The kwargs.
        """

        # Configure the environment.
        self.env = gym.make(env_id)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Configure the model.
        self.device = kwargs.get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self.model: BaseModel = None

        # Initialize the buffer.
        self.buffer = ReplayBuffer()

        # Save kwargs
        self.kwargs = kwargs
    
    def _set_model_settings(self,):
        """Set the model settings."""
        # Save settings.
        self.algorithm_name: str = self.model.__class__.__name__
        self.model_save_interval = self.kwargs.pop("model_save_interval", 100)
        self.model_save_dir = self.kwargs.pop(
            "model_save_dir",
            Path("results") / self.env.unwrapped.spec.id / \
                self.algorithm_name / \
                datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # Configure the optimizer.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.kwargs.pop("lr", 3e-4)
        )

    def rollout(
        self,
        num_steps: int,
        requires_grad: bool = True,
    ) -> ReplayBuffer:
        """Rollout the agent for a given number of steps.
        During rollout, the gradient should be enabled,
        because we need to compute the gradient of the policy.
        
        Args:
            num_steps: Number of steps to rollout the agent for.
            requires_grad: Whether to require gradients for the policy.
 
        Returns:
            buffer: The replay buffer containing the rollout.
        """
        state, info = self.env.reset()

        for _ in range(num_steps):
            action, action_info = self.model.select_action(
                state,
                requires_grad=requires_grad
            )

            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

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
    
    def get_discounted_returns(
        self,
        gamma: float,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Get the discounted returns of the agent.
        Usually represented as G_t in the literature.
        
        Args:
            gamma: The discount factor.
            normalize: Whether to normalize the returns.
        """
        rewards = torch.tensor(self.buffer.get_reward_list())
        masks = torch.tensor(self.buffer.get_mask_list())
        returns = torch.zeros_like(rewards)

        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * masks[t]
            returns[t] = running_return

        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def get_advantages(
        self,
        gamma: float,
        gae_lambda: float,
        normalize: bool,
    ) -> torch.Tensor:
        advantages = torch.zeros(self.buffer.buffer_size)

        rewards = torch.tensor(self.buffer.get_reward_list()).to(self.device)
        masks = torch.tensor(self.buffer.get_mask_list()).to(self.device)
        values = torch.stack([info["value"] for info in self.buffer.get_action_info_list()]).to(self.device)

        running_gae = 0
        for t in reversed(range(self.buffer.buffer_size - 1)):
        # https://discuss.pytorch.org/t/categorical-distribution-returning-breaking/165343
            td_error = rewards[t] \
                + gamma * values[t + 1] * masks[t] \
                - values[t]
            running_gae = td_error \
                + running_gae * gamma * gae_lambda * masks[t]
            advantages[t] = running_gae

        # Normalizing the advantages greatly improves the stability of the training process.
        # Notice that when it's normalized, the value loss will always be 1,
        # and the policy loss is always close to 0.(learn about this later)
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages
    
    def save_model(
        self,
        epoch: int,
    ) -> None:
        """Save the model."""
        torch.save(
            self.model.state_dict(),
            self.model_save_dir / f"{self.algorithm_name}_{epoch}.pth"
        )

    def load_model(
        self,
        epoch: int,
    ) -> None:
        """Load the model."""
        self.model.load_state_dict(
            torch.load(self.model_save_dir / f"{self.algorithm_name}_{epoch}.pth")
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

    def update(self,):
        """One epoch of training."""
        raise NotImplementedError("This method should be implemented by the subclass.")

    def train(
        self,
        plot: bool = True,
        window_size: int = 10,
    ):
        """Train the agent.
        Args:
            plot: Whether to plot the results.
            window_size: Size of the window to smooth the results.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")