import gymnasium as gym
import torch
import numpy as np
from typing import Tuple
from .BaseAgent import BaseAgent
from .models import PPO

class PPOAgent(BaseAgent):

    def __init__(
        self,
        env_id: str | gym.envs.registration.EnvSpec,
        **kwargs,
    ) -> None:
        """Initialize the PPO agent."""
        super().__init__(env_id, **kwargs)

        # Configure the model.
        self.model = PPO(
            observation_space=self.observation_space,
            action_space=self.action_space,
            **kwargs,
        ).to(self.device)

        # Parameters for loss functions.
        self.clip_ratio = kwargs.get("clip_ratio", 0.2)
        self.value_coef = kwargs.get("value_coef", 0.5)
        self.gamma = kwargs.get("gamma", 0.99)
        self.gae_lambda = kwargs.get("gae_lambda", 0.98)
        self.entropy_coef = kwargs.get("entropy_coef", 0.01)
        self.normalize = kwargs.get("normalize", True)

        # Parameters for training.
        self.num_epochs = kwargs.get("num_epochs", 100)
        self.num_steps = kwargs.get("num_steps", 1000)
        self.num_minibatches = kwargs.get("num_minibatches", 50)
        self.batch_size = self.num_steps // self.num_minibatches

        # Parameters for gradient clipping.
        self.max_grad_norm = kwargs.get("max_grad_norm", 0.5)

        # Save settings.
        self._set_model_settings()
    
    def update(
        self,
    ) -> Tuple[float, float, float]:
        """Update the model using the PPO algorithm."""
        shuffled_indices = torch.randperm(self.buffer.buffer_size)
        o_returns = self.get_discounted_returns(
            gamma=self.gamma,
            normalize=False,
        ).to(self.device)
        
        o_advantages = self.get_advantages(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            normalize=self.normalize,
        ).to(self.device)

        policy_losses, value_losses, entropy_losses = [], [], []
        for start in range(0, self.buffer.buffer_size, self.batch_size):
            end = start + self.batch_size
            b_indices = shuffled_indices[start:end]

            # ob means old batch.
            ob_observations = torch.tensor(
                np.array([
                    self.buffer.get_observation_list()[i] for i in b_indices
                ]),
                dtype=torch.float32
            ).to(self.device)

            ob_actions = torch.tensor(
                np.array([
                    self.buffer.get_action_list()[i] for i in b_indices
                ]),
                dtype=torch.long
            ).to(self.device)

            ob_log_probs = torch.stack([
                self.buffer.get_action_info_list()[i]["log_prob"] for i in b_indices
            ]).to(self.device)

            ob_returns = o_returns[b_indices]
            ob_advantages = o_advantages[b_indices]
            
            # Compute the logits, values and entropies of current policy.
            _, nb_action_info = self.model.select_action(
                observation=ob_observations,
                action=ob_actions,
                requires_grad=True,
            )
            nb_log_probs = nb_action_info["log_prob"]
            nb_values = nb_action_info["value"]
            nb_entropies = nb_action_info["entropy"]

            # Compute the ratio (pi_theta / pi_theta_old)
            b_ratio = torch.exp(nb_log_probs - ob_log_probs)
            
            # Compute the policy loss.
            surr1 = b_ratio * ob_advantages
            surr2 = torch.clamp(
                b_ratio, 
                1 - self.clip_ratio,
                1 + self.clip_ratio) * ob_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Compute the value loss.
            value_loss = (nb_values - ob_returns).pow(2).mean()

            # Compute the entropy loss.
            entropy_loss = nb_entropies.mean()

            # Compute the total loss.
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
            
            # Update the model.
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_grad_norm
            )
            self.optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
        
        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropy_losses)
    
    def train(
        self,
        plot: bool = True,
        window_size: int = 10,
    ) -> None:
        """Train the agent."""
        epoch_returns = []
        for epoch in range(self.num_epochs):
            self.rollout(self.num_steps, requires_grad=False)
            policy_loss, value_loss, entropy_loss = self.update()

            epoch_return = self.get_mean_episode_return()
            print(f"Epoch {epoch + 1}/{self.num_epochs} - "
                  f"Return: {epoch_return:.2f} - "
                  f"Policy Loss: {policy_loss:.2f} - "
                  f"Value Loss: {value_loss:.2f} - "
                  f"Entropy Loss: {entropy_loss:.2f}")
            
            if (epoch + 1) % self.model_save_interval == 0:
                self.save_model(epoch + 1)
            
            epoch_returns.append(epoch_return)

            # Clear buffer after update.
            self.buffer.clear()
        
        if plot:
            self.plot_results(epoch_returns, window_size)
