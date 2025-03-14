import gymnasium as gym
import torch
from typing import Tuple

from .BaseAgent import BaseAgent
from .models import A2C
import torch.optim as optim
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

    def update_GAN(self, true_actions, discriminator, optimizer_discriminator, optimizer_generator) -> Tuple[
        float, float]:
        """Update both generator and discriminator in the GAN framework,
           using the true actions from another agent."""
        torch.autograd.set_detect_anomaly(True)

        # 获取当前代理生成的动作 (生成器的输出)
        action_info = self.buffer.get_action_info_list()  # 1000条
        print(len(action_info))
        generated_actions = []

        # 提取当前代理的log_probs，并构造生成器的动作
        for info in action_info:
            generated_actions.append(info["log_prob"])  # 当前代理生成器的输出，log_probs

        # 将生成器输出和真实动作转换为tensor
        generated_actions = torch.stack(generated_actions).to(self.device)
        true_actions = true_actions.to(self.device)

        # 判别器的输出
        discriminator_output_generated = discriminator(generated_actions)  # 判别生成器输出
        discriminator_output_true = discriminator(true_actions)  # 判别真实动作

        # 计算生成器损失（使用生成器输出的判别器评分）
        loss_generator = -torch.mean(discriminator_output_generated)

        # 计算判别器损失（真实动作和生成器输出的损失）
        loss_discriminator = -torch.mean(discriminator_output_true) + torch.mean(discriminator_output_generated)

        # 优化判别器
        optimizer_discriminator.zero_grad()
        loss_discriminator.backward(retain_graph=True)  # 保持计算图以便后续反向传播
        #loss_discriminator.backward()
        optimizer_discriminator.step()

        # 优化生成器
        optimizer_generator.zero_grad()
        loss_generator.backward()
        optimizer_generator.step()

        # 返回生成器和判别器的损失
        return loss_generator.item(), loss_discriminator.item()


    def train_GAN(
            self,
            true_actions=None,  # 传入另一个代理
            discriminator=None,
            plot: bool = True,
            window_size: int = 10,
    ) -> None:
        """Train the agent with both generator and discriminator, using another agent's true actions."""
        epoch_returns = []
        discriminator = discriminator.to(self.device)


        # Initialize optimizer for discriminator and generator
        optimizer_discriminator = optim.Adam(discriminator.parameters(), 1e-3)
        optimizer_generator = optim.Adam(self.model.parameters(),  1e-3)

        for epoch in range(self.num_epochs):
            self.rollout(self.num_steps, requires_grad=True)

            # 更新生成器和判别器，传入另一个代理
            G_loss, D_loss = self.update_GAN(true_actions,discriminator,optimizer_discriminator,optimizer_generator)

            epoch_return = self.get_mean_episode_return()
            print(f"Epoch {epoch + 1}/{self.num_epochs} - "
                  f"Return: {epoch_return:.2f}, "
                  f"Generator Loss: {G_loss:.2f}, Discriminator Loss: {D_loss:.2f}")

            if (epoch + 1) % self.model_save_interval == 0:
                self.save_model(epoch + 1)

            epoch_returns.append(epoch_return)

            # Clear buffer after update.
            self.buffer.clear()

        if plot:
            self.plot_results(epoch_returns, window_size)


