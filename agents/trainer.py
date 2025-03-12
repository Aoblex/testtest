import torch
import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, List, Type
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

from models.base import BaseAgent
from models.ppo import PPO
from utils.collector import TrajectoryCollector

class RLTrainer:
    """A trainer class that handles the complete RL training process"""
    
    def __init__(
        self,
        env_id: str,
        agent_class: Type[BaseAgent],
        agent_kwargs: Dict[str, Any],
        save_dir: str = "results",
        seed: Optional[int] = None
    ):
        """
        Args:
            env_id: Gymnasium environment ID
            agent_class: Class of the RL agent to use
            agent_kwargs: Arguments to initialize the agent
            save_dir: Directory to save results and models
            seed: Random seed for reproducibility
        """
        # Set up environment
        self.env = gym.make(env_id)
        if seed is not None:
            self.env.reset(seed=seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Initialize agent
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
        else:
            action_dim = self.env.action_space.shape[0]
            
        self.agent = agent_class(
            state_dim=state_dim,
            action_dim=action_dim,
            **agent_kwargs
        )
        
        # Set up collector
        self.collector = TrajectoryCollector(self.env, self.agent)
        
        # Set up saving
        self.save_dir = Path(save_dir) / env_id / agent_class.__name__ / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.train_returns: List[float] = []
        self.eval_returns: List[float] = []
        self.episode_lengths: List[int] = []
        
    def train(
        self,
        num_episodes: int,
        max_steps: int = 1000,
        eval_interval: int = 10,
        num_eval_episodes: int = 5,
        save_interval: int = 100
    ) -> None:
        """Train the agent.
        
        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            eval_interval: Episodes between evaluations
            num_eval_episodes: Number of episodes for evaluation
            save_interval: Episodes between saving checkpoints
        """

        if isinstance(self.agent, PPO):
            enable_grad = False
        else:
            enable_grad = True

        for episode in range(num_episodes):
            # Collect trajectory without gradients for old policy
            if enable_grad:
                self.agent.train()  # Still in train mode, but will collect without gradients
            else:
                self.agent.eval()
            
            # if current agent uses PPO algorithm, we need to collect trajectory without gradients
            trajectory = self.collector.collect_trajectory(max_steps, enable_grad=enable_grad)
            
            # Update agent
            metrics = self.agent.update(trajectory)
            
            # Log metrics
            episode_return = TrajectoryCollector.compute_mean_episode_return(trajectory)
            episode_length = TrajectoryCollector.compute_mean_episode_length(trajectory)
            self.train_returns.append(episode_return)
            self.episode_lengths.append(episode_length)
            
            # Print progress
            print(f"Episode {episode + 1}/{num_episodes} - Mean Return: {episode_return:.2f}, Mean Length: {episode_length}")
            for name, value in metrics.items():
                print(f"  {name}: {value:.4f}")
            
            # Evaluate
            if (episode + 1) % eval_interval == 0:
                eval_return = self.evaluate(num_eval_episodes, max_steps)
                self.eval_returns.append(eval_return)
                print(f"Evaluation Return: {eval_return:.2f}")
                
            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                self.save_checkpoint(episode + 1)
                self.plot_results()
                
        # Final save and plot
        self.save_checkpoint(num_episodes)
        self.plot_results()
        
    def evaluate(self, num_episodes: int, max_steps: int = 1000) -> float:
        """Evaluate the agent.
        
        Args:
            num_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            
        Returns:
            Mean return over evaluation episodes
        """
        self.agent.eval()
        returns = []
        
        for _ in range(num_episodes):
            # During evaluation, we don't need gradients
            trajectory = self.collector.collect_trajectory(max_steps, enable_grad=False)
            returns.append(
                TrajectoryCollector.compute_mean_episode_return(trajectory)
            )
            
        return np.mean(returns)
    
    def save_checkpoint(self, episode: int) -> None:
        """Save a training checkpoint.
        
        Args:
            episode: Current episode number
        """
        checkpoint = {
            'episode': episode,
            'agent_state': self.agent.state_dict() if hasattr(self.agent, 'state_dict') else None,
            'train_returns': self.train_returns,
            'eval_returns': self.eval_returns,
            'episode_lengths': self.episode_lengths
        }
        
        torch.save(checkpoint, self.save_dir / f"checkpoint_{episode}.pt")
        
    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path)
        if hasattr(self.agent, 'load_state_dict') and checkpoint['agent_state'] is not None:
            self.agent.load_state_dict(checkpoint['agent_state'])
        self.train_returns = checkpoint['train_returns']
        self.eval_returns = checkpoint['eval_returns']
        self.episode_lengths = checkpoint['episode_lengths']
        
    def plot_results(self) -> None:
        """Plot and save training results."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot returns
        ax1.plot(self.train_returns, label='Train Returns', alpha=0.6)
        if self.eval_returns:
            eval_episodes = list(range(0, len(self.train_returns), len(self.train_returns) // len(self.eval_returns)))
            ax1.plot(eval_episodes, self.eval_returns, label='Eval Returns', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Return')
        ax1.set_title('Training and Evaluation Returns')
        ax1.legend()
        ax1.grid(True)
        
        # Plot episode lengths
        ax2.plot(self.episode_lengths, label='Episode Length')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Lengths')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_results.png')
        plt.close()
        
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict action for a given state.
        
        Args:
            state: Environment state
            
        Returns:
            Selected action
        """
        self.agent.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.agent.select_action(state_tensor)[0]
        return action 