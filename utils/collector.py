import torch
import gymnasium as gym
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

from models.base import BaseAgent

class TrajectoryCollector:
    """Collects trajectories and computes various RL metrics"""
    
    def __init__(self, env: gym.Env, agent: BaseAgent):
        """
        Args:
            env: Gymnasium environment
            agent: RL agent that implements BaseAgent
        """
        self.env = env
        self.agent = agent
        
    def collect_trajectory(self, max_steps: int = 1000, enable_grad: bool = False) -> Dict[str, Any]:
        """Collect a single trajectory of exactly max_steps unless the environment has no more episodes.
        
        Args:
            max_steps: Number of steps to collect
            enable_grad: Whether to enable gradient tracking during collection
            
        Returns:
            Dictionary containing:
                states: List of states
                actions: List of actions
                rewards: List of rewards
                next_states: List of next states
                terminated: List of termination flags
                truncated: List of truncation flags
                masks: List of continuation masks (0 for episode end, 1 otherwise)
                info: List of info dicts from environment
                action_info: List of additional info from action selection
        """
        states = []
        actions = []
        rewards = []
        next_states = []
        terminateds = []
        truncateds = []
        masks = []
        infos = []
        action_infos = []
        
        state, info = self.env.reset()
        steps_taken = 0
        
        while steps_taken < max_steps:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Select action with appropriate gradient context
            with torch.set_grad_enabled(enable_grad):
                action_tuple = self.agent.select_action(state_tensor)
                action = action_tuple[0]
                action_info = action_tuple[1:]
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
            action_infos.append(action_info)
            
            # Add mask (0 if episode ended, 1 if continuing)
            masks.append(0.0 if (terminated or truncated) else 1.0)
            
            steps_taken += 1
            
            # Reset environment if episode ended and we need more steps
            if (terminated or truncated) and steps_taken < max_steps:
                state, info = self.env.reset()
            else:
                state = next_state
                
        return {
            'states': torch.FloatTensor(np.array(states)),
            'actions': torch.tensor(actions),
            'rewards': rewards,
            'next_states': torch.FloatTensor(np.array(next_states)),
            'terminated': terminateds,
            'truncated': truncateds,
            'masks': torch.FloatTensor(masks),
            'infos': infos,
            'action_infos': list(zip(*action_infos))  # Unzip the action infos
        }
    
    def collect_trajectories(self, num_trajectories: int,
                           max_steps: int = 1000,
                           enable_grad: bool = False) -> List[Dict[str, Any]]:
        """Collect multiple trajectories.
        
        Args:
            num_trajectories: Number of trajectories to collect
            max_steps: Maximum steps per trajectory
            enable_grad: Whether to enable gradient tracking during collection
            
        Returns:
            List of trajectory dictionaries
        """
        trajectories = []
        for _ in range(num_trajectories):
            trajectory = self.collect_trajectory(max_steps, enable_grad)
            trajectories.append(trajectory)
        return trajectories

    @classmethod
    def compute_mean_episode_length(
        self,
        trajectory: Dict[str, Any]
    ) -> float:
        """Compute the mean episode length of a trajectory.
        
        Args:
            trajectory: Trajectory dictionary containing rewards and masks,
                        each episode is separated by a mask value of 0
            
        Returns:
            Mean episode length
        """
        # First extract the episodes by mask value
        episodes = []
        current_episode = []
        assert len(trajectory['rewards']) == len(trajectory['masks']), \
               "The length of rewards and masks must be the same"
        for reward, mask in zip(trajectory['rewards'], trajectory['masks']):
            current_episode.append(reward)
            if mask == 0:
                episodes.append(current_episode)
                current_episode = []
        return np.mean([len(episode) for episode in episodes])
    
    @classmethod
    def compute_mean_episode_return(
        self,
        trajectory: Dict[str, Any]
    ) -> float:
        """Compute the mean episode return of a trajectory.
        
        Args:
            trajectory: Trajectory dictionary containing rewards and masks,
                        each episode is separated by a mask value of 0
            
        Returns:
            Mean episode return
        """ 
        # First extract the episodes by mask value
        episodes_returns = []
        current_episode_returns = []
        assert len(trajectory['rewards']) == len(trajectory['masks']), \
               "The length of rewards and masks must be the same"
        for reward, mask in zip(trajectory['rewards'], trajectory['masks']):
            current_episode_returns.append(reward)
            if mask == 0:
                episodes_returns.append(current_episode_returns)
                current_episode_returns = []
        return np.mean([sum(returns) for returns in episodes_returns])
    
    @classmethod
    def compute_returns(
        self,
        trajectory: Dict[str, Any],
        gamma: float,
    ) -> torch.Tensor:
        """Compute discounted returns with proper masking for episode boundaries.
        
        Args:
            trajectory: Trajectory dictionary containing rewards and masks
            gamma: Discount factor
            
        Returns:
            Tensor of discounted returns
        """
        rewards = torch.tensor(trajectory['rewards'])
        masks = trajectory['masks']
        returns = torch.zeros_like(rewards)
        
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * masks[t]
            returns[t] = running_return
            
        return returns

    @classmethod
    def compute_advantages(
        self,
        trajectory: Dict[str, Any],
        gamma: float,
        gae_lambda: float
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation (GAE).
        
        Args:
            trajectory: Trajectory dictionary containing rewards, values, and masks
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Tensor of advantage estimates
        """
        rewards = torch.tensor(trajectory['rewards'])
        values = torch.stack(trajectory['action_infos'][1])  # Assuming values are second item
        masks = trajectory['masks']
        
        advantages = torch.zeros_like(rewards)
        running_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value * masks[t] - values[t]
            running_advantage = delta + gamma * gae_lambda * running_advantage * masks[t]
            advantages[t] = running_advantage
            
        return advantages

    @classmethod
    def compute_td_errors(
        self,
        trajectory: Dict[str, Any],
        gamma: float
    ) -> torch.Tensor:
        """Compute TD errors for value function learning.
        
        Args:
            trajectory: Trajectory dictionary containing rewards, values, and masks
            gamma: Discount factor
            
        Returns:
            Tensor of TD errors
        """
        rewards = torch.tensor(trajectory['rewards'])
        values = torch.stack(trajectory['action_infos'][1])  # Assuming values are second item
        masks = trajectory['masks']
        
        td_errors = torch.zeros_like(rewards)
        
        for t in range(len(rewards) - 1):
            td_target = rewards[t] + gamma * values[t + 1] * masks[t]
            td_errors[t] = td_target - values[t]
            
        # Handle last step
        td_errors[-1] = rewards[-1] - values[-1]  # No next state for last step
        
        return td_errors 