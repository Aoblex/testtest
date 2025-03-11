import torch
import gymnasium as gym
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

from models.base import BaseAgent

class TrajectoryCollector:
    """Collects trajectories by running an agent in an environment"""
    
    def __init__(self, env: gym.Env, agent: BaseAgent):
        """
        Args:
            env: Gymnasium environment
            agent: RL agent that implements BaseAgent
        """
        self.env = env
        self.agent = agent
        
    def collect_trajectory(self, max_steps: int = 1000) -> Dict[str, Any]:
        """Collect a single trajectory.
        
        Args:
            max_steps: Maximum number of steps to take
            
        Returns:
            Dictionary containing:
                states: List of states
                actions: List of actions
                rewards: List of rewards
                next_states: List of next states
                terminated: List of termination flags
                truncated: List of truncation flags
                info: List of info dicts from environment
                action_info: List of additional info from action selection
        """
        state, info = self.env.reset()
        terminated = False
        truncated = False
        
        states = []
        actions = []
        rewards = []
        next_states = []
        terminateds = []
        truncateds = []
        infos = []
        action_infos = []
        
        for _ in range(max_steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Select action
            action_tuple = self.agent.select_action(state_tensor)
            action = action_tuple[0]
            action_info = action_tuple[1:]  # Additional info (e.g., log_probs, values)
            
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
            
            state = next_state
            
            if terminated or truncated:
                break
                
        return {
            'states': torch.FloatTensor(np.array(states)),
            'actions': torch.tensor(actions),
            'rewards': rewards,
            'next_states': torch.FloatTensor(np.array(next_states)),
            'terminated': terminateds,
            'truncated': truncateds,
            'infos': infos,
            'action_infos': list(zip(*action_infos))  # Unzip the action infos
        }
    
    def collect_trajectories(self, num_trajectories: int,
                           max_steps: int = 1000) -> List[Dict[str, Any]]:
        """Collect multiple trajectories.
        
        Args:
            num_trajectories: Number of trajectories to collect
            max_steps: Maximum steps per trajectory
            
        Returns:
            List of trajectory dictionaries
        """
        trajectories = []
        for _ in range(num_trajectories):
            trajectory = self.collect_trajectory(max_steps)
            trajectories.append(trajectory)
        return trajectories 