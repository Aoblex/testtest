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