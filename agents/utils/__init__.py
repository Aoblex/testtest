import random
import numpy as np
import torch
from typing import Generator


class ReplayBuffer:

    def __init__(self) -> None:
        self.next_state_list = []
        self.reward_list = []
        self.terminated_list = []
        self.truncated_list = []
        self.info_list = []

        self.state_list = []
        self.action_list = []
        self.action_info_list = []
        self.mask_list = []
        self.buffer_size = 0
    
    def get_next_state_list(self) -> list:
        """Get the next state list of the buffer."""
        return self.next_state_list
    
    def get_reward_list(self) -> list:
        """Get the reward list of the buffer."""
        return self.reward_list
    
    def get_terminated_list(self) -> list:
        """Get the terminated list of the buffer."""
        return self.terminated_list
    
    def get_truncated_list(self) -> list:
        """Get the truncated list of the buffer."""
        return self.truncated_list  
    
    def get_info_list(self) -> list:
        """Get the info list of the buffer."""
        return self.info_list
    
    def get_state_list(self) -> list:
        """Get the state list of the buffer."""
        return self.state_list  
    
    def get_action_list(self) -> list:
        """Get the action list of the buffer."""
        return self.action_list
    
    def get_action_info_list(self) -> list:
        """Get the action info list of the buffer."""
        return self.action_info_list
    
    def get_mask_list(self) -> list:
        """Get the mask list of the buffer."""
        return self.mask_list
    
    def push(
        self,
        next_state: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
        state: np.ndarray,
        action: any,
        action_info: dict,
    ):
        self.next_state_list.append(next_state)
        self.reward_list.append(reward)
        self.terminated_list.append(terminated)
        self.truncated_list.append(truncated)
        self.info_list.append(info)
        self.state_list.append(state)
        self.action_list.append(action)
        self.action_info_list.append(action_info)

        if terminated or truncated:
            self.mask_list.append(0.0)
        else:
            self.mask_list.append(1.0)
        
        self.buffer_size += 1

        assert len(self.next_state_list) \
            == len(self.reward_list) \
            == len(self.terminated_list) \
            == len(self.truncated_list) \
            == len(self.info_list) \
            == len(self.state_list) \
            == len(self.action_list) \
            == len(self.action_info_list) \
            == len(self.mask_list) \
            == self.buffer_size, \
            "The length of the buffer is not consistent."
        
    def clear(self) -> None:
        self.next_state_list = []
        self.reward_list = []
        self.terminated_list = []
        self.truncated_list = []
        self.info_list = []

        self.state_list = []
        self.action_list = []
        self.action_info_list = []
        self.mask_list = []

        self.buffer_size = 0
    
    def _shuffle(self) -> None:
        indices = np.random.permutation(len(self.state_list))
        self.state_list = [self.state_list[i] for i in indices]
        self.action_list = [self.action_list[i] for i in indices]
        self.reward_list = [self.reward_list[i] for i in indices]
        self.next_state_list = [self.next_state_list[i] for i in indices]
        self.terminated_list = [self.terminated_list[i] for i in indices]
        self.truncated_list = [self.truncated_list[i] for i in indices] 
        self.info_list = [self.info_list[i] for i in indices]
        self.action_info_list = [self.action_info_list[i] for i in indices]
        self.mask_list = [self.mask_list[i] for i in indices]

    def sample(
        self,
        batch_size: int
    ) -> Generator[dict, None, None]:
        self._shuffle()

        for i in range(0, len(self.state_list), batch_size):    
            yield {
                "state": self.state_list[i:i+batch_size],
                "action": self.action_list[i:i+batch_size],
                "reward": self.reward_list[i:i+batch_size],
                "next_state": self.next_state_list[i:i+batch_size],
                "terminated": self.terminated_list[i:i+batch_size],
                "truncated": self.truncated_list[i:i+batch_size],
                "info": self.info_list[i:i+batch_size],
                "action_info": self.action_info_list[i:i+batch_size],
                "mask": self.mask_list[i:i+batch_size],
            }