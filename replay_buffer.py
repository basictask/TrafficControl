"""
 ▄▄▄▄▄▄   ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄▄     ▄▄▄▄▄▄ ▄▄   ▄▄ ▄▄▄▄▄▄▄ ▄▄   ▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄
█   ▄  █ █       █       █   █   █      █  █ █  █  ▄    █  █ █  █       █       █       █   ▄  █
█  █ █ █ █    ▄▄▄█    ▄  █   █   █  ▄   █  █▄█  █ █▄█   █  █ █  █    ▄▄▄█    ▄▄▄█    ▄▄▄█  █ █ █
█   █▄▄█▄█   █▄▄▄█   █▄█ █   █   █ █▄█  █       █       █  █▄█  █   █▄▄▄█   █▄▄▄█   █▄▄▄█   █▄▄█▄
█    ▄▄  █    ▄▄▄█    ▄▄▄█   █▄▄▄█      █▄     ▄█  ▄   ██       █    ▄▄▄█    ▄▄▄█    ▄▄▄█    ▄▄  █
█   █  █ █   █▄▄▄█   █   █       █  ▄   █ █   █ █ █▄█   █       █   █   █   █   █   █▄▄▄█   █  █ █
█▄▄▄█  █▄█▄▄▄▄▄▄▄█▄▄▄█   █▄▄▄▄▄▄▄█▄█ █▄▄█ █▄▄▄█ █▄▄▄▄▄▄▄█▄▄▄▄▄▄▄█▄▄▄█   █▄▄▄█   █▄▄▄▄▄▄▄█▄▄▄█  █▄█
This is the class for the replaybuffer that the reinforcement learning agents use to sample memory.
"""
import torch
import random
import numpy as np
import pandas as pd
from collections import deque
from collections import namedtuple


class ReplayBuffer:
    """
    Replay memory for the Agent. The replay memory gets sampled each update_every iteration that can be set in config.ini
    """
    def __init__(self, buffer_size: int, batch_size: int):
        """
        Create the replaybuffer
        :param buffer_size: How many actions shall fit into the memory at once
        :param batch_size: Size of minibatches
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = int(batch_size)
        self.experience = namedtuple('experience', field_names=['state', 'start', 'end', 'action', 'reward', 'next_state', 'successful'])

    def add(self, state: pd.DataFrame, start: int, end: int, action: int, reward: float, next_state: pd.DataFrame, successful: int):
        """
        Adds a new experience to the replay buffer.
        :param state: State-definition matrix
        :param start: Starting node for the infrastructure
        :param end: Ending node for the infrastructure
        :param action: Action taken in the state
        :param reward: Reward received for the action
        :param next_state: Next state of the environment
        :param successful: If the operation has completed successfully
        :return: None
        """
        self.memory.append(self.experience(state, start, end, action, reward, next_state, successful))

    def sample(self) -> tuple:
        """
        Takes a sample from the replaybuffer
        :return: Batch of previously experienced (states, starts, ends, actions, rewards, next_states, successfuls)
        """
        experiences = random.sample(self.memory, k=self.batch_size)  # Sample the experience buffer

        states = torch.from_numpy(np.vstack([np.array(e.state).flatten() for e in experiences if e is not None])).float()
        starts = torch.from_numpy(np.vstack([e.start for e in experiences if e is not None])).long()
        ends = torch.from_numpy(np.vstack([e.end for e in experiences if e is not None])).long()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([np.array(e.next_state).flatten() for e in experiences if e is not None])).float()
        successfuls = torch.from_numpy(np.vstack([e.successful for e in experiences if e is not None])).long()

        return states, starts, ends, actions, rewards, next_states, successfuls

    def __len__(self):
        """
        :return: Returns the number of elements inside the replay buffer. Overwrites the __len__ property of the object
        """
        return len(self.memory)
