"""
 ▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄    ▄ ▄▄▄▄▄▄▄
█      █       █       █  █  █ █       █
█  ▄   █   ▄▄▄▄█    ▄▄▄█   █▄█ █▄     ▄█
█ █▄█  █  █  ▄▄█   █▄▄▄█       █ █   █
█      █  █ █  █    ▄▄▄█  ▄    █ █   █
█  ▄   █  █▄▄█ █   █▄▄▄█ █ █   █ █   █
█▄█ █▄▄█▄▄▄▄▄▄▄█▄▄▄▄▄▄▄█▄█  █▄▄█ █▄▄▄█
This is the file that defines the behavior specific to the agent in the Reinforcement learning environment.
There can be multiple types of agents defined depending on the task.
The baseline model that was defined was a DDDQN agent
"""
# Own
from suppl import check_all_attributes_initialized
# Generic
import os
import random
import numpy as np
import pandas as pd
from collections import deque
from collections import namedtuple
# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fn
# Arguments
import configparser
args = configparser.ConfigParser()
args.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini'))


class QNetwork(nn.Module):
    """
    Q-network model for the reinforcement learning agent.
    The structure of the network can be defined in the __init__ function.
    """
    def __init__(self, state_shape: tuple, action_size: int, state_high: pd.DataFrame, n_neurons: list):
        """
        Defines the architecture of the model. The model needs to predict 3 values at each iteration:
            - Which action to take
            - Starting node (scaled to a range between 0 and 1)
            - Ending node (scaled to a range between 0 and 1)
        The architecture is a Dueling Double Deep Q-learning with 3 different heads to predict the 3 target variables
        :param state_shape: Shape of the state-definition matrix. Should be n*n
        :param action_size: Integer value on how many actions can the agent take
        :param state_high: highest possible values for all the states (from Environment)
        :param n_neurons: A list of the sizes of each neuron layer. Defined in config.ini
        """
        self.state_shape = state_shape
        self.state_high = state_high
        self.n_nodes = self.state_shape[0]
        self.input_size = self.n_nodes * self.n_nodes
        self.action_size = action_size
        self.n_neurons = n_neurons

        # Neural network architecture #
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(self.input_size, n_neurons[0])  # Input layer
        self.fc2 = nn.Linear(n_neurons[0], n_neurons[1])  # H1
        self.fc3 = nn.Linear(n_neurons[1], n_neurons[2])  # H2

        # Start
        self.fc4_start_value = nn.Linear(n_neurons[2], n_neurons[3])  # H3
        self.start_value = nn.Linear(n_neurons[3], 1)  # V(s) for start

        self.fc4_start_adv = nn.Linear(n_neurons[2], n_neurons[3])  # H3
        self.start_adv = nn.Linear(n_neurons[3], self.n_nodes)  # A(s,a) for start

        # End
        self.fc4_end_value = nn.Linear(n_neurons[2], n_neurons[3])  # H3
        self.end_value = nn.Linear(n_neurons[3], 1)  # V(s) for end

        self.fc4_end_adv = nn.Linear(n_neurons[2], n_neurons[3])  # H3
        self.end_adv = nn.Linear(n_neurons[3] + 1, self.n_nodes)  # A(s,a) for end

        # Action
        self.fc4_action_value = nn.Linear(n_neurons[2], n_neurons[3])  # H3
        self.action_value = nn.Linear(n_neurons[3], 1)  # V(s) for action

        self.fc4_action_adv = nn.Linear(n_neurons[2], n_neurons[3])  # H3
        self.action_adv = nn.Linear(n_neurons[3] + 3, self.action_size)  # A(s,a) for action

    def forward(self, state: pd.DataFrame) -> (int, int, int):
        """
        One pass of the neural network model to predict: start, end, action, state-action value
        The state gets flattened and converted into floating-point values.
        Then passes through the hidden layers before splitting into 3 different heads.
        Each head has a separate hidden layer that predicts the value function V(s) and the advante function A(s,a)
        The Q(s,a) state-action value function gets calculated using the formula Q(s,a) = V(s) + A(s,a) - avg(A(s,a))
        :param state: State-definition matrix as a pandas DataFrame
        :return: (start, end, action): for nodes A --> B build x infrastructure
        """
        # Preprocess
        if state.shape == self.state_shape:  # Prediction pass
            state_tensor = torch.flatten(torch.tensor(np.array(state).astype(np.float32)))  # Convert the state to float and flatten
        else:  # Learning pass
            state_tensor = state.clone().detach().requires_grad_(True)

        # Hidden
        x = fn.relu(self.fc1(state_tensor))
        x = fn.relu(self.fc2(x))
        x = fn.relu(self.fc3(x))

        # Start
        start_v = fn.relu(self.fc4_start_value(x))
        start_v = self.start_value(start_v)  # V(s) for start

        start_a = fn.relu(self.fc4_start_adv(x))
        start_a = self.start_adv(start_a)  # A(s,a) for start

        start_i = torch.argmax(fn.softmax(start_a, dim=-1), dim=-1, keepdim=True)  # Starting node index
        start_n = start_i.float() / self.n_nodes  # Starting node tensor
        start_avg = torch.mean(start_a, dim=-1, keepdim=True)  # avg(A(s,a))
        start_q = start_v + start_a - start_avg  # Q(s,a) for start

        # End
        end_v = fn.relu(self.fc4_end_value(x))
        end_v = self.end_value(end_v)  # V(s) for end

        end_a = fn.relu(self.fc4_end_adv(x))
        end_input = torch.cat([end_a, start_n], dim=-1)  # Append start to state-tensor
        end_a = self.end_adv(end_input)  # A(s,a) for end

        end_i = torch.argmax(fn.softmax(end_a, dim=-1), dim=-1, keepdim=True)  # Ending node index
        end_n = end_i.float() / self.n_nodes  # Ending node tensor
        end_avg = torch.mean(end_a, dim=-1, keepdim=True)  # avg(A(s,a))
        end_q = end_v + end_a - end_avg  # Q(s,a) for end

        # Action
        action_v = fn.relu(self.fc4_action_value(x))
        action_v = self.action_value(action_v)  # V(s) for action

        action_a = fn.relu(self.fc4_action_adv(x))
        current_state = self.get_current_state(start_i, end_i, state)
        action_input = torch.cat([action_a, start_n, end_n, current_state], dim=-1).float()  # Append start, end and current state
        action_a = self.action_adv(action_input)  # A(s,a) for action

        action_avg = torch.mean(action_a, dim=-1, keepdim=True)  # avg(A(s,a))
        action_q = action_v + action_a - action_avg  # Q(s,a) for action

        return start_q, start_v, end_q, end_v, action_q, action_v

    def get_current_state(self, start_i: torch.tensor, end_i: torch.tensor, state: pd.DataFrame) -> torch.tensor:
        """
        Find the current element in the state-representation matrix. This method is needed because the input size varies at learning and prediction
        :param start_i: Starting node tensor
        :param end_i: Ending node tensor
        :param state: State tensor
        :return: A 1-dimensional torch tensor: either 1*1 or 1*batch_size
        """
        if state.shape == self.state_shape:
            current_state = torch.tensor(state.loc[int(start_i), int(end_i)] / self.state_high.loc[int(start_i), int(end_i)]).unsqueeze(-1)  # x(start, end)
        else:
            state_unpacked = []
            for i in range(state.shape[0]):
                state_unpacked.append(state[i, :].reshape(self.state_shape)[start_i[i][0], end_i[i][0]])
            current_state = torch.tensor(state_unpacked).unsqueeze(-1)
        return current_state


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


class Agent:
    """
    This class defines the reinforcement learning agent
    """
    def __init__(self, state_size: tuple, action_size: int, state_high: pd.DataFrame):
        """
        Setup the agent by reading the learning parameters from the config
        :param state_size: Size of the state-definition matrix
        :param action_size: Number of actions that is possible to take
        :param state_high: Numpy array of the
        """
        self.tau = args['learning'].getfloat('tau')
        self.gamma = args['learning'].getfloat('gamma')
        self.batch_size = args['learning'].getint('batch_size')
        self.buffer_size = int(args['learning'].getfloat('buffer_size'))
        self.update_every = args['learning'].getfloat('update_every')
        self.learning_rate = args['learning'].getfloat('learning_rate')
        self.n_neurons_local = [int(x) for x in args['learning'].get('n_neurons_local').split(',')]
        self.n_neurons_target = [int(x) for x in args['learning'].get('n_neurons_target').split(',')]

        self.state_high = state_high
        self.state_size = state_size
        self.n_nodes = state_size[0]
        self.action_size = action_size
        self.action_stack = np.zeros((0, 4))  # Initialize the action stack to 0x4 dimensions: [start, end, action, reward]
        self.history = np.zeros((0, 3))  # Initialize the stack to 0x3 dimensions: [start, end, action]
        self.state_stack = np.zeros((0, 3))  # Initialize the stack to 0x3 dimensions: [V(start), V(end), V(action)]

        self.qnetwork_local = QNetwork(state_size, action_size, state_high, self.n_neurons_local)  # Local network (for every step)
        self.qnetwork_target = QNetwork(state_size, action_size, state_high, self.n_neurons_target)  # Target network (for self.update_every)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        self.t_step = 0  # Initialize time step (for self.update_every)

        check_all_attributes_initialized(self)  # Check if all attributes have been set up properly

    def act(self, state: pd.DataFrame, eps: float):
        """
        Take some action based on state
        :param state: State-definition matrix that serves as the input for the model
        :param eps: Epsilon value (probability of exploration)
        :return: None
        """
        self.qnetwork_local.eval()
        with torch.no_grad():  # Disabled gradient calculation (for predictions)
            start_q, start_v, end_q, end_v, action_q, action_v = self.qnetwork_local.forward(state)  # Get a prediction from the neural network

        self.state_stack = np.vstack([self.state_stack, np.array([float(start_v), float(end_v), float(action_v)])])  # Keeping track of state values
        self.qnetwork_local.train()  # Apply gradients if necessary

        # Epsilon-greedy action selection
        if random.random() > eps:
            start = np.argmax(start_q.cpu().data.numpy())
            end = np.argmax(end_q.cpu().data.numpy())
            action = np.argmax(action_q.cpu().data.numpy())
            return start, end, action

        # Random action selection
        else:
            start = np.random.randint(self.n_nodes)
            end = np.random.randint(self.n_nodes)
            action = np.random.randint(self.action_size)
            return start, end, action

    def step(self, state: pd.DataFrame, start: int, end: int, action: int, reward: int, next_state: pd.DataFrame, successful: bool):
        """
        Save one step in the reinforcement learning environment
        :param state: State before stepping
        :param start: Starting node for the infrastructure
        :param end: Ending node for the infrastructure
        :param action: Action that was the prediction of the model in the state
        :param reward: Reward that was received for the action
        :param next_state: Next state of the environment
        :param successful: If the operation has completed successfully
        :return: None
        """
        self.memory.add(state, start, end, action, reward, next_state, int(successful))  # Save experience in replay memory

        # This variable is only used for inference - it gets filled here
        self.action_stack = np.vstack([self.action_stack, np.array([start, end, action, reward])])

        self.t_step = (self.t_step + 1) % self.update_every  # Update the time step
        if self.t_step == 0 and len(self.memory) > self.batch_size:  # If there's enough experience in the memory we will sample it and learn
            self.learn()

    def learn(self):
        """
        Apply the gradients based on previoyus experience
        :return: None
        """
        states, starts, ends, actions, rewards, next_states, successfuls = self.memory.sample()  # Obtain a random mini-batch

        # Compute and minimize the loss
        target_start_q, target_start_v, target_end_q, target_end_v, target_action_q, target_action_v = self.qnetwork_target(next_states)
        local_start_q, local_start_v, local_end_q, local_end_v, local_action_q, local_action_v = self.qnetwork_local(states)

        # Start
        targets_start_q_next = target_start_q.detach().max(1)[0].unsqueeze(1)
        targets_start = rewards + self.gamma * targets_start_q_next * successfuls
        expected_start = local_start_q.gather(1, starts)
        loss_start = fn.mse_loss(expected_start, targets_start)

        # End
        targets_end_q_next = target_end_q.detach().max(1)[0].unsqueeze(1)
        targets_end = rewards + self.gamma * targets_end_q_next * successfuls
        expected_end = local_end_q.gather(1, ends)
        loss_end = fn.mse_loss(expected_end, targets_end)

        # Action
        targets_action_q_next = target_action_q.detach().max(1)[0].unsqueeze(1)
        targets_action = rewards + self.gamma * targets_action_q_next * successfuls
        expected_action = local_action_q.gather(1, actions)
        loss_action = fn.mse_loss(expected_action, targets_action)

        # Calculate loss and step in optimizer
        loss_tensor = torch.tensor([loss_start, loss_end, loss_action])
        loss = torch.sum(loss_tensor, 0)
        loss.requires_grad = True
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self):
        """
        For dual network architectures one network's parameters are copied over to the other one. Tau defines the ratio of copying
        Iterates over the parameters in the local and target network and copies them over
        :return: None
        """
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
