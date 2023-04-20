"""
This is a new agent definition that uses three separate Q-networks to predict the start, end and action for each rL iteration
"""
# Own
from suppl import JUNCTION_TYPES, ACTION_NAMES, check_all_attributes_initialized, count_incoming_lanes
from replay_buffer import ReplayBuffer
# Generic
import os
import random
import numpy as np
import pandas as pd
# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fn
# Arguments
import configparser
args = configparser.ConfigParser()
args.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config.ini'))


class QNetwork(nn.Module):
    """
    Q-network model for the reinforcement learning agent.
    The structure of the network can be defined in the __init__ function.
    """
    def __init__(self, state_size: int, action_size: int, n_neurons: list):
        """
        Defines the architecture of the model. The model needs to predict 3 values at each iteration. For this purpose this agent will use 3 neural networks.
        The architecture is a Dueling Double Deep Q-learning with 3 different heads to predict the 3 target variables
        :param state_size: Input dimension for the neural network
        :param action_size: Integer value on how many actions can the agent take
        :param n_neurons: A list of the sizes of each neuron layer. Defined in config.ini
        """
        # Neural network architecture #
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, n_neurons[0])  # States are input here
        self.fc2 = nn.Linear(n_neurons[0], n_neurons[1])  # States are input here
        self.fc3 = nn.Linear(n_neurons[1], n_neurons[2])  # States are input here

        self.fc_value = nn.Linear(n_neurons[2], n_neurons[3])
        self.value = nn.Linear(n_neurons[3], 1)  # This will estimate the state-value function (independent of action --> size=1)

        self.fc_adv = nn.Linear(n_neurons[2], n_neurons[3])
        self.adv = nn.Linear(n_neurons[3], action_size)  # This will predict the state-action-value function (size must equal the number of actions)
        
        check_all_attributes_initialized(self)  # Check if all data members have been set up properly

    def forward(self, state: torch.tensor) -> (torch.tensor, torch.tensor):  # Get predictions from the layers
        x = fn.relu(self.fc1(state))
        x = fn.relu(self.fc2(x))
        x = fn.relu(self.fc3(x))

        # Predictions for state-value function: V(s)
        value = fn.relu(self.fc_value(x))
        value = self.value(value)

        # Predictions for advantage function: A(s,a) = Q(s,a) - V(s)
        adv = fn.relu(self.fc_adv(x))
        adv = self.adv(adv)

        adv_avg = torch.mean(adv, dim=-1, keepdim=True)
        q = value + adv - adv_avg

        return q, value


class Agent:
    """
    This class defines the reinforcement learning agent
    """
    def __init__(self, state_shape: tuple, action_size: int, state_high: pd.DataFrame):
        """
        Setup the agent by reading the learning parameters from the config
        :param state_shape: Size of the state-definition matrix
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
        self.trafficlight_inbound = [int(x) for x in args['trafficlight'].get('allow_inbound').split(',')]

        self.action_size = action_size
        self.state_high = state_high
        self.n_nodes = state_shape[0]
        self.state_size = state_shape[0] * state_shape[1]
        self.state_shape = state_shape
        self.action_stack = np.zeros((0, 4))  # Initialize the action stack to 0x4 dimensions: [start, end, action, reward]
        self.state_stack = np.zeros((0, 3))  # Initialize the stack to 0x3 dimensions: [V(start), V(end), V(action)]
        self.history = np.zeros((0, 3))  # Initialize the stack to 0x3 dimensions: [start, end, action]

        self.start_qnetwork_local = QNetwork(self.state_size, self.n_nodes, self.n_neurons_local)  # Local network for start node
        self.start_qnetwork_target = QNetwork(self.state_size, self.n_nodes, self.n_neurons_target)  # Target network for start node
        self.start_optimizer = optim.Adam(self.start_qnetwork_local.parameters(), lr=self.learning_rate)  # Optimizer for start node

        self.end_qnetwork_local = QNetwork(self.state_size + 1, self.n_nodes, self.n_neurons_local)  # Local network for end node
        self.end_qnetwork_target = QNetwork(self.state_size + 1, self.n_nodes, self.n_neurons_target)  # Target network for end node
        self.end_optimizer = optim.Adam(self.end_qnetwork_local.parameters(), lr=self.learning_rate)  # Optimizer for end node

        self.action_qnetwork_local = QNetwork(self.state_size + 2, self.action_size, self.n_neurons_local)  # Local network for action
        self.action_qnetwork_target = QNetwork(self.state_size + 2, self.action_size, self.n_neurons_target)  # Target network for action
        self.action_optimizer = optim.Adam(self.action_qnetwork_local.parameters(), lr=self.learning_rate)  # Optimizer for action

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        self.t_step = 0  # Initialize time step (for self.update_every)

        check_all_attributes_initialized(self)  # Check if all attributes have been set up properly

    def act(self, state: pd.DataFrame, eps: float) -> (int, int, int, bool):
        """
        Take some action based on state
        :param state: State-definition matrix that serves as the input for the model
        :param eps: Epsilon value (probability of exploration)
        :return: None
        """
        self.start_qnetwork_local.eval()
        self.end_qnetwork_local.eval()
        self.action_qnetwork_local.eval()

        state_tensor = torch.flatten(torch.tensor(np.array(state).astype(np.float32)))  # Convert the state to float and flatten

        with torch.no_grad():
            start_q, start_v = self.start_qnetwork_local.forward(state_tensor)
            start_i = torch.argmax(start_q, dim=-1, keepdim=True)
            start_n = start_i.float() / self.n_nodes

            state_tensor = torch.cat([state_tensor, start_n], dim=-1)

            end_q, end_v = self.end_qnetwork_local.forward(state_tensor)
            end_i = torch.argmax(end_q, dim=-1, keepdim=True)
            end_n = end_i.float() / self.n_nodes

            state_tensor = torch.cat([state_tensor, end_n], dim=-1)

            action_q, action_v = self.action_qnetwork_local.forward(state_tensor)
            action_q = self.get_valid_actions(start_i, end_i, action_q, state)
            action_i = torch.argmax(action_q, dim=-1, keepdim=True)

        self.start_qnetwork_local.train()  # Apply gradients if necessary
        self.end_qnetwork_local.train()  # Apply gradients if necessary
        self.action_qnetwork_local.train()  # Apply gradients if necessary

        # Epsilon-greedy action selection
        if random.random() > eps:
            return int(start_i), int(end_i), int(action_i), False

        # Random action selection
        else:
            start = np.random.randint(self.n_nodes)
            end = np.random.randint(self.n_nodes)
            action = self.choose_random_action(start, end, state)
            return start, end, action, True

    def choose_random_action(self, start: int, end: int, state: pd.DataFrame) -> int:
        """
        Chooses a valid random action depending on the state
        :param start: Index of the starting node
        :param end: Index of the anding node
        :param state: State-definition matrix
        :return: Integer representation of valid action
        """
        current_state_start_end = state.loc[start, end]
        current_state_end_end = state.loc[end, end]
        current_junction = JUNCTION_TYPES[current_state_end_end]
        all_actions = list(range(self.action_size))

        if start != end and current_state_start_end == self.state_high.loc[start, end]:  # Maximum number of lanes reached
            all_actions.remove(ACTION_NAMES['add_lane'])

        elif start != end and current_state_start_end == 0:
            all_actions.remove(ACTION_NAMES['remove_lane'])

        if start == end:
            all_actions.remove(ACTION_NAMES['add_lane'])
            all_actions.remove(ACTION_NAMES['remove_lane'])

        if current_junction == 'righthand':
            all_actions.remove(ACTION_NAMES['add_righthand'])

        elif current_junction == 'roundabout':
            all_actions.remove(ACTION_NAMES['add_roundabout'])

        elif current_junction == 'trafficlight':
            all_actions.remove(ACTION_NAMES['add_trafficlight'])

        if ACTION_NAMES['add_trafficlight'] in all_actions and not count_incoming_lanes(state, end) in self.trafficlight_inbound:
            all_actions.remove(ACTION_NAMES['add_trafficlight'])

        return np.random.choice(all_actions)

    def get_current_state(self, start_i: torch.tensor, end_i: torch.tensor, state, normalize: bool) -> torch.tensor:
        """
        Find the current element in the state-representation matrix. This method is needed because the input size varies at learning and prediction
        :param start_i: Starting node tensor
        :param end_i: Ending node tensor
        :param state: State tensor
        :param normalize: The state will be divided by the corresponding state_high value
        :return: A 1-dimensional torch tensor: either 1*1 or 1*batch_size
        """
        if state.shape == self.state_shape:
            if normalize:
                current_state = torch.tensor(state.loc[int(start_i), int(end_i)] / self.state_high.loc[int(start_i), int(end_i)]).unsqueeze(-1)  # x(start, end)
            else:
                current_state = torch.tensor(state.loc[int(start_i), int(end_i)]).unsqueeze(-1)
        else:
            state = state[:, :100]  # Remove the previously appended part (unique to this model)
            state_unpacked = []
            for i in range(state.shape[0]):
                state_unpacked.append(state[i, :].reshape(self.state_shape)[start_i[i][0], end_i[i][0]])
            current_state = torch.tensor(state_unpacked).unsqueeze(-1)
        return current_state

    def get_valid_actions(self, start_i: torch.tensor, end_i: torch.tensor, action_q: torch.tensor, state) -> torch.tensor:
        """
        Inspect the context of the starting and ending node and eliminate all the invalid actions
        :param start_i: Starting node index
        :param end_i: Ending node index
        :param action_q: Torch tensor for the action values
        :param state: State-definition matrix
        :return: A new torch tensor of Q-values with invalid actions set to -inf
        """
        current_state_start_end = self.get_current_state(start_i, end_i, state, normalize=False)
        current_state_end_end = self.get_current_state(end_i, end_i, state, normalize=False)

        if action_q.dim() == 1:
            action_q = action_q.unsqueeze(0)
            n_incoming = [count_incoming_lanes(state, int(end_i))]
        else:
            n_incoming = []
            for i in range(len(start_i)):
                n_incoming.append(count_incoming_lanes(pd.DataFrame(state[i][:100].detach().reshape(self.state_shape), dtype=int), int(end_i[i])))

        for i in range(len(start_i)):
            start = int(start_i[i])
            end = int(end_i[i])
            current_junction = JUNCTION_TYPES[int(current_state_end_end[i])]

            if start != end and int(current_state_start_end[i]) == self.state_high.loc[start, end]:  # Maximum number of lanes reached
                action_q[i, ACTION_NAMES['add_lane']] = float('-inf')

            elif start != end and int(current_state_start_end[i]) == 0:
                action_q[i, ACTION_NAMES['remove_lane']] = float('-inf')

            if start == end:
                action_q[i, ACTION_NAMES['add_lane']] = float('-inf')
                action_q[i, ACTION_NAMES['remove_lane']] = float('-inf')

            if current_junction == 'righthand':
                action_q[i, ACTION_NAMES['add_righthand']] = float('-inf')

            elif current_junction == 'roundabout':
                action_q[i, ACTION_NAMES['add_roundabout']] = float('-inf')

            elif current_junction == 'trafficlight':
                action_q[i, ACTION_NAMES['add_trafficlight']] = float('-inf')

            if not n_incoming[i] in self.trafficlight_inbound:
                action_q[i, ACTION_NAMES['add_trafficlight']] = float('-inf')

        return action_q

    def step(self, state: pd.DataFrame, start: int, end: int, action: int, reward: int, next_state: pd.DataFrame, successful: bool) -> None:
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
        self.t_step = (self.t_step + 1) % self.update_every  # Update the time step

        if self.t_step == 0 and len(self.memory) > self.batch_size:  # If there's enough experience in the memory we will sample it and learn
            self.learn()

    def learn(self):
        """
        Apply the gradients based on previoyus experience
        :return: None
        """
        states, starts, ends, actions, rewards, next_states, successfuls = self.memory.sample()  # Obtain a random mini-batch

        # Make a copy of the states and convert them to state tensors
        target_next_states = next_states.clone().detach().requires_grad_(True)
        local_states = states.clone().detach().requires_grad_(True)

        # Target networks 
        target_start_q, target_start_v = self.start_qnetwork_target(target_next_states)
        start_i = torch.argmax(target_start_q, dim=-1, keepdim=True)
        start_n = start_i.float() / self.n_nodes

        target_next_states = torch.cat([target_next_states, start_n], dim=-1)

        target_end_q, target_end_v = self.end_qnetwork_target(target_next_states)
        end_i = torch.argmax(target_end_q, dim=-1, keepdim=True)
        end_n = end_i.float() / self.n_nodes
        
        target_next_states = torch.cat([target_next_states, end_n], dim=-1)
        
        target_action_q, target_action_v = self.action_qnetwork_target(target_next_states)
        target_action_q = self.get_valid_actions(start_i, end_i, target_action_q, target_next_states)

        # Local networks
        local_start_q, local_start_v = self.start_qnetwork_local(local_states)
        start_i = torch.argmax(local_start_q, dim=-1, keepdim=True)
        start_n = start_i.float() / self.n_nodes

        local_states = torch.cat([local_states, start_n], dim=-1)

        local_end_q, local_end_v = self.end_qnetwork_local(local_states)
        end_i = torch.argmax(local_end_q, dim=-1, keepdim=True)
        end_n = end_i.float() / self.n_nodes

        local_states = torch.cat([local_states, end_n], dim=-1)

        local_action_q, local_action_v = self.action_qnetwork_local(local_states)
        local_action_q = self.get_valid_actions(start_i, end_i, local_action_q, local_states)

        # Start
        targets_start_q_next = target_start_q.detach().max(-1)[0].unsqueeze(-1)
        targets_start = rewards + self.gamma * targets_start_q_next  # * successfuls
        expected_start = local_start_q.gather(0, starts)
        start_loss = fn.mse_loss(expected_start, targets_start)
        self.start_optimizer.zero_grad()
        start_loss.backward()
        self.start_optimizer.step()

        # End
        targets_end_q_next = target_end_q.detach().max(-1)[0].unsqueeze(-1)
        targets_end = rewards + self.gamma * targets_end_q_next  # * successfuls
        expected_end = local_end_q.gather(0, ends)
        end_loss = fn.mse_loss(expected_end, targets_end)
        self.end_optimizer.zero_grad()
        end_loss.backward()
        self.end_optimizer.step()

        # Action
        targets_action_q_next = target_action_q.detach().max(-1)[0].unsqueeze(-1)
        targets_action = rewards + self.gamma * targets_action_q_next  # * successfuls
        expected_action = local_action_q.gather(0, actions)
        action_loss = fn.mse_loss(expected_action, targets_action)
        self.action_optimizer.zero_grad()
        action_loss.backward()
        self.action_optimizer.step()

    def soft_update(self):
        """
        For dual network architectures one network's parameters are copied over to the other one. Tau defines the ratio of copying
        Iterates over the parameters in the local and target network and copies them over
        :return: None
        """
        for target_param, local_param in zip(self.start_qnetwork_target.parameters(), self.start_qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

        for target_param, local_param in zip(self.end_qnetwork_target.parameters(), self.end_qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

        for target_param, local_param in zip(self.action_qnetwork_target.parameters(), self.action_qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save_models(self):
        """
        Iterates over all the models and saves their states
        :return: None
        """
        torch.save(self.start_qnetwork_local.state_dict(), './models/start_local.pth')
        torch.save(self.start_qnetwork_target.state_dict(), './models/start_target.pth')

        torch.save(self.end_qnetwork_local.state_dict(), './models/end_local.pth')
        torch.save(self.end_qnetwork_target.state_dict(), './models/end_target.pth')

        torch.save(self.action_qnetwork_local.state_dict(), './models/action_local.pth')
        torch.save(self.action_qnetwork_target.state_dict(), './models/action_target.pth')
