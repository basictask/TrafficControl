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
from suppl import ACTION_NAMES, JUNCTION_TYPES, count_incoming_lanes, check_all_attributes_initialized
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
        self.trafficlight_inbound = [int(x) for x in args['trafficlight'].get('allow_inbound').split(',')]
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
        self.action_adv = nn.Linear(n_neurons[3] + 2, self.action_size)  # A(s,a) for action

    def forward(self, state) -> (int, int, int):
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

        start_avg = torch.mean(start_a, dim=-1, keepdim=True)  # avg(A(s,a))
        start_q = start_v + start_a - start_avg  # Q(s,a) for start

        start_i = torch.argmax(start_q, dim=-1, keepdim=True)  # Starting node index
        start_n = start_i.float() / self.n_nodes  # Starting node tensor

        # End
        end_v = fn.relu(self.fc4_end_value(x))
        end_v = self.end_value(end_v)  # V(s) for end

        end_a = fn.relu(self.fc4_end_adv(x))
        end_input = torch.cat([end_a, start_n], dim=-1)  # Append start to state-tensor
        end_a = self.end_adv(end_input)  # A(s,a) for end

        end_avg = torch.mean(end_a, dim=-1, keepdim=True)  # avg(A(s,a))
        end_q = end_v + end_a - end_avg  # Q(s,a) for end

        end_i = torch.argmax(end_q, dim=-1, keepdim=True)  # Ending node index
        end_n = end_i.float() / self.n_nodes  # Ending node tensor

        # Action
        action_v = fn.relu(self.fc4_action_value(x))
        action_v = self.action_value(action_v)  # V(s) for action

        action_a = fn.relu(self.fc4_action_adv(x))
        # current_state = self.get_current_state(start_i, end_i, state)  -->  removed from torch.cat command
        action_input = torch.cat([action_a, start_n, end_n], dim=-1).float()  # Append start, end and current state
        action_a = self.action_adv(action_input)  # A(s,a) for action

        action_avg = torch.mean(action_a, dim=-1, keepdim=True)  # avg(A(s,a))
        action_q = action_v + action_a - action_avg  # Q(s,a) for action

        action_q = self.get_valid_actions(start_i, end_i, action_q, state)

        return start_q, start_v, end_q, end_v, action_q, action_v

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
                n_incoming.append(count_incoming_lanes(pd.DataFrame(state[i].reshape(self.state_shape), dtype=int), int(end_i[i])))

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


class Agent1Net:
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
        self.max_lanes = args['reader'].getint('max_lanes')
        self.batch_size = args['learning'].getint('batch_size')
        self.buffer_size = int(args['learning'].getfloat('buffer_size'))
        self.update_every = args['learning'].getfloat('update_every')
        self.learning_rate = args['learning'].getfloat('learning_rate')
        self.n_neurons_local = [int(x) for x in args['learning'].get('n_neurons_local').split(',')]
        self.n_neurons_target = [int(x) for x in args['learning'].get('n_neurons_target').split(',')]
        self.trafficlight_inbound = [int(x) for x in args['trafficlight'].get('allow_inbound').split(',')]

        self.state_high = state_high
        self.state_size = state_size
        self.n_nodes = state_size[0]
        self.action_size = action_size
        self.action_stack = np.zeros((0, 4))  # Initialize the action stack to 0x4 dimensions: [start, end, action, reward]
        self.history = np.zeros((0, 3))  # Initialize the stack to 0x3 dimensions: [start, end, action]
        self.state_stack = np.zeros((0, 3))  # Initialize the stack to 0x3 dimensions: [V(start), V(end), V(action)]

        self.qnetwork_local = QNetwork(self.state_size, self.action_size, self.state_high, self.n_neurons_local)  # Local network (for every step)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.state_high, self.n_neurons_target)  # Target network (for self.update_every)
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
            return start, end, action, False

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
        targets_start = rewards + self.gamma * targets_start_q_next  # * successfuls
        expected_start = local_start_q.gather(1, starts)
        loss_start = fn.mse_loss(expected_start, targets_start)

        # End
        targets_end_q_next = target_end_q.detach().max(1)[0].unsqueeze(1)
        targets_end = rewards + self.gamma * targets_end_q_next  # * successfuls
        expected_end = local_end_q.gather(1, ends)
        loss_end = fn.mse_loss(expected_end, targets_end)

        # Action
        targets_action_q_next = target_action_q.detach().max(1)[0].unsqueeze(1)
        targets_action = rewards + self.gamma * targets_action_q_next  # * successfuls
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

    def save_models(self) -> None:
        """
        Saves the model files into the given path
        :return: None
        """
        torch.save(self.qnetwork_local.state_dict(), '../models/local.pth')
        torch.save(self.qnetwork_target.state_dict(), '../models/target.pth')
