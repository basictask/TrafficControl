"""
This is a Graph-Convolution agent
"""
# Own
from suppl import JUNCTION_TYPES, ACTION_NAMES, \
    check_all_attributes_initialized, count_incoming_lanes, choose_random_action, get_batch_graph_features, get_batch_embeddings
from agents.agent_1_net import ReplayBuffer
# Generic
import os
import math
import random
import numpy as np
import pandas as pd
# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
# Arguments
import configparser
args = configparser.ConfigParser()
args.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config.ini'))


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, features, adj):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Net(nn.Module):
    def __init__(self, n_nodes: int, action_size: int, n_feat: int, n_hid: int, n_class: int):
        super(Net, self).__init__()
        self.n_nodes = n_nodes
        self.action_size = action_size
        self.n_neurons = [int(x) for x in args['learning'].get('n_neurons_local').split(',')]
        # Convolutional layers
        self.gc1 = GraphConvolution(n_feat, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_class)
        # Dense layers
        self.fc1 = nn.Linear(n_nodes * n_class, self.n_neurons[0])
        self.fc2 = nn.Linear(self.n_neurons[0], self.n_neurons[1])
        self.d1 = nn.Dropout(p=0.8)
        self.fc3 = nn.Linear(self.n_neurons[1], self.n_neurons[2])
        self.d2 = nn.Dropout(p=0.8)
        self.fc4 = nn.Linear(self.n_neurons[2], self.n_neurons[3])
        self.fc5 = nn.Linear(self.n_neurons[3], self.action_size)

    def forward(self, state: torch.tensor):
        # Converting features
        x = torch.diag(state).unsqueeze(1).float()
        adj = state.fill_diagonal_(0).float()
        # Convolution pass
        x = fn.relu(self.gc1(x, adj))
        x = fn.dropout(x, training=self.training, p=0.5)
        x = self.gc2(x, adj)
        x = fn.log_softmax(x, dim=1)
        # Dense pass
        x = torch.flatten(x)
        x = fn.relu(self.fc1(x))
        x = fn.relu(self.fc2(x))
        x = fn.dropout(x, training=self.training, p=0.5)
        x = fn.relu(self.fc3(x))
        x = fn.dropout(x, training=self.training, p=0.5)
        x = fn.relu(self.fc4(x))
        q = fn.relu(self.fc5(x))
        return q


if __name__ == '__main__':
    obs = torch.tensor(np.random.randint(low=0, high=4, size=(16, 10, 10)))

    model = Net(n_nodes=10, action_size=5, n_feat=1, n_hid=64, n_class=10)
    print(model(obs))
