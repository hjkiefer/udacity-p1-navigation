import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from typing import List

class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, 
                 state_size: int, 
                 action_size: int,
                 seed: int,
                 hidden_layer_neurons: List[int] = [32,16,8]):
        """Creates a multilayer perceptron neural network with relu activation (and softmax output)

        param: int state_size: length of the state space vector
        param: int action_size: length of the action space vector
        param: int seed: random seed
        param: List[int] hidden_layer_neurons: list of integers for the neural network hidden layers
        return: PyTorch network
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        inputs = [state_size]+hidden_layer_neurons
        outputs = hidden_layer_neurons+[action_size]
        self.fully_connected_layers = nn.ModuleList()
        for inp,outp in zip(inputs,outputs):
            self.fully_connected_layers.append(nn.Linear(inp, outp))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for layer in self.fully_connected_layers[:-1]:
            x = F.relu(layer(x))
        x = F.softmax(self.fully_connected_layers[-1](x), dim=1)
        return x

class QNetwork_with_connection_from_input(nn.Module):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 seed: int,
                 hidden_layer_neurons: List[int]=[32,16,8]):
        """Creates a multilayer perceptron neural network with relu activation (and softmax output)
        An additional connection from input to every layer is created. 
        These connections prevents diminishing gradient for deeper networks

        param: int state_size: length of the state space vector
        param: int action_size: length of the action space vector
        param: int seed: random seed
        param: List[int] hidden_layer_neurons: list of integers for the neural network hidden layers
        return: PyTorch network
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        inputs = [0]+hidden_layer_neurons[:-1]
        outputs = hidden_layer_neurons
        self.fully_connected_layers = nn.ModuleList()
        for inp,outp in zip(inputs,outputs):
            new_layer = nn.Linear(inp+state_size, outp)
            self.fully_connected_layers.append(new_layer)
        self.fully_connected_layers.append(nn.Linear(outp+state_size,action_size))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x0 = np.array(state)
        x = F.relu(self.fully_connected_layers[0](state))
        for layer in self.fully_connected_layers[1:-1]:
            x = F.relu(layer(torch.cat([state,x],1)))
        x = F.softmax(self.fully_connected_layers[-1](torch.cat([state,x],1)
                     ), dim=1)
        return x