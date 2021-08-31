import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, hidden_layer_neurons=[32,16,8]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
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
    def __init__(self, state_size, action_size, seed, hidden_layer_neurons=[32,16,8]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        inputs = [0]+hidden_layer_neurons[:-1]
        outputs = hidden_layer_neurons
        self.fully_connected_layers = nn.ModuleList()
        for inp,outp in zip(inputs,outputs):
            print(inp)
            new_layer = nn.Linear(inp+state_size, outp)
            print(new_layer)
            self.fully_connected_layers.append(new_layer)
        self.fully_connected_layers.append(nn.Linear(outp+state_size,action_size))
        print(self.fully_connected_layers[-1])

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x0 = np.array(state)
        x = F.relu(self.fully_connected_layers[0](state))
        #print(x.size())
        for layer in self.fully_connected_layers[1:-1]:
            #print(x.size())
            x = F.relu(layer(torch.cat([state,x],1)))
        x = F.softmax(self.fully_connected_layers[-1](torch.cat([state,x],1)
                     ), dim=0)
        return x
    
class QNetwork_resnet(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_layer_neurons=[32,32,16]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
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
        x = self.fully_connected_layers[-1](x)
        return x
