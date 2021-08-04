import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layer_neurons=[32,32,16]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
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

