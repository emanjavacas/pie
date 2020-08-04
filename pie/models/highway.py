
import torch
import torch.nn as nn
import torch.nn.functional as F

from pie import initialization


class Highway(nn.Module):
    """
    Highway network
    """
    def __init__(self, in_features, num_layers, act='relu'):
        self.in_features = in_features

        self.act = act
        super().__init__()

        self.layers = nn.ModuleList(
            [nn.Linear(in_features, in_features*2) for _ in range(num_layers)])

        self.init()

    def init(self):
        for layer in self.layers:
            initialization.init_linear(layer)
            # bias gate to let information go untouched
            nn.init.constant_(layer.bias[self.in_features:], 1.)

    def forward(self, inp):
        current = inp
        for layer in self.layers:
            inp, gate = layer(current).chunk(2, dim=-1)
            inp, gate = getattr(F, self.act)(inp), torch.sigmoid(gate)
            current = gate * current + (1 - gate) * inp

        return current


