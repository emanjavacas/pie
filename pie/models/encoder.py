
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from pie import torch_utils
from pie import initialization


class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=1, bidirectional=True,
                 cell='GRU', dropout=0.0):

        # TODO: types of merging

        if bidirectional and hidden_size % 2 != 0:
            raise ValueError("Bidirectional RNN needs even `hidden_size` "
                             "but got {}".format(hidden_size))

        self.hidden_size = hidden_size
        self.num_dirs = 1 + int(bidirectional)
        self.num_layers = num_layers
        self.cell = cell
        super().__init__()

        self.rnn = getattr(nn, cell)(
            in_size, hidden_size // self.num_dirs,
            num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)

        self.init()

    def init(self):
        initialization.init_rnn(self.rnn)

    def forward(self, inp, lengths):
        hidden = torch_utils.init_hidden_for(
            inp, self.num_dirs, self.num_layers,
            self.hidden_size // self.num_dirs, self.cell)

        inp, unsort = torch_utils.pack_sort(inp, lengths)
        inp, _ = self.rnn(inp)
        inp, _ = unpack(inp)
        inp = inp[:, unsort]

        return inp
