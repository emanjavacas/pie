
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from pie import torch_utils
from pie import initialization


class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=1, cell='GRU'):

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = cell
        super().__init__()

        self.rnn = getattr(nn, cell)(
            in_size, hidden_size, num_layers=num_layers, bidirectional=True)

        self.init()

    def init(self):
        initialization.init_rnn(self.rnn)

    def forward(self, inp, lengths):
        hidden = torch_utils.init_hidden_for(
            inp, 2, self.num_layers, self.hidden_size, self.cell,
            add_init_jitter=True)

        inp, unsort = torch_utils.pack_sort(inp, lengths)
        inp, _ = self.rnn(inp, hidden)
        inp, _ = unpack(inp)
        inp = inp[:, unsort]

        return inp
