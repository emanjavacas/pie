
import torch.nn as nn

from pie import torch_utils
from pie import initialization


class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=1, cell='GRU', dropout=0.0):

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = cell
        self.dropout = dropout
        super().__init__()

        rnn = []
        for layer in range(num_layers):
            rnn_inp = in_size if layer == 0 else hidden_size * 2
            rnn.append(getattr(nn, cell)(rnn_inp, hidden_size, bidirectional=True))
        self.rnn = nn.ModuleList(rnn)

        self.init()

    def init(self):
        for rnn in self.rnn:
            initialization.init_rnn(rnn)

    def forward(self, inp, lengths):
        hidden = [
            torch_utils.init_hidden_for(
                inp, 2, 1, self.hidden_size, self.cell, add_init_jitter=True)
            for _ in range(len(self.rnn))]

        inp, unsort = torch_utils.pack_sort(inp, lengths)
        outs = []

        for layer, rnn in enumerate(self.rnn):
            louts, _ = rnn(inp, hidden[layer])
            if layer != len(self.rnn) - 1:
                louts, lengths = nn.utils.rnn.pad_packed_sequence(louts)
                louts = torch_utils.sequential_dropout(
                    louts, self.dropout, self.training)
                louts = nn.utils.rnn.pack_padded_sequence(louts, lengths)
            outs.append(louts)
            inp = louts

        # unpack
        for layer in range(len(self.rnn)):
            louts, _ = nn.utils.rnn.pad_packed_sequence(outs[layer])
            outs[layer] = louts[:, unsort]

        return outs
