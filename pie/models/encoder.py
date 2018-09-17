
import torch.nn as nn

from pie import torch_utils
from pie import initialization


class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=1, cell='GRU', dropout=0.0,
                 init_rnn='default'):

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = cell
        self.dropout = dropout
        self.init_rnn = init_rnn
        super().__init__()

        rnn = []
        for layer in range(num_layers):
            rnn_inp = in_size if layer == 0 else hidden_size * 2
            rnn.append(getattr(nn, cell)(rnn_inp, hidden_size, bidirectional=True))
        self.rnn = nn.ModuleList(rnn)

        self.init()

    def init(self):
        for rnn in self.rnn:
            initialization.init_rnn(rnn, scheme=self.init_rnn)

    def forward(self, inp, lengths):
        hidden = [
            torch_utils.init_hidden_for(
                inp, 2, 1, self.hidden_size, self.cell, add_init_jitter=True)
            for _ in range(len(self.rnn))]

        inp, unsort = torch_utils.pack_sort(inp, lengths)
        outs = []

        for layer, rnn in enumerate(self.rnn):
            # apply dropout only in between layers (not on the output)
            if layer > 0 and layer != len(self.rnn) - 1:
                inp, lengths = nn.utils.rnn.pad_packed_sequence(inp)
                inp = torch_utils.sequential_dropout(inp, self.dropout, self.training)
                inp = nn.utils.rnn.pack_padded_sequence(inp, lengths)
            # run layer
            louts, _ = rnn(inp, hidden[layer])
            # unpack
            outs.append(nn.utils.rnn.pad_packed_sequence(louts)[0][:, unsort])
            inp = louts

        return outs
