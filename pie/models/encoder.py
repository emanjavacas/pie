
import torch
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

    def forward(self, inp, lengths, hidden=None, only_last=False, return_hidden=False):
        if hidden is None:
            hidden = [
                torch_utils.init_hidden_for(
                    inp, 2, 1, self.hidden_size, self.cell, add_init_jitter=False)
                for _ in range(len(self.rnn))]

        _, sort = torch.sort(lengths, descending=True)
        _, unsort = sort.sort()
        inp = nn.utils.rnn.pack_padded_sequence(inp[:, sort], lengths[sort].cpu())
        outs, hiddens = [], []

        for layer, rnn in enumerate(self.rnn):
            # apply dropout only in between layers (not on the output)
            if layer > 0:
                inp, lengths = nn.utils.rnn.pad_packed_sequence(inp)
                inp = torch_utils.sequential_dropout(inp, self.dropout, self.training)
                inp = nn.utils.rnn.pack_padded_sequence(inp, lengths.cpu())
            # run layer
            louts, lhidden = rnn(inp, hidden[layer])
            # unpack
            louts_, _ = nn.utils.rnn.pad_packed_sequence(louts)
            outs.append(louts_[:, unsort])
            if isinstance(lhidden, tuple):
                lhidden = lhidden[0][:, unsort, :], lhidden[1][:, unsort, :]
            else:
                lhidden = lhidden[:, unsort, :]
            hiddens.append(lhidden)
            # recur
            inp = louts

        if only_last:
            outs, hiddens = outs[-1], hiddens[-1]

        if return_hidden:
            return outs, hiddens
        else:
            return outs
