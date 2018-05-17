
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from pie import torch_utils as tutils


class RNNEncoder(nn.Module):
    # TODO: perhaps add token-level context vectors to input?
    def __init__(self, embs, hidden_size, num_layers=1, bidirectional=True,
                 cell='GRU', dropout=0.0):

        if bidirectional and hidden_size % 2 != 0:
            raise ValueError("Bidirectional RNN needs even `hidden_size` "
                             "but got {}".format(hidden_size))

        self.hidden_size = hidden_size
        self.num_dirs = 1 + int(bidirectional)
        self.num_layers = num_layers
        self.cell = cell
        super().__init__()

        self.embs = embs
        self.rnn = nn.GRU(
            embs.embedding_dim, hidden_size // self.num_dirs,
            num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)

    def forward(self, inp, lengths, *args):
        if isinstance(self.embs, nn.Embedding):
            inp = self.embs(inp)
        else:
            inp = self.embs(inp, lengths, *args)

        hidden = tutils.init_hidden_for(
            inp, self.num_dirs, self.num_layers,
            self.hidden_size // self.num_dirs, self.cell)

        inp, unsort = tutils.pack_sort(inp, lengths)
        inp, _ = self.rnn(inp, hidden)
        inp, _ = unpack(inp)
        inp = inp[:, unsort]

        return inp
