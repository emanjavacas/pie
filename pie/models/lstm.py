
import math

import torch
import torch.nn as nn
from pie import torch_utils


class CustomLSTMCell(nn.Module):
    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialization from dynet
        """
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.bias[self.hidden_size: 2 * self.hidden_size], 1.0)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.xavier_uniform_(self.weight_ih)

    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).

        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias = self.bias.unsqueeze(0).expand(batch_size, *self.bias.size())
        wi = torch.mm(input_, self.weight_ih)
        wh = torch.mm(h_0, self.weight_hh)
        i, f, o, g = torch.split(wh + wi + bias, self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f + 1) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class CustomLSTM(nn.Module):
    """A module that runs multiple steps of LSTM."""
    def __init__(self, input_size, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = CustomLSTMCell(input_size=input_size, hidden_size=hidden_size)

    @staticmethod
    def _forward_rnn(cell, input_, hx, lengths, backward=False):
        max_time = input_.size(0)
        output = []
        for time in reversed(range(max_time)) if backward else range(max_time):
            h_next, c_next = cell(input_=input_[time], hx=hx)
            mask = (time < lengths).float().unsqueeze(1).expand_as(h_next)
            h_next = h_next * mask + hx[0] * (1 - mask)
            c_next = c_next * mask + hx[1] * (1 - mask)
            hx_next = (h_next, c_next)
            output.append(h_next * mask)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, hidden=None, lengths=None, backward=False):
        max_time, batch_size, _ = input_.size()

        if hidden is None:
            h0 = input_.new(batch_size, self.hidden_size).zero_()
            c0 = input_.new(batch_size, self.hidden_size).zero_()
        else:
            h0, c0 = hidden
            if h0.dim() == 3:
                h0, c0 = h0.squeeze(0), c0.squeeze(0)

        if lengths is None:
            lengths = input_.new([max_time] * batch_size).long()

        output, (h_n, c_n) = CustomLSTM._forward_rnn(
            self.cell, input_, (h0, c0), lengths, backward=backward)

        return output, (h_n.unsqueeze(0), c_n.unsqueeze(0))


class CustomBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        super().__init__()

        layers = []
        rnn_inp = input_size
        for layer in range(num_layers):
            fwd = CustomLSTM(rnn_inp, hidden_size)
            bwd = CustomLSTM(rnn_inp, hidden_size)
            layers.append((fwd, bwd))
            self.add_module('fwd_{}'.format(layer), fwd)
            self.add_module('bwd_{}'.format(layer), bwd)
            rnn_inp = hidden_size * 2
        self.layers = layers

    def forward(self, inputs, hidden=None, lengths=None):
        if hidden is not None:
            h0, c0 = hidden
            h0, c0 = h0.view(len(self.layers), 2, -1), c0.view(len(self.layers), 2, -1)
            (fwd_h0, bwd_h0), (fwd_c0, bwd_c0) = h0.split(1, dim=1), c0.split(1, dim=1)
            fwd_hidden = tuple(zip(fwd_h0, fwd_c0))
            bwd_hidden = tuple(zip(fwd_h0, fwd_c0))
        else:
            fwd_hidden = bwd_hidden = [None] * len(self.layers)

        hn, cn = [], []
        for layer, (fwd, bwd) in enumerate(self.layers):
            fwd_outs, (fwd_h1, fwd_c1) = fwd(
                inputs, fwd_hidden[layer], lengths=lengths)
            bwd_outs, (bwd_h1, bwd_c1) = bwd(
                inputs, bwd_hidden[layer], lengths=lengths, backward=True)
            # compute new input (seq_len x batch x 2 * hidden_size)
            inputs = torch.cat([fwd_outs, bwd_outs], dim=2)
            # apply dropout
            if layer < self.num_layers - 1:
                inputs = torch_utils.sequential_dropout(
                    inputs, self.dropout, self.training)
            # store hidden
            hn.append(torch.cat([fwd_h1, bwd_h1], dim=0))
            cn.append(torch.cat([fwd_c1, bwd_c1], dim=0))

        return inputs, (torch.cat(hn, dim=0), torch.cat(cn, dim=0))
