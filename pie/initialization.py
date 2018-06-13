
import torch.nn as nn


def init_embeddings(embeddings):
    nn.init.uniform_(embeddings.weight, -0.05, 0.05)


def init_linear(linear):
    nn.init.uniform_(linear.weight, -0.05, 0.05)
    nn.init.constant_(linear.bias, 0.)


def init_rnn(rnn):
    if isinstance(rnn, (nn.GRUCell, nn.LSTMCell, nn.RNNCell)):
        nn.init.xavier_uniform_(rnn.weight_hh)
        nn.init.xavier_uniform_(rnn.weight_ih)
        nn.init.constant_(rnn.bias_hh, 0.)
        nn.init.constant_(rnn.bias_ih, 0.)

    else:
        for layer in range(rnn.num_layers):
            nn.init.xavier_uniform_(getattr(rnn, f'weight_hh_l{layer}'))
            nn.init.xavier_uniform_(getattr(rnn, f'weight_ih_l{layer}'))
            nn.init.constant_(getattr(rnn, f'bias_hh_l{layer}'), 0.)
            nn.init.constant_(getattr(rnn, f'bias_ih_l{layer}'), 0.)


def init_conv(conv):
    nn.init.xavier_uniform_(conv.weight)
    nn.init.constant_(conv.bias, 0.)


def init_sequential_linear(sequential):
    for child in sequential.children():
        if isinstance(child, nn.Linear):
            init_linear(child)
