
import torch.nn as nn


def init_embeddings(embeddings):
    embeddings.reset_parameters()


def init_linear(linear):
    linear.reset_parameters()
    nn.init.constant_(linear.bias, 0.)
    pass


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
    pass


def init_conv(conv):
    conv.reset_parameters()
    nn.init.xavier_uniform_(conv.weight)
    nn.init.constant_(conv.bias, 0.)
    pass
