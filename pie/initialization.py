
import torch
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


def init_pretrained_embeddings(path, encoder, embedding):
    with open(path) as f:
        nemb, dim = next(f).split()

        if int(dim) != embedding.weight.data.size(1):
            raise ValueError("Unexpected embeddings size: {}".format(dim))

        inits = 0
        for line in f:
            word, *vec = line.split()
            if word in encoder.table:
                embedding.weight.data[encoder.table[word], :].copy_(
                    torch.tensor([float(v) for v in vec]))
                inits += 1

    if embedding.padding_idx is not None:
        embedding.weight.data[embedding.padding_idx].zero_()

    print("Initialized {}/{} embeddings".format(inits, embedding.num_embeddings))
