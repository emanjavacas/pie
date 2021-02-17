
import torch
import torch.nn as nn
import torch.nn.functional as F

from pie import torch_utils
from pie import initialization

from .lstm import CustomBiLSTM
from .highway import Highway


class CNNEmbedding(nn.Module):
    """
    Character-level Embeddings with Convolutions following Kim 2014.
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 highway_layers=2, kernel_sizes=(5, 4, 3), out_channels=32):
        self.num_embeddings = num_embeddings
        self.embedding_dim = out_channels * len(kernel_sizes)
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        super().__init__()

        self.emb = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx)

        convs = []
        for W in kernel_sizes:
            padding = ((W//2) - 1, W - (W//2), 0, 0)
            conv = torch.nn.Sequential(
                torch.nn.ZeroPad2d(padding),
                torch.nn.Conv2d(1, out_channels, (embedding_dim, W)))
            convs.append(conv)
        self.convs = nn.ModuleList(convs)

        self.highway = None
        if highway_layers > 0:
            self.highway = Highway(self.embedding_dim, highway_layers)

        self.init()

    def init(self):
        initialization.init_embeddings(self.emb)
        for conv_seq in self.convs:
            for conv in conv_seq:
                if isinstance(conv, torch.nn.ZeroPad2d):
                    continue
            initialization.init_conv(conv)

    def forward(self, char, nchars, nwords):
        emb = self.emb(char)

        emb = emb.transpose(0, 1)  # (batch x seq_len x emb_dim)
        emb = emb.transpose(1, 2)  # (batch x emb_dim x seq_len)
        emb = emb.unsqueeze(1)     # (batch x 1 x emb_dim x seq_len)

        conv_outs = []
        for conv in self.convs:
            # (batch x C_o x seq_len)
            conv_outs.append(F.relu(conv(emb).squeeze(2)))

        # (batch * nwords x C_o * len(kernel_sizes) x seq_len)
        conv_outs = torch.cat(conv_outs, dim=1)
        # (batch * nwords  x C_o * len(kernel_sizes) x 1)
        conv_out = F.max_pool1d(conv_outs, conv_outs.size(2)).squeeze(2)
        if self.highway is not None:
            conv_out = self.highway(conv_out)
        conv_out = torch_utils.pad_flat_batch(
            conv_out, nwords, maxlen=max(nwords).item())

        conv_outs = conv_outs.transpose(0, 2).transpose(1, 2)

        return conv_out, conv_outs


class RNNEmbedding(nn.Module):
    """
    Character-level Embeddings with BiRNNs.
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 custom_lstm=False, cell='LSTM', init_rnn='default',
                 num_layers=1, dropout=0.0):
        self.num_embeddings = num_embeddings
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim * 2  # bidirectional
        super().__init__()

        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        initialization.init_embeddings(self.emb)

        if custom_lstm:
            self.rnn = CustomBiLSTM(
                embedding_dim, embedding_dim, num_layers=num_layers, dropout=dropout)
        else:
            self.rnn = getattr(nn, cell)(
                embedding_dim, embedding_dim, bidirectional=True,
                num_layers=num_layers, dropout=dropout if num_layers > 1 else 0)
            initialization.init_rnn(self.rnn, scheme=init_rnn)

    def forward(self, char, nchars, nwords):
        """
        Parameters
        ===========
        char : tensor((seq_len x) batch)
        nchars : tensor(batch)
        nwords : tensor(output batch)
        """
        # embed
        char = self.emb(char)
        # rnn
        hidden = None
        _, sort = torch.sort(nchars, descending=True)
        _, unsort = sort.sort()
        char, nchars = char[:, sort], nchars[sort]
        if isinstance(self.rnn, nn.RNNBase):
            outs, emb = self.rnn(
                nn.utils.rnn.pack_padded_sequence(char, nchars.cpu()), hidden)
            outs, _ = nn.utils.rnn.pad_packed_sequence(outs)
            if isinstance(emb, tuple):
                emb, _ = emb
        else:
            outs, (emb, _) = self.rnn(char, hidden, nchars)
        # (max_seq_len x batch * nwords x emb_dim)
        outs, emb = outs[:, unsort], emb[:, unsort]
        # (layers * 2 x batch x hidden) -> (layers x 2 x batch x hidden)
        emb = emb.view(self.num_layers, 2, len(nchars), -1)
        # use only last layer
        emb = emb[-1]
        # (2 x batch x hidden) - > (batch x 2 * hidden)
        emb = emb.transpose(0, 1).contiguous().view(len(nchars), -1)
        # (batch x 2 * hidden) -> (nwords x batch x 2 * hidden)
        emb = torch_utils.pad_flat_batch(emb, nwords, maxlen=max(nwords).item())

        return emb, outs


class EmbeddingMixer(nn.Module):
    def __init__(self, emb_dim):
        self.embedding_dim = emb_dim
        super().__init__()

        # mix parameter
        self.alpha = nn.Parameter(
            torch.Tensor(self.embedding_dim * 2, 1).uniform_(-0.05, 0.05))

    def forward(self, wembs, cembs):
        # ((seq_len x) batch x emb_dim * 2)
        alpha_in = torch.cat([wembs, cembs], dim=-1)
        # ((seq_len x) batch)
        if wembs.dim() == 3:
            alpha = torch.sigmoid(torch.einsum('do,mbd->mb', [self.alpha, alpha_in]))
        else:
            alpha = torch.sigmoid(torch.einsum('do,bd->b', [self.alpha, alpha_in]))

        wembs = alpha.unsqueeze(-1).expand_as(wembs) * wembs
        cembs = (1 - alpha).unsqueeze(-1).expand_as(cembs) * cembs

        return wembs + cembs


def EmbeddingConcat():
    def func(wemb, cemb):
        return torch.cat([wemb, cemb], dim=-1)
    return func


def build_embeddings(label_encoder, wemb_dim,
                     cemb_dim, cemb_type, custom_cemb_cell, cemb_layers, cell, init_rnn,
                     merge_type, dropout):
    """
    Utility function to build embedding layers
    """
    wemb = None
    if wemb_dim > 0:
        wemb = nn.Embedding(len(label_encoder.word), wemb_dim,
                            padding_idx=label_encoder.word.get_pad())
        # init embeddings
        initialization.init_embeddings(wemb)

    cemb = None
    if cemb_type.upper() == 'RNN':
        cemb = RNNEmbedding(len(label_encoder.char), cemb_dim,
                            padding_idx=label_encoder.char.get_pad(),
                            custom_lstm=custom_cemb_cell, dropout=dropout,
                            num_layers=cemb_layers, cell=cell, init_rnn=init_rnn)
    elif cemb_type.upper() == 'CNN':
        cemb = CNNEmbedding(len(label_encoder.char), cemb_dim,
                            padding_idx=label_encoder.char.get_pad())

    merger = None
    if cemb is not None and wemb is not None:
        if merge_type.lower() == 'mixer':
            if cemb.embedding_dim != wemb.embedding_dim:
                raise ValueError("EmbeddingMixer needs equal embedding dims")
            merger = EmbeddingMixer(wemb_dim)
            in_dim = wemb_dim
        elif merge_type == 'concat':
            merger = EmbeddingConcat()
            in_dim = wemb_dim + cemb.embedding_dim
        else:
            raise ValueError("Unknown merge method: {}".format(merge_type))
    elif cemb is not None:
        in_dim = cemb.embedding_dim
    else:
        in_dim = wemb_dim

    return (wemb, cemb, merger), in_dim


if __name__ == '__main__':
    from pie.settings import settings_from_file
    from pie.data import Dataset

    settings = settings_from_file('./config.json')
    data = Dataset(settings)
    ((word, wlen), (char, clen)), tasks = next(data.batch_generator())
    print("lemma", tasks['lemma'][0].size(), tasks['lemma'][1])
    print("char", char.size(), clen)
    print("word", word.size(), wlen)

    emb_dim = 20
    wemb = nn.Embedding(len(data.label_encoder.word), emb_dim)
    cemb = RNNEmbedding(len(data.label_encoder.char), emb_dim)
    cnncemb = CNNEmbedding(len(data.label_encoder.char), emb_dim)

    mixer = EmbeddingMixer(20)
    w, (c, _) = wemb(word), cemb(char, clen, wlen)
    output = mixer(w, c)

    output2 = []
    for w, c in zip(w, c):
        output2.append(mixer(w, c))
    output2 = torch.stack(output2)

    print(output.size())
