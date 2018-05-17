
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import pie.torch_utils as tutils
from pie.encoder import RNNEncoder


class CNNEmbedding(nn.Module):
    """
    Character-level Embeddings with Convolutions following Kim 2014.

    TODO: make sure output embedding dim is equal to `embedding_dim`,
          otherwise char-level embeddings can't be element-wise combined
          with word-level embeddings.
          This basically means to tweak kernel_sizes and out_channels.
          Alternatively, we could add a projection to target embedding_dim.
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 kernel_sizes=(5, 4, 3), out_channels=100, dropout=0.0):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        self.dropout = dropout
        super().__init__()

        self.emb = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx)

        convs = []
        for W in kernel_sizes:
            wide_pad = (0, math.floor(W / 2))
            conv = nn.Conv2d(
                1, out_channels, (embedding_dim, W), padding=wide_pad)
            convs.append(conv)
        self.convs = nn.ModuleList(convs)

    def forward(self, char, nchars, nwords):
        emb = self.emb(char)

        emb = emb.transpose(0, 1)  # (batch x seq_len x emb_dim)
        emb = emb.transpose(1, 2)  # (batch x emb_dim x seq_len)
        emb = emb.unsqueeze(1)     # (batch x 1 x emb_dim x seq_len)

        emb = F.dropout(emb, p=self.dropout, training=self.training)

        conv_outs = []
        for conv in self.convs:
            conv_out = F.relu(conv(emb).squeeze(2))  # (batch x C_o x seq_len)
            conv_out = F.max_pool1d(conv_out, conv_out.size(2))
            conv_out = conv_out.squeeze(2)  # (batch x C_0)
            conv_outs.append(conv_out)

        # (batch * nwords x C_o * len(kernel_sizes))
        output = torch.cat(conv_outs, dim=1)
        return tutils.pad_batch(output, nwords)


class RNNEmbedding(RNNEncoder):
    """
    Character-level Embeddings with RNNs.
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, **kwargs):
        embs = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        super().__init__(embs, hidden_size=embedding_dim, **kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, char, nchars, nwords):
        # (max_seq_len x batch * nwords x emb_dim)
        emb = super().forward(char, nchars)
        # (batch * nwords x emb_dim)
        emb = tutils.get_last_token(emb, nchars)

        return tutils.pad_batch(emb, nwords)


class CombinedEmbedding(nn.Module):
    def __init__(self, label_encoder, emb_dim, char_emb_type='RNN', **char_kwargs):
        self.embedding_dim = emb_dim
        super().__init__()

        # word embeddings
        self.wemb = nn.Embedding(len(label_encoder.word), emb_dim,
                                 padding_idx=label_encoder.word.get_pad())

        # char embeddings
        if char_emb_type.upper() == 'RNN':
            char_emb_cls = RNNEmbedding
        elif char_emb_type.upper() == 'CNN':
            char_emb_cls = CNNEmbedding
        else:
            raise ValueError("Unkonwn embedding class: {}".format(char_emb_type))

        self.cemb = char_emb_cls(
            len(label_encoder.char), emb_dim,
            padding_idx=label_encoder.char.get_pad(), **char_kwargs)

    def mix_embeddings(self, wembs, cembs):
        return torch.cat([wembs, cembs], dim=2)

    def forward(self, words, nwords, chars, nchars):
        # (seq_len x batch x wemb_dim)
        wembs = self.wemb(words)

        # (seq_len x batch x cemb_dim)
        cembs = self.cemb(chars, nchars, nwords)

        return self.mix_embeddings(wembs, cembs)


class MixedEmbedding(CombinedEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # mix parameter
        self.alpha = nn.Parameter(
            torch.Tensor(self.embedding_dim * 2, 1).uniform_(-0.05, 0.05))

    def mix_embeddings(self, wembs, cembs):
        # (seq_len x batch x emb_dim * 2)
        alpha_in = torch.cat([wembs, cembs], dim=2)
        # (seq_len x batch)
        alpha = F.sigmoid(torch.einsum('do,mbd->mb', [self.alpha, alpha_in]))

        wembs = alpha.unsqueeze(2).expand_as(wembs) * wembs
        cembs = (1 - alpha).unsqueeze(2).expand_as(cembs) * cembs

        return wembs + cembs


if __name__ == '__main__':
    from pie.settings import settings_from_file
    from pie.data import Dataset

    settings = settings_from_file('./config.json')
    data = Dataset(settings)
    ((word, wlen), (char, clen)), tasks = next(data.batch_generator())
    print("lemma", tasks['lemma'][0].size(), tasks['lemma'][1])
    print("char", char.size(), clen)
    print("word", word.size(), wlen)
    emb = MixedEmbedding(data.label_encoder, 20)
    output = emb(word, wlen, char, clen)
    print(output.size())
