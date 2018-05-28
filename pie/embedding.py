
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pie import torch_utils
from pie.encoder import RNNEncoder


class CNNEmbedding(nn.Module):
    """
    Character-level Embeddings with Convolutions following Kim 2014.
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 kernel_sizes=(5, 4, 3), out_channels=100, dropout=0.0):
        self.num_embeddings = num_embeddings
        self.embedding_dim = out_channels * len(kernel_sizes)
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
        output = torch_utils.pad_flat_batch(output, nwords-1, maxlen=max(nwords).item())
        return output, None


class RNNEmbedding(RNNEncoder):
    """
    Character-level Embeddings with RNNs.
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, **kwargs):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        super().__init__(embedding_dim, hidden_size=embedding_dim, **kwargs)

        self.embs = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, char, nchars, nwords):
        """
        Parameters
        ===========
        char : tensor((seq_len x) batch)
        nchars : tensor(batch)
        nwords : tensor(output batch)
        """
        char = self.embs(char)
        # (max_seq_len x batch * nwords x emb_dim)
        emb_outs = super().forward(char, nchars)
        # (batch * nwords x emb_dim)
        emb = torch_utils.get_last_token(emb_outs, nchars)
        emb = torch_utils.pad_flat_batch(emb, nwords-1, maxlen=max(nwords).item())

        return emb, emb_outs


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
            alpha = F.sigmoid(torch.einsum('do,mbd->mb', [self.alpha, alpha_in]))
        else:
            alpha = F.sigmoid(torch.einsum('do,bd->b', [self.alpha, alpha_in]))

        wembs = alpha.unsqueeze(-1).expand_as(wembs) * wembs
        cembs = (1 - alpha).unsqueeze(-1).expand_as(cembs) * cembs

        return wembs + cembs


def EmbeddingConcat():
    def func(wemb, cemb):
        return torch.cat([wemb, cemb], dim=-1)
    return func


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
    cemb = RNNEmbedding(len(data.label_encoder.char), emb_dim, bidirectional=True)
    cnncemb = CNNEmbedding(len(data.label_encoder.char), emb_dim)

    mixer = EmbeddingMixer(20)
    w, (c, _) = wemb(word), cemb(char, clen, wlen)
    output = mixer(w, c)

    output2 = []
    for w, c in zip(w, c):
        output2.append(mixer(w, c))
    output2 = torch.stack(output2)

    print(output.size())
