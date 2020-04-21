
import torch
import torch.nn as nn
import torch.nn.functional as F

from pie import initialization, torch_utils


def DotScorer(dec_out, enc_outs, **kwargs):
    """
    Score for query decoder state and the ith encoder state is given
    by their dot product.

    dec_outs: (trg_seq_len x batch x hid_dim)
    enc_outs: (src_seq_len x batch x hid_dim)

    output: ((trg_seq_len x) batch x src_seq_len)
    """
    score = torch.bmm(
        # (batch x src_seq_len x hid_dim)
        enc_outs.transpose(0, 1),
        # (batch x hid_dim x trg_seq_len)
        dec_out.transpose(0, 1).transpose(1, 2))
    # (batch x src_seq_len x trg_seq_len) => (trg_seq_len x batch x src_seq_len)
    return score.transpose(0, 1).transpose(0, 2)


class GeneralScorer(nn.Module):
    """
    Inserts a linear projection to the query state before the dot product
    """
    def __init__(self, dim):
        super(GeneralScorer, self).__init__()
        self.W_a = nn.Linear(dim, dim, bias=False)
        self.init()

    def init(self):
        initialization.init_linear(self.W_a)

    def forward(self, dec_out, enc_outs, **kwargs):
        return DotScorer(self.W_a(dec_out), enc_outs)


class BahdanauScorer(nn.Module):
    """
    Projects both query decoder state and encoder states to an attention space.
    The scores are computed by a dot product with a learnable param v_a after
    transforming the sum of query decoder state and encoder state with a tanh.

    `score(a_i_j) = a_v \dot tanh(W_s @ h_s_j + W_t @ h_t_i)`
    """
    def __init__(self, hid_dim, att_dim):
        self.att_dim = att_dim
        super(BahdanauScorer, self).__init__()
        # params
        self.W_s = nn.Linear(hid_dim, att_dim, bias=False)
        self.W_t = nn.Linear(hid_dim, att_dim, bias=True)
        self.v_a = nn.Parameter(torch.Tensor(att_dim, 1))
        self.init()

    def init(self):
        torch.nn.init.uniform_(self.v_a, -0.05, 0.05)
        initialization.init_linear(self.W_s)
        initialization.init_linear(self.W_t)

    def project_enc_outs(self, enc_outs):
        """
        mapping: (seq_len x batch x hid_dim) -> (seq_len x batch x att_dim)

        Returns:
        --------
        torch.Tensor (seq_len x batch x att_dim),
            Projection of encoder output onto attention space
        """
        seq_len, batch, hid_dim = enc_outs.size()

        return self.W_s(enc_outs.view(-1, hid_dim)).view(seq_len, batch, -1)

    def forward(self, dec_out, enc_outs, enc_att=None):
        # get projected encoder sequence
        if enc_att is None:
            # (seq_len x batch x att dim)
            enc_att = self.project_enc_outs(enc_outs)

        # get projected decoder step (or sequence in ffw mode)
        # ((trg_len x) batch x att_dim)
        dec_att = self.W_t(dec_out)

        (src_len, batch, dim), (trg_len, *_) = enc_att.size(), dec_att.size()

        # see seqmod/test/modules/attention.py
        enc_att = enc_att.view(src_len, 1, batch, -1)
        # (src_len x trg_len x batch x att_dim) +
        # (1           x trg_len x batch x att_dim)
        # => (src_len x trg_len x batch x att_dim)
        dec_enc_att = torch.tanh(enc_att + dec_att.unsqueeze(0))

        # (src_len * trg_len x batch x dim)
        dec_enc_att = dec_enc_att.view(src_len * trg_len, batch, dim)
        # (batch x src_len * trg_len x dim)
        dec_enc_att = dec_enc_att.transpose(0, 1).contiguous()
        # (batch x seq_len x dim) * (1 x dim x 1) -> (batch x seq_len)
        scores = dec_enc_att @ self.v_a.unsqueeze(0)
        # (batch x src_len x trg_len)
        scores = scores.view(batch, src_len, trg_len)
        # (trg_len x batch x src_len)
        return scores.transpose(0, 2).transpose(1, 2)


class Attention(nn.Module):
    """
    Global attention implementing the three scorer modules from Luong 15.

    Parameters:
    -----------
    - hid_dim: int, dimensionality of the query vector
    - att_dim: (optional) int, dimensionality of the attention space (only
        used by the bahdanau scorer). If not given it will default to hid_dim.
    - scorer: str, one of ('dot', 'general', 'bahdanau')
    - hid_dim2: (optional), int, dimensionality of the key vectors (optionally
        used by the bahdanau scorer if given)
    """
    def __init__(self, hid_dim, att_dim=None, scorer='general'):
        super(Attention, self).__init__()

        # Scorer
        if scorer.lower() == 'dot':
            self.scorer = DotScorer
        elif scorer.lower() == 'general':
            self.scorer = GeneralScorer(hid_dim)
        elif scorer.lower() == 'bahdanau':
            self.scorer = BahdanauScorer(hid_dim, att_dim or hid_dim)

        # Output layer (Luong 15. eq (5))
        self.linear_out = nn.Linear(
            hid_dim * 2, hid_dim, bias=scorer.lower() == 'bahdanau')

        self.init()

    def init(self):
        initialization.init_linear(self.linear_out)

    def forward(self, dec_out, enc_outs, lengths):
        """
        Parameters:
        -----------

        - dec_outs: torch.Tensor(trg_seq_len x batch_size x hid_dim)
        - enc_outs: torch.Tensor(seq_len x batch_size x hid_dim)
        - lengths: torch.LongTensor(batch), source lengths

        Returns:
        --------
        - context: (trg_seq_len x batch x hid_dim)
        - weights: (trg_seq_len x batch x seq_len)
        """
        # get scores
        # (trg_seq_len x batch x seq_len)
        weights = self.scorer(dec_out, enc_outs)

        # apply source length mask
        mask = torch_utils.make_length_mask(lengths)
        # (batch x src_seq_len) => (trg_seq_len x batch x src_seq_len)
        mask = mask.unsqueeze(0).expand_as(weights)
        # weights = weights * mask.float()
        # Torch 1.1 -> 1.2: (1 - mask) becomes ~(mask)
        weights.masked_fill_(~mask, -float('inf'))

        # normalize
        weights = F.softmax(weights, dim=2)

        # (eq 7) (batch x trg_seq_len x seq_len) * (batch x seq_len x hid_dim)
        # => (batch x trg_seq_len x hid_dim) => (trg_seq_len x batch x hid_dim)
        context = torch.bmm(
            weights.transpose(0, 1), enc_outs.transpose(0, 1)
        ).transpose(0, 1)
        # (eq 5) linear out combining context and hidden
        context = torch.tanh(self.linear_out(torch.cat([context, dec_out], 2)))

        return context, weights
