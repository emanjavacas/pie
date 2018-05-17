
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from pie import inits
from pie.beam_search import Beam
from pie.torch_utils import pack_sort


class LinearDecoder(nn.Module):
    """
    Simple Linear Decoder that outputs a probability distribution
    over the vocabulary

    Parameters
    ===========
    label_encoder : LabelEncoder
    in_features : int, input dimension
    """
    def __init__(self, label_encoder, in_features, dropout=0.0):
        self.label_encoder = label_encoder
        self.dropout = dropout
        super(LinearDecoder, self).__init__()

        nll_weight = torch.ones(len(label_encoder))
        nll_weight[label_encoder.get_pad()] = 0.
        self.register_buffer('nll_weight', nll_weight)
        self.decoder = nn.Linear(in_features, len(label_encoder))
        self.init()

    def init(self):
        # linear
        inits.init_linear(self.decoder)

    def forward(self, enc_outs):
        linear_out = self.decoder(enc_outs)
        linear_out = F.dropout(
            linear_out, p=self.dropout, training=self.training)

        return linear_out

    def loss(self, enc_outs, targets):
        logits = self(enc_outs).view(-1, len(self.label_encoder))
        loss = F.cross_entropy(logits, targets.view(-1), weight=self.nll_weight,
                               size_average=False)

        return loss / targets.ne(self.label_encoder.get_pad()).sum().item()


class Attention(nn.Module):
    """
    Attention module.

    Parameters
    ===========
    hidden_size : int, size of both the encoder output/attention
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size)
        self.init()

    def init(self):
        inits.init_linear(self.linear_in)
        inits.init_linear(self.linear_out)

    def forward(self, dec_outs, enc_outs):
        """
        Parameters
        ===========
        dec_outs : (out_seq_len x batch x hidden_size)
            Output of the rnn decoder.
        enc_outs : (inp_seq_len x batch x hidden_size)
            Output of the encoder over the entire sequence.

        Returns
        ========
        context : (out_seq_len x batch x hidden_size)
            Context vector combining current rnn output and the entire
            encoded sequence.
        weights : (out_seq_len x batch x inp_seq_len)
            Weights computed by the attentional module over the input seq.
        """
        out_seq, batch, hidden_size = dec_outs.size()
        # (out_seq_len x batch x hidden_size)
        att_proj = self.linear_in(
            dec_outs.view(out_seq * batch, -1)
        ).view(out_seq, batch, -1)
        # (batch x out_seq_len x hidden) * (batch x hidden x inp_seq_len)
        # -> (batch x out_seq_len x inp_seq_len)
        weights = torch.bmm(
            att_proj.transpose(0, 1),
            enc_outs.transpose(0, 1).transpose(1, 2))
        # apply softmax
        weights = F.softmax(weights, dim=2)
        # (batch x out_seq_len x inp_seq_len) * (batch x inp_seq_len x hidden)
        # -> (batch x out_seq_len x hidden_size)
        weighted = torch.bmm(weights, enc_outs.transpose(0, 1))
        # (out_seq_len x batch x hidden * 2)
        combined = torch.cat([weighted.transpose(0, 1), dec_outs], 2)
        # (out_seq_len x batch x hidden)
        combined = self.linear_out(
            combined.view(out_seq * batch, -1)
        ).view(out_seq, batch, -1)

        context = F.tanh(combined)

        return context, weights


class AttentionalDecoder(nn.Module):

    """
    Decoder using attention over the entire input sequence

    Parameters
    ===========
    label_encoder : LabelEncoder of the task
    in_dim : int, embedding dimension of the task.
        It should be the same as the corresponding encoder to ensure that weights
        can be shared.
    hidden_size : int, hidden size of the encoder, decoder and attention modules.
    context_dim : int (optional), dimensionality of additional context vectors
    """
    def __init__(self, label_encoder, in_dim, hidden_size, context_dim=0, dropout=0.0):
        self.label_encoder = label_encoder
        self.context_dim = context_dim
        super(AttentionalDecoder, self).__init__()

        nll_weight = torch.ones(len(label_encoder))
        nll_weight[label_encoder.get_pad()] = 0.
        self.register_buffer('nll_weight', nll_weight)
        self.embs = nn.Embedding(len(label_encoder), in_dim)
        self.rnn = nn.GRU(in_dim + context_dim, hidden_size, dropout=dropout)
        self.attn = Attention(hidden_size)
        self.proj = nn.Linear(hidden_size, len(label_encoder))

        self.init()

    def init(self):
        # embeddings
        inits.init_embeddings(self.embs)
        # rnn
        inits.init_rnn(self.rnn)
        # linear
        inits.init_linear(self.proj)

    def forward(self, inp, lengths, enc_outs, context=None):
        """
        Decoding routine for training
        """
        embs = self.embs(inp)

        if self.context_dim > 0:
            if context is None:
                raise ValueError("Contextual Decoder needs `context`")
            # (seq_len x batch x emb_dim) + (batch x context_dim)
            embs = torch.cat(
                [embs, context.unsqueeze(0).repeat(embs.size(0), 1, 1)],
                dim=2)
        embs, unsort = pack_sort(embs, lengths)

        outs, _ = self.rnn(embs)
        outs, _ = unpack(outs)
        outs = outs[:, unsort]

        context, _ = self.attn(outs, enc_outs)

        return self.proj(context)

    def loss(self, logits, targets):
        logits = logits.view(-1, len(self.label_encoder))
        loss = F.cross_entropy(logits, targets.view(-1), weight=self.nll_weight,
                               size_average=False)
        return loss / targets.ne(self.label_encoder.get_pad()).sum().item()

    def generate(self, enc_outs, max_seq_len=20, beam_width=5):
        """
        Decoding routine for inference with beam search
        """
        hidden = None
        batch = enc_outs.size(1)
        beams = [Beam(beam_width, self.label_encoder.char.get_eos(),
                      device=enc_outs.device) for _ in range(batch)]

        # expand data along beam width
        enc_outs = enc_outs.repeat(1, beam_width, 1)

        for _ in range(max_seq_len):
            if all(not beam.active for beam in beams):
                break

            inp = torch.cat([beam.get_current_state() for beam in beams])
            emb = self.embs(inp)
            outs, hidden = self.rnn(emb, hidden)
            context, _ = self.attn(outs, enc_outs)
            probs = F.log_softmax(self.proj(outs))

            # (batch x beam_width x vocab)
            probs = probs.view(beam_width, batch, -1)
            hidden = hidden.view(1, beam_width, batch, -1)
            enc_outs = enc_outs.view(enc_outs.size(0), beam_width, batch, -1)

            for i, beam in enumerate(beams):
                source_beam = beam.get_source_beam()
                # advance
                beam.advance(probs[:, i])
                # rearrange
                hidden[:, :, i].copy_(
                    hidden[:, :, i].index_select(1, source_beam))
                enc_outs[:, :, i].copy_(
                    enc_outs[:, :, i].index_select(1, source_beam))

            hidden = hidden.view(1, beam_width * batch, -1)
            enc_outs = enc_outs.view(enc_outs.size(0), beam_width * batch, -1)

        scores, hyps = [], []
        for beam in beams:
            bscores, bhyps = beam.decode(n=1)
            bscores, bhyps = bscores[0], bhyps[0]
            # unwrap best k beams dimension
            scores.append(bscores)
            hyps.append(bhyps)

        return scores, hyps
