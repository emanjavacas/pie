
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from pie import inits
from pie.beam_search import Beam
from pie import torch_utils
from pie.constants import TINY


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

    def loss(self, logits, targets):
        loss = F.cross_entropy(
            logits.view(-1, len(self.label_encoder)), targets.view(-1),
            weight=self.nll_weight, size_average=False)

        return loss / targets.ne(self.label_encoder.get_pad()).sum().item()

    def predict(self, enc_outs, lengths):
        """
        Parameters
        ==========
        enc_outs : torch.tensor(seq_len x batch x hidden_size)
        """
        probs = F.softmax(self.decoder(enc_outs), dim=-1)
        probs, preds = torch.max(probs.transpose(0, 1), dim=-1)

        lengths = lengths - 1   # remove <eos>
        preds = [self.label_encoder.inverse_transform(pred)[:length]
                 for pred, length in zip(preds.tolist(), lengths.tolist())]
        probs = probs.tolist()

        return preds, probs


class CRFDecoder(nn.Module):
    def __init__(self, label_encoder, hidden_size):
        self.label_encoder = label_encoder
        super().__init__()

        BOS = label_encoder.get_bos()
        EOS = label_encoder.get_eos()
        PAD = label_encoder.get_pad()
        vocab = len(label_encoder)

        self.projection = nn.Linear(hidden_size, vocab)

        # matrix of transition scores from j to i
        self.trans = nn.Parameter(torch.randn(vocab, vocab))
        self.trans[BOS, :] = -10000. # no transition to <bos>
        self.trans[:, EOS] = -10000. # no transition from <eos> except to <pad>
        self.trans[:, PAD] = -10000. # no transition from <pad> except to <pad>
        self.trans[PAD, :] = -10000.
        self.trans[PAD, EOS] = 0.
        self.trans[PAD, PAD] = 0.

    def init(self):
        inits.init_linear(self.projection)

    def forward(self, enc_outs, lengths):
        # (seq_len x batch x vocab)
        feats = self.projection(enc_outs)
        seq_len, batch, vocab = feats.size()

        # initialize forward variables in log space
        score = enc_outs.new(batch, vocab).fill_(-10000.)
        score[:, self.label_encoder.get_bos()] = 0.

        # mask on padding
        mask = torch_utils.make_length_mask(lengths)  # (batch x seq_len)
        feats = feats * mask.unsqueeze(2).expand_as(feats)

        for t in range(vocab): # iterate through the sequence
            mask_t = mask[:, t].unsqueeze(-1).expand_as(score)
            score_t = score.unsqueeze(1).expand(-1, *self.trans.size())
            emit = y[:, t].unsqueeze(-1).expand_as(score_t)
            trans = self.trans.unsqueeze(0).expand_as(score_t)
            score_t = torch_utils.log_sum_exp(score_t + emit + trans)
            score = score_t * mask_t + score * (1 - mask_t)

        score = torch_utils.log_sum_exp(score)

        return score

    def score(self, feats, targets, lengths):
        # calculate the score of a given sequence
        seq_len, batch, vocab = feats.size()
        score = feats.new(batch).fill_(0.)

        # prepend <bos>
        bos = targets.new(1, batch).fill_(self.label_encoder.get_bos())
        targets = torch.cat([bos, targets])

        # mask
        mask = torch_utils.make_length_mask(lengths)  # (batch x seq_len)

        for t in range(seq_len): # iterate through the sequence
            mask_t = mask[:, t]
            emit = torch.cat([y[b, t, y0[b, t + 1]] for b in range(BATCH_SIZE)])
            trans = torch.cat([self.trans[seq[t + 1], seq[t]] for seq in y0]) * mask_t
            score = score + emit + trans
        return score

    def decode(self, y): # Viterbi decoding
        # initialize backpointers and viterbi variables in log space
        bptr = []
        score = Tensor(self.num_tags).fill_(-10000.)
        score[SOS_IDX] = 0.
        score = Var(score)

        for emit in y: # iterate through the sequence
            # backpointers and viterbi variables at this timestep
            bptr_t = []
            score_t = []
            for i in range(self.num_tags): # for each next tag
                z = score + self.trans[i]
                best_tag = argmax(z) # find the best previous tag
                bptr_t.append(best_tag)
                score_t.append(z[best_tag])
            bptr.append(bptr_t)
            score = torch.cat(score_t) + emit
        best_tag = argmax(score)
        best_score = score[best_tag]

        # back-tracking
        best_path = [best_tag]
        for bptr_t in reversed(bptr):
            best_path.append(bptr_t[best_tag])
        best_path = reversed(best_path[:-1])

        return best_path


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

    def forward(self, dec_outs, enc_outs, lengths):
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
        att_proj = self.linear_in(dec_outs)
        # (batch x out_seq_len x hidden) * (batch x hidden x inp_seq_len)
        # -> (batch x out_seq_len x inp_seq_len)
        weights = torch.bmm(
            att_proj.transpose(0, 1),
            enc_outs.transpose(0, 1).transpose(1, 2))
        # downweight scores for source padding (mask) (batch x inp_seq_len)
        mask = torch_utils.make_length_mask(lengths)
        weights.masked_fill_(1 - mask.unsqueeze(1).expand_as(weights), -float('inf'))
        # apply softmax
        weights = F.softmax(weights, dim=2)
        # (batch x out_seq_len x inp_seq_len) * (batch x inp_seq_len x hidden)
        # -> (batch x out_seq_len x hidden_size)
        weighted = torch.bmm(weights, enc_outs.transpose(0, 1))
        # (out_seq_len x batch x hidden * 2)
        context = torch.cat([weighted.transpose(0, 1), dec_outs], 2)
        # (out_seq_len x batch x hidden)
        context = F.tanh(self.linear_out(context))

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

    def forward(self, targets, lengths, enc_outs, src_lengths, context=None):
        """
        Decoding routine for training. Returns the logits corresponding to
        the targets for the `loss` method. Takes care of padding.
        """
        eos = self.label_encoder.get_eos()
        embs = self.embs(torch_utils.prepad(targets[:-1], pad=eos))

        if self.context_dim > 0:
            if context is None:
                raise ValueError("Contextual Decoder needs `context`")
            # (seq_len x batch x emb_dim) + (batch x context_dim)
            embs = torch.cat(
                [embs, context.unsqueeze(0).repeat(embs.size(0), 1, 1)],
                dim=2)
        embs, unsort = torch_utils.pack_sort(embs, lengths)

        outs, _ = self.rnn(embs)
        outs, _ = unpack(outs)
        outs = outs[:, unsort]

        context, _ = self.attn(outs, enc_outs, src_lengths)

        return self.proj(context)

    def loss(self, logits, targets):
        """
        Compute loss from logits (output of forward)

        Parameters
        ===========
        logits : tensor(seq_len x batch x vocab)
        targets : tensor(seq_len x batch)
        """
        loss = F.cross_entropy(
            logits.view(-1, len(self.label_encoder)), targets.view(-1),
            weight=self.nll_weight, size_average=False)

        return loss / targets.ne(self.label_encoder.get_pad()).sum().item()

    def predict_sequence(self, enc_outs, lengths, context=None):
        """
        Decoding routine with step-wise argmax for fixed output lengths

        Parameters
        ===========
        enc_outs : tensor(src_seq_len x batch x hidden_size)
        context : tensor(batch x hidden_size), optional
        """
        hidden = None
        batch = enc_outs.size(1)
        device = enc_outs.device
        mask = torch.ones(batch, dtype=torch.int64, device=device)
        inp = torch.zeros(batch, dtype=torch.int64, device=device)
        inp += self.label_encoder.get_eos()
        hyps, scores = [], 0

        for i in range(max(lengths.tolist())):
            emb = self.embs(inp)
            if context is not None:
                emb = torch.cat([emb, context], dim=1)
            emb = emb.unsqueeze(0)
            outs, hidden = self.rnn(emb, hidden)
            outs, _ = self.attn(outs, enc_outs, lengths)
            outs = self.proj(outs).squeeze(0)
            probs = F.log_softmax(outs, dim=1)
            score, inp = probs.max(1)
            hyps.append(inp.tolist())
            mask = mask * (i != lengths).long()
            score[mask == 0] = 0
            scores += score

        lengths = lengths - 1   # remove <eos>
        hyps = [self.label_encoder.inverse_transform(hyp)[:length]
                for hyp, length in zip(zip(*hyps), lengths.tolist())]
        scores = (scores / lengths.float()).tolist()

        return hyps, scores

    def predict_max(self, enc_outs, lengths, context=None, max_seq_len=20):
        """
        Decoding routine for inference with step-wise argmax procedure

        Parameters
        ===========
        enc_outs : tensor(src_seq_len x batch x hidden_size)
        context : tensor(batch x hidden_size), optional
        """
        hidden = None
        batch = enc_outs.size(1)
        device = enc_outs.device
        mask = torch.ones(batch, dtype=torch.int64, device=device)
        inp = torch.zeros(batch, dtype=torch.int64, device=device)
        inp += self.label_encoder.get_eos()
        hyps, scores = [], 0

        for _ in range(max_seq_len):
            if mask.sum().item() == 0:
                break

            emb = self.embs(inp)
            if context is not None:
                emb = torch.cat([emb, context], dim=1)
            emb = emb.unsqueeze(0)
            outs, hidden = self.rnn(emb, hidden)
            outs, _ = self.attn(outs, enc_outs, lengths)
            outs = self.proj(outs).squeeze(0)
            probs = F.log_softmax(outs, dim=1)
            score, inp = probs.max(1)
            hyps.append(inp.tolist())
            mask = mask * (inp != self.label_encoder.get_eos()).long()
            score = score.cpu()
            score[mask == 0] = 0
            scores += score

        hyps = [self.label_encoder.stringify(hyp) for hyp in zip(*hyps)]
        scores = [s/(len(hyp) + TINY) for s, hyp in zip(scores.tolist(), hyps)]

        return hyps, scores

    def predict_beam(self, enc_outs, lengths, context=None,
                     max_seq_len=20, beam_width=5):
        """
        Decoding routine for inference with beam search

        Parameters
        ===========
        enc_outs : tensor(src_seq_len x batch x hidden_size)
        context : tensor(batch x hidden_size), optional
        """
        hidden = None
        seq_len, batch, _ = enc_outs.size()
        beams = [Beam(beam_width, eos=self.label_encoder.get_eos(),
                      device=enc_outs.device) for _ in range(batch)]

        # expand data along beam width
        enc_outs = enc_outs.repeat(1, beam_width, 1)
        lengths = lengths.repeat(beam_width)
        if context is not None:
            context = context.repeat(beam_width, 1)

        for _ in range(max_seq_len):
            if all(not beam.active for beam in beams):
                break

            inp = [beam.get_current_state() for beam in beams]
            inp = torch.stack(inp).t().contiguous().view(-1)
            emb = self.embs(inp)
            if context is not None:
                emb = torch.cat([emb, context], dim=1)
            emb = emb.unsqueeze(0)
            outs, hidden = self.rnn(emb, hidden)
            outs, _ = self.attn(outs, enc_outs, lengths)
            outs = self.proj(outs)
            outs = outs.squeeze(0)
            probs = F.log_softmax(outs, dim=1)

            # (beam_width x batch x vocab)
            probs = probs.view(beam_width, batch, -1)
            hidden = hidden.view(1, beam_width, batch, -1)
            enc_outs = enc_outs.view(seq_len, beam_width, batch, -1)

            for i, beam in enumerate(beams):
                # advance
                beam.advance(probs[:, i])
                # rearrange
                source_beam = beam.get_source_beam()
                hidden[:, :, i].copy_(
                    hidden[:, :, i].index_select(1, source_beam))
                enc_outs[:, :, i].copy_(
                    enc_outs[:, :, i].index_select(1, source_beam))

            hidden = hidden.view(1, beam_width * batch, -1)
            enc_outs = enc_outs.view(seq_len, beam_width * batch, -1)

        scores, hyps = [], []
        for beam in beams:
            bscores, bhyps = beam.decode(n=1)
            bscores, bhyps = bscores[0], bhyps[0]
            # unwrap best k beams dimension
            scores.append(bscores)
            hyps.append(bhyps)

        hyps = [self.label_encoder.stringify(hyp) for hyp in zip(*hyps)]
        scores = [s/(len(hyp) + TINY) for s, hyp in zip(scores.tolist(), hyps)]

        return hyps, scores
