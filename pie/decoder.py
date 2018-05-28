
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

        preds = [self.label_encoder.inverse_transform(pred)[:length]
                 for pred, length in zip(preds.tolist(), lengths.tolist())]
        probs = probs.tolist()

        return preds, probs


class CRFDecoder(nn.Module):
    """
    CRF decoder layer
    """
    def __init__(self, label_encoder, hidden_size):
        self.label_encoder = label_encoder
        super().__init__()

        vocab = len(label_encoder)
        self.projection = nn.Linear(hidden_size, vocab)
        self.transition = nn.Parameter(torch.tensor(vocab, vocab))

    def init(self):
        inits.init_linear(self.projection)
        nn.init.xavier_normal(self.transition)

    def forward(self, enc_outs):
        # (seq_len x batch x vocab)
        logits = self.projection(enc_outs)

        return F.log_softmax(logits)

    def partition(self, logits, mask):
        seq_len, batch, vocab = logits.size()

        # (batch x vocab)
        Z = logits[0]

        for t in range(1, seq_len):
            emit_score = logits[t].view(batch, 1, vocab)
            trans_score = self.transition.view(1, vocab, vocab)
            # mask and update
            Z_t = Z.view(batch, vocab, 1)
            # (batch x vocab x vocab) => (batch x vocab)
            Z_t = torch_utils.log_sum_exp(Z_t + emit_score + trans_score, 1)
            mask_t = mask[t].view(batch, 1)
            Z = Z_t * mask_t + Z * (1 - mask_t)

        # (batch x vocab) => (batch)
        Z = torch_utils.log_sum_exp(Z)

        return Z

    def score(self, logits, mask, targets):
        # calculate the score of a given sequence
        seq_len, batch, vocab = logits.size()
        score = 0.
        # (batch x vocab x vocab)
        trans = self.transition.unsqueeze(0).expand(batch, vocab, vocab)

        # iterate from tag to next tag
        for t in range(seq_len - 1):
            curr_tag, next_tag = targets[t], targets[t+1]
            # from current transition scores (batch, 1, vocab) => (batch, vocab)
            trans_score = trans.gather(
                1, curr_tag.view(batch, 1, 1).expand(batch, 1, vocab)
            ).squeeze(1)
            # from current to next transition scores (batch, 1) => (batch)
            trans_score = trans_score.gather(1, next_tag.view(batch, 1)).squeeze(1)
            # (batch)
            emit_score = logits.gather(1, next_tag.view(batch, 1)).squeeze(1)
            score = score + (trans_score * mask[t+1]) + (emit_score * mask[t])
            
        # last step
        last_target_index = mask.sum(0).long() - 1
        # (batch)
        last_target = targets.gather(0, last_target_index.expand(seq_len, batch))[0]
        # (batch x 1) => (batch)
        last_score = logits[-1].gather(1, last_target.unsqueeze(1)).squeeze(1)
        # (batch)
        score = last_score * mask[-1]

        return score

    def loss(self, enc_outs, targets, lengths):
        # mask on padding
        mask = torch_utils.make_length_mask(lengths).float()
        # (batch x seq_len) => (seq_len x batch)
        mask = mask.t()

        logits = self.projection(enc_outs)
        # logits = logits * mask.unsqueeze(2).expand_as(logits)

        Z = self.partition(logits, mask)
        score = self.score(logits, mask, targets)

        return torch.mean(Z - score)

    def predict(self, enc_outs, lengths):
        # (seq_len x batch x vocab)
        logits = self.projection(enc_outs)
        seq_len, _, vocab = logits.size()
        start_tag, end_tag = vocab, vocab + 1

        # mask on padding (batch x seq_len)
        mask = torch_utils.make_length_mask(lengths).float()

        # variables
        trans = logits.new(vocab + 2, vocab + 2).fill_(-10000.)
        trans[:vocab, :vocab] = self.transition.data
        hyps, scores = [], []
        tag_sequence = logits.new(seq_len + 2, vocab + 2)

        # go over batches
        for logits_b, mask_b in zip(logits.t(), mask.t()):
            seq_len = mask_b.sum()
            # get this batch logits
            tag_sequence.fill_(-10000)
            tag_sequence[0, start_tag] = 0.
            tag_sequence[1:seq_len+1, :vocab] = logits_b[:seq_len]
            tag_sequence[seq_len+1, end_tag] = 0.

            path, score = torch_utils.viterbi_decode(tag_sequence[:seq_len+2], trans)
            hyps.append(self.label_encoder.inverse_transform(path[1:-1]))
            scores.append(score)

        return hyps, scores


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

        hyps = [self.label_encoder.stringify(hyp) for hyp in hyps]
        scores = [s/(len(hyp) + TINY) for s, hyp in zip(scores, hyps)]

        return hyps, scores
