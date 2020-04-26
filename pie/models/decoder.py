
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from pie import initialization
from pie import torch_utils
from pie.constants import TINY

from .beam_search import Beam
from .attention import Attention
from .highway import Highway


class LinearDecoder(nn.Module):
    """
    Simple Linear Decoder that outputs a probability distribution
    over the vocabulary

    Parameters
    ===========
    label_encoder : LabelEncoder
    in_features : int, input dimension
    """
    def __init__(self, label_encoder, in_features, dropout=0.0,
                 highway_layers=0, highway_act='relu'):
        self.label_encoder = label_encoder
        super().__init__()

        # nll weight
        nll_weight = torch.ones(len(label_encoder))
        nll_weight[label_encoder.get_pad()] = 0.
        self.register_buffer('nll_weight', nll_weight)
        # highway
        self.highway = None
        if highway_layers > 0:
            self.highway = Highway(in_features, highway_layers, highway_act)
        # decoder output
        self.decoder = nn.Linear(in_features, len(label_encoder))
        self.init()

    def init(self):
        # linear
        initialization.init_linear(self.decoder)

    def forward(self, enc_outs):
        if self.highway is not None:
            enc_outs = self.highway(enc_outs)
        linear_out = self.decoder(enc_outs)

        return linear_out

    def loss(self, logits, targets):
        loss = F.cross_entropy(
            logits.view(-1, len(self.label_encoder)), targets.view(-1),
            weight=self.nll_weight, reduction="mean",
            ignore_index=self.label_encoder.get_pad())

        return loss

    def predict(self, enc_outs, lengths):
        """
        Parameters
        ==========
        enc_outs : torch.tensor(seq_len x batch x hidden_size)
        """
        if self.highway is not None:
            enc_outs = self.highway(enc_outs)
        probs = F.softmax(self.decoder(enc_outs), dim=-1)
        probs, preds = torch.max(probs.transpose(0, 1), dim=-1)
        output_probs, output_preds = [], []
        for idx, length in enumerate(lengths.tolist()):
            output_preds.append(
                self.label_encoder.inverse_transform(preds[idx])[:length])
            output_probs.append(probs[idx].tolist())

        return output_preds, output_probs


class CRFDecoder(nn.Module):
    """
    CRF decoder layer
    """
    def __init__(self, label_encoder, hidden_size, highway_layers=0, highway_act='relu'):
        self.label_encoder = label_encoder
        super().__init__()

        vocab = len(label_encoder)
        self.highway = None
        if highway_layers > 0:
            self.highway = Highway(hidden_size, highway_layers, highway_act)
        self.projection = nn.Linear(hidden_size, vocab)
        self.transition = nn.Parameter(torch.Tensor(vocab, vocab))
        self.start_transition = nn.Parameter(torch.Tensor(vocab))
        self.end_transition = nn.Parameter(torch.Tensor(vocab))

        self.init()

    def init(self):
        # transitions
        nn.init.xavier_normal_(self.transition)
        nn.init.normal_(self.start_transition)
        nn.init.normal_(self.end_transition)

    def forward(self, enc_outs):
        """get logits of the input features"""
        # (seq_len x batch x vocab)
        if self.highway is not None:
            enc_outs = self.highway(enc_outs)
        logits = self.projection(enc_outs)

        return F.log_softmax(logits, -1)

    def partition(self, logits, mask):
        seq_len, batch, vocab = logits.size()

        # (batch x vocab)
        Z = self.start_transition.view(1, vocab) + logits[0]

        for t in range(1, seq_len):
            emit_score = logits[t].view(batch, 1, vocab)
            trans_score = self.transition.view(1, vocab, vocab)
            # mask and update
            Z_t = Z.view(batch, vocab, 1)
            # (batch x vocab x vocab) => (batch x vocab)
            Z_t = torch_utils.log_sum_exp(Z_t + emit_score + trans_score, 1)
            # mask & update
            mask_t = mask[t].view(batch, 1)
            Z = Z_t * mask_t + Z * (1 - mask_t)

        Z = Z + self.end_transition.view(1, vocab)
        # (batch x vocab) => (batch)
        Z = torch_utils.log_sum_exp(Z)

        return Z

    def score(self, logits, mask, targets):
        # calculate the score of a given sequence
        seq_len, batch, vocab = logits.size()
        # transition from start tag to first tag
        score = self.start_transition.index_select(0, targets[0])
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
            emit_score = logits[t].gather(1, curr_tag.view(batch, 1)).squeeze(1)
            score = score + (trans_score * mask[t+1]) + (emit_score * mask[t])
            
        # last step
        last_target_index = mask.sum(0).long() - 1
        # (batch)
        last_target = targets.gather(0, last_target_index.expand(seq_len, batch))[0]
        # (batch)
        last_trans_score = self.end_transition.index_select(0, last_target)
        # (batch x 1) => (batch)
        last_emit_score = logits[-1].gather(1, last_target.unsqueeze(1)).squeeze(1)
        # (batch)
        score = score + last_trans_score + last_emit_score * mask[-1]

        return score

    def loss(self, logits, targets, lengths):
        # mask on padding
        mask = torch_utils.make_length_mask(lengths).float()
        # (batch x seq_len) => (seq_len x batch)
        mask = mask.t()
        # logits = logits * mask.unsqueeze(2).expand_as(logits)
        Z = self.partition(logits, mask)
        score = self.score(logits, mask, targets)

        # FIXME: this gives the average loss per sentence (perhaps it should)
        # be weighted down to make it also per word?
        return torch.mean(Z - score)

    def predict(self, enc_outs, lengths):
        # (seq_len x batch x vocab)
        logits = self(enc_outs)
        seq_len, _, vocab = logits.size()
        start_tag, end_tag = vocab, vocab + 1

        # mask on padding (batch x seq_len)
        mask = torch_utils.make_length_mask(lengths).float()
        # Mask is not used !

        # variables
        trans = logits.new(vocab + 2, vocab + 2).fill_(-10000.)
        trans[:vocab, :vocab] = self.transition.data
        hyps, scores = [], []
        tag_sequence = logits.new(seq_len + 2, vocab + 2)

        # iterate over batches
        for logits_b, len_b in zip(logits.transpose(0, 1), lengths):
            seq_len = len_b.item()
            # get this batch logits
            tag_sequence.fill_(-10000)
            tag_sequence[0, start_tag] = 0.
            tag_sequence[1:seq_len+1, :vocab] = logits_b[:seq_len]
            tag_sequence[seq_len+1, end_tag] = 0.

            path, score = torch_utils.viterbi_decode(tag_sequence[:seq_len+2], trans)
            hyps.append(self.label_encoder.inverse_transform(path[1:-1]))
            scores.append(score)

        return hyps, scores


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
    def __init__(self, label_encoder, in_dim, hidden_size, dropout=0.0,
                 # rnn
                 num_layers=1, cell='LSTM', init_rnn='default',
                 # attention
                 scorer='general',
                 # sentence context
                 context_dim=0):

        self.label_encoder = label_encoder
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.init_rnn = init_rnn
        super().__init__()

        if label_encoder.get_eos() is None or label_encoder.get_bos() is None:
            raise ValueError("AttentionalDecoder needs <eos> and <bos>")

        # nll weight
        nll_weight = torch.ones(len(label_encoder))
        nll_weight[label_encoder.get_pad()] = 0.
        self.register_buffer('nll_weight', nll_weight)
        # emb
        self.embs = nn.Embedding(len(label_encoder), in_dim)
        # rnn
        self.rnn = getattr(nn, cell)(in_dim + context_dim, hidden_size,
                                     num_layers=num_layers,
                                     dropout=dropout if num_layers > 1 else 0)
        self.attn = Attention(hidden_size)
        self.proj = nn.Linear(hidden_size, len(label_encoder))

        self.init()

    def init(self):
        # embeddings
        initialization.init_embeddings(self.embs)
        # rnn
        initialization.init_rnn(self.rnn, scheme=self.init_rnn)
        # linear
        initialization.init_linear(self.proj)

    def forward(self, targets, lengths, enc_outs, src_lengths, context=None):
        """
        Decoding routine for training. Returns the logits corresponding to
        the targets for the `loss` method. Takes care of padding.
        """
        targets, lengths = targets[:-1], lengths - 1
        embs = self.embs(targets)

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
        targets = targets[1:]  # remove <bos> from targets

        loss = F.cross_entropy(
            logits.view(-1, len(self.label_encoder)), targets.view(-1),
            weight=self.nll_weight, reduction="mean",
            ignore_index=self.label_encoder.get_pad())

        # FIXME: normalize loss to be word-level

        return loss

    def predict_max(self, enc_outs, lengths,
                    max_seq_len=20, bos=None, eos=None,
                    context=None):
        """
        Decoding routine for inference with step-wise argmax procedure

        Parameters
        ===========
        enc_outs : tensor(src_seq_len x batch x hidden_size)
        context : tensor(batch x hidden_size), optional
        """
        eos = eos or self.label_encoder.get_eos()
        bos = bos or self.label_encoder.get_bos()
        hidden, batch, device = None, enc_outs.size(1), enc_outs.device
        mask = torch.ones(batch, dtype=torch.int64, device=device)
        inp = torch.zeros(batch, dtype=torch.int64, device=device) + bos
        hyps, scores = [], 0

        for _ in range(max_seq_len):
            if mask.sum().item() == 0:
                break

            # prepare input
            emb = self.embs(inp)
            if context is not None:
                emb = torch.cat([emb, context], dim=1)
            # run rnn
            emb = emb.unsqueeze(0)
            outs, hidden = self.rnn(emb, hidden)
            outs, _ = self.attn(outs, enc_outs, lengths)
            outs = self.proj(outs).squeeze(0)
            # get logits
            probs = F.log_softmax(outs, dim=1)
            # sample and accumulate
            score, inp = probs.max(1)
            hyps.append(inp.tolist())
            mask = mask * (inp != eos).long()
            score = score.cpu()
            score[mask == 0] = 0
            scores += score

        hyps = [self.label_encoder.stringify(hyp) for hyp in zip(*hyps)]
        scores = [s/(len(hyp) + TINY) for s, hyp in zip(scores.tolist(), hyps)]

        return hyps, scores

    def predict_beam(self, enc_outs, lengths,
                     max_seq_len=50, width=12, eos=None, bos=None,
                     context=None):
        """
        Decoding routine for inference with beam search

        Parameters
        ===========
        enc_outs : tensor(src_seq_len x batch x hidden_size)
        context : tensor(batch x hidden_size), optional
        """
        eos = eos or self.label_encoder.get_eos()
        bos = bos or self.label_encoder.get_bos()
        hidden, device = None, enc_outs.device
        seq_len, batch, _ = enc_outs.size()
        beams = [Beam(width, eos=eos, bos=bos, device=device) for _ in range(batch)]

        # expand data along beam width
        # (seq_len x beam * batch x hidden_size)
        enc_outs = enc_outs.repeat(1, width, 1)
        lengths = lengths.repeat(width)
        if context is not None:
            # (beam * batch x context_dim)
            context = context.repeat(width, 1)

        for _ in range(max_seq_len):
            if all(not beam.active for beam in beams):
                break
            # (beam x batch)
            inp = torch.stack([beam.get_current_state() for beam in beams], dim=1)
            # (beam * batch)
            inp = inp.view(-1)
            # (beam * batch x emb_dim)
            emb = self.embs(inp)
            if context is not None:
                # (beam * batch x emb_dim + context_dim)
                emb = torch.cat([emb, context], dim=1)
            # run rnn
            emb = emb.unsqueeze(0)  # add singleton seq_len dim
            outs, hidden = self.rnn(emb, hidden)

            # (1 x beam * batch x hidden)
            outs, _ = self.attn(outs, enc_outs, lengths)
            # (beam * batch x vocab)
            outs = self.proj(outs).squeeze(0)
            # (beam * batch x vocab)
            probs = F.log_softmax(outs, dim=1)
            # (beam x batch x vocab)
            probs = probs.view(width, batch, -1)

            # expose beam dim for swaping
            if isinstance(hidden, tuple):
                hidden = hidden[0].view(self.num_layers, width, batch, -1), \
                         hidden[1].view(self.num_layers, width, batch, -1)
            else:
                hidden = hidden.view(self.num_layers, width, batch, -1)

            # advance and swap
            for i, beam in enumerate(beams):
                if not beam.active:
                    continue
                # advance
                beam.advance(probs[:, i])
                # rearrange
                sbeam = beam.get_source_beam()
                if isinstance(hidden, tuple):
                    hidden[0][:, :, i].copy_(hidden[0][:, :, i].index_select(1, sbeam))
                    hidden[1][:, :, i].copy_(hidden[1][:, :, i].index_select(1, sbeam))
                else:
                    hidden[:, :, i].copy_(hidden[:, :, i].index_select(1, sbeam))

            # collapse beam and batch
            if isinstance(hidden, tuple):
                hidden = hidden[0].view(self.num_layers, width * batch, -1), \
                         hidden[1].view(self.num_layers, width * batch, -1)
            else:
                hidden = hidden.view(self.num_layers, width * batch, -1)

        scores, hyps = [], []
        for beam in beams:
            bscores, bhyps = beam.decode(n=1)
            bscores, bhyps = bscores[0], bhyps[0]
            scores.append(bscores)
            hyps.append(bhyps)

        hyps = [self.label_encoder.stringify(hyp) for hyp in hyps]
        scores = [s/(len(hyp) + TINY) for s, hyp in zip(scores, hyps)]

        return hyps, scores
