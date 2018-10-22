
import unittest

import torch


BOS = 0
vocab, batch = 20, 4
lengths = torch.Tensor(batch).random_(12).long()
seq_len = max(lengths).item()
feats = torch.randn(seq_len, batch, vocab)
trans_ = torch.randn(vocab, vocab)
targets = torch.Tensor(seq_len, batch).random_(vocab).long()
mask = torch.arange(seq_len).unsqueeze(0).expand(batch, seq_len).t()
mask = (mask < lengths).float()
targets = torch.nn.functional.pad(targets, (0, 0, 1, 0))


def original_score(score):
    for t in range(seq_len):
        emit = torch.stack(
            [feats[t, b, targets[t + 1, b]] for b in range(batch)])
        trans = torch.stack(
            [trans_[targets[t + 1, b], targets[t, b]] for b in range(batch)])
        score = score + emit + (trans * mask[t])

    return score


def opt_score(score):
    for t in range(seq_len):
        emit = feats[t, :, targets[t+1]].diag()
        trans = trans_[targets[t+1], targets[t]]
        score = score + emit + (trans * mask[t])
    return score


def original_predict(feats, lengths):
    hyps, scores = [], []

    # TODO: don't iterate over batches
    # iterate over batches
    for feat, length in zip(feats.chunk(feats.size(1), dim=1), lengths.tolist()):
        # (seq_len x batch x vocab) => (real_len x vocab)
        feat = feat.squeeze(1)[:length]
        bptr = []
        score = feats.new(vocab).fill_(-10000.)
        score[BOS] = 0.

        # iterate over sequence
        for emit in feat[:length]:
            bptr_t, score_t = [], []

            # TODO: don't iterate over tags
            # for each next tag
            for i in range(vocab):
                prob, best = torch.max(score + trans_[i], dim=0)
                bptr_t.append(best.item())
                score_t.append(prob)
            # accumulate
            bptr.append(bptr_t)
            score = torch.stack(score_t) + emit

        score, best = torch.max(score, dim=0)
        score, best = score.item(), best.item()

        # back-tracking
        hyp = [best]
        for bptr_t in reversed(bptr):
            hyp.append(bptr_t[best])
        hyp = list(reversed(hyp[:-1]))

        scores.append(score)
        hyps.append(hyp)

    return hyps, scores


def opt_predict1(feats, lengths):
    hyps, scores = [], []

    _bptr = []
    # TODO: don't iterate over batches
    # iterate over batches
    for feat, length in zip(feats.chunk(feats.size(1), dim=1), lengths.tolist()):
        # (seq_len x batch x vocab) => (real_len x vocab)
        feat = feat.squeeze(1)[:length]
        bptr = []
        score = feats.new(vocab).fill_(-10000.)
        score[BOS] = 0.

        # iterate over sequence
        for emit in feat[:length]:
            # (vocab) => (vocab x vocab)
            z = score.unsqueeze(0).expand_as(trans_) + trans_
            # (vocab)
            score_t, bptr_t = torch.max(z, -1)
            bptr.append(bptr_t.tolist())
            score = score_t + emit

        # (1)
        score, best = torch.max(score, dim=0)
        score, best = score.item(), best.item()

        # back-tracking
        # bptr => (seq_len x vocab)
        hyp = [best]
        for bptr_t in reversed(bptr):
            hyp.append(bptr_t[best])
        hyp = list(reversed(hyp[:-1]))

        scores.append(score)
        hyps.append(hyp)
        _bptr.append(bptr)

    return hyps, scores


def opt_predict(feats, lengths, mask):
    bptr = []
    score = feats.new(batch, vocab).fill_(-10000.)
    score[:, BOS] = 0.
    # (vocab x vocab) => (batch x vocab x vocab)
    trans = trans_.unsqueeze(0).expand(batch, -1, -1)

    # iterate over sequence
    # emit => (batch x vocab)
    for t, emit in enumerate(feats):
        # (batch) => (batch x vocab)
        mask_t = mask[t].unsqueeze(1).expand_as(score)
        # (batch x vocab) => (batch x vocab x vocab) + (batch x vocab x vocab)
        z = score.unsqueeze(1).expand_as(trans) + trans
        # (batch x vocab)
        score_t, bptr_t = torch.max(z, -1)
        # accumulate
        bptr.append(bptr_t)
        # apply mask & update
        score = (score_t + emit) * mask_t + score * (1 - mask_t)

    # (batch)
    score, best = torch.max(score, dim=1)

    # back-tracking
    bptr = torch.stack(bptr).transpose(0, 1)  # (batch x seq_len x vocab)

    hyps = []
    for b, length in enumerate(lengths.tolist()):
        hyp = [best[b].item()]
        for bptr_t in reversed(bptr[b][:length].tolist()):
            hyp.append(bptr_t[best[b].item()])
        hyp = list(reversed(hyp[:-1]))
        hyps.append(hyp)

    return hyps, score.tolist()


class TestCRFOptimization(unittest.TestCase):
    def test_score(self):
        score = torch.zeros(batch)
        orig = original_score(score)
        opt = opt_score(score)
        for idx, b in enumerate(range(len(score))):
            self.assertEqual(orig[b].item(), opt[b].item())

    def test_predict(self):
        orig_hyp, orig_score = original_predict(feats, lengths)
        opt_hyp, opt_score = opt_predict(feats, lengths, mask)
        self.assertEqual(orig_hyp, opt_hyp)
        self.assertEqual(orig_score, opt_score)
