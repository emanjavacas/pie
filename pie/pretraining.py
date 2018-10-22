
import os
import random
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from pie.models import RNNEmbedding, CNNEmbedding
from pie import initialization
from pie import torch_utils
from pie import utils
from pie.data import pack_batch, MultiLabelEncoder


def repackage_hidden(hidden):
    for l in range(len(hidden)):
        hidden[l] = hidden[l][0].detach(), hidden[l][1].detach()

    return hidden


def reset_hidden(hidden):
    for layer in range(len(hidden)):
        if isinstance(hidden[layer], tuple):
            hidden[layer][0].normal_(0, 0.01)
            hidden[layer][1].zero_()
        else:
            hidden[layer].normal_(0, 0.01)


def transpose(lists):
    return list(map(list, zip(*lists)))


def prepare_batch(src, trg, label_encoder, device):
    src, trg = transpose(src), transpose(trg)
    (word, char), _ = label_encoder.transform(src)
    (trg, _), _ = label_encoder.transform(trg)
    w, wlen = torch_utils.pad_batch(word, label_encoder.word.get_pad(), device=device)
    c, clen = torch_utils.pad_batch(char, label_encoder.char.get_pad(), device=device)
    trg, _ = torch_utils.pad_batch(trg, label_encoder.word.get_pad(), device=device)

    return ((w, wlen), (c, clen)), trg


def readlines_reverse(filename):
    from file_read_backwards import FileReadBackwards

    with FileReadBackwards(filename) as f:
        yield from f


def readlines(filename):
    with open(filename) as f:
        for line in f:
            yield line.strip()


def word_iterator(*paths, reverse=False):
    reader = readlines_reverse if reverse else readlines
    for path in paths:
        for line in reader(path):
            line = line.strip().split()
            if not line:
                continue
            if reverse:
                yield from reversed(line)
            else:
                yield from line


def make_iterator(*paths, reverse=None):
    def func():
        yield from word_iterator(*paths, reverse=reverse)
    return func


class Dataset:
    def __init__(self, iterator, batch_size, bptt, buffer_size):
        if buffer_size < batch_size:
            raise ValueError("`buffer_size` must be greater than `batch_size`")

        self.batch_size = batch_size
        self.bptt = bptt
        self.buffer_size = buffer_size
        self.iterator = iterator

    def prepare_buffer(self, words):
        nbatches = len(words) // self.batch_size
        # words = words[:nbatches * self.batch_size]
        words = [words[i:i+nbatches] for i in range(0, len(words), nbatches)]
        words = transpose(words)

        for i in range(0, len(words) - 1, self.bptt):
            seq_len = min(self.bptt, len(words) - 1 - i)
            yield words[i:i+seq_len], words[i+1:i+1+seq_len]

    def get_batches(self):
        words = []
        for w in self.iterator():
            if len(words) >= self.buffer_size:
                nbatches, rwords = divmod(len(words), self.batch_size)
                yield from self.prepare_buffer(words[:-rwords or None])
                words = words[-rwords:]
            words.append(w)

        nbatches, rwords = divmod(len(words), self.batch_size)
        if nbatches > 0:
            yield from self.prepare_buffer(words[:-rwords or None])

        raise StopIteration


class LM(nn.Module):
    def __init__(self, label_encoder, wemb_dim, cemb_dim, hidden_size, num_layers=1,
                 cemb_type='RNN', cell='LSTM', dropout=0.0, word_dropout=0.0,
                 init_rnn='default'):

        self.label_encoder = label_encoder
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.word_dropout = word_dropout
        super().__init__()

        input_size = 0
        # Embedding
        self.wemb = None
        if wemb_dim > 0:
            self.wemb = nn.Embedding(len(label_encoder.word), wemb_dim,
                                     padding_idx=label_encoder.word.get_pad())
            input_size += self.wemb.embedding_dim

        self.cemb = None
        if cemb_dim > 0:
            if cemb_type.lower() == 'rnn':
                self.cemb = RNNEmbedding(len(label_encoder.char), cemb_dim,
                                         padding_idx=label_encoder.char.get_pad(),
                                         cell=cell, init_rnn=init_rnn)
            elif cemb_type.lower() == 'cnn':
                self.cemb = CNNEmbedding(len(label_encoder.char), cemb_dim,
                                         padding_idx=label_encoder.char.get_pad())
            input_size += self.cemb.embedding_dim

        # RNN
        fwd, bwd = [], []
        rnn_inp, rnn_hid = input_size, hidden_size
        for layer in range(num_layers):

            if wemb_dim > 0 and num_layers > 1 and layer == (num_layers - 1):
                rnn_hid = wemb_dim

            l = nn.LSTM(rnn_inp, rnn_hid)
            self.add_module('fwd_{}'.format(layer), l)
            fwd.append(l)

            l = nn.LSTM(rnn_inp, rnn_hid)
            self.add_module('bwd_{}'.format(layer), l)
            bwd.append(l)

            rnn_inp = hidden_size

        self.fwd = fwd
        self.bwd = bwd

        # Output
        self.output = nn.Linear(rnn_hid, len(label_encoder.word))
        if rnn_hid == wemb_dim:
            print("Tying weights")
            self.output.weight = self.wemb.weight

        self.init()

    def init(self):
        if self.wemb is not None:
            initialization.init_embeddings(self.wemb)
        else:
            initialization.init_linear(self.output)

        for l in range(self.num_layers):
            initialization.init_rnn(self.fwd[l])
            initialization.init_rnn(self.bwd[l])

    def device(self):
        return next(self.parameters()).device

    def embeddings(self, w, wlen, c, clen):
        emb = []
        if self.wemb is not None:
            # word dropout
            w = torch_utils.word_dropout(
                w, self.word_dropout, self.training, self.label_encoder.word)
            emb.append(self.wemb(w))

        if self.cemb is not None:
            (cemb, _) = self.cemb(c, clen, wlen)
            emb.append(cemb)

        # merge
        emb = torch.cat(emb, dim=-1)

        return emb

    def forward(self, w, wlen, c, clen, hidden=None, run='fwd'):
        emb = self.embeddings(w, wlen, c, clen)

        # input dropout (dropouti)
        emb = torch_utils.sequential_dropout(emb, self.dropout, self.training)
        _, sort = torch.sort(wlen, descending=True)
        _, unsort = sort.sort()
        emb = nn.utils.rnn.pack_padded_sequence(emb[:, sort], wlen[sort])

        if hidden is None:
            hidden = [None] * self.num_layers
        else:
            for l in range(self.num_layers):
                hidden[l] = hidden[l][0][:, sort], hidden[l][1][:, sort]
        new_hidden = []

        inp = emb
        for layer in range(self.num_layers):
            outs, lhidden = getattr(self, "{}_{}".format(run, layer))(inp, hidden[layer])
            # hidden dropout (dropouth)
            if layer < (self.num_layers - 1):
                outs, new_wlen = nn.utils.rnn.pad_packed_sequence(outs)
                torch_utils.sequential_dropout(outs, self.dropout, self.training)
                outs = nn.utils.rnn.pack_padded_sequence(outs, new_wlen)
            inp = outs
            new_hidden.append(lhidden)

        outs, _ = nn.utils.rnn.pad_packed_sequence(outs)
        outs = outs[:, unsort]
        for l in range(self.num_layers):
            new_hidden[l] = new_hidden[l][0][:, unsort], new_hidden[l][1][:, unsort]
        # output dropout (dropouto)
        outs = F.dropout(outs, p=self.dropout, training=self.training)

        return outs, new_hidden

    def loss(self, outs, targets):
        seqlen, batch, _ = outs.size()
        return F.cross_entropy(self.output(outs).view(seqlen * batch, -1),
                               targets.view(-1),
                               ignore_index=self.label_encoder.word.get_pad())

    def evaluate(self, fwd_dev, bwd_dev):
        nbatches = 0
        fwd_hidden, bwd_hidden = None, None
        tfwd_loss, tbwd_loss = 0, 0
        device = self.device()

        for fwd_batch, bwd_batch in zip(fwd_dev.get_batches(), bwd_dev.get_batches()):
            nbatches += 1

            # => run forward
            src, trg = fwd_batch
            ((w, wlen), (c, clen)), trg = prepare_batch(
                src, trg, self.label_encoder, device)
            # get loss
            outs, fwd_hidden = self(w, wlen, c, clen, fwd_hidden, run='fwd')
            tfwd_loss += self.loss(outs, trg).item()

            # => run backward
            src, trg = bwd_batch
            ((w, wlen), (c, clen)), trg = prepare_batch(
                src, trg, self.label_encoder, device)
            # get loss
            outs, bwd_hidden = self(w, wlen, c, clen, bwd_hidden, run='bwd')
            tbwd_loss += self.loss(outs, trg).item()

        return tfwd_loss / nbatches, tbwd_loss / nbatches


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--devpath', required=True)
    parser.add_argument('--vocabpath')
    # model
    parser.add_argument('--max_size', type=int, default=50000)
    parser.add_argument('--wemb_dim', type=int, default=64)
    parser.add_argument('--cemb_dim', type=int, default=100)
    parser.add_argument('--cemb_type', default='rnn')
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--cell', default='LSTM')
    # training
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--clipping', type=float, default=2.5)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--word_dropout', type=float, default=0.0)
    parser.add_argument('--drop_hidden', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--bptt', type=int, default=50)
    parser.add_argument('--optim', default='SGD')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_weight', type=float, default=0.75)
    parser.add_argument('--buffer_size', type=int, default=1e+7)
    parser.add_argument('--weight_decay', type=float, default=2e-6)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--repfreq', type=int, default=100)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    # label encoder
    label_encoder = MultiLabelEncoder(word_max_size=args.max_size)
    label_encoder.fit(line.split() for line in readlines(args.path))
    print(label_encoder)

    # datasets
    fwd_train = Dataset(make_iterator(args.path), args.batch_size,
                        args.bptt, args.buffer_size)
    bwd_train = Dataset(make_iterator(args.path, reverse=True), args.batch_size,
                        args.bptt, args.buffer_size)
    fwd_dev = Dataset(make_iterator(args.devpath), args.batch_size,
                      args.bptt, args.buffer_size)
    bwd_dev = Dataset(make_iterator(args.devpath, reverse=True), args.batch_size,
                      args.bptt, args.buffer_size)

    # model
    model = LM(label_encoder, args.wemb_dim, args.cemb_dim,
               args.hidden_size, num_layers=args.num_layers, cell=args.cell,
               cemb_type=args.cemb_type, dropout=args.dropout,
               word_dropout=args.word_dropout)
    print(model)
    print(" * Number of parameters", sum(p.nelement() for p in model.parameters()))

    model.to(args.device)
    print("Starting training")
    print()

    bwd_hidden, fwd_hidden = None, None

    # optim
    if args.optim.lower() == 'sgd':
        optim = torch.optim.SGD(list(model.parameters()), lr=args.lr, # momentum=0.9,
                                weight_decay=args.weight_decay)
    else:
        optim = getattr(torch.optim, args.optim)(list(model.parameters()), lr=args.lr,
                                                 weight_decay=args.weight_decay)

    # report
    best_loss, best_params, fails = float('inf'), None, 0

    for epoch in range(args.epochs):
        print("Starting epoch:", epoch + 1)
        tbatches, tfwd_loss, tbwd_loss, titems, ttime = 0, 0, 0, 0, time.time()
        rbatches, rfwd_loss, rbwd_loss, ritems, rtime = 0, 0, 0, 0, time.time()

        for fwd_batch, bwd_batch in zip(fwd_train.get_batches(), bwd_train.get_batches()):
                # => run forward
                src, trg = fwd_batch
                ((w, wlen), (c, clen)), trg = prepare_batch(
                    src, trg, label_encoder, args.device)
                # get loss
                outs, fwd_hidden = model(w, wlen, c, clen, fwd_hidden, run='fwd')
                fwd_loss = model.loss(outs, trg)
            
                # => run backward
                src, trg = bwd_batch
                ((w, wlen), (c, clen)), trg = prepare_batch(
                    src, trg, label_encoder, args.device)
                # get loss
                outs, bwd_hidden = model(w, wlen, c, clen, bwd_hidden, run='bwd')
                bwd_loss = model.loss(outs, trg)
            
                # optimize
                optim.zero_grad()
                nn.utils.clip_grad_norm_(model.parameters(), args.clipping)
                ((fwd_loss + bwd_loss) / 2).backward()
                optim.step()
            
                # repackage hidden
                fwd_hidden = repackage_hidden(fwd_hidden)
                bwd_hidden = repackage_hidden(bwd_hidden)
            
                # reset hidden
                if random.random() < args.drop_hidden:
                    reset_hidden(fwd_hidden)
                    reset_hidden(bwd_hidden)
            
                # accumulate
                tfwd_loss += fwd_loss.item()
                tbwd_loss += bwd_loss.item()
                titems += sum(wlen)
                tbatches += 1
                rfwd_loss += fwd_loss.item()
                rbwd_loss += bwd_loss.item()
                ritems += sum(wlen)
                rbatches += 1
            
                if rbatches == args.repfreq:
                    print('{}/{}: fwd loss=>{:.3f} bwd loss=>{:.3f} '.format(
                        epoch + 1, tbatches,
                        math.exp(rfwd_loss/rbatches),
                        math.exp(rbwd_loss/rbatches)) +
                          'speed=>{:.3f} w/s'.format(ritems / (time.time()-rtime)))
                    rbatches, rfwd_loss, rbwd_loss, ritems = 0, 0, 0, 0
                    rtime = time.time()

        # epoch
        print("Epoch: {}; loss: fwd=>{:.3f}, bwd=>{:.3f}; speed=>{:.3f} w/sec".format(
            epoch + 1,
            math.exp(tfwd_loss / tbatches),
            math.exp(tbwd_loss / tbatches),
            titems / (time.time()-ttime)))

        # do validation
        model.eval()
        with torch.no_grad():
            dev_fwd_loss, dev_bwd_loss = model.evaluate(fwd_dev, bwd_dev)
        model.train()

        print("Evaluation: fwd=>{:.3f}, bwd=>{:.3f}".format(
            math.exp(dev_fwd_loss), math.exp(dev_bwd_loss)))

        if (dev_fwd_loss + dev_bwd_loss) / 2 < best_loss:
            best_loss = (dev_fwd_loss + dev_bwd_loss) / 2
            best_params = model.to('cpu').state_dict()
            model.to(args.device)
            fails = 0
        else:
            fails += 1
            # reduce LR
            for pgroup in optim.param_groups:
                pgroup['lr'] = pgroup['lr'] * args.lr_weight
            print(optim)

        if fails == args.patience:
            print("Stop training with best loss=>{:.3f}".format(math.exp(best_loss)))
            break

    if args.save:
        filename = '-'.join(['pretraining', ])
