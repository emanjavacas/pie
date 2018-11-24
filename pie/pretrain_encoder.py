
import os
import tarfile
import datetime
import json
import time
import math
import random
import logging

import torch
import torch.nn.functional as F
import torch.nn as nn

import pie
from pie import utils, torch_utils, initialization
from pie.models import RNNEmbedding, CNNEmbedding, EmbeddingConcat, EmbeddingMixer
from pie.models import RNNEncoder, LinearDecoder


class EarlyStop(Exception):
    def __init__(self):
        pass


def load_sentences(path, bptt=35, buffer_size=100000):
    buf = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            buf.extend(line.split())

            if len(buf) >= buffer_size:
                sents = [buf[i:i+bptt] for i in range(0, len(buf), bptt)]
                random.shuffle(sents)
                yield from sents
                buf = buf[len(sents)*bptt:]


def get_batches(lines, batch_size, label_encoder, device):
    batch = []
    for line in lines:
        if len(batch) == batch_size:
            (word, char), _ = label_encoder.transform(batch)
            w, wlen = torch_utils.pad_batch(
                word, label_encoder.word.get_pad(), device=device)
            c, clen = torch_utils.pad_batch(
                char, label_encoder.char.get_pad(), device=device)
            yield (w, wlen), (c, clen)
            batch = []
        batch.append(line)


class Encoder(nn.Module):
    def __init__(self, label_encoder, wemb_dim, cemb_dim, hidden_size, num_layers,
                 dropout=0.0, word_dropout=0.0, merge_type='concat', cemb_type='RNN',
                 cemb_layers=1, cell='LSTM', custom_cemb_cell=False,
                 init_rnn='xavier_uniform'):

        self.label_encoder = label_encoder
        self.wemb_dim = wemb_dim
        self.cemb_dim = cemb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # kwargs
        self.cell = cell
        self.dropout = dropout
        self.word_dropout = word_dropout
        self.merge_type = merge_type
        self.cemb_type = cemb_type
        self.cemb_layers = cemb_layers
        self.custom_cemb_cell = custom_cemb_cell
        super().__init__()

        self.wemb = None
        if self.wemb_dim > 0:
            self.wemb = nn.Embedding(len(label_encoder.word), wemb_dim,
                                     padding_idx=label_encoder.word.get_pad())
            # init embeddings
            initialization.init_embeddings(self.wemb)

        self.cemb = None
        if cemb_type.upper() == 'RNN':
            self.cemb = RNNEmbedding(
                len(label_encoder.char), cemb_dim,
                padding_idx=label_encoder.char.get_pad(),
                custom_lstm=custom_cemb_cell, dropout=dropout,
                num_layers=cemb_layers, cell=cell, init_rnn=init_rnn)
        elif cemb_type.upper() == 'CNN':
            self.cemb = CNNEmbedding(
                len(label_encoder.char), cemb_dim,
                padding_idx=label_encoder.char.get_pad())

        self.merger = None
        if self.cemb is not None and self.wemb is not None:
            if merge_type.lower() == 'mixer':
                if self.cemb.embedding_dim != self.wemb.embedding_dim:
                    raise ValueError("EmbeddingMixer needs equal embedding dims")
                self.merger = EmbeddingMixer(wemb_dim)
                in_dim = wemb_dim
            elif merge_type == 'concat':
                self.merger = EmbeddingConcat()
                in_dim = wemb_dim + self.cemb.embedding_dim
            else:
                raise ValueError("Unknown merge method: {}".format(merge_type))
        elif self.cemb is None:
            in_dim = wemb_dim
        else:
            in_dim = self.cemb.embedding_dim

        # Encoder
        self.encoder = RNNEncoder(
            in_dim, hidden_size, num_layers=num_layers, cell=cell, dropout=dropout,
            init_rnn=init_rnn)

        # decoders
        self.lm_fwd_decoder = LinearDecoder(label_encoder.word, hidden_size)
        self.lm_bwd_decoder = LinearDecoder(label_encoder.word, hidden_size)

    def get_args_and_kwargs(self):
        return {'args': (self.wemb_dim, self.cemb_dim, self.hidden_size, self.num_layers),
                'kwargs': {'merge_type': self.merge_type, 'cemb_type': self.cemb_type,
                           'cemb_layers': self.cemb_layers, 'cell': self.cell,
                           'custom_cemb_cell': self.custom_cemb_cell}}

    def save(self, path):
        path = utils.ensure_ext(path, 'tar')
        # create dir if needed
        dirname = os.path.dirname(path)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

        with tarfile.open(path, 'w') as tar:
            # serialize label_encoder
            string, path = json.dumps(self.label_encoder.jsonify()), 'label_encoder.zip'
            utils.add_gzip_to_tar(string, path, tar)

            # serialize parameters
            string, path = json.dumps(self.get_args_and_kwargs()), 'parameters.zip'
            utils.add_gzip_to_tar(string, path, tar)

            # serialize weights
            with utils.tmpfile() as tmppath:
                torch.save(self.state_dict(), tmppath)
                tar.add(tmppath, arcname='state_dict.pt')

            # serialize current pie commit
            string, path = pie.__commit__, 'pie-commit.zip'
            utils.add_gzip_to_tar(string, path, tar)

    @classmethod
    def load(cls, path):
        with tarfile.open(utils.ensure_ext(path, 'tar'), 'r') as tar:
            commit = utils.get_gzip_from_tar(tar, 'pie-commit.zip')
            if pie.__commit__ != commit:
                logging.warn(
                    ("Model {} was serialized with a previous "
                     "version of `pie`. This might result in issues. "
                     "Model commit is {}, whereas current `pie` commit is {}."
                    ).format(path, commit, pie.__commit__))

            # load label encoder
            le = pie.dataset.MultiLabelEncoder.load_from_string(
                utils.get_gzip_from_tar(tar, 'label_encoder.zip'))

            # load model parameters
            params = json.loads(utils.get_gzip_from_tar(tar, 'parameters.zip'))

            # instantiate model
            model = Encoder(le, *params['args'], **params['kwargs'])

            # load state_dict
            with utils.tmpfile() as tmppath:
                tar.extract('state_dict.pt', path=tmppath)
                dictpath = os.path.join(tmppath, 'state_dict.pt')
                model.load_state_dict(torch.load(dictpath, map_location='cpu'))

        model.eval()

        return model

    def embedding(self, word, wlen, char, clen):
        wemb = cemb = None
        if self.wemb is not None:
            # set words to unknown with prob `p` depending on word frequency
            word = torch_utils.word_dropout(
                word, self.word_dropout, self.training, self.label_encoder.word)
            wemb = self.wemb(word)
        if self.cemb is not None:
            # cemb_outs: (seq_len x batch x emb_dim)
            cemb, _ = self.cemb(char, clen, wlen)

        if wemb is None:
            emb = cemb
        elif cemb is None:
            emb = wemb
        else:
            emb = self.merger(wemb, cemb)

        return emb

    def loss(self, word, wlen, char, clen):
        # Embedding
        emb = self.embedding(word, wlen, char, clen)

        # Encoder
        emb = F.dropout(emb, p=self.dropout, training=self.training)
        enc_outs = self.encoder(emb, wlen)

        fwd, bwd = F.dropout(enc_outs[-1], p=0, training=self.training).chunk(2, dim=2)
        # forward logits
        logits = self.lm_fwd_decoder(torch_utils.pad(fwd[:-1], pos='pre'))
        fwd_lm = self.lm_fwd_decoder.loss(logits, word)
        # backward logits
        logits = self.lm_bwd_decoder(torch_utils.pad(bwd[1:], pos='post'))
        bwd_lm = self.lm_bwd_decoder.loss(logits, word)

        return fwd_lm, bwd_lm

    def evaluate(self, dev):
        nbatches = 0
        tfwd_loss, tbwd_loss = 0, 0
        for (w, wlen), (c, clen) in dev:
            nbatches += 1
            fwd_loss, bwd_loss = self.loss(w, wlen, c, clen)
            tfwd_loss += fwd_loss.item()
            tbwd_loss += bwd_loss.item()

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
    parser.add_argument('--load_pretrained_embeddings')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--clipping', type=float, default=2.5)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--word_dropout', type=float, default=0.0)
    parser.add_argument('--drop_hidden', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--bptt', type=int, default=35)
    parser.add_argument('--optim', default='SGD')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_weight', type=float, default=0.75)
    parser.add_argument('--buffer_size', type=int, default=1e+7)
    parser.add_argument('--weight_decay', type=float, default=2e-6)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--repfreq', type=int, default=100)
    parser.add_argument('--checkfreq', type=int, default=5000)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    print("Fitting encoder")
    label_encoder = pie.dataset.MultiLabelEncoder(word_max_size=args.max_size)
    vocabpath = args.vocabpath or args.path
    label_encoder.fit(load_sentences(vocabpath))
    print(label_encoder)

    # model
    print("Building model")
    model = Encoder(label_encoder, args.wemb_dim, args.cemb_dim,
                    args.hidden_size, num_layers=args.num_layers, cell=args.cell,
                    cemb_type=args.cemb_type, dropout=args.dropout,
                    word_dropout=args.word_dropout)
    print(model)
    print(" * Number of parameters", sum(p.nelement() for p in model.parameters()))

    if args.load_pretrained_embeddings:
        print("Loading pretrained embeddings from", args.load_pretrained_embeddings)
        pie.initialization.init_pretrained_embeddings(
            args.load_pretrained_embeddings, label_encoder.word, model.wemb)

    model.to(args.device)
    print("Starting training")
    print()

        # optim
    if args.optim.lower() == 'sgd':
        optim = torch.optim.SGD(list(model.parameters()), lr=args.lr, # momentum=0.9,
                                weight_decay=args.weight_decay)
    else:
        optim = getattr(torch.optim, args.optim)(list(model.parameters()), lr=args.lr,
                                                 weight_decay=args.weight_decay)

    # report
    best_loss, best_params, fails = float('inf'), None, 0

    try:
        for epoch in range(args.epochs):
            print("Starting epoch:", epoch + 1)
            tbatches, tfwd_loss, tbwd_loss, titems, ttime = 0, 0, 0, 0, time.time()
            rbatches, rfwd_loss, rbwd_loss, ritems, rtime = 0, 0, 0, 0, time.time()

            for (w, wlen), (c, clen) in get_batches(
                    load_sentences(args.path), args.batch_size,
                    label_encoder, args.device):
    
                fwd_loss, bwd_loss = model.loss(w, wlen, c, clen)
    
                # optimize
                optim.zero_grad()
                nn.utils.clip_grad_norm_(model.parameters(), args.clipping)
                ((fwd_loss + bwd_loss) / 2).backward()
                optim.step()
    
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
    
                if tbatches % args.checkfreq == 0:
                    # do validation
                    model.eval()
                    with torch.no_grad():
                        dev_fwd_loss, dev_bwd_loss = model.evaluate(
                            get_batches(load_sentences(args.devpath),
                                        args.batch_size, label_encoder, args.device))
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
                        raise EarlyStop

            # epoch
            print("Epoch: {}; loss: fwd=>{:.3f}, bwd=>{:.3f}; speed=>{:.3f} w/sec".format(
                epoch + 1,
                math.exp(tfwd_loss / tbatches),
                math.exp(tbwd_loss / tbatches),
                titems / (time.time()-ttime)))

    except EarlyStop:
        print("Stop training with best loss=>{:.3f}".format(math.exp(best_loss)))

    if args.save:
        model.to('cpu').load_state_dict(best_params)
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        model.save(os.path.join('pretrained-encoder', timestamp))
