
import math
from string import ascii_letters
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pie.encoder import RNNEncoder
from pie.decoder import AttentionalDecoder
from pie.data import LabelEncoder
from pie import torch_utils


def random_string():
    return ''.join(ascii_letters[random.randint(0, len(ascii_letters)-1)]
                   for _ in range(random.randint(5, 25)))

def random_dataset(nitems=10000):
    label_encoder = LabelEncoder(level='char', eos=True, bos=True)

    dataset = []
    for _ in range(nitems):
        item = random_string()
        dataset.append((item, item[::-1]))

    src, trg = zip(*dataset)
    for item in src:
        label_encoder.add(item)
    label_encoder.compute_vocab()
    src = [label_encoder.transform(s) for s in src]
    trg = [label_encoder.transform(s) for s in trg]

    return list(zip(src, trg)), label_encoder


class EncoderDecoder(nn.Module):
    def __init__(self, label_encoder, emb_dim=24, hidden_size=64, dropout=0.25):
        self.label_encoder = label_encoder
        super().__init__()

        self.embs = nn.Embedding(len(label_encoder), emb_dim,
                                 padding_idx=label_encoder.get_pad())
        nn.init.uniform_(self.embs.weight, -0.05, 0.05)
        self.encoder = RNNEncoder(
            emb_dim, hidden_size, bidirectional=True, dropout=dropout)
        self.decoder = AttentionalDecoder(
            label_encoder, emb_dim, hidden_size, dropout=dropout)

    def forward(self, src, src_len, trg, trg_len):
        enc_outs = self.encoder(self.embs(src), src_len)
        logits = self.decoder(trg, trg_len, enc_outs, src_len)
        return self.decoder.loss(logits, trg)

    def predict(self, src, src_len):
        enc_outs = self.encoder(self.embs(src), src_len)
        hyps, scores = self.decoder.predict_beam(enc_outs, src_len, max_seq_len=50)
        return hyps, scores

    def train_epoch(self, dataset, batch_size=20):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        dataset, dev = dataset[:-5], dataset[-5:]
        rep_loss, rep_batches = 0, 0

        for i in range(0, len(dataset), batch_size):
            src, trg = zip(*dataset[i:i+batch_size])
            src, src_len = torch_utils.pad_batch(src, self.label_encoder.get_pad())
            trg, trg_len = torch_utils.pad_batch(trg, self.label_encoder.get_pad())

            optimizer.zero_grad()
            loss = self(src, src_len, trg, trg_len)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            optimizer.step()

            rep_loss += loss.item()
            rep_batches += 1

            if (i // batch_size) % 100 == 0:
                self.eval()
                with torch.no_grad():
                    print(math.exp(rep_loss / rep_batches))
                    src, trg = zip(*dev)
                    src, src_len = torch_utils.pad_batch(
                        src, self.label_encoder.get_pad())
                    hyps, scores = self.predict(src, src_len)
                    print()
                    for hyp, score, true in zip(hyps, scores, trg):
                        target = ''.join(self.label_encoder.stringify(true))
                        hyp = ''.join(hyp)
                        print("Score [{:.2f}]: {} ==> {}".format(score, target, hyp))
                    print()
                self.train()

                rep_loss, rep_batches = 0, 0


if __name__ == '__main__':
    dataset, label_encoder = random_dataset()
    model = EncoderDecoder(label_encoder)
    for _ in range(5):          # train epochs
        model.train_epoch(dataset)
