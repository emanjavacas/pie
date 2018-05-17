
import torch
import torch.nn as nn

from pie.embedding import MixedEmbedding
from pie.decoder import AttentionalDecoder, LinearDecoder


class SimpleModel(nn.Module):
    def __init__(self, label_encoder, emb_dim, hidden_size, dropout=0.0):
        self.label_encoder = label_encoder
        super().__init__()

        # embeddings
        self.emb = MixedEmbedding(label_encoder, emb_dim)

        # feature extractor
        self.feats = nn.GRU(emb_dim, hidden_size // 2, bidirectional=True)

        # decoders
        self.pos = AttentionalDecoder(
            label_encoder.pos, emb_dim, hidden_size, dropout=dropout)

        self.lemma = LinearDecoder(label_encoder.lemma, hidden_size,
                                   dropout=dropout)

    def loss(self, batch_data):
        ((token, tlen), (char, clen), lengths), (lemma, pos, morph) = batch_data

        embs = self.emb(token, char, clen, lengths)
        enc_outs, _ = self.feats(embs)

        # POS
        pos, plen = pos
        pos_inp, pos_targets = pos[:-1], pos[1:]
        plen = torch.tensor(plen) - 1
        pos_loss = self.pos.loss(pos_inp, plen, enc_outs, pos_targets)

        # lemma
        lemma, _ = lemma
        lemma_loss = self.lemma.loss(enc_outs, lemma)

        return pos_loss, lemma_loss


if __name__ == '__main__':
    from pie.settings import settings_from_file
    from pie.data import Dataset

    settings = settings_from_file('./config.json')
    data = Dataset(settings)
    model = SimpleModel(data.label_encoder, settings.emb_dim, settings.hidden_size)
    for batch in data.batch_generator():
        print(model.loss(batch))
    ((token, tlen), (char, clen), lengths), _ = next(data.batch_generator())
