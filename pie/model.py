
import torch.nn as nn
import torch.nn.functional as F

from pie.embedding import MixedEmbedding
from pie.decoder import AttentionalDecoder, LinearDecoder
from pie.encoder import RNNEncoder
from pie import torch_utils


class SimpleModel(nn.Module):
    def __init__(self, label_encoder, emb_dim, hidden_size, dropout=0.0):
        self.label_encoder = label_encoder
        super().__init__()

        # embeddings
        emb = MixedEmbedding(label_encoder, emb_dim)

        # encoder
        self.encoder = RNNEncoder(emb, hidden_size, dropout=dropout)

        # decoders
        self.pos_decoder = AttentionalDecoder(
            label_encoder.tasks['pos'], emb_dim, hidden_size, dropout=dropout)

        if label_encoder.tasks['lemma'].level == 'word':
            self.lemma_decoder = LinearDecoder(
                label_encoder.tasks['lemma'], hidden_size, dropout=dropout)
        else:
            self.lemma_decoder = AttentionalDecoder(
                label_encoder.tasks['lemma'], emb_dim, hidden_size,
                context_dim=hidden_size, dropout=dropout)
            self.lemma_encoder = RNNEncoder(self.lemma_decoder.embs, hidden_size)

    def loss(self, batch_data):
        ((word, wlen), (char, clen)), tasks = batch_data

        enc_outs = self.encoder(word, wlen, char, clen)

        # POS
        pos, plen = tasks['pos']
        pos_inp = F.pad(pos[:-1], (0, 0, 1, 0))  # pad the first step
        pos_logits = self.pos_decoder(pos_inp, plen, enc_outs)
        pos_loss = self.pos_decoder.loss(pos_logits, pos)

        # lemma
        lemma, llen = tasks['lemma']
        if isinstance(self.lemma_decoder, AttentionalDecoder):
            lemma_inp = F.pad(lemma[:-1], (0, 0, 1, 0))
            lemma_context = torch_utils.pad_flatten_batch(enc_outs, wlen)
            lemma_enc_outs = self.lemma_encoder(lemma_inp, llen)
            lemma_logits = self.lemma_decoder(
                lemma_inp, llen, lemma_enc_outs, context=lemma_context)
            lemma_loss = self.lemma_decoder.loss(lemma_logits, lemma)
        else:
            lemma_logits = self.lemma_decoder(enc_outs)
            lemma_loss = self.lemma_decoder.loss(lemma_logits, lemma)

        return pos_loss, lemma_loss


if __name__ == '__main__':
    from pie.settings import settings_from_file
    from pie.data import Dataset

    settings = settings_from_file('./config.json')
    data = Dataset(settings)
    model = SimpleModel(data.label_encoder, settings.emb_dim, settings.hidden_size)
    for batch in data.batch_generator():
        model.loss(batch)
        break
    ((word, wlen), (char, clen)), tasks = next(data.batch_generator())

    enc_outs = model.encoder(word, wlen, char, clen)
    scores, hyps = model.pos_decoder.generate(enc_outs)
    print(scores, hyps)
