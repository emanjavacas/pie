
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from pie.embedding import RNNEmbedding, EmbeddingMixer, EmbeddingConcat
from pie.decoder import AttentionalDecoder, LinearDecoder, CRFDecoder
from pie.encoder import RNNEncoder
from pie.evaluation import Scorer
from pie import torch_utils


class SimpleModel(nn.Module):
    """
    Parameters
    ==========
    label_encoder : MultiLabelEncoder
    emb_dim : int, embedding dimension for all embedding layers
    hidden_size : int, hidden_size for all hidden layers
    dropout : float
    merge_type : str, one of "concat", "mixer", method to merge word-level and
        char-level embeddings
    cemb_type : str, one of "RNN", "CNN", layer to use for char-level embeddings
    """
    def __init__(self, label_encoder, emb_dim, hidden_size, num_layers, dropout=0.0,
                 merge_type='concat', cemb_type='RNN', include_self=True):
        self.label_encoder = label_encoder
        self.include_self = include_self
        super().__init__()

        # Embeddings
        self.wemb = nn.Embedding(len(label_encoder.word), emb_dim,
                                 padding_idx=label_encoder.word.get_pad())
        if cemb_type == 'RNN':
            self.cemb = RNNEmbedding(len(label_encoder.char), emb_dim,
                                     padding_idx=label_encoder.char.get_pad())
        elif cemb_type == 'CNN':
            self.cemb = CNNEmbedding(len(label_encoder.char), emb_dim,
                                     padding_idx=label_encoder.char.get_pad())

        if merge_type == 'mixer':
            if self.cemb.embedding_dim != self.wemb.embedding_dim:
                raise ValueError("EmbeddingMixer needs equal embedding dims")
            self.merger = EmbeddingMixer(emb_dim)
            in_dim = emb_dim

        elif merge_type == 'concat':
            self.merger = EmbeddingConcat()
            in_dim = self.wemb.embedding_dim + self.cemb.embedding_dim

        # Encoder
        self.encoder = RNNEncoder(
            in_dim, hidden_size, num_layers=num_layers, dropout=dropout)

        # Decoders
        # - POS
        self.pos_decoder = LinearDecoder(
            label_encoder.tasks['pos'], hidden_size, dropout=dropout)
        # self.pos_decoder = CRFDecoder(label_encoder.tasks['pos'], hidden_size)

        # - Lemma
        self.lemma_sequential = label_encoder.tasks['lemma'].level == 'char'
        if self.lemma_sequential:
            # TODO: check using char-level input as encoder instead of extra one
            self.lemma_emb = nn.Embedding(len(label_encoder.char), emb_dim)
            self.lemma_encoder = RNNEncoder(emb_dim, hidden_size)
            self.lemma_decoder = AttentionalDecoder(
                label_encoder.tasks['lemma'], emb_dim, hidden_size,
                context_dim=hidden_size, dropout=dropout)
        else:
            self.lemma_decoder = LinearDecoder(
                label_encoder.tasks['lemma'], hidden_size, dropout=dropout)

        # - Self
        if self.include_self:
            self.self_decoder = LinearDecoder(
                label_encoder.word, hidden_size, dropout=dropout)

    def loss(self, batch_data):
        ((word, wlen), (char, clen)), tasks = batch_data
        output = {}

        wemb, cemb = self.wemb(word), self.cemb(char, clen, wlen)
        enc_outs = self.encoder(self.merger(wemb, cemb), wlen)

        # POS
        pos, plen = tasks['pos']
        pos_logits = self.pos_decoder(enc_outs)
        pos_loss = self.pos_decoder.loss(pos_logits, pos)
        # pos_feats = self.pos_decoder(enc_outs)
        # pos_loss = self.pos_decoder.loss(pos_feats, pos, plen)
        output['pos'] = pos_loss

        # lemma
        lemma, llen = tasks['lemma']
        if self.lemma_sequential:
            lemma_context = torch_utils.flatten_padded_batch(enc_outs, wlen-1)
            lemma_enc_outs = self.lemma_encoder(self.lemma_emb(char), clen)
            lemma_logits = self.lemma_decoder(
                lemma, llen, lemma_enc_outs, clen, context=lemma_context)
            lemma_loss = self.lemma_decoder.loss(lemma_logits, lemma)
        else:
            lemma_logits = self.lemma_decoder(enc_outs)
            lemma_loss = self.lemma_decoder.loss(lemma_logits, lemma)
        output['lemma'] = lemma_loss

        # self (autoregressive language-model like loss)
        if self.include_self:
            self_logits = self.self_decoder(torch_utils.prepad(enc_outs[:-1]))
            self_loss = self.self_decoder.loss(self_logits, word)
            output['self'] = self_loss

        return output

    def predict(self, batch):
        # unpack
        ((word, wlen), (char, clen)), tasks = batch

        # forward
        wemb, cemb = self.wemb(word), self.cemb(char, clen, wlen)
        enc_outs = self.encoder(self.merger(wemb, cemb), wlen)

        # remove <eos> during decoding
        wlen = wlen-1

        # pos
        pos_hyps, _ = self.pos_decoder.predict(enc_outs, wlen)
        pos, _ = tasks['pos']
        pos_trues = [self.label_encoder.tasks['pos'].stringify(p)
                     for p in pos.t().tolist()]

        # lemma
        lemma, _ = tasks['lemma']
        lemma_trues = [self.label_encoder.tasks['lemma'].stringify(l)
                       for l in lemma.t().tolist()]
        if self.lemma_sequential:
            lemma_context = torch_utils.flatten_padded_batch(enc_outs, wlen)
            lemma_enc_outs = self.lemma_encoder(self.lemma_emb(char), clen)
            lemma_hyps, _ = self.lemma_decoder.predict_max(
                lemma_enc_outs, clen, context=lemma_context)
            lemma_hyps = [''.join(hyp) for hyp in lemma_hyps]
            lemma_trues = [''.join(l) for l in lemma_trues]
        else:
            lemma_hyps, _ = self.lemma_decoder.predict(enc_outs, wlen)

        return (pos_hyps, pos_trues), (lemma_hyps, lemma_trues)

    def evaluate(self, dataset):
        """
        Get scores per task
        """
        pos_scorer = Scorer(self.label_encoder.tasks['pos'])
        lemma_scorer = Scorer(self.label_encoder.tasks['lemma'],
                              compute_unknown=self.lemma_sequential)

        with torch.no_grad():
            for batch in dataset.batch_generator():
                (pos_hyps, pos_trues), (lemma_hyps, lemma_trues) = self.predict(batch)
                pos_scorer.register_batch(pos_hyps, pos_trues)
                lemma_scorer.register_batch(lemma_hyps, lemma_trues)

        return {'pos': pos_scorer.get_scores(), 'lemma': lemma_scorer.get_scores()}


if __name__ == '__main__':
    from pie.settings import settings_from_file
    from pie.data import Dataset

    settings = settings_from_file('./config.json')
    data = Dataset(settings)
    model = SimpleModel(data.label_encoder, settings.emb_dim, settings.hidden_size,
                        settings.num_layers)
    model.to(settings.device)

    for batch in data.batch_generator():
        model.loss(batch)
        break
    ((word, wlen), (char, clen)), tasks = next(data.batch_generator())

    model.evaluate(data)

    wemb, cemb = model.wemb(word), model.cemb(char, clen, wlen)
    emb = model.merger(wemb, cemb)
    enc_outs = model.encoder(emb, wlen)
    hyps, scores = model.pos_decoder.predict_sequence(enc_outs, wlen)
    print(scores, hyps)
