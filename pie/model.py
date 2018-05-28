
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
                 merge_type='concat', cemb_type='RNN', include_self=True, pos_crf=True):
        self.label_encoder = label_encoder
        self.include_self = include_self
        self.pos_crf = pos_crf
        self.dropout = dropout
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
        else:
            raise ValueError("Unknown merge method: {}".format(merge_type))

        # Encoder
        self.encoder = RNNEncoder(
            in_dim, hidden_size, num_layers=num_layers, dropout=dropout)

        # Decoders
        # - POS
        if self.pos_crf:
            self.pos_decoder = CRFDecoder(label_encoder.tasks['pos'], hidden_size)
        else:
            self.pos_decoder = LinearDecoder(
                label_encoder.tasks['pos'], hidden_size, dropout=dropout)

        # - lemma
        self.lemma_sequential = label_encoder.tasks['lemma'].level == 'char'
        if self.lemma_sequential:
            self.lemma_decoder = AttentionalDecoder(
                label_encoder.tasks['lemma'], emb_dim, emb_dim, #hidden_size,
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

        wemb, (cemb, cemb_outs) = self.wemb(word), self.cemb(char, clen, wlen)
        # cemb_outs: (seq_len x batch x emb_dim)
        wemb = F.dropout(wemb, p=self.dropout, training=self.training)
        cemb = F.dropout(cemb, p=self.dropout, training=self.training)
        enc_outs = self.encoder(self.merger(wemb, cemb), wlen)

        # POS
        pos, plen = tasks['pos']
        pos_logits = self.pos_decoder(enc_outs)
        if self.pos_crf:
            pos_loss = self.pos_decoder.loss(pos_logits, pos, plen)
        else:
            pos_loss = self.pos_decoder.loss(pos_logits, pos)
        output['pos'] = pos_loss

        # lemma
        lemma, llen = tasks['lemma']
        if self.lemma_sequential:
            cemb_outs = F.dropout(cemb_outs, p=self.dropout, training=self.training)
            lemma_context = torch_utils.flatten_padded_batch(enc_outs, wlen)
            lemma_logits = self.lemma_decoder(
                lemma, llen, cemb_outs, clen, context=lemma_context)
        else:
            lemma_logits = self.lemma_decoder(enc_outs)
        lemma_loss = self.lemma_decoder.loss(lemma_logits, lemma)
        output['lemma'] = lemma_loss

        # self (autoregressive language-model like loss)
        if self.include_self:
            self_logits = self.self_decoder(torch_utils.prepad(enc_outs[:-1]))
            self_loss = self.self_decoder.loss(self_logits, word)
            output['self'] = self_loss * 0.2

        return output

    def predict(self, inp):
        # unpack
        (word, wlen), (char, clen) = inp

        # forward
        wemb, (cemb, cemb_outs) = self.wemb(word), self.cemb(char, clen, wlen)
        enc_outs = self.encoder(self.merger(wemb, cemb), wlen)

        # pos
        pos_hyps, _ = self.pos_decoder.predict(enc_outs, wlen)

        # lemma
        if self.lemma_sequential:
            lemma_context = torch_utils.flatten_padded_batch(enc_outs, wlen)
            lemma_hyps, _ = self.lemma_decoder.predict_max(
                cemb_outs, clen, context=lemma_context)
            lemma_hyps = [''.join(hyp) for hyp in lemma_hyps]

        else:
            lemma_hyps, _ = self.lemma_decoder.predict(enc_outs, wlen)

        return pos_hyps, lemma_hyps

    def evaluate(self, dataset, total=None):
        """
        Get scores per task
        """
        pos_scorer = Scorer(self.label_encoder.tasks['pos'])
        lemma_scorer = Scorer(
            self.label_encoder.tasks['lemma'], compute_unknown=self.lemma_sequential)
        pos_le = self.label_encoder.tasks['pos']
        lemma_le = self.label_encoder.tasks['lemma']

        with torch.no_grad():

            for inp, tasks in tqdm.tqdm(dataset, total=total):
                # get trues
                (pos, plen), (lemma, llen) = tasks['pos'], tasks['lemma']
                pos, lemma = pos.t().tolist(), lemma.t().tolist()
                plen, llen = plen.tolist(), llen.tolist()
                pos_true = [pos_le.stringify(p, l) for p, l in zip(pos, plen)]
                if self.lemma_sequential:
                    lemma_true = [''.join(lemma_le.stringify(l)) for l in lemma]
                else:
                    lemma_true = [lemma_le.stringify(l, ll) for l, ll in zip(lemma, llen)]

                # get preds
                pos_hyps, lemma_hyps = self.predict(inp)

                # accumulate
                pos_scorer.register_batch(pos_hyps, pos_true)
                lemma_scorer.register_batch(lemma_hyps, lemma_true)

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

    wemb, (cemb, _) = model.wemb(word), model.cemb(char, clen, wlen)
    emb = model.merger(wemb, cemb)
    enc_outs = model.encoder(emb, wlen)
    model.pos_decoder.predict(enc_outs, wlen)
