
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from pie import torch_utils

from .embedding import RNNEmbedding, CNNEmbedding, EmbeddingMixer, EmbeddingConcat
from .decoder import AttentionalDecoder, LinearDecoder, CRFDecoder
from .encoder import RNNEncoder
from .base_model import BaseModel


class SimpleModel(BaseModel):
    """
    Parameters
    ==========
    label_encoder : MultiLabelEncoder
    wemb_dim : int, embedding dimension for word-level embedding layer
    cemb_dim : int, embedding dimension for char-level embedding layer
    hidden_size : int, hidden_size for all hidden layers
    dropout : float
    merge_type : str, one of "concat", "mixer", method to merge word-level and
        char-level embeddings
    cemb_type : str, one of "RNN", "CNN", layer to use for char-level embeddings
    """
    def __init__(self, label_encoder, wemb_dim, cemb_dim, hidden_size, num_layers,
                 dropout=0.0, merge_type='concat', cemb_type='RNN', cell='GRU',
                 include_self=True, pos_crf=True, word_dropout=0.0):
        # args
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
        self.include_self = include_self
        self.pos_crf = pos_crf
        super().__init__(label_encoder)

        # Embeddings
        self.wemb = None
        if self.wemb_dim > 0:
            self.wemb = nn.Embedding(len(label_encoder.word), wemb_dim,
                                     padding_idx=label_encoder.word.get_pad())
        self.cemb = None
        if cemb_type.upper() == 'RNN':
            self.cemb = RNNEmbedding(len(label_encoder.char), cemb_dim,
                                     padding_idx=label_encoder.char.get_pad(),
                                     cell=cell)
        elif cemb_type.upper() == 'CNN':
            self.cemb = CNNEmbedding(len(label_encoder.char), cemb_dim,
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
            in_dim, hidden_size, num_layers=num_layers, dropout=dropout, cell=cell)

        # Decoders
        # - POS
        if 'pos' in label_encoder.tasks:
            if self.pos_crf:
                self.pos_decoder = CRFDecoder(
                    label_encoder.tasks['pos'], hidden_size * 2)
            else:
                self.pos_decoder = LinearDecoder(
                    label_encoder.tasks['pos'], hidden_size * 2)

        # - lemma
        if 'lemma' in label_encoder.tasks:
            self.lemma_sequential = label_encoder.tasks['lemma'].level == 'char'
            if self.lemma_sequential:
                if self.cemb is None:
                    raise ValueError("Sequential lemmatizer requires char embeddings")
                self.lemma_decoder = AttentionalDecoder(
                    label_encoder.tasks['lemma'],
                    self.cemb.embedding_dim, self.cemb.embedding_dim,  # hidden_size * 2,
                    context_dim=hidden_size * 2, dropout=dropout)
            else:
                self.lemma_decoder = LinearDecoder(
                    label_encoder.tasks['lemma'], hidden_size * 2)

        self.linear_tasks, linear_tasks = None, OrderedDict()
        for task in label_encoder.tasks:
            if task in ('pos', 'lemma'):
                continue
            linear_tasks[task] = LinearDecoder(label_encoder.tasks[task], hidden_size * 2)
        if len(linear_tasks) > 0:
            self.linear_tasks = nn.Sequential(linear_tasks)

        # - Self
        if self.include_self:
            self.self_decoder = LinearDecoder(label_encoder.word, hidden_size * 2)

    def get_args_and_kwargs(self):
        return {'args': (self.wemb_dim, self.cemb_dim, self.hidden_size, self.num_layers),
                'kwargs': {'dropout': self.dropout,
                           'word_dropout': self.word_dropout,
                           'cell': self.cell,
                           'merge_type': self.merge_type,
                           'cemb_type': self.cemb_type,
                           'include_self': self.include_self,
                           'pos_crf': self.pos_crf}}

    def embedding(self, word, wlen, char, clen):
        wemb, cemb, cemb_outs = None, None, None
        if self.wemb is not None:
            # set words to unknown with prob `p` depending on word frequency
            word = torch_utils.word_dropout(
                word, self.word_dropout, self.training, self.label_encoder.word)
            wemb = self.wemb(word)
        if self.cemb is not None:
            # cemb_outs: (seq_len x batch x emb_dim)
            cemb, cemb_outs = self.cemb(char, clen, wlen)

        if wemb is None:
            emb = cemb
        elif cemb is None:
            emb = wemb
        else:
            emb = self.merger(wemb, cemb)

        return emb, cemb_outs

    def loss(self, batch_data):
        ((word, wlen), (char, clen)), tasks = batch_data
        output = {}

        emb, cemb_outs = self.embedding(word, wlen, char, clen)
        emb = F.dropout(emb, p=self.dropout, training=self.training)
        enc_outs = self.encoder(emb, wlen)
        enc_outs = F.dropout(enc_outs, p=self.dropout, training=self.training)

        # POS
        if 'pos' in tasks:
            pos, plen = tasks['pos']
            pos_logits = self.pos_decoder(enc_outs)
            if self.pos_crf:
                pos_loss = self.pos_decoder.loss(pos_logits, pos, plen)
            else:
                pos_loss = self.pos_decoder.loss(pos_logits, pos)
            output['pos'] = pos_loss

        # lemma
        if 'lemma' in tasks:
            lemma, llen = tasks['lemma']
            if self.lemma_sequential:
                cemb_outs = F.dropout(cemb_outs, p=self.dropout, training=self.training)
                lemma_context = torch_utils.flatten_padded_batch(enc_outs, wlen)
                lemma_logits = self.lemma_decoder(
                    lemma, llen, cemb_outs, clen, context=lemma_context)
            else:
                lemma_logits = self.lemma_decoder(enc_outs)
            output['lemma'] = self.lemma_decoder.loss(lemma_logits, lemma)

        # linear tasks
        if self.linear_tasks is not None:
            for task, decoder in self.linear_tasks._modules.items():
                logits = decoder(enc_outs)
                tinp, tlen = tasks[task]
                output[task] = decoder.loss(logits, tinp)

        # self (autoregressive language-model like loss)
        if self.include_self:
            if len(enc_outs) > 1:  # can't compute loss for 1-length batches
                self_logits = self.self_decoder(torch_utils.prepad(enc_outs[:-1]))
                self_loss = self.self_decoder.loss(self_logits, word)
                output['self'] = self_loss * 0.2

        return output

    def predict(self, inp, *tasks):
        tasks = set(self.label_encoder.tasks if not len(tasks) else tasks)
        preds = {}
        # unpack
        (word, wlen), (char, clen) = inp

        # forward
        emb, cemb_outs = self.embedding(word, wlen, char, clen)
        enc_outs = self.encoder(emb, wlen)

        # pos
        if 'pos' in self.label_encoder.tasks and 'pos' in tasks:
            pos_hyps, _ = self.pos_decoder.predict(enc_outs, wlen)
            preds['pos'] = pos_hyps

        # lemma
        if 'lemma' in self.label_encoder.tasks and 'lemma' in tasks:
            if self.lemma_sequential:
                lemma_context = torch_utils.flatten_padded_batch(enc_outs, wlen)
                lemma_hyps, _ = self.lemma_decoder.predict_max(
                    cemb_outs, clen, context=lemma_context)
                lemma_hyps = [''.join(hyp) for hyp in lemma_hyps]

            else:
                lemma_hyps, _ = self.lemma_decoder.predict(enc_outs, wlen)
            preds['lemma'] = lemma_hyps

        if self.linear_tasks is not None:
            for task, decoder in self.linear_tasks._modules.items():
                if task in tasks:
                    hyps, _ = decoder.predict(enc_outs, wlen)
                    preds[task] = hyps

        return preds


if __name__ == '__main__':
    from pie.settings import settings_from_file
    from pie.data import Dataset, Reader, MultiLabelEncoder

    settings = settings_from_file('./config.json')
    reader = Reader(settings, settings.input_path)
    label_encoder = MultiLabelEncoder.from_settings(settings)
    label_encoder.fit_reader(reader)
    data = Dataset(settings, reader, label_encoder)
    model = SimpleModel(data.label_encoder, settings.wemb_dim, settings.cemb_dim,
                        settings.hidden_size, settings.num_layers)
    model.to(settings.device)

    for batch in data.batch_generator():
        model.loss(batch)
        break
    ((word, wlen), (char, clen)), tasks = next(data.batch_generator())

    wemb, (cemb, cemb_outs) = model.wemb(word), model.cemb(char, clen, wlen)
    emb = model.merger(wemb, cemb)
    enc_outs = model.encoder(emb, wlen)
    model.pos_decoder.predict(enc_outs, wlen)
    lemma_hyps, _ = model.lemma_decoder.predict_max(
        cemb_outs, clen, context=torch_utils.flatten_padded_batch(enc_outs, wlen))
    print(lemma_hyps)

    model.evaluate(data.get_dev_split())
