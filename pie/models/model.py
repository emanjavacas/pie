
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pie import torch_utils, initialization

from .embedding import RNNEmbedding, CNNEmbedding, EmbeddingMixer, EmbeddingConcat
from .decoder import AttentionalDecoder, LinearDecoder, CRFDecoder
from .encoder import RNNEncoder
from .base_model import BaseModel


def get_context(outs, wemb, wlen, lemma_context):
    if lemma_context.lower() == 'sentence':
        return torch_utils.flatten_padded_batch(outs, wlen)
    elif lemma_context.lower() == 'word':
        return torch_utils.flatten_padded_batch(wemb, wlen)
    elif lemma_context.lower() == 'both':
        outs = torch_utils.flatten_padded_batch(outs, wlen)
        wemb = torch_utils.flatten_padded_batch(wemb, wlen)
        return torch.cat([outs, wemb], -1)
    else:
        return None


class SimpleModel(BaseModel):
    """
    Parameters
    ==========
    label_encoder : MultiLabelEncoder
    tasks : settings.tasks
    wemb_dim : int, embedding dimension for word-level embedding layer
    cemb_dim : int, embedding dimension for char-level embedding layer
    hidden_size : int, hidden_size for all hidden layers
    dropout : float
    merge_type : str, one of "concat", "mixer", method to merge word-level and
        char-level embeddings
    cemb_type : str, one of "RNN", "CNN", layer to use for char-level embeddings
    """
    def __init__(self, label_encoder, tasks, wemb_dim, cemb_dim, hidden_size, num_layers,
                 dropout=0.0, word_dropout=0.0, merge_type='concat', cemb_type='RNN',
                 cell='LSTM', custom_cemb_cell=False, init_rnn='xavier_uniform',
                 include_lm=True):
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
        self.cemb_layers = cemb_layers
        self.include_lm = include_lm
        self.custom_cemb_cell = custom_cemb_cell
        # only during training
        self.init_rnn = init_rnn
        super().__init__(label_encoder, tasks)


        # Embeddings
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
        self.encoder = None
        needs_encoder = False
        for task in self.tasks:
            if task['level'] == 'token':
                needs_encoder = True
                break
            elif task.get('context', '').lower() in ('sentence', 'both'):
                needs_encoder = True
                break
        if not needs_encoder:
            print("Model doesn't need sentence encoder, leaving uninitialized")
        else:
            self.encoder = RNNEncoder(
                in_dim, hidden_size, num_layers=num_layers, cell=cell, dropout=dropout,
                init_rnn=init_rnn)

        # Decoders
        decoders = OrderedDict()
        for tname, task in self.tasks.items():
            # linear
            if task['decoder'].lower() == 'linear':
                decoders[task] = LinearDecoder(
                    label_encoder.tasks[tname], hidden_size * 2)
            # crf
            elif task['decoder'].lower() == 'crf':
                decoders[task] = CRFDecoder(
                    label_encoder.tasks[tname], hidden_size * 2)
            # attentional
            elif task['decoder'].lower() == 'attentional':
                # checks
                if task['level'].lower() != 'char':
                    raise ValueError(
                        "Got task {} with token-level attentional decoder".format(tname))
                if self.cemb is None:
                    raise ValueError(
                        "Attentional decoder requires char embeddings. Task", tname)
                # get context size
                context_dim = 0
                if task['context'].lower() == 'sentence':
                    context_dim = hidden_size * 2  # bidirectional encoder
                elif task['context'].lower() == 'word':
                    context_dim = wemb_dim
                elif task['context'].lower() == 'both':
                    context_dim = hidden_size * 2 + wemb_dim

                decoders[task] = AttentionalDecoder(
                    label_encoder.tasks[task], self.cemb.embedding_dim,
                    self.cemb.embedding_dim, context_dim=context_dim, dropout=dropout,
                    num_layers=cemb_layers, dropout=dropout, init_rnn=init_rnn)

            else:
                raise ValueError(
                    "Unknown decoder type {}. Task: {}".format(task['decoder'], tname))

        self.decoders = nn.Sequential(decoders)

        # - LM
        if self.include_lm:
            self.lm_decoder_fwd = LinearDecoder(label_encoder.word, hidden_size)
            self.lm_decoder_bwd = LinearDecoder(label_encoder.word, hidden_size)

    def get_args_and_kwargs(self):
        return {'args': (self.wemb_dim, self.cemb_dim, self.hidden_size, self.num_layers),
                'kwargs': {'dropout': self.dropout,
                           'word_dropout': self.word_dropout,
                           'cell': self.cell,
                           'merge_type': self.merge_type,
                           'cemb_type': self.cemb_type,
                           'cemb_layers': self.cemb_layers,
                           'include_lm': self.include_lm,
                           'custom_cemb_cell': self.custom_cemb_cell}}

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

        return wemb, cemb, cemb_outs

    def loss(self, batch_data, *target_tasks):
        ((word, wlen), (char, clen)), tasks = batch_data
        output = {}

        # Embedding
        wemb, cemb, cemb_outs = self.embedding(word, wlen, char, clen)
        if wemb is None:
            emb = cemb
        elif cemb is None:
            emb = wemb
        else:
            emb = self.merger(wemb, cemb)

        # Encoder
        emb = F.dropout(emb, p=self.dropout, training=self.training)
        enc_outs = None
        if self.encoder is not None:
            # TODO: check if we need encoder for this particular batch
            enc_outs = self.encoder(emb, wlen)

        # Decoder
        for task in target_tasks:
            (target, length), at_layer = tasks[task], self.tasks[task]['layer']
            # prepare input layer
            outs = None
            if enc_outs is not None:
                outs = F.dropout(enc_outs[at_layer], p=0, training=self.training)

            decoder = self.decoders[task]

            # prepare logits
            if isinstance(decoder, LinearDecoder):
                logits = decoder(outs)
                output[task] = decoder.loss(logits, target)
            elif isinstance(decoder, CRFDecoder):
                logits = decoder(outs)
                output[task] = decoder.loss(logits, target, length)
            else:
                cemb_outs = F.dropout(cemb_outs, p=self.dropout, training=self.training)
                context = get_context(outs, wemb, wlen, self.tasks[task]['context'])
                logits = decoder(inp, length, cemb_outs, clen, context)
                output[task] = decoder.loss(logits, target)

        # (LM)
        if self.include_lm:
            if len(emb) > 1:  # can't compute loss for 1-length batches
                # always at first layer
                fwd, bwd = F.dropout(
                    enc_outs[0], p=0, training=self.training
                ).chunk(2, dim=2)
                # forward logits
                logits = self.lm_decoder_fwd(torch_utils.pad(fwd[:-1], pos='pre'))
                output['fwd_lm'] = self.lm_decoder_fwd.loss(logits, word)
                # backward logits
                logits = self.lm_decoder_fwd(torch_utils.pad(bwd[1:], pos='post'))
                output['bwd_lm'] = self.lm_decoder_bwd.loss(logits, word)

        return output

    def predict(self, inp, *tasks):
        tasks = set(self.label_encoder.tasks if not len(tasks) else tasks)
        preds = {}
        (word, wlen), (char, clen) = inp

        # Embedding
        wemb, cemb, cemb_outs = self.embedding(word, wlen, char, clen)
        if wemb is None:
            emb = cemb
        elif cemb is None:
            emb = wemb
        else:
            emb = self.merger(wemb, cemb)

        # Encoder
        enc_outs = None
        if self.encoder is not None:
            # TODO: check if we need encoder for this particular batch
            enc_outs = self.encoder(emb, wlen)

        # Decoders
        for task in tasks:

            decoder, at_layer = self.decoders[task], self.tasks[task]['layer']
            outs = None
            if enc_outs is not None:
                outs = enc_outs[at_layer]

            if isinstance(decoder, LinearDecoder):
                hyps, _ = decoder.predict(outs, wlen)
            elif isinstance(decoder, CRFDecoder):
                hyps, _ = decoder.predict(outs, wlen)
            else:
                context = get_context(outs, wemb, wlen, self.tasks[task]['context'])
                hyps, _ = decoder.predict_max(cemb_outs, clen, context=context)

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
    model = SimpleModel(data.label_encoder, settings.tasks,
                        settings.wemb_dim, settings.cemb_dim,
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
    lemma_hyps, _ = model.decoders['lemma'].predict_max(
        cemb_outs, clen, context=torch_utils.flatten_padded_batch(enc_outs, wlen))
    print(lemma_hyps)
