
import torch
import torch.nn as nn
import torch.nn.functional as F

from pie import torch_utils, initialization

from .embedding import build_embeddings
from .decoder import AttentionalDecoder, LinearDecoder, CRFDecoder
from .encoder import RNNEncoder
from .base_model import BaseModel


def get_context(outs, wemb, wlen, context_type):
    if context_type.lower() == 'sentence':
        return torch_utils.flatten_padded_batch(outs, wlen)
    elif context_type.lower() == 'word':
        return torch_utils.flatten_padded_batch(wemb, wlen)
    elif context_type.lower() == 'both':
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
                 cell='LSTM', init_rnn='xavier_uniform',
                 # dropout
                 dropout=0.0, word_dropout=0.0,
                 # word embeddings
                 merge_type='concat', cemb_type='RNN', cemb_layers=1,
                 custom_cemb_cell=False,
                 # lm joint loss
                 include_lm=True, lm_shared_softmax=True,
                 # decoder
                 scorer='general', linear_layers=1,
                 # kwargs
                 **kwargs):
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
        self.custom_cemb_cell = custom_cemb_cell
        self.include_lm = include_lm
        self.lm_shared_softmax = lm_shared_softmax
        self.scorer = scorer
        self.linear_layers = linear_layers
        # only during training
        self.init_rnn = init_rnn
        super().__init__(label_encoder, tasks)

        # Embeddings
        (self.wemb, self.cemb, self.merger), in_dim = build_embeddings(
            label_encoder, wemb_dim,
            cemb_dim, cemb_type, custom_cemb_cell, cemb_layers, cell, init_rnn,
            merge_type, dropout)

        # Encoder
        self.encoder = None
        needs_encoder = False
        for task in self.tasks.values():
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
        decoders = {}
        for tname, task in self.tasks.items():

            if task['level'].lower() == 'char':
                if task['decoder'].lower() == 'linear':
                    decoder = LinearDecoder(
                        label_encoder.tasks[tname], self.cemb.embedding_dim)
                elif task['decoder'].lower() == 'crf':
                    decoder = CRFDecoder(
                        label_encoder.tasks[tname], self.cemb.embedding_dim)
                elif task['decoder'].lower() == 'attentional':
                    context_dim = 0
                    if task['context'].lower() == 'sentence':
                        context_dim = hidden_size * 2  # bidirectional encoder
                    elif task['context'].lower() == 'word':
                        context_dim = wemb_dim
                    elif task['context'].lower() == 'both':
                        context_dim = hidden_size * 2 + wemb_dim
                    decoder = AttentionalDecoder(
                        label_encoder.tasks[tname], cemb_dim, self.cemb.embedding_dim,
                        context_dim=context_dim, scorer=scorer, num_layers=cemb_layers,
                        cell=cell, dropout=dropout, init_rnn=init_rnn)
                else:
                    raise ValueError(
                        "Unknown decoder type {} for char-level task: {}".format(
                            task['decoder'], tname))

            elif task['level'].lower() == 'token':
                # linear
                if task['decoder'].lower() == 'linear':
                    decoder = LinearDecoder(
                        label_encoder.tasks[tname], hidden_size * 2, dropout=dropout,
                        highway_layers=linear_layers - 1)
                # crf
                elif task['decoder'].lower() == 'crf':
                    decoder = CRFDecoder(
                        label_encoder.tasks[tname], hidden_size * 2,
                        highway_layers=linear_layers - 1)

            else:
                raise ValueError(
                    "Unknown decoder type {} for token-level task: {}".format(
                        task['decoder'], tname))

            self.add_module('{}_decoder'.format(tname), decoder)
            decoders[tname] = decoder

        self.decoders = decoders

        # - LM
        if self.include_lm:
            self.lm_fwd_decoder = LinearDecoder(label_encoder.word, hidden_size)
            if lm_shared_softmax:
                self.lm_bwd_decoder = self.lm_fwd_decoder
            else:
                self.lm_bwd_decoder = LinearDecoder(label_encoder.word, hidden_size)

    def get_args_and_kwargs(self):
        return {'args': (self.wemb_dim, self.cemb_dim,
                         self.hidden_size, self.num_layers),
                'kwargs': {'dropout': self.dropout,
                           'word_dropout': self.word_dropout,
                           'cell': self.cell,
                           'merge_type': self.merge_type,
                           'linear_layers': self.linear_layers,
                           'cemb_type': self.cemb_type,
                           'cemb_layers': self.cemb_layers,
                           'include_lm': self.include_lm,
                           'scorer': self.scorer,
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

        if wemb is None:
            emb = cemb
        elif cemb is None:
            emb = wemb
        else:
            emb = self.merger(wemb, cemb)

        return emb, (wemb, cemb, cemb_outs)

    def init_from_encoder(self, encoder):
        # wemb
        total = 0
        for w, idx in encoder.label_encoder.word.table.items():
            if w in self.label_encoder.word.table:
                self.wemb.weight.data[self.label_encoder.word.table[w]].copy_(
                    encoder.wemb.weight.data[idx])
                total += 1
        print("Initialized {}/{} word embs".format(total, len(self.wemb.weight)))
        # cemb
        total = 0
        for w, idx in encoder.label_encoder.char.table.items():
            if w in self.label_encoder.char.table:
                self.cemb.emb.weight.data[self.label_encoder.char.table[w]].copy_(
                    encoder.cemb.emb.weight.data[idx])
                total += 1
        print("Initialized {}/{} char embs".format(total, len(self.cemb.emb.weight)))
        # cemb rnn
        self.cemb.rnn.load_state_dict(encoder.cemb.rnn.state_dict())
        # sentence rnn
        self.encoder.load_state_dict(encoder.encoder.state_dict())

        if self.include_lm:
            pass

    def loss(self, batch_data, *target_tasks):
        ((word, wlen), (char, clen)), tasks = batch_data
        output = {}

        # Embedding
        emb, (wemb, cemb, cemb_outs) = self.embedding(word, wlen, char, clen)

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

            if self.tasks[task]['level'].lower() == 'char':
                if isinstance(decoder, LinearDecoder):
                    logits = decoder(cemb_outs)
                    output[task] = decoder.loss(logits, target)
                elif isinstance(decoder, CRFDecoder):
                    logits = decoder(cemb_outs)
                    output[task] = decoder.loss(logits, target, length)
                elif isinstance(decoder, AttentionalDecoder):
                    cemb_outs = F.dropout(
                        cemb_outs, p=self.dropout, training=self.training)
                    context = get_context(outs, wemb, wlen, self.tasks[task]['context'])
                    logits = decoder(target, length, cemb_outs, clen, context=context)
                    output[task] = decoder.loss(logits, target)
            else:
                if isinstance(decoder, LinearDecoder):
                    logits = decoder(outs)
                    output[task] = decoder.loss(logits, target)
                elif isinstance(decoder, CRFDecoder):
                    logits = decoder(outs)
                    output[task] = decoder.loss(logits, target, length)

        # (LM)
        if self.include_lm:
            if len(emb) > 1:  # can't compute loss for 1-length batches
                # always at first layer
                fwd, bwd = F.dropout(
                    enc_outs[0], p=0, training=self.training
                ).chunk(2, dim=2)
                # forward logits
                logits = self.lm_fwd_decoder(torch_utils.pad(fwd[:-1], pos='pre'))
                output['lm_fwd'] = self.lm_fwd_decoder.loss(logits, word)
                # backward logits
                logits = self.lm_bwd_decoder(torch_utils.pad(bwd[1:], pos='post'))
                output['lm_bwd'] = self.lm_bwd_decoder.loss(logits, word)

        return output

    def predict(self, inp, *tasks, return_probs=False,
                use_beam=False, beam_width=10, **kwargs):
        """
        inp : (word, wlen), (char, clen), text input
        tasks : list of str, target tasks
        """
        tasks = set(self.tasks if not len(tasks) else tasks)
        preds, probs = {}, {}
        (word, wlen), (char, clen) = inp

        # Embedding
        emb, (wemb, cemb, cemb_outs) = self.embedding(word, wlen, char, clen)

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

            if self.label_encoder.tasks[task].level.lower() == 'char':
                if isinstance(decoder, LinearDecoder):
                    hyps, prob = decoder.predict(cemb_outs, clen)
                elif isinstance(decoder, CRFDecoder):
                    hyps, prob = decoder.predict(cemb_outs, clen)
                else:
                    context = get_context(outs, wemb, wlen, self.tasks[task]['context'])
                    if use_beam:
                        hyps, prob = decoder.predict_beam(
                            cemb_outs, clen, width=beam_width, context=context)
                    else:
                        hyps, prob = decoder.predict_max(
                            cemb_outs, clen, context=context)
                    if self.label_encoder.tasks[task].preprocessor_fn is None:
                        hyps = [''.join(hyp) for hyp in hyps]
            else:
                if isinstance(decoder, LinearDecoder):
                    hyps, prob = decoder.predict(outs, wlen)
                elif isinstance(decoder, CRFDecoder):
                    hyps, prob = decoder.predict(outs, wlen)

            preds[task] = hyps
            probs[task] = prob

        if return_probs:
            return preds, probs

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
