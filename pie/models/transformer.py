
import random
import logging
from datetime import datetime

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

import pie
from pie import torch_utils
from pie.settings import settings_from_file
from pie.trainer import Trainer
from pie import initialization
from pie.data import Dataset, Reader, MultiLabelEncoder
from pie.data.dataset import pack_batch
from pie.models import (BaseModel, LinearDecoder, CRFDecoder,
                        build_embeddings, AttentionalDecoder)
from pie import optimize


def get_instance_spans(tokenizer, text):
    index = []
    tokens = []
    for (i, token) in enumerate(text.split()):
        index.append(len(tokens))
        for sub_token in tokenizer.tokenize(token):
            tokens.append(sub_token)
    index.append(len(tokens))
    spans = list(zip(index[:-1], index[1:]))
    return spans


def get_spans(tokenizer, texts, batch):
    spans = [get_instance_spans(tokenizer, inp) for inp in texts]
    max_span_len = max(end - start for sent in spans for start, end in sent)
    max_spans = max(map(len, spans))
    batch_size, _, emb_dim = batch.shape
    output = torch.zeros(
        batch_size, max_spans, max_span_len, emb_dim, device=batch.device)
    mask = torch.zeros(batch_size, max_spans, max_span_len)

    for i in range(batch_size):
        for span, (start, end) in enumerate(spans[i]):
            output[i, span, 0:end-start].copy_(batch[i, start:end])
            mask[i, span, 0:end-start] = 1

    return output, mask.bool()


def check_alignment(tokenizer, text):
    spans = get_instance_spans(tokenizer, text)
    orig_tokens = text.split()
    assert len(spans) == len(orig_tokens)
    tokens = tokenizer.tokenize(text)
    output = []
    for idx, (start, end) in enumerate(spans):
        output.append((tokens[start:end], orig_tokens[idx]))
    return output


class TransformerDataset(Dataset):
    def __init__(self, settings, reader, label_encoder, tokenizer, model):
        super().__init__(settings, reader, label_encoder)

        self.tokenizer = tokenizer
        self.model = model

    def get_transformer_output(self, text, device):
        encoded = self.tokenizer.batch_encode_plus(
            text, return_tensors='pt', pad_to_max_length=True)
        encoded = {k: val.to(self.model.device) for k, val in encoded.items()}
        with torch.no_grad():
            batch = self.model(**encoded)[0]  # some models return 2 items, others 1
        # remove <s>, </s> tokens
        batch = batch[:, 1:-1]
        # get spans
        context, mask = get_spans(self.tokenizer, text, batch)
        context, mask = context.to(device), mask.to(device)

        return context, mask

    def pack_batch(self, batch, device=None):
        device = device or self.device
        (word, char), tasks = pack_batch(self.label_encoder, batch, device)
        context, mask = self.get_transformer_output(
            [' '.join(inp) for inp, _ in batch], device)
        
        return (word, char, (context, mask)), tasks


class SpanSelfAttention(nn.Module):
    def __init__(self, context_dim, hidden_size, dropout=0.0):
        self.context_dim = context_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        super().__init__()

        self.W = nn.Linear(context_dim, hidden_size)
        self.v_a = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.init()

    def init(self):
        self.v_a.data.uniform_(-0.05, 0.05)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, context, mask):
        # (batch, num_spans, max_span_len, 1)
        weights = self.W(context) @ self.v_a.unsqueeze(0).unsqueeze(0)
        weights = weights.squeeze(3)
        # apply mask
        weights.masked_fill_(~mask, -float('inf'))
        # softmax
        weights = F.softmax(weights, dim=-1)
        # remove nans that arise in padding
        weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
        # weighted sum (batch, num_spans, max_span_len, dim) -> (batch, num_spans, dim)
        context = (context * weights.unsqueeze(-1)).sum(2)
        context = F.dropout(context, p=self.dropout, training=self.training)
        # transpose to batch-second
        context = context.transpose(0, 1)
        return context


class Model(BaseModel):
    def __init__(self, label_encoder, tasks, context_dim,
                 # input embeddings
                 wemb_dim=0, cemb_dim=0, cemb_type='RNN', custom_cemb_cell=False,
                 cemb_layers=1, cell='GRU', init_rnn='default', merge_type='concat',
                 # decoder
                 linear_layers=1, dropout=0.0, scorer='general'):
        self.context_dim = context_dim
        self.linear_layers = linear_layers
        self.dropout = dropout
        super().__init__(label_encoder, tasks)

        hidden_size = context_dim

        # embeddings
        (self.wemb, self.cemb, self.merger), in_dim = build_embeddings(
            label_encoder, wemb_dim,
            cemb_dim, cemb_type, custom_cemb_cell, cemb_layers, cell, init_rnn,
            merge_type, dropout)

        # self attention
        self.self_att = SpanSelfAttention(
            context_dim, hidden_size, dropout=dropout)

        # decoders
        decoders = {}
        for tname, task in self.tasks.items():

            if task['level'].lower() == 'char':
                if task['decoder'].lower() == 'attentional':                
                    decoder = AttentionalDecoder(
                        label_encoder.tasks[tname], cemb_dim, self.cemb.embedding_dim,
                        context_dim=hidden_size + in_dim, scorer=scorer,
                        num_layers=cemb_layers, cell=cell, dropout=dropout,
                        init_rnn=init_rnn)

            elif task['level'].lower() == 'token':
                # linear
                if task['decoder'].lower() == 'linear':
                    decoder = LinearDecoder(
                        label_encoder.tasks[tname], hidden_size + in_dim,
                        highway_layers=linear_layers - 1)
                # crf
                elif task['decoder'].lower() == 'crf':
                    decoder = CRFDecoder(
                        label_encoder.tasks[tname], hidden_size + in_dim,
                        highway_layers=linear_layers - 1)

            else:
                raise ValueError(
                    "Unknown decoder type {} for token-level task: {}".format(
                        task['decoder'], tname))

            self.add_module('{}_decoder'.format(tname), decoder)
            decoders[tname] = decoder

            self.decoders = decoders

    def get_args_and_kwargs(self):
        return {'args': (self.context_dim, ),
                'kwargs': {'linear_layers': self.linear_layers}}

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
        elif self.merger is not None:
            emb = self.merger(wemb, cemb)
        else:
            emb = None

        return emb, (wemb, cemb, cemb_outs)

    def loss(self, batch_data, *target_tasks):
        ((word, wlen), (char, clen), (context, mask)), tasks = batch_data
        output = {}

        emb, (_, _, cemb_outs) = self.embedding(word, wlen, char, clen)

        outs = self.self_att(context, mask)
        if emb is not None:
            outs = torch.cat([outs, emb], dim=-1)

        for task in target_tasks:
            (target, length), decoder = tasks[task], self.decoders[task]

            if self.tasks[task]['level'].lower() == 'char':
                cemb_outs = F.dropout(
                    cemb_outs, p=self.dropout, training=self.training)
                logits = decoder(target, length, cemb_outs, clen,
                                 context=torch_utils.flatten_padded_batch(outs, wlen))
                output[task] = decoder.loss(logits, target)
            else:
                if isinstance(decoder, LinearDecoder):
                    logits = decoder(outs)
                    output[task] = decoder.loss(logits, target)
                elif isinstance(decoder, CRFDecoder):
                    logits = decoder(outs)
                    output[task] = decoder.loss(logits, target, length)

        return output

    def predict(self, inp, *tasks, use_beam=False, beam_width=10, **kwargs):
        tasks = set(self.label_encoder.tasks if not len(tasks) else tasks)
        (word, wlen), (char, clen), (context, mask) = inp

        emb, (_, _, cemb_outs) = self.embedding(word, wlen, char, clen)
        
        outs = self.self_att(context, mask)
        if emb is not None:
            outs = torch.cat([outs, emb], dim=-1)

        preds = {}
        for task in tasks:
            decoder = self.decoders[task]

            if self.label_encoder.tasks[task].level.lower() == 'char':
                if not use_beam:
                    hyps, _ = decoder.predict_max(
                        cemb_outs, clen,
                        context=torch_utils.flatten_padded_batch(outs, wlen))
                else:
                    hyps, _ = decoder.predict_beam(
                        cemb_outs, clen,
                        context=torch_utils.flatten_padded_batch(outs, wlen),
                        width=beam_width)
                if self.label_encoder.tasks[task].preprocessor_fn is None:
                    hyps = [''.join(hyp) for hyp in hyps]
            else:
                if isinstance(decoder, LinearDecoder):
                    hyps, _ = decoder.predict(outs, wlen)
                elif isinstance(decoder, CRFDecoder):
                    hyps, _ = decoder.predict(outs, wlen)
                else:
                    raise ValueError()

            preds[task] = hyps

        return preds


def run(settings, transformer_path):
    now = datetime.now()
    seed = now.hour * 10000 + now.minute * 100 + now.second
    print("Using seed:", seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    reader = Reader(settings, settings.input_path)
    tasks = reader.check_tasks(expected=None)

    label_encoder = MultiLabelEncoder.from_settings(settings, tasks=tasks)
    label_encoder.fit_reader(reader)
    if settings.verbose:
        print("::: Tasks :::")
        print()
        for task, le in label_encoder.tasks.items():
            print("- {:<15} target={:<6} level={:<6} vocab={:<6}"
                  .format(task, le.target, le.level, len(le)))
        print()

    tokenizer = AutoTokenizer.from_pretrained(transformer_path)
    transformer = AutoModel.from_pretrained(transformer_path)
    trainset = TransformerDataset(
        settings, reader, label_encoder, tokenizer, transformer)
    devset = None
    if settings.dev_path:
        devset = TransformerDataset(
            settings, Reader(settings, settings.dev_path), label_encoder,
            tokenizer, transformer)
    else:
        logging.warning("No devset: cannot monitor/optimize training")

    model = Model(label_encoder, settings.tasks, trainset.model.config.hidden_size,
                  wemb_dim=settings.wemb_dim, cemb_dim=settings.cemb_dim,
                  cemb_type=settings.cemb_type,
                  custom_cemb_cell=settings.custom_cemb_cell,
                  cemb_layers=settings.cemb_layers, cell=settings.cell,
                  init_rnn=settings.init_rnn, merge_type=settings.merge_type,
                  linear_layers=settings.linear_layers, dropout=settings.dropout,
                  scorer=settings.scorer)
    model.to(settings.device)

    print("::: Model :::")
    print()
    print(model)
    print()
    print("::: Model parameters :::")
    print()
    trainable = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    total = sum(p.nelement() for p in model.parameters())
    print("{}/{} trainable/total".format(trainable, total))
    print()

    # training
    print("Starting training")
    trainer = Trainer(settings, model, trainset, reader.get_nsents())
    scores = None
    try:
        scores = trainer.train_epochs(settings.epochs, devset=devset)
    except KeyboardInterrupt:
        print("Stopping training")
    finally:
        model.eval()

    if devset is not None:
        scorers = model.evaluate(
            devset, trainset=trainset, use_beam=True, beam_width=10)
        for task, scorer in scorers.items():
            print(task)
            scorer.print_summary()
            print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('transformer_path')
    parser.add_argument('--opt_path', help='Path to optimization file (see opt.json)')
    parser.add_argument('--n_iter', type=int, default=20)
    args = parser.parse_args()

    settings = settings_from_file(args.config_path)

    if args.opt_path:
        opt = optimize.read_opt(args.opt_path)
        optimize.run_optimize(
            run, settings, opt, args.n_iter, transformer_path=args.transformer_path)
    else:
        run(settings, args.transformer_path)
