
import re
import string
import os
import itertools

import tqdm

from pie.models import BaseModel
from pie.data import pack_batch
from pie import utils


def regexsplitter(regex):
    def func(line):
        for line in re.split(regex, line):
            line = line.strip()
            if line:
                yield line

    return func


SECTION = r'([A-Z\. ]+\.)'
FULLSTOP = r'([^\.]+\.)'
WORD = r'([{}])'.format(string.punctuation)


def simple_tokenizer(text):
    section = regexsplitter(SECTION)
    fullstop = regexsplitter(FULLSTOP)
    word = regexsplitter(WORD)

    for line in section(text):
        for sent in fullstop(line):
            yield [w for raw in sent.split() for w in word(raw)]


def get_sentences_vrt(fpath, max_sent_len):
    with open(fpath) as f:
        sent = []
        for line in f:
            line = line.strip()
            if sent and not line or len(sent) >= max_sent_len:
                yield sent
                sent = []
            w, *_ = line.split()
            sent.append(w)
        if sent:
            yield sent


def get_sentences_line(fpath, max_sent_len):
    with open(fpath) as f:
        sent = []
        for line in f:
            line = line.strip().split()
            if not line and sent:
                yield sent
                sent = []
            for w in line:
                if len(sent) >= max_sent_len:
                    yield sent
                    sent = []
                sent.append(w)
            if sent:
                yield sent
                sent = []
        if sent:
            yield sent            


def get_sentences(fpath, max_sent_len, vrt):
    if vrt:
        yield from get_sentences_vrt(fpath, max_sent_len)
    else:
        yield from get_sentences_line(fpath, max_sent_len)


def lines_from_file(fpath, tokenize=False, max_sent_len=35, vrt=False):
    """
    tokenize : bool, whether to use simple_tokenizer
    max_sent_len : int, only applicable if tokenize is False
    """
    if tokenize:
        # ignore vrt
        with open(fpath) as f:
            for sent in simple_tokenizer(line):
                yield sent, len(sent)
    else:
        for sent in get_sentences(fpath, max_sent_len, vrt):
            yield sent, len(sent)


class Tagger():
    def __init__(self, device='cpu', batch_size=100,
                 lower=False, tokenize=False, vrt=False, max_sent_len=35):
        self.device = device
        self.batch_size = batch_size
        self.lower = lower
        self.tokenize = tokenize
        self.vrt = vrt
        self.max_sent_len = max_sent_len
        self.models = []

    def add_model(self, model_path, *tasks):
        model = BaseModel.load(model_path)
        for task in tasks:
            if task not in model.label_encoder.tasks:
                raise ValueError("Model [{}] doesn't have task: {}".format(
                    model_path, task))

        self.models.append((model, tasks))

    def tag(self, sents, lengths, **kwargs):
        # lower if needed
        batch_sents = sents
        if self.lower:
            batch_sents = [[w.lower() for w in sent] for sent in sents]
        # add dummy input tasks (None)
        batch = list(zip(batch_sents, itertools.repeat(None)))
        # [token1, token2, ...]
        tokens = [token for sent in sents for token in sent]
        # output
        output = {}
        for model, tasks in self.models:
            model.to(self.device)

            inp, _ = pack_batch(model.label_encoder, batch, self.device)

            # inference
            preds = model.predict(inp, *tasks, **kwargs)

            for task in preds:
                # flatten all sentences (since some tasks return flattened output)
                if model.label_encoder.tasks[task].level != 'char':
                    preds[task] = [hyp for sent in preds[task] for hyp in sent]
                # postprocess if needed
                if model.label_encoder.tasks[task].preprocessor_fn is not None:
                    post = []
                    assert len(tokens) == len(preds[task])
                    for tok, hyp in zip(tokens, preds[task]):
                        pred = model.label_encoder.tasks[task].preprocessor_fn \
                               .inverse_transform(hyp, tok)
                        post.append(pred)
                    preds[task] = post
                # append
                output[task] = preds[task]
            model.to('cpu')

        tasks = sorted(output)

        # [(task1, task2, ...), (task1, task2, ...), ...]
        output = list(zip(*tuple(output[task] for task in tasks)))
        assert len(tokens) == len(output), "{} != {}".format(len(tokens), len(output))

        # segment sentences
        tagged = []
        for length in lengths:
            sent = []
            for _ in range(length):
                sent.append((tokens.pop(0), output.pop(0)))
            tagged.append(sent)

        return tagged, tasks

    def tag_file(self, fpath, sep='\t', keep_boundaries=False, **kwargs):
        _, ext = os.path.splitext(fpath)
        header = False

        with open(utils.ensure_ext(fpath, ext, 'pie'), 'w+') as f:
            lines = lines_from_file(
                fpath, tokenize=self.tokenize,
                max_sent_len=self.max_sent_len, vrt=self.vrt)
            lines = list(lines)

            with tqdm.tqdm(total=len(lines)) as pbar:
                for chunk in utils.chunks(lines, self.batch_size):

                    tagged, tasks = self.tag(*zip(*chunk), **kwargs)
                    pbar.update(n=len(tagged))

                    for sent in tagged:
                        if not header:
                            f.write(sep.join(['token'] + tasks) + '\n')
                            header = True
                        for token, tags in sent:
                            f.write(sep.join([token] + list(tags)) + '\n')
                        if keep_boundaries:
                            f.write('\n')
