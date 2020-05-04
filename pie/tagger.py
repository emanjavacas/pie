
import re
import string
import os
import itertools

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


def simple_tokenizer(text, lower):
    section, fullstop = regexsplitter(SECTION), regexsplitter(FULLSTOP)
    word = regexsplitter(WORD)
    for line in section(text):
        for sentence in fullstop(line):
            sentence = [w for raw in sentence.split() for w in word(raw)]
            if lower:
                sentence = [w.lower() for w in sentence]
            yield sentence


def lines_from_file(fpath, lower=False, tokenize=False, max_sent_len=35):
    """
    lower : bool, whether to lowercase output
    tokenize : bool, whether to use simple_tokenizer
    max_sent_len : int, only applicable if tokenize is False
    """
    with open(fpath) as f:
        for line in f:
            sentence = []
            if not tokenize:
                for w in line.split():
                    if len(sentence) >= max_sent_len:
                        yield sentence, len(sentence)
                        sentence = []
                    sentence.append(w.lower() if lower else w)
            else:
                for sentence in simple_tokenizer(line, lower):
                    yield sentence, len(sentence)
                sentence = []

    if sentence: # yield remaining words when tokenize is False
        yield sentence, len(sentence)


class Tagger():
    def __init__(self, device='cpu', batch_size=100, lower=False, tokenize=False):
        self.device = device
        self.batch_size = batch_size
        self.lower = lower
        self.tokenize = tokenize
        self.models = []

    def add_model(self, model_path, *tasks):
        model = BaseModel.load(model_path)
        for task in tasks:
            if task not in model.label_encoder.tasks:
                raise ValueError("Model [{}] doesn't have task: {}".format(
                    model_path, task))

        self.models.append((model, tasks))

    def tag(self, sents, lengths, **kwargs):
        # add dummy input tasks (None)
        batch = list(zip(sents, itertools.repeat(None)))
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

    def tag_file(self, fpath, sep='\t', **kwargs):
        _, ext = os.path.splitext(fpath)
        header = False

        with open(utils.ensure_ext(fpath, ext, 'pie'), 'w+') as f:
            lines = lines_from_file(fpath, lower=self.lower, tokenize=self.tokenize)
            for chunk in utils.chunks(lines, self.batch_size):
                tagged, tasks = self.tag(*zip(*chunk), **kwargs)

                for sent in tagged:
                    if not header:
                        f.write(sep.join(['token'] + tasks) + '\n')
                        header = True
                    for token, tags in sent:
                        f.write(sep.join([token] + list(tags)) + '\n')

                    if self.tokenize:
                        f.write('\n')
