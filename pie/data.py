
import os
import glob
import json
from collections import namedtuple
import random

from .utils import chunks


Sent = namedtuple('Sent', ('token', 'pos', 'lemma', 'morph'))


class BaseReader(object):
    """
    Abstract reader class
    """
    def reset(self):
        raise NotImplementedError

    def readlines(self):
        raise NotImplementedError


class TabReader(BaseReader):
    """
    Docstr
    """
    def __init__(self, settings):
        self.indir = os.path.abspath(settings.input_dir)
        self.filenames = glob.glob(self.indir + '/*.{}'.format(settings.extension))
        # ("sent_length", max_sent_length)
        # ("full_stop", full_stop_pos)
        self.breaktype, self.breakdata = settings.breaktype
        self.shuffle_files = settings.shuffle_files

        # attributes
        self.current_line = 0
        self.current_fpath = None

    def _parse_line(self, line):
        tok, lemma, pos, morph = line.split()
        return (tok, lemma, pos, morph)

    def reset(self):
        self.current_line = 0
        self.current_fpath = None

        if self.shuffle_files:
            random.shuffle(self.filenames)

    def readsents(self):
        for fpath in self.filenames:
            self.current_fpath = fpath

            with open(fpath, 'r+') as f:
                token, pos, lemma, morph = [], [], [], []

                for line in f:
                    # update counter to filename
                    self.current_line += 1
                    t, p, l, m = self._parse_line(line.strip())
                    token.append(t)
                    pos.append(p)
                    lemma.append(l)
                    morph.append(m)

                    stop = False
                    if self.breaktype == 'full_stop':
                        if p == self.breakdata:
                            stop = True
                    elif self.breaktype == 'max_sent_length':
                        if len(token) == self.breakdata:
                            stop = True

                    if stop:
                        yield Sent(token, pos, lemma, morph)
                        token, pos, lemma, morph = [], [], [], []


class LabelEncoder(object):
    EOS = '<eos>'
    PAD = '<pad>'
    UNK = '<unk>'

    def fit(self, sents):
        for sent in sents:
            # fit sent-by-sent all sent.token, sent.pos, sent.lemma, sent.morph
            pass

    def transform(self, sents):  # sents will be a batch
        pass

    def save(self, path):
        with open(path, 'w+') as f:
            json.dumps(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r+') as f:
            obj = json.load(f)

        inst = cls(obj)
        return inst


class Dataset(object):
    """
    Docstr
    """
    def __init__(self, settings, reader, evaluation=False, label_encoder=None):
        self.buffer_size = settings.buffer_size
        self.batch_size = settings.batch_size
        if self.batch_size > self.buffer_size:
            raise ValueError("Not enough buffer capacity {} for batch_size of {}"
                             .format(self.buffer_size, self.batch_size))
        self.device = settings.device

        self.reader = reader
        self.label_encoder = label_encoder or LabelEncoder().fit(self.reader)

    def pack_batch(self, batch):
        # sort by lenght, transform into tensors, move to device
        pass

    def batches(self):
        buf = []
        for sent in self.reader.readsents():

            if len(buf) == self.buffer_size:
                for batch in chunks(buf, self.batch_size):
                    yield self.pack_batch(batch)

                self.buf = []

            self.buf.append(sent)

        if len(buf) > 0:
            for batch in chunks(buf, self.batch_size):
                yield self.pack_batch(batch)
