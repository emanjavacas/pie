
import os
import glob
import json
import logging
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
    Reader for files in tab format where each line has annotations for a given token
    and each annotation is located in a column separated by tabs.

    ...
    italiae	italia	NE	gender=FEMININE|case=GENITIVE|number=SINGULAR
    eo	is	PRO	gender=MASCULINE|case=ABLATIVE|number=SINGULAR
    modo	modus	NN	gender=MASCULINE|case=ABLATIVE|number=SINGULAR
    ...

    Relevant settings:
      - input_dir: str
      - extension: str, type of csv
      - breakline_type: str, one of "LENGTH" or "FULLSTOP".
      - breakline_data: str or int, if breakline_type is LENGTH it will be assumed
            to be an integer defining the number of words per sentence, and the
            dataset will be break into equally sized chunks. If breakline_type is
            FULLSTOP it will be assumed to be a POS tag to use as criterion to
            split sentences.
      - shuffle: bool, whether to shuffle files after each iteration.
    """
    def __init__(self, settings):
        self.indir = os.path.abspath(settings.input_dir)
        self.filenames = glob.glob(self.indir + '/*.{}'.format(settings.extension))
        self.breakline_type = settings.breakline_type
        self.breakline_data = settings.breakline_data
        self.shuffle = settings.shuffle

        # attributes
        self.current_line = 0
        self.current_fpath = None

    def _parse_line(self, line):
        tok, lemma, pos, morph = line.split()
        return (tok, lemma, pos, morph)

    def _check_breakline(self, pos):
        if self.breakline_type == 'FULLSTOP':
            if pos[-1] == self.breakline_data:
                return True
        elif self.breakline_type == 'LENGTH':
            if len(pos) == self.breakline_data:
                return True

    def reset(self):
        """
        Must be called after a full run over `readsents`
        """
        self.current_line = 0
        self.current_fpath = None

        if self.shuffle:
            random.shuffle(self.filenames)

    def readsents(self):
        """
        Generator over dataset sentences. Each output will be a Sent object with
        the attributes "token", "pos", "lemma" and "morph", each of which is a list
        of strings.
        """
        for fpath in self.filenames:
            self.current_fpath = fpath

            with open(fpath, 'r+') as f:
                token, pos, lemma, morph = [], [], [], []

                for line in f:
                    # update counter to filename
                    self.current_line += 1
                    try:
                        t, p, l, m = self._parse_line(line.strip())
                    except Exception:
                        logging.warning("Parse error at [{}:n{}]".format(
                            self.current_fpath, self.current_line + 1))
                        continue
                    token.append(t)
                    pos.append(p)
                    lemma.append(l)
                    morph.append(m)

                    if self._check_breakline(pos):
                        yield Sent(token, pos, lemma, morph)
                        token, pos, lemma, morph = [], [], [], []


class LabelEncoder(object):
    EOS = '<eos>'
    PAD = '<pad>'
    UNK = '<unk>'

    def fit(self, sents):
        for sent in sents:
            # TODO: fit sent-by-sent all sent.token, sent.pos, sent.lemma, sent.morph
            pass

    def transform(self, sents):  # sents will be a batch
        # TODO
        pass

    def save(self, path):
        with open(path, 'w+') as f:
            # TODO: extract necessary info
            json.dumps(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r+') as f:
            obj = json.load(f)
        # TODO: load necessary info
        inst = cls(obj)
        return inst


class Dataset(object):
    """
    Dataset class to encode files into integers and compute batches.

    Relevant settings:
      - buffer_size: int, maximum number of sentences in memory at any given time.
           The larger the buffer size the more memory instensive the dataset will
           be but also the more effective the shuffling over instances.
      - batch_size: int, number of sentences per batch
      - device: str, target device to put the processed batches on
      - shuffle: bool, whether to shuffle items in the buffer

    Arguments:
      - evaluation: bool, whether the dataset is an evaluation dataset or not
      - label_encoder: optional, prefitted LabelEncoder object
    """
    def __init__(self, settings, reader, evaluation=False, label_encoder=None):
        self.buffer_size = settings.buffer_size
        self.batch_size = settings.batch_size
        if self.batch_size > self.buffer_size:
            raise ValueError("Not enough buffer capacity {} for batch_size of {}"
                             .format(self.buffer_size, self.batch_size))
        self.device = settings.device
        self.shuffle = settings.shuffle

        self.reader = reader
        self.label_encoder = label_encoder or LabelEncoder().fit(self.reader)

    def pack_batch(self, batch):
        batch = sorted(batch, key=lambda sent: len(sent.token), reverse=True)
        lengths = [len(sent.token) for sent in batch]
        token, pos, lemma, morph = zip(*self.label_encoder.transform(batch))

        # TODO: transform to tensors

        return (token, pos, lemma, morph), lengths

    def batches(self):
        """
        Generator over dataset batches. Each batch is a tuple of (data, lengths),
        where data is itself a tuple of (token, pos, lemma, morph) where each is a
        torch Tensor.
        """
        buf = []
        for sent in self.reader.readsents():

            if len(buf) == self.buffer_size:
                if self.shuffle:
                    random.shuffle(buf)

                for batch in chunks(buf, self.batch_size):
                    yield self.pack_batch(batch)

                self.buf = []

            self.buf.append(sent)

        if len(buf) > 0:
            for batch in chunks(buf, self.batch_size):
                yield self.pack_batch(batch)
