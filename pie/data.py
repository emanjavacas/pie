
import os
import glob
import json
import logging
from collections import namedtuple, Counter
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
        if len(self.filenames) == 0:
            raise ValueError("Couldn't find matching files in {}".format(self.indir))
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

        self.reset()


class ModalityLabelEncoder(object):
    """
    Label encoder for single modality.
    """
    def __init__(self, sequential, vocabsize=None):
        self.sequential = sequential
        self.vocabsize = vocabsize
        self._reserved = (LabelEncoder.UNK,)
        if sequential:
            self._reserved += (LabelEncoder.EOS, LabelEncoder.PAD)

        self.freqs = Counter()
        self.table = None
        self.inverse_table = None
        self.fitted = False

    def __len__(self):
        if not self.fitted:
            return -1
        return len(self.table)

    def add(self, sent):
        if self.fitted:
            raise ValueError("Already fitted")
        self.freqs.update(sent)

    def compute_vocab(self):
        if self.fitted:
            raise ValueError("Cannot compute vocabulary, already fitted")

        self.inverse_table = list(self._reserved)
        if self.sequential:
            self.inverse_table += [
                sym for sym, _ in
                self.freqs.most_common(n=self.vocabsize or len(self.freqs))]
        else:
            self.inverse_table += list(self.freqs)

        self.table = {sym: idx for idx, sym in enumerate(self.inverse_table)}
        self.fitted = True

    def transform(self, sent):
        if not self.fitted:
            raise ValueError("Vocabulary hasn't been computed yet")

        sent = [self.table.get(tok, self.table[LabelEncoder.UNK]) for tok in sent]

        if self.sequential:
            sent.append(self.table.get(LabelEncoder.EOS))

        return sent

    def jsonify(self):
        if not self.fitted:
            raise ValueError("Attempted to serialize unfitted encoder")

        return json.dump({'sequential': self.sequential,
                          'vocabsize': self.vocabsize,
                          'freqs': dict(self.freqs),
                          'table': dict(self.table),
                          'inverse_table': self.inverse_table})

    @classmethod
    def from_json(cls, obj):
        inst = cls(obj['sequential'], obj['vocabsize'])
        inst.freqs = Counter(obj['freqs'])
        inst.table = dict(obj['table'])
        inst.inverse_table = list(obj['inverse_table'])
        inst.fitted = True


class LabelEncoder(object):
    """
    Complex Label encoder for all modalities.
    """

    EOS = '<eos>'
    PAD = '<pad>'
    UNK = '<unk>'

    def __init__(self, vocabsize):
        # TODO: set off modalities if required
        self.token = ModalityLabelEncoder(sequential=True, vocabsize=vocabsize)
        self.pos = ModalityLabelEncoder(sequential=True)
        # TODO: lemma-only vocab size?
        self.lemma = ModalityLabelEncoder(sequential=True, vocabsize=vocabsize)
        self.morph = ModalityLabelEncoder(sequential=False)
        self._all_encoders = [self.token, self.pos, self.lemma, self.morph]

    @classmethod
    def from_settings(cls, settings):
        return cls(settings.vocabsize)

    def fit(self, sents):
        """
        Arguments:
          - sents: list of Sent
        """
        for sent in sents:
            self.token.add(sent.token)
            self.pos.add(sent.pos)
            self.lemma.add(sent.lemma)
            self.morph.add(sent.morph)

        for le in self._all_encoders:
            le.compute_vocab()

        return self

    def transform(self, sents):
        """
        Arguments:
           - sents: list of Sent
        """
        token, pos, lemma, morph = [], [], [], []
        for sent in sents:
            token.append(self.token.transform(sent.token))
            pos.append(self.pos.transform(sent.pos))
            lemma.append(self.lemma.transform(sent.lemma))
            morph.append(self.morph.transform(sent.morph))

        return token, pos, lemma, morph

    def save(self, path):
        with open(path, 'w+') as f:
            json.dump({'token': self.token.jsonify(),
                       'pos': self.pos.jsonify(),
                       'lemma': self.lemma.jsonify(),
                       'morph': self.morph.jsonify()}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r+') as f:
            obj = json.load(f)

        inst = cls(vocabsize=None)  # dummy instance to overwrite

        # set encoders
        inst.token = ModalityLabelEncoder.from_json(obj['token'])
        inst.pos = ModalityLabelEncoder.from_json(obj['pos'])
        inst.lemma = ModalityLabelEncoder.from_json(obj['lemma'])
        inst.morph = ModalityLabelEncoder.from_json(obj['morph'])
        inst._all_encoders = [inst.token, inst.pos, inst.lemma, inst._all_encoders]

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
    def __init__(self, settings, evaluation=False, label_encoder=None):
        self.buffer_size = settings.buffer_size
        self.batch_size = settings.batch_size
        if self.batch_size > self.buffer_size:
            raise ValueError("Not enough buffer capacity {} for batch_size of {}"
                             .format(self.buffer_size, self.batch_size))
        self.device = settings.device
        self.shuffle = settings.shuffle

        self.reader = TabReader(settings)
        self.label_encoder = label_encoder
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder.from_settings(settings) \
                                             .fit(self.reader.readsents())

    def pack_batch(self, batch):
        batch = sorted(batch, key=lambda sent: len(sent.token), reverse=True)
        lengths = [len(sent.token) for sent in batch]
        token, pos, lemma, morph = self.label_encoder.transform(batch)

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

                buf = []

            buf.append(sent)

        if len(buf) > 0:
            for batch in chunks(buf, self.batch_size):
                yield self.pack_batch(batch)
