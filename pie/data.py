
import os
import glob
import json
import logging
from collections import namedtuple, Counter
import random

import torch

from .utils import chunks

# All currently supported modalities in expected order
MODALITIES = 'token', 'lemma', 'pos', 'morph'
# Wrapper enum-like class to store sentence info
Sent = namedtuple('Sent', ('token', 'lemma', 'pos', 'morph'))


class LineParseException(Exception):
    pass


class BaseReader(object):
    """
    Abstract reader class

    Settings
    ==========
    input_dir : str, root directory with data files
    filenames : list, data files to be processed
    extension : str, type of csv
    shuffle : bool, whether to shuffle files after each iteration

    Attributes
    ===========
    current_sent : int, counter on number of sents processed in total (over all files)
    current_fpath : str, name of the file being currently processed
    """
    def __init__(self, settings):
        self.input_dir = os.path.abspath(settings.input_dir)
        self.filenames = glob.glob(self.input_dir + '/*.{}'.format(settings.extension))
        if len(self.filenames) == 0:
            raise ValueError("Couldn't find matching files in {}".format(self.input_dir))
        self.shuffle = settings.shuffle

        # attributes
        self.current_sent = 0
        self.current_fpath = None

    def reset(self):
        """
        Called after a full run over `readsents`
        """
        self.current_sent = 0

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

            lines = self.parselines(fpath, self.get_modalities(fpath))

            while True:
                try:
                    token, lemma, pos, morph = next(lines)
                    yield Sent(token, lemma, pos, morph)
                    self.current_sent += 1

                except LineParseException as e:
                    logging.warning(
                        "Parse error at [{}:sent={}]\n  => {}"
                        .format(self.current_fpath,
                                self.current_sent + 1,
                                str(e)))
                    continue

                except StopIteration:
                    break

        self.reset()

    def parselines(self, fpath, modalities):
        """
        Generator over tuples of (token, lemma, pos, morph), where each item is a list
        of strings with data from the corresponding modality. Some modalities might be
        missing from a file, in which case the corresponding value is None.

        ParseError can be thrown if an issue is encountered during processing.
        The issue can be documented by passing an error message as second argument
        to ParseError
        """
        raise NotImplementedError

    def get_modalities(self, fpath):
        """
        Reader is responsible for extracting the expected modalities in the file.

        Returns
        =========
        Set of modalities in a file
        """
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

    Settings
    ===========
    breakline_type : str, one of "LENGTH" or "FULLSTOP".
    breakline_data : str or int, if breakline_type is LENGTH it will be assumed
        to be an integer defining the number of words per sentence, and the
        dataset will be break into equally sized chunks. If breakline_type is
        FULLSTOP it will be assumed to be a POS tag to use as criterion to
        split sentences.
    """
    def __init__(self, settings):
        super(TabReader, self).__init__(settings)
        self.breakline_type = settings.breakline_type
        self.breakline_data = settings.breakline_data

    class LineParser(object):
        """
        Inner class to handle sentence breaks
        """
        def __init__(self, modalities, breakline_type, breakline_data):
            if breakline_type == 'FULLSTOP' and 'pos' not in modalities:
                raise ValueError("Cannot use FULLSTOP info to break lines. "
                                 "Modality POS is missing.")

            self.breakline_type = breakline_type
            self.breakline_data = breakline_data
            self.data = {mod: [] for mod in modalities}

        def add(self, line, linenum):
            """
            Adds line to current sentence.
            """
            mods = line.split()
            if len(mods) != len(self.data):
                raise LineParseException(
                    "Not enough number of modalities. "
                    "Expected {} but got {} at line {}.".format(
                        len(self.data), len(mods), linenum))

            # TODO: assumes modalities are always in order (token, lemma, pos, morph)
            for mod, data in zip(MODALITIES, line.split()):
                # TODO: parse morph into something meaningful
                self.data[mod].append(data)

        def check_breakline(self):
            """
            Check if sentence is finished.
            """
            if self.breakline_type == 'FULLSTOP':
                if self.data['pos'][-1] == self.breakline_data:
                    return True
            elif self.breakline_type == 'LENGTH':
                if len(self.data['token']) == self.breakline_data:
                    return True

        def get_data(self):
            """
            Return sentence as data (tuple of modalities)
            """
            return tuple(self.data.get(mod) for mod in MODALITIES)

        def reset(self):
            """
            Reset sentence data
            """
            self.data = {mod: [] for mod in self.data}

    def parselines(self, fpath, modalities):
        """
        Generator over sentences in a single file
        """
        with open(fpath, 'r+') as f:
            parser = self.LineParser(
                modalities, self.breakline_type, self.breakline_data)

            for line_num, line in enumerate(f):
                line = line.strip()

                if not line:    # avoid empty line
                    continue

                parser.add(line, line_num)

                if parser.check_breakline():
                    yield parser.get_data()
                    parser.reset()

    def get_modalities(self, fpath):
        """
        Guess modalities from file assuming expected order
        """
        with open(fpath, 'r+') as f:

            # move to first non empty line
            line = next(f).strip()
            while not line:
                line = next(f).strip()

            line = line.split()
            if len(line) == 0:
                raise ValueError("Not enough modalities in file [{}]".format(fpath))
            else:
                return set(MODALITIES[:len(line)])  # TODO: modalities in expected order


class ModalityLabelEncoder(object):
    """
    Label encoder for single modality.
    """
    def __init__(self, sequential, vocabsize=None, name='Unknown'):
        self.sequential = sequential
        self.vocabsize = vocabsize
        self.name = name
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

    def __eq__(self, other):
        if type(other) != ModalityLabelEncoder:
            return False

        return self.sequential == other.sequential and \
            self.vocabsize == other.vocabsize and \
            self.freqs == other.freqs and \
            self.table == other.table and \
            self.inverse_table == other.inverse_table and \
            self.fitted == other.fitted

    def add(self, sent):
        if self.fitted:
            raise ValueError("Already fitted")
        self.freqs.update(sent)

    def compute_vocab(self):
        if self.fitted:
            raise ValueError("Cannot compute vocabulary, already fitted")

        if len(self.freqs) == 0:
            logging.warning(
                "Computing vocabulary for empty encoder {}".format(self.name))

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

    def _get_sym(self, sym):
        if not self.fitted:
            raise ValueError("Vocabulary hasn't been computed yet")

        if not self.sequential:
            raise ValueError("No PAD token for non-sequential Label Encoder")

        return self.table[getattr(LabelEncoder, sym)]

    def get_pad(self):
        return self._get_sym('PAD')

    def get_eos(self):
        return self._get_sym('EOS')

    def jsonify(self):
        if not self.fitted:
            raise ValueError("Attempted to serialize unfitted encoder")

        return {'sequential': self.sequential,
                'vocabsize': self.vocabsize,
                'freqs': dict(self.freqs),
                'table': dict(self.table),
                'inverse_table': self.inverse_table}

    @classmethod
    def from_json(cls, obj):
        inst = cls(obj['sequential'], obj['vocabsize'])
        inst.freqs = Counter(obj['freqs'])
        inst.table = dict(obj['table'])
        inst.inverse_table = list(obj['inverse_table'])
        inst.fitted = True

        return inst


class LabelEncoder(object):
    """
    Complex Label encoder for all modalities.
    """

    EOS = '<eos>'
    PAD = '<pad>'
    UNK = '<unk>'

    def __init__(self, vocabsize):
        self.token = ModalityLabelEncoder(
            sequential=True, vocabsize=vocabsize, name='token')
        self.pos = ModalityLabelEncoder(sequential=True, name='pos')
        # TODO: lemma-only vocab size?
        self.lemma = ModalityLabelEncoder(
            sequential=True, vocabsize=vocabsize, name='lemma')
        self.morph = ModalityLabelEncoder(sequential=False, name='morph')
        self._all_encoders = [self.token, self.lemma, self.pos, self.morph]

    @classmethod
    def from_settings(cls, settings):
        return cls(settings.vocabsize)

    def fit(self, sents):
        """
        Parameters
        ===========
        sents : list of Sent
        """
        for sent in sents:
            self.token.add(sent.token)
            if sent.lemma is not None:
                self.lemma.add(sent.lemma)
            if sent.pos is not None:
                self.pos.add(sent.pos)
            if sent.morph is not None:
                self.morph.add(sent.morph)

        for le in self._all_encoders:
            le.compute_vocab()

        return self

    def transform(self, sents):
        """
        Parameters
        ===========
        sents : list of Sent

        Returns
        ===========
        tuple of modalities. Each modality is either a list of integer or None.
        """
        token, pos, lemma, morph = [], [], [], []
        for sent in sents:
            token.append(self.token.transform(sent.token))
            if sent.pos is not None:
                pos.append(self.pos.transform(sent.pos))
            if sent.lemma is not None:
                lemma.append(self.lemma.transform(sent.lemma))
            if sent.morph is not None:
                morph.append(self.morph.transform(sent.morph))

        return token, pos or None, lemma or None, morph or None

    def save(self, path):
        with open(path, 'w+') as f:
            json.dump({self.token.name: self.token.jsonify(),
                       self.pos.name: self.pos.jsonify(),
                       self.lemma.name: self.lemma.jsonify(),
                       self.morph.name: self.morph.jsonify()}, f)

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
        inst._all_encoders = [inst.token, inst.lemma, inst.pos, inst.morph]

        return inst


class Dataset(object):
    """
    Dataset class to encode files into integers and compute batches.

    Settings
    ===========
    buffer_size : int, maximum number of sentences in memory at any given time.
       The larger the buffer size the more memory instensive the dataset will
       be but also the more effective the shuffling over instances.
    batch_size : int, number of sentences per batch
    device : str, target device to put the processed batches on
    shuffle : bool, whether to shuffle items in the buffer

    Parameters
    ===========
    label_encoder : optional, prefitted LabelEncoder object
    """
    def __init__(self, settings, label_encoder=None):
        if settings.batch_size > settings.buffer_size:
            raise ValueError("Not enough buffer capacity {} for batch_size of {}"
                             .format(settings.buffer_size, settings.batch_size))

        self.buffer_size = settings.buffer_size
        self.batch_size = settings.batch_size
        self.device = settings.device
        self.shuffle = settings.shuffle

        self.reader = TabReader(settings)
        self.label_encoder = label_encoder or \
            LabelEncoder.from_settings(settings).fit(self.reader.readsents())

    @staticmethod
    def pad_batch(self, batch, padding_id, device):
        """
        Pad batch into tensor
        """
        maxlen, batch_size = max(map(len, batch)), len(batch)
        output = torch.zeros(maxlen, batch_size).long() + padding_id
        for i, inst in enumerate(batch):
            output[i, 0:len(inst)].copy_(
                torch.tensor(inst, dtype=torch.int64, device=device))

        return output

    def pack_batch(self, batch):
        batch = sorted(batch, key=lambda sent: len(sent.token), reverse=True)
        # assumes sent.token is always given
        lengths = [len(sent.token) for sent in batch]
        token, pos, lemma, morph = self.label_encoder.transform(batch)

        token = Dataset.pad_batch(token, self.label_encoder.token.get_pad)

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
