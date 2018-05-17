
import os
import glob
import json
import logging
from collections import namedtuple, Counter
import random

import torch

from pie import utils
from pie import consts

# All currently supported tasks in internally expected order
TASKS = 'lemma', 'pos', 'morph'
# Wrapper enum-like classes to store sentence info
Input = namedtuple('Input', ('token', ))
Tasks = namedtuple('Tasks', ('lemma', 'pos', 'morph'))


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
            raise ValueError("Couldn't find files in {}".format(self.input_dir))
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
        Generator over dataset sentences. Each output is a tuple of (Input, Tasks)
        objects with, where each entry is a list of strings.
        """
        for fpath in self.filenames:
            self.current_fpath = fpath

            lines = self.parselines(fpath, self.get_tasks(fpath))

            while True:
                try:
                    yield next(lines)
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

    def parselines(self, fpath, tasks):
        """
        Generator over tuples of (Input, Tasks) holding input and target data
        as lists of strings. Some tasks might be missing from a file, in which
        case the corresponding value is None.

        ParseError can be thrown if an issue is encountered during processing.
        The issue can be documented by passing an error message as second argument
        to ParseError
        """
        raise NotImplementedError

    def get_tasks(self, fpath):
        """
        Reader is responsible for extracting the expected tasks in the file.

        Returns
        =========
        Set of tasks in a file
        """
        raise NotImplementedError


class TabReader(BaseReader):
    """
    Reader for files in tab format where each line has annotations for a given token
    and each annotation is located in a column separated by tabs.

    ...
    italiae	italia	NE	gender=FEMININE|case=GENITIVE|number=SINGULAR
    eo	is	PRO	gender=MASCULINE|case=ABLATIVE|number=SINGULAR
    tasko	taskus	NN	gender=MASCULINE|case=ABLATIVE|number=SINGULAR
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
        self.tasks = settings.tasks  # TODO: add setting to setting docs

    class LineParser(object):
        """
        Inner class to handle sentence breaks
        """
        def __init__(self, tasks, breakline_type, breakline_data):
            if breakline_type == 'FULLSTOP' and 'pos' not in tasks:
                raise ValueError("Cannot use FULLSTOP info to break lines. "
                                 "Task POS is missing.")

            self.breakline_type = breakline_type
            self.breakline_data = breakline_data
            self.inp = []
            self.tasks = {task: [] for task in tasks}

        def add(self, line, linenum):
            """
            Adds line to current sentence.
            """
            inp, *tasks = line.split()
            if len(tasks) != len(self.tasks):
                raise LineParseException(
                    "Not enough number of tasks. Expected {} but got {} at line {}."
                    .format(len(self.tasks), len(tasks), linenum))

            self.inp.append(inp)
            for task, data in zip(self.tasks.keys(), tasks):
                # TODO: parse morph into something meaningful
                self.tasks[task].append(data)

        def check_breakline(self):
            """
            Check if sentence is finished.
            """
            if self.breakline_type == 'FULLSTOP':
                if self.tasks['pos'][-1] == self.breakline_data:
                    return True
            elif self.breakline_type == 'LENGTH':
                if len(self.inp) == self.breakline_data:
                    return True

        def get_data(self):
            """
            Return data tuple
            """
            return (Input(self.inp),
                    Tasks(*[self.tasks.get(task, None) for task in TASKS]))

        def reset(self):
            """
            Reset sentence data
            """
            self.tasks = {task: [] for task in self.tasks}
            self.inp = []

    def parselines(self, fpath, tasks):
        """
        Generator over sentences in a single file
        """
        with open(fpath, 'r+') as f:
            parser = self.LineParser(tasks, self.breakline_type, self.breakline_data)

            for line_num, line in enumerate(f):
                line = line.strip()

                if not line:    # avoid empty line
                    continue

                parser.add(line, line_num)

                if parser.check_breakline():
                    yield parser.get_data()
                    parser.reset()

    def get_tasks(self, fpath):
        """
        Guess tasks from file assuming expected order
        """
        with open(fpath, 'r+') as f:

            # move to first non empty line
            line = next(f).strip()
            while not line:
                line = next(f).strip()

            _, *tasks = line.split()
            if len(tasks) == 0:
                raise ValueError("Not enough input in file [{}]".format(fpath))

            if self.tasks is not None:
                return self.tasks

            # default to order specified by TASKS
            return TASKS[:len(tasks)]


class SingleLabelEncoder(object):
    """
    Label encoder for single taskality.
    """
    def __init__(self, pad=True, eos=True, vocabsize=None, name='Unknown'):
        self.eos = consts.EOS if eos else None
        self.pad = consts.PAD if pad else None
        self.vocabsize = vocabsize
        self.name = name
        self.reserved = (consts.UNK,)
        self.reserved += tuple([sym for sym in [self.eos, self.pad] if sym])
        self.freqs = Counter()
        self.table = None
        self.inverse_table = None
        self.fitted = False

    def __len__(self):
        if not self.fitted:
            raise ValueError("Cannot get length of unfitted LabelEncoder")
        return len(self.table)

    def __eq__(self, other):
        if type(other) != SingleLabelEncoder:
            return False

        return self.pad == other.pad and \
            self.eos == other.eos and \
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
            logging.warning("Computing vocabulary for empty encoder {}"
                            .format(self.name))

        most_common = self.freqs.most_common(n=self.vocabsize or len(self.freqs))
        self.inverse_table = list(self.reserved) + [sym for sym, _ in most_common]
        self.table = {sym: idx for idx, sym in enumerate(self.inverse_table)}
        self.fitted = True

    def transform(self, sent):
        if not self.fitted:
            raise ValueError("Vocabulary hasn't been computed yet")

        sent = [self.table.get(tok, self.table[consts.UNK]) for tok in sent]

        if self.eos:
            sent.append(self.get_eos())

        return sent

    def _get_sym(self, sym):
        if not self.fitted:
            raise ValueError("Vocabulary hasn't been computed yet")

        return self.table[sym]

    def get_pad(self):
        return self._get_sym(consts.PAD)

    def get_eos(self):
        return self._get_sym(consts.EOS)

    def jsonify(self):
        if not self.fitted:
            raise ValueError("Attempted to serialize unfitted encoder")

        return {'eos': self.eos,
                'pad': self.pad,
                'vocabsize': self.vocabsize,
                'freqs': dict(self.freqs),
                'table': dict(self.table),
                'inverse_table': self.inverse_table}

    @classmethod
    def from_json(cls, obj):
        inst = cls(pad=obj['pad'], eos=obj['eos'], vocabsize=obj['vocabsize'])
        inst.freqs = Counter(obj['freqs'])
        inst.table = dict(obj['table'])
        inst.inverse_table = list(obj['inverse_table'])
        inst.fitted = True

        return inst


class LabelEncoder(object):
    """
    Complex Label encoder for all tasks.
    """
    def __init__(self, vocabsize):
        # TODO: per task, input settings
        self.token = SingleLabelEncoder(vocabsize=vocabsize, name='token')
        self.char = SingleLabelEncoder(name='char')
        self.pos = SingleLabelEncoder(name='pos')
        # TODO: lemma char-level
        self.lemma = SingleLabelEncoder(vocabsize=vocabsize, name='lemma')
        self.morph = SingleLabelEncoder(name='morph')
        self._all_encoders = [self.token, self.char, self.lemma, self.pos, self.morph]

    @classmethod
    def from_settings(cls, settings):
        # TODO: per task, input settings
        return cls(vocabsize=settings.vocabsize)

    def fit(self, lines):
        """
        Parameters
        ===========
        lines : iterator over tuples of (Input, Tasks)
        """
        for idx, (inp, tasks) in enumerate(lines):
            # input
            self.token.add(inp.token)
            self.char.add([char for word in inp.token for char in word])
            # tasks
            # TODO: lemma char-level vs token-level?
            if tasks.lemma is not None:
                self.lemma.add(tasks.lemma)
            if tasks.pos is not None:
                self.pos.add(tasks.pos)
            if tasks.morph is not None:
                self.morph.add(tasks.morph)

        for le in self._all_encoders:
            le.compute_vocab()

        return self

    def transform(self, sents):
        """
        Parameters
        ===========
        sents : list of Example's

        Returns
        ===========
        tuple of tuples (token, char, lengths), (lemma, pos, morph)

        Each item in a tuple is a list of the following:
            * Input
                - token: list of integers
                - char: list of integers where each list represents a word
                    at the character level
                - lengths: integers representing the length of the original
                    sentence

            * Tasks
                - lemma: optional, list of integers
                - pos: optional, list of integers
                - morph: #TODO
        """
        token, char, lengths, lemma, pos, morph = [], [], [], [], [], []

        for inp, tasks in sents:
            # input data
            token.append(self.token.transform(inp.token))
            char.extend([self.char.transform(w) for w in inp.token])
            lengths.append(len(inp.token))
            # task data
            if tasks.pos is not None:
                pos.append(self.pos.transform(tasks.pos))
            # TODO: lemma char-level instead of token-level
            if tasks.lemma is not None:
                lemma.append(self.lemma.transform(tasks.lemma))
            if tasks.morph is not None:
                morph.append(self.morph.transform(tasks.morph))

        return (token, char, lengths), (lemma or None, pos or None, morph or None)

    def save(self, path):
        with open(path, 'w+') as f:
            json.dump({le.name: le.jsonify() for le in self._all_encoders}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r+') as f:
            obj = json.load(f)

        inst = cls(vocabsize=None)  # dummy instance to overwrite

        # set encoders
        inst.token = SingleLabelEncoder.from_json(obj['token'])
        inst.char = SingleLabelEncoder.from_json(obj['char'])
        inst.pos = SingleLabelEncoder.from_json(obj['pos'])
        inst.lemma = SingleLabelEncoder.from_json(obj['lemma'])
        inst.morph = SingleLabelEncoder.from_json(obj['morph'])
        inst._all_encoders = [inst.token, inst.char, inst.lemma, inst.pos, inst.morph]

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
    def __init__(self, settings, label_encoder=None, verbose=True):
        if settings.batch_size > settings.buffer_size:
            raise ValueError("Not enough buffer capacity {} for batch_size of {}"
                             .format(settings.buffer_size, settings.batch_size))

        # attributes
        self.buffer_size = settings.buffer_size
        self.batch_size = settings.batch_size
        self.device = settings.device
        self.shuffle = settings.shuffle

        # data
        self.reader = TabReader(settings)
        # label encoder
        self.label_encoder = label_encoder
        if self.label_encoder is None:
            if verbose:
                print("Fitting label encoders...")
            self.label_encoder = LabelEncoder \
                .from_settings(settings) \
                .fit(self.reader.readsents())

    def pad_batch(self, batch, padding_id):
        """
        Pad batch into tensor
        """
        lengths = [len(example) for example in batch]
        maxlen, batch_size = max(lengths), len(batch)
        output = torch.zeros(maxlen, batch_size).long() + padding_id
        for i, example in enumerate(batch):
            output[0:lengths[i], i].copy_(
                torch.tensor(example, dtype=torch.int64, device=self.device))

        return output, lengths

    def pack_batch(self, batch):
        "Transform batch data to tensors"
        inp, tasks = self.label_encoder.transform(batch)
        (token, char, lengths), (lemma, pos, morph) = inp, tasks

        token = self.pad_batch(token, self.label_encoder.token.get_pad())
        char = self.pad_batch(char, self.label_encoder.char.get_pad())

        if pos is not None:
            pos = self.pad_batch(pos, self.label_encoder.pos.get_pad())
        if lemma is not None:
            lemma = self.pad_batch(lemma, self.label_encoder.lemma.get_pad())

        return (token, char, lengths), (lemma, pos, morph)

    def prepare_buffer(self, buf):
        "Transform buffer into batch generator"

        def key(data):
            inp, tasks = data
            return len(inp.token)

        buf = sorted(buf, key=key, reverse=True)
        batches = list(utils.chunks(buf, self.batch_size))

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield self.pack_batch(batch)

    def batch_generator(self):
        """
        Generator over dataset batches. Each batch is a tuple of (input, tasks):
            * (token, char, lengths)
                - token : tensor(length, batch_size), padded lengths
                - char : tensor(length, batch_size * words), padded lengths
                - lengths : original number of words per sentence (to rearrange
                    character-level embeddings to sentence orm)

            * (lemma, pos, morph), each is a tensor(length, batch_size)
        """
        buf = []
        for data in self.reader.readsents():

            # check if buffer if full and yield
            if len(buf) == self.buffer_size:
                yield from self.prepare_buffer(buf)
                buf = []

            # fill buffer
            buf.append(data)

        if len(buf) > 0:
            yield from self.prepare_buffer(buf)
