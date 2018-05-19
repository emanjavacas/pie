
import os
import glob
import time
import json
import logging
from collections import Counter, defaultdict
import random

import torch

from pie import utils
from pie import consts


class LineParseException(Exception):
    pass


class BaseReader(object):
    """
    Abstract reader class

    Parameters
    ==========
    input_dir : str (optional), root directory with data files
    extension : str (optional), format extension

    Settings
    ========
    shuffle : bool, whether to shuffle files after each iteration

    Attributes
    ===========
    current_sent : int, counter on number of sents processed in total (over all files)
    current_fpath : str, name of the file being currently processed
    """
    def __init__(self, settings, input_dir=None, extension=None):
        input_dir = input_dir or settings.input_dir
        extension = extension or settings.extension
        input_dir = os.path.abspath(input_dir)
        self.filenames = glob.glob(input_dir + '/*.{}'.format(extension))
        if len(self.filenames) == 0:
            raise ValueError("Couldn't find files in {}".format(input_dir))

        # settings
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

    def check_tasks(self, expected=None):
        """
        Returns the union of the tasks across files. If `expected` is passed,
        it is understood to be a subset of all available tasks and it will check
        whether all `expected` are in fact available in all files as per the output
        of `get_tasks`.

        Returns
        ========
        List of tasks available in all files (or a subset if `expected` is passed)
        """
        tasks = None
        for fpath in self.filenames:
            if tasks is None:
                tasks = set(self.get_tasks(fpath))
            else:
                tasks = tasks.intersection(set(self.get_tasks(fpath)))

        if expected is not None:
            diff = set(expected).difference(tasks)
            if diff:
                raise ValueError("Following expected tasks are missing "
                                 "from at least one file: '{}'"
                                 .format('"'.join(diff)))

        return expected or list(tasks)

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


class LineParser(object):
    """
    Inner class to handle sentence breaks
    """
    def __init__(self, tasks, breakline_type, breakline_ref, breakline_data):
        # breakline info
        self.breakline_type = breakline_type
        self.breakline_ref = breakline_ref
        self.breakline_data = breakline_data
        # data
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
            try:
                data = getattr(self, 'process_{}'.format(task.lower()))(data)
            except AttributeError:
                pass
            finally:
                self.tasks[task].append(data)

    def check_breakline(self):
        """
        Check if sentence is finished.
        """
        if self.breakline_ref == 'input':
            ref = self.inp
        else:
            ref = self.tasks[self.breakline_ref]

        if self.breakline_type == 'FULLSTOP':
            if ref[-1] == self.breakline_data:
                return True
        elif self.breakline_type == 'LENGTH':
            if len(ref) == self.breakline_data:
                return True

    def reset(self):
        """
        Reset sentence data
        """
        self.tasks, self.inp = {task: [] for task in self.tasks}, []


class CustomLineParser(LineParser):
    # TODO: parse morphology into some data structure
    def process_morph(self, data):
        pass


class TabReader(BaseReader):
    """
    Reader for files in tab format where each line has annotations for a given word
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
    def __init__(self, settings, line_parser=LineParser, **kwargs):
        super(TabReader, self).__init__(settings, **kwargs)

        self.line_parser = line_parser
        self.breakline_type = settings.breakline_type
        self.breakline_ref = settings.breakline_ref
        self.breakline_data = settings.breakline_data
        self.max_sent_len = settings.max_sent_len
        self.tasks_order = settings.tasks_order

    def parselines(self, fpath, tasks):
        """
        Generator over sentences in a single file
        """
        with open(fpath, 'r+') as f:
            parser = self.line_parser(
                tasks, self.breakline_type, self.breakline_ref, self.breakline_data)

            for line_num, line in enumerate(f):
                line = line.strip()

                if not line:    # avoid empty line
                    continue

                parser.add(line, line_num)

                if parser.check_breakline() or len(parser.inp) >= self.max_sent_len:
                    yield parser.inp, parser.tasks
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
                raise ValueError("Not enough input tasks: [{}]".format(fpath))

            return self.tasks_order[:len(tasks)]


class LabelEncoder(object):
    """
    Label encoder
    """
    def __init__(self, pad=True, eos=True, vocabsize=None, level='word', name='Unk'):
        if level.lower() not in ('word', 'char'):
            raise ValueError("`level` must be 'word' or 'char'")
        self.eos = consts.EOS if eos else None
        self.pad = consts.PAD if pad else None
        self.vocabsize = vocabsize
        self.level = level.lower()
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
        if type(other) != LabelEncoder:
            return False

        return self.pad == other.pad and \
            self.eos == other.eos and \
            self.vocabsize == other.vocabsize and \
            self.level == other.level and \
            self.freqs == other.freqs and \
            self.table == other.table and \
            self.inverse_table == other.inverse_table and \
            self.fitted == other.fitted

    def add(self, sent):
        if self.fitted:
            raise ValueError("Already fitted")

        if self.level == 'word':
            self.freqs.update(sent)
        else:
            self.freqs.update(utils.flatten(sent))

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
                'level': self.level,
                'vocabsize': self.vocabsize,
                'freqs': dict(self.freqs),
                'table': dict(self.table),
                'inverse_table': self.inverse_table}

    @classmethod
    def from_json(cls, obj):
        inst = cls(pad=obj['pad'], eos=obj['eos'],
                   level=obj['level'], vocabsize=obj['vocabsize'])
        inst.freqs = Counter(obj['freqs'])
        inst.table = dict(obj['table'])
        inst.inverse_table = list(obj['inverse_table'])
        inst.fitted = True

        return inst


class MultiLabelEncoder(object):
    """
    Complex Label encoder for all tasks.
    """
    def __init__(self, word_vocabsize=None, char_vocabsize=None):
        self.word = LabelEncoder(vocabsize=word_vocabsize, name='word')
        self.char = LabelEncoder(vocabsize=char_vocabsize, name='char')
        self.tasks = {}

    def add_task(self, name, **kwargs):
        self.tasks[name] = LabelEncoder(name=name, **kwargs)
        return self

    @classmethod
    def from_settings(cls, settings):
        return cls(word_vocabsize=settings.word_vocabsize,
                   char_vocabsize=settings.char_vocabsize)

    def fit(self, lines):
        """
        Parameters
        ===========
        lines : iterator over tuples of (Input, Tasks)
        """
        for idx, (inp, tasks) in enumerate(lines):
            # input
            self.word.add(inp)
            self.char.add(utils.flatten(inp))

            for task, le in self.tasks.items():
                if le.level == 'char':
                    le.add(utils.flatten(tasks[task]))
                else:
                    le.add(tasks[task])

        self.word.compute_vocab()
        self.char.compute_vocab()
        for le in self.tasks.values():
            le.compute_vocab()

        return self

    def transform(self, sents):
        """
        Parameters
        ===========
        sents : list of Example's

        Returns
        ===========
        tuple of (word, char), task_dict

            - word: list of integers
            - char: list of integers where each list represents a word at the
                character level
            - task_dict: Dict to corresponding integer output for each task
        """
        word, char, tasks_dict = [], [], defaultdict(list)

        for inp, tasks in sents:
            # input data
            word.append(self.word.transform(inp))
            for w in inp:
                char.append(self.char.transform(w))

            # task data
            for task, le in self.tasks.items():
                if le.level == 'word':
                    tasks_dict[task].append(le.transform(tasks[task]))
                else:
                    for w in tasks[task]:
                        tasks_dict[task].append(le.transform(w))

        return (word, char), tasks_dict

    def save(self, path):
        with open(path, 'w+') as f:
            obj = {'word': self.word.jsonify(),
                   'char': self.char.jsonify(),
                   'tasks': {le.name: le.jsonify() for le in self.tasks.values()}}
            json.dump(obj, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r+') as f:
            obj = json.load(f)

        inst = cls()  # dummy instance to overwrite

        inst.word = LabelEncoder.from_json(obj['word'])
        inst.char = LabelEncoder.from_json(obj['char'])

        for task, le_obj in obj['tasks'].items():
            inst.tasks[task] = LabelEncoder.from_json(le_obj)

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
    def __init__(self, settings, reader=None, label_encoder=None, expected_tasks=None):
        if settings.batch_size > settings.buffer_size:
            raise ValueError("Not enough buffer capacity {} for batch_size of {}"
                             .format(settings.buffer_size, settings.batch_size))

        # attributes
        self.buffer_size = settings.buffer_size
        self.batch_size = settings.batch_size
        self.device = settings.device
        self.shuffle = settings.shuffle

        # data
        # TODO: this assumes TabReader
        self.reader = reader or TabReader(settings)
        tasks = self.reader.check_tasks(expected=expected_tasks)
        # label encoder
        if label_encoder is None:
            label_encoder = MultiLabelEncoder.from_settings(settings)
            for task in tasks:
                label_encoder.add_task(task, **settings.tasks[task])
            if settings.verbose:
                print("\n::: Fitting data... :::\n")
            start = time.time()
            label_encoder.fit(self.reader.readsents())
            if settings.verbose:
                print("\tDone in {:g} secs".format(time.time() - start))
        if settings.verbose:
            print("\n::: Available tasks :::\n")
            for task in tasks:
                print("\t{}".format(task))
            print()
        self.label_encoder = label_encoder

    def get_nelement(self, batch):
        """
        Returns the number of elements in a batch (based on word-level length)
        """
        return batch[0][0][1].sum().item()

    def pad_batch(self, batch, padding_id):
        """
        Pad batch into tensor
        """
        lengths = [len(example) for example in batch]
        maxlen, batch_size = max(lengths), len(batch)
        output = torch.zeros(
            maxlen, batch_size, device=self.device, dtype=torch.int64
        ) + padding_id

        for i, example in enumerate(batch):
            output[0:lengths[i], i].copy_(
                torch.tensor(example, dtype=torch.int64, device=self.device))

        lengths = torch.tensor(lengths, dtype=torch.int64, device=self.device)

        return output, lengths

    def pack_batch(self, batch):
        """
        Transform batch data to tensors
        """
        (word, char), tasks = self.label_encoder.transform(batch)

        word = self.pad_batch(word, self.label_encoder.word.get_pad())
        char = self.pad_batch(char, self.label_encoder.char.get_pad())

        output_tasks = {}
        for task, data in tasks.items():
            output_tasks[task] = self.pad_batch(
                data, self.label_encoder.tasks[task].get_pad())

        return (word, char), output_tasks

    def prepare_buffer(self, buf):
        "Transform buffer into batch generator"

        def key(data):
            inp, tasks = data
            return len(inp)

        buf = sorted(buf, key=key, reverse=True)
        batches = list(utils.chunks(buf, self.batch_size))

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield self.pack_batch(batch)

    def batch_generator(self):
        """
        Generator over dataset batches. Each batch is a tuple of (input, tasks):
            * (word, char)
                - word : tensor(length, batch_size), padded lengths
                - char : tensor(length, batch_size * words), padded lengths
            * (tasks) dictionary with tasks
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
