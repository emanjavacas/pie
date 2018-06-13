
import json
import logging
from collections import Counter, defaultdict
import random

import torch

from pie import utils, torch_utils, constants


class LabelEncoder(object):
    """
    Label encoder
    """
    def __init__(self, level='word', target=None, name=None,
                 pad=True, eos=False, bos=False,
                 max_size=None, min_freq=1, **kwargs):

        if level.lower() not in ('word', 'char'):
            raise ValueError("`level` must be 'word' or 'char'")

        self.eos = constants.EOS if eos else None
        self.pad = constants.PAD if pad else None
        self.bos = constants.BOS if bos else None
        self.max_size = max_size
        self.min_freq = min_freq
        self.level = level.lower()
        self.target = target
        self.name = name
        self.reserved = (constants.UNK,)  # always use <unk>
        self.reserved += tuple([sym for sym in [self.eos, self.pad, self.bos] if sym])
        self.freqs = Counter()
        self.known_tokens = set()  # for char-level dicts, keep word-level known tokens
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
            self.bos == other.bos and \
            self.max_size == other.max_size and \
            self.level == other.level and \
            self.target == other.target and \
            self.freqs == other.freqs and \
            self.table == other.table and \
            self.inverse_table == other.inverse_table and \
            self.fitted == other.fitted

    def __repr__(self):
        try:
            length = len(self)
        except Exception:
            length = 0

        return (
            '<LabelEncoder name="{}" target="{}" level="{}" vocab="{}" fitted="{}">'
        ).format(self.name, self.target, self.level, length, self.fitted)

    def add(self, seq):
        if self.fitted:
            raise ValueError("Already fitted")

        if self.level == 'word':
            self.freqs.update(seq)
        else:
            self.freqs.update(utils.flatten(seq))
            self.known_tokens.update(seq)

    def compute_vocab(self):
        if self.fitted:
            raise ValueError("Cannot compute vocabulary, already fitted")

        if len(self.freqs) == 0:
            logging.warning("Computing vocabulary for empty encoder {}"
                            .format(self.name))

        if self.max_size:
            most_common = self.freqs.most_common(n=self.max_size)
        else:
            most_common = [it for it in self.freqs.items() if it[1] >= self.min_freq]
        self.inverse_table = list(self.reserved) + [sym for sym, _ in most_common]
        self.table = {sym: idx for idx, sym in enumerate(self.inverse_table)}
        self.fitted = True

    def transform(self, seq):
        if not self.fitted:
            raise ValueError("Vocabulary hasn't been computed yet")

        output = []
        if self.bos:
            output.append(self.get_bos())

        output += [self.table.get(tok, self.table[constants.UNK]) for tok in seq]

        if self.eos:
            output.append(self.get_eos())

        return output

    def inverse_transform(self, seq):
        if not self.fitted:
            raise ValueError("Vocabulary hasn't been computed yet")

        return [self.inverse_table[i] for i in seq]

    def stringify(self, seq, length=None):
        if not self.fitted:
            raise ValueError("Vocabulary hasn't been computed yet")

        # compute length based on <eos>
        if length is None:
            eos = self.get_eos()
            if eos is None:
                raise ValueError("Don't know how to compute input length")
            try:
                length = seq.index(eos)
            except ValueError:  # eos not found in input
                length = -1

        seq = seq[:length]

        # eventually remove <bos> if required
        if self.get_bos() is not None:
            if len(seq) > 0 and seq[0] == self.get_bos():
                seq = seq[1:]

        seq = self.inverse_transform(seq)

        return seq

    def _get_sym(self, sym):
        if not self.fitted:
            raise ValueError("Vocabulary hasn't been computed yet")

        return self.table.get(sym)

    def get_pad(self):
        return self._get_sym(constants.PAD)

    def get_eos(self):
        return self._get_sym(constants.EOS)

    def get_bos(self):
        return self._get_sym(constants.BOS)

    def jsonify(self):
        if not self.fitted:
            raise ValueError("Attempted to serialize unfitted encoder")

        return {'name': self.name,
                'eos': self.eos,
                'bos': self.bos,
                'pad': self.pad,
                'level': self.level,
                'target': self.target,
                'max_size': self.max_size,
                'min_freq': self.min_freq,
                'freqs': dict(self.freqs),
                'table': dict(self.table),
                'inverse_table': self.inverse_table,
                'known_tokens': list(self.known_tokens)}

    @classmethod
    def from_json(cls, obj):
        inst = cls(pad=obj['pad'], eos=obj['eos'], bos=obj['bos'],
                   level=obj['level'], target=obj['target'],
                   max_size=obj['max_size'], min_freq=['min_freq'], name=obj['name'])
        inst.freqs = Counter(obj['freqs'])
        inst.table = dict(obj['table'])
        inst.inverse_table = list(obj['inverse_table'])
        inst.known_tokens = set(obj['known_tokens'])
        inst.fitted = True

        return inst


class MultiLabelEncoder(object):
    """
    Complex Label encoder for all tasks.
    """
    def __init__(self, word_max_size=None, char_max_size=None,
                 word_min_freq=1, char_min_freq=None):
        self.word = LabelEncoder(max_size=word_max_size, min_freq=word_min_freq,
                                 name='word')
        self.char = LabelEncoder(max_size=char_max_size, min_freq=char_min_freq,
                                 name='char', level='char', eos=True, bos=True)
        self.tasks = {}

    def __repr__(self):
        return (
            '<MultiLabelEncoder>\n\t' +
            '\n\t'.join(map(str, [self.word, self.char] + list(self.tasks.values()))) +
            '\n</MultiLabelEncoder>')

    def __eq__(self, other):
        if not (self.word == other.word and self.char == other.char):
            return False

        for task in self.tasks:
            if task not in other.tasks:
                return False
            if self.tasks[task] != other.tasks[task]:
                return False

        return True

    def add_task(self, name, **kwargs):
        self.tasks[name] = LabelEncoder(name=name, **kwargs)
        return self

    @classmethod
    def from_settings(cls, settings, tasks=None):
        le = cls(word_max_size=settings.word_max_size,
                 word_min_freq=settings.word_min_freq,
                 char_max_size=settings.char_max_size,
                 char_min_freq=settings.char_min_freq)

        for task in settings.tasks:
            task_settings = task.get("settings", {})
            task_target = task_settings.get('target', task['name'])
            if tasks is not None and task_target not in tasks:
                logging.warning("Ignoring task [{}]: no available data"
                                .format(task_target))
                continue

            task_settings['target'] = task_target
            le.add_task(task['name'], **task_settings)

        return le

    def fit(self, lines):
        """
        Parameters
        ===========
        lines : iterator over tuples of (Input, Tasks)
        """
        ninsts = 0
        for idx, (inp, tasks) in enumerate(lines):
            # input
            self.word.add(inp)
            self.char.add(inp)

            for le in self.tasks.values():
                le.add(tasks[le.target])

            # increment counter
            ninsts += 1

        self.word.compute_vocab()
        self.char.compute_vocab()
        for le in self.tasks.values():
            le.compute_vocab()

        return ninsts

    def fit_reader(self, reader):
        """
        fit reader in a non verbose way (to warn about parsing issues)
        """
        return self.fit(line for (_, line) in reader.readsents(silent=False))

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
            if tasks is None:
                # during inference there is no task data (pass None)
                continue

            for le in self.tasks.values():
                if le.level == 'word':
                    tasks_dict[le.name].append(le.transform(tasks[le.target]))
                else:
                    for w in tasks[le.target]:
                        tasks_dict[le.name].append(le.transform(w))

        return (word, char), tasks_dict

    def jsonify(self):
        return {'word': self.word.jsonify(),
                'char': self.char.jsonify(),
                'tasks': {le.name: le.jsonify() for le in self.tasks.values()}}

    def save(self, path):
        with open(path, 'w+') as f:
            json.dump(self.jsonify(), f)

    @staticmethod
    def _init(inst, obj):
        inst.word = LabelEncoder.from_json(obj['word'])
        inst.char = LabelEncoder.from_json(obj['char'])

        for task, le in obj['tasks'].items():
            inst.tasks[task] = LabelEncoder.from_json(le)

        return inst

    @classmethod
    def load_from_string(cls, string):
        inst, obj = cls(), json.loads(string)
        return cls._init(inst, obj)

    @classmethod
    def load_from_file(cls, path):
        with open(path, 'r+') as f:
            obj = json.load(f)

        inst = cls()  # dummy instance to overwrite
        return cls._init(inst, obj)


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
    input_path : str (optional), either a path to a directory, a path to a file
        or a unix style pathname pattern expansion for glob. If given, the
        input_path in settings will be overwritten by the new value
    label_encoder : optional, prefitted LabelEncoder object
    """
    def __init__(self, settings, reader, label_encoder):

        if settings.batch_size > settings.buffer_size:
            raise ValueError("Not enough buffer capacity {} for batch_size of {}"
                             .format(settings.buffer_size, settings.batch_size))

        # attributes
        self.buffer_size = settings.buffer_size
        self.batch_size = settings.batch_size
        self.device = settings.device
        self.shuffle = settings.shuffle

        # data
        self.dev_sents = defaultdict(set)
        self.reader = reader
        self.label_encoder = label_encoder

    @staticmethod
    def get_nelement(batch):
        """
        Returns the number of elements in a batch (based on word-level length)
        """
        return batch[0][0][1].sum().item()

    def pack_batch(self, batch, device=None):
        """
        Transform batch data to tensors
        """
        return pack_batch(self.label_encoder, batch, device or self.device)

    def prepare_buffer(self, buf, **kwargs):
        "Transform buffer into batch generator"

        def key(data):
            inp, tasks = data
            return len(inp)

        buf = sorted(buf, key=key, reverse=True)
        batches = list(utils.chunks(buf, self.batch_size))

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield self.pack_batch(batch, **kwargs)

    def get_dev_split(self, ninsts, split=0.05):
        "Grab a dev split from the dataset"
        if len(self.dev_sents) > 0:
            raise RuntimeError("A dev-split has already been created!")

        buf = []
        split_size = ninsts * split
        split_prop = split_size / ninsts

        for sent in self.reader.readsents():
            if len(buf) >= split_size:
                break

            if random.random() > split_prop:
                continue

            (fpath, line_num), data = sent
            buf.append(data)
            self.dev_sents[fpath].add(line_num)

        # get batches on cpu
        batches = list(self.prepare_buffer(buf, device='cpu'))

        return device_wrapper(batches, device=self.device)

    def batch_generator(self):
        """
        Generator over dataset batches. Each batch is a tuple of (input, tasks):
            * (word, char)
                - word : tensor(length, batch_size), padded lengths
                - char : tensor(length, batch_size * words), padded lengths
            * (tasks) dictionary with tasks
        """
        buf = []
        for (fpath, line_num), data in self.reader.readsents():

            # check if buffer is full and yield
            if len(buf) == self.buffer_size:
                yield from self.prepare_buffer(buf)
                buf = []

            # don't use dev sentences
            if fpath in self.dev_sents and line_num in self.dev_sents[fpath]:
                continue

            # fill buffer
            buf.append(data)

        if len(buf) > 0:
            yield from self.prepare_buffer(buf)


def pack_batch(label_encoder, batch, device=None):
    """
    Transform batch data to tensors
    """
    (word, char), tasks = label_encoder.transform(batch)

    word = torch_utils.pad_batch(word, label_encoder.word.get_pad(), device=device)
    char = torch_utils.pad_batch(char, label_encoder.char.get_pad(), device=device)

    output_tasks = {}
    for task, data in tasks.items():
        output_tasks[task] = torch_utils.pad_batch(
            data, label_encoder.tasks[task].get_pad(), device=device)

    return (word, char), output_tasks


def wrap_device(it, device):
    for i in it:
        if isinstance(i, torch.Tensor):
            yield i.to(device)
        elif isinstance(i, dict):
            yield {k: tuple(wrap_device(v, device)) for k, v in i.items()}
        else:
            yield tuple(wrap_device(i, device))


class device_wrapper(object):
    def __init__(self, batches, device):
        self.batches = batches
        self.device = device

    def __getitem__(self, idx):
        return tuple(wrap_device(self.batches[idx], self.device))

    def __len__(self):
        return len(self.batches)
