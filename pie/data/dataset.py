
import warnings
from functools import partial
import tarfile
import json
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import logging
from collections import Counter, defaultdict
import random

import torch

from pie import utils, torch_utils, constants
from . import preprocessors


class LabelEncoder(object):
    """
    Label encoder
    """
    def __init__(self, level='token', name=None, target=None,
                 lower=False, utfnorm=False, utfnorm_type='NFKD', drop_diacritics=False,
                 preprocessor=None, max_size=None, min_freq=1,
                 pad=True, eos=False, bos=False, reserved=(), **meta):

        if level.lower() not in ('token', 'char'):
            raise ValueError("`level` must be 'token' or 'char'. Got ", level)

        self.meta = meta  # dictionary with other task-relevant information
        self.pad = constants.PAD if pad else None
        self.eos = constants.EOS if eos else None
        self.bos = constants.BOS if bos else None
        self.lower = lower
        self.utfnorm = utfnorm
        self.utfnorm_type = utfnorm_type
        self.drop_diacritics = drop_diacritics
        self.text_preprocess_fn = None
        if lower or utfnorm or drop_diacritics:
            self.text_preprocess_fn = self._get_text_preprocess_fn(
                lower, utfnorm, utfnorm_type, drop_diacritics)
        self.preprocessor = preprocessor
        self.preprocessor_fn = \
            getattr(preprocessors, preprocessor) if preprocessor else None
        self.max_size = max_size
        self.min_freq = min_freq
        self.level = level.lower()
        self.target = target
        self.name = name
        self.reserved = reserved + (constants.UNK,)  # always use <unk>
        self.reserved += tuple([sym for sym in [self.eos, self.pad, self.bos] if sym])
        self.freqs = Counter()
        self.known_tokens = set()  # for char-level dicts, keep word-level known tokens
        self.table = None
        self.inverse_table = None
        self.fitted = False

    def _get_text_preprocess_fn(self, lower, utfnorm, utfnorm_type, drop_diacritics):
        fns = []
        if lower:
            fns.append(utils.lower_str)
        if utfnorm:
            fns.append(partial(utils.apply_utfnorm, form=utfnorm_type))
        if drop_diacritics:
            fns.append(utils.drop_diacritics)

        return utils.compose(*fns)

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
            self.preprocessor == other.preprocessor and \
            self.max_size == other.max_size and \
            self.level == other.level and \
            self.lower == other.lower and \
            self.utfnorm == other.utfnorm and \
            self.drop_diacritics == other.drop_diacritics and \
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
            '<LabelEncoder name="{}" lower="{}" utfnorm="{}" utfnorm_type="{}" ' +
            'target="{}" vocab="{}" level="{}" fitted="{}"/>'
        ).format(
            self.name, self.lower, self.utfnorm, self.utfnorm_type,
            self.target, self.level, length, self.fitted)

    def get_type_stats(self):
        """
        Compute number of known types, total number of types and ratio
        """
        if not self.fitted:
            raise ValueError("Vocabulary hasn't been computed yet")

        total_types = len(self.freqs)
        known_types = len(self) - len(self.reserved)
        return known_types, total_types, known_types / total_types

    def get_token_stats(self):
        """
        Compute number of known tokens, total number of types and ratio
        """
        if not self.fitted:
            raise ValueError("Vocabulary hasn't been computed yet")

        total_tokens = sum(self.freqs.values())
        known_tokens = sum(self.freqs[w] for w in self.table)
        return known_tokens, total_tokens, known_tokens / total_tokens

    def add(self, seq, rseq=None):
        if self.fitted:
            raise ValueError("Already fitted")

        postseq = self.preprocess(seq, rseq)

        if self.level == 'token':
            self.freqs.update(postseq)
        else:
            self.freqs.update(c for tok in postseq for c in tok)
            # always use original sequence for known tokens
            self.known_tokens.update(seq)

    def compute_vocab(self):
        if self.fitted:
            raise ValueError("Cannot compute vocabulary, already fitted")

        if len(self.freqs) == 0:
            logging.warning("Computing vocabulary for empty encoder {}"
                            .format(self.name))

        if self.max_size:
            most_common = self.freqs.most_common(n=self.max_size)
        elif self.min_freq:
            most_common = [it for it in self.freqs.items() if it[1] >= self.min_freq]
        else:
            most_common = self.freqs.most_common()

        self.inverse_table = list(self.reserved) + [sym for sym, _ in most_common]
        self.table = {sym: idx for idx, sym in enumerate(self.inverse_table)}
        self.fitted = True

    def preprocess_text(self, seq):
        """
        Apply surface level preprocessing such as lowering, unicode normalization
        """
        if self.text_preprocess_fn:
            seq = list(map(self.text_preprocess_fn, seq))
        return seq

    def preprocess(self, tseq, rseq=None):
        """
        Full preprocessing pipeline including possible token-level transformations
        """
        tseq = self.preprocess_text(tseq)

        if self.preprocessor_fn is not None:
            if rseq is None:
                raise ValueError("Expected ref sequence for preprocessor")

            return [self.preprocessor_fn.transform(t, r) for t, r in zip(tseq, rseq)]

        return tseq

    def transform(self, seq, rseq=None):
        if not self.fitted:
            raise ValueError("Vocabulary hasn't been computed yet")

        def transform_seq(s):
            output = []
            if self.bos:
                output.append(self.get_bos())

            for tok in s:
                output.append(self.table.get(tok, self.table[constants.UNK]))

            if self.eos:
                output.append(self.get_eos())

            return output

        # preprocess
        seq = self.preprocess(seq, rseq)

        if self.level == 'token':
            output = transform_seq(seq)
        else:
            output = [transform_seq(w) for w in seq]

        return output

    def inverse_transform(self, seq):
        if not self.fitted:
            raise ValueError("Vocabulary hasn't been computed yet")

        return [self.inverse_table[i] for i in seq]

    def stringify(self, seq, length=None):
        if not self.fitted:
            raise ValueError("Vocabulary hasn't been computed yet")

        eos, bos = self.get_eos(), self.get_bos()
        if length is not None:
            if eos is not None or bos is not None:
                warnings.warn("Length was passed to stringify but LabelEncoder "
                              "has <eos> and/or <bos> tokens")
            seq = seq[:length]
        else:
            if eos is None:
                raise ValueError("Don't know how to compute input length")
            try:
                # some generations might fail to produce the <eos> symbol
                seq = seq[:seq.index(eos)]
            except ValueError:
                pass

            # eventually remove <bos> if required
            if bos is not None:
                if len(seq) > 0 and seq[0] == bos:
                    seq = seq[1:]

        return self.inverse_transform(seq)

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

    def get_unk(self):
        return self._get_sym(constants.UNK)

    def jsonify(self):
        if not self.fitted:
            raise ValueError("Attempted to serialize unfitted encoder")

        return {'name': self.name,
                'eos': self.eos,
                'bos': self.bos,
                'pad': self.pad,
                'meta': self.meta,
                'level': self.level,
                'preprocessor': self.preprocessor,
                'lower': self.lower,
                'utfnorm': self.utfnorm,
                'utfnorm_type': self.utfnorm_type,
                'drop_diacritics': self.drop_diacritics,
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
                   level=obj['level'], target=obj['target'], lower=obj['lower'],
                   max_size=obj['max_size'], min_freq=['min_freq'],
                   drop_diacritics=obj.get('drop_diacritics', False),
                   utfnorm=obj.get('utfnorm', False),
                   utfnorm_type=obj.get('utfnorm_type', False),
                   preprocessor=obj.get('preprocessor'),
                   name=obj['name'], meta=obj.get('meta', {}))
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
    def __init__(self, word_max_size=None, word_min_freq=1, word_lower=False,
                 char_max_size=None, char_min_freq=None, char_lower=False,
                 char_eos=True, char_bos=True, utfnorm=False, utfnorm_type='NFKD',
                 drop_diacritics=False):
        self.word = LabelEncoder(max_size=word_max_size, min_freq=word_min_freq,
                                 lower=word_lower, utfnorm=utfnorm,
                                 utfnorm_type=utfnorm_type,
                                 drop_diacritics=drop_diacritics, name='word')
        self.char = LabelEncoder(max_size=char_max_size, min_freq=char_min_freq,
                                 level='char', lower=char_lower, name='char',
                                 eos=char_eos, bos=char_bos, utfnorm_type=utfnorm_type,
                                 utfnorm=utfnorm, drop_diacritics=drop_diacritics)
        self.tasks = {}
        self.nsents = None

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

    def add_task(self, name, **meta):
        self.tasks[name] = LabelEncoder(name=name, **meta)

        # check <eos> <bos> (not suitable for linear models)
        if meta['level'].lower() != 'char' and (meta.get('eos') or meta.get('bos')):
            raise ValueError(
                ('[Task: {task}] => `bos` and `eos` options are '
                 'only compatible with char-level tasks but got '
                 'level: "{level}". Aborting!!!').format(
                    task=name, level=meta['level']))

        return self

    @classmethod
    def from_settings(cls, settings, tasks=None):
        le = cls(word_max_size=settings.word_max_size,
                 word_min_freq=settings.word_min_freq,
                 word_lower=settings.word_lower,
                 char_max_size=settings.char_max_size,
                 char_min_freq=settings.char_min_freq,
                 char_lower=settings.char_lower,
                 char_eos=settings.char_eos,
                 char_bos=settings.char_bos,
                 utfnorm=settings.utfnorm,
                 utfnorm_type=settings.utfnorm_type,
                 drop_diacritics=settings.drop_diacritics)

        for task in settings.tasks:
            if tasks is not None and task['settings']['target'] not in tasks:
                raise ValueError("No available data for task [{}]".format(
                    task['settings']['target']))
            le.add_task(task['name'], level=task['level'], **task['settings'])

        return le

    def fit(self, lines):
        """
        Parameters
        ===========
        lines : iterator over tuples of (Input, Tasks)
        """
        for idx, inp in enumerate(lines):
            tasks = None
            if isinstance(inp, tuple):
                inp, tasks = inp

            # input
            self.word.add(inp)
            self.char.add(inp)

            for le in self.tasks.values():
                le.add(tasks[le.target], inp)

        self.word.compute_vocab()
        self.char.compute_vocab()
        for le in self.tasks.values():
            le.compute_vocab()

        return self

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

        for inp in sents:
            tasks = None

            # task might not be passed
            if isinstance(inp, tuple):
                inp, tasks = inp

            # input data
            word.append(self.word.transform(inp))
            char.extend(self.char.transform(inp))

            # task data
            if tasks is None:
                # during inference there is no task data (pass None)
                continue

            for le in self.tasks.values():
                task_data = le.transform(tasks[le.target], inp)
                # add data
                if le.level == 'char':
                    tasks_dict[le.name].extend(task_data)
                else:
                    tasks_dict[le.name].append(task_data)

        return (word, char), tasks_dict

    def jsonify(self):
        return {'word': self.word.jsonify(),
                'char': self.char.jsonify(),
                'tasks': {le.name: le.jsonify() for le in self.tasks.values()}}

    def save(self, path):
        with open(path, 'w+') as f:
            yaml.dump(self.jsonify(), f, Dumper=Dumper)

    @staticmethod
    def _init(inst, obj):
        inst.word = LabelEncoder.from_json(obj['word'])
        inst.char = LabelEncoder.from_json(obj['char'])

        for task, le in obj['tasks'].items():
            inst.tasks[task] = LabelEncoder.from_json(le)

        return inst

    @classmethod
    def load_from_string(cls, string):
        inst = cls()
        try:
            obj = json.loads(string)
        except ValueError:      # use yaml
            obj = yaml.load(string, Loader=Loader)
        return cls._init(inst, obj)

    @classmethod
    def load_from_file(cls, path):
        with open(path, 'r+') as f:
            return cls.load_from_string(f.read())

    @classmethod
    def load_from_pretrained_model(cls, path):
        with tarfile.open(utils.ensure_ext(path, 'tar'), 'r') as tar:
            return cls.load_from_string(utils.get_gzip_from_tar(tar, 'label_encoder'))


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
    minimize_pad : bool, whether to pack batches with sentences of similar length
       in order to minimize padding.

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
        self.minimize_pad = settings.minimize_pad
        self.cache_dataset = settings.cache_dataset

        # data
        self.reader = reader
        self.label_encoder = label_encoder
        self.cached = []

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

    def prepare_buffer(self, buf, return_raw=False, **kwargs):
        "Transform buffer into batch generator"

        def key(data):
            inp, tasks = data
            return len(inp)

        if self.minimize_pad:
            buf = sorted(buf, key=key, reverse=True)
        elif self.shuffle:
            random.shuffle(buf)

        batches = list(utils.chunks(buf, self.batch_size))

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            packed = self.pack_batch(batch, **kwargs)
            if return_raw:
                inp, tasks = zip(*batch)
                yield packed, (inp, tasks)
            else:
                yield packed

    def batch_generator(self, return_raw=False):
        """
        Generator over dataset batches. Each batch is a tuple of (input, tasks):
            * (word, char)
                - word : tensor(length, batch_size), padded lengths
                - char : tensor(length, batch_size * words), padded lengths
            * (tasks) dictionary with tasks
        """
        if self.cache_dataset:
            if not self.cached:
                self.cache_batches()
            if self.shuffle:
                random.shuffle(self.cached)

            for batch, raw in self.cached:
                # move to device
                batch = tuple(list(wrap_device(batch, self.device)))
                if return_raw:
                    yield batch, raw
                else:
                    yield batch
        else:
            yield from self.batch_generator_(return_raw=return_raw)

    def batch_generator_(self, return_raw=False):
        buf = []
        for (fpath, line_num), data in self.reader.readsents():

            # fill buffer
            buf.append(data)

            # check if buffer is full and yield
            if len(buf) == self.buffer_size:
                yield from self.prepare_buffer(buf, return_raw=return_raw)
                buf = []

        if len(buf) > 0:
            yield from self.prepare_buffer(buf, return_raw=return_raw)

    def cache_batches(self):
        if self.cached:
            return

        buf = [data for _, data in self.reader.readsents()]
        for batch, raw in self.prepare_buffer(buf, return_raw=True, device='cpu'):
            self.cached.append((batch, raw))


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
