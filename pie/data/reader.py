
import os
import random

from pie.utils import get_filenames

from .tabreader import TabReader
from .conll_reader import CONLLReader, CONLLUReader
from .tei_reader import TEIReader


class Reader(object):
    """
    Dispatcher class

    Parameters
    ==========
    input_path : str (optional), either a path to a directory, a path to a file
        or a unix style pathname pattern expansion for glob

    Settings
    ========
    shuffle : bool, whether to shuffle files after each iteration
    max_sents : int, maximum number of sentences to read (note that depending
        on shuffle the result might be non-determinitic)
    """
    def __init__(self, settings, *input_paths):
        filenames = []
        for input_path in input_paths:
            if input_path is not None:
                filenames.extend(get_filenames(input_path))

        if len(filenames) == 0:
            raise RuntimeError("Couldn't find files [{}]".format(''.join(input_paths)))

        self.readers = [self.get_reader(fpath)(settings, fpath) for fpath in filenames]

        # settings
        self.shuffle = settings.shuffle
        self.max_sents = settings.max_sents
        # cache
        self.nsents = None

    def get_reader(self, fpath):
        """
        Decide on reader type based on filename
        """
        # imports

        if fpath.endswith('tab') or fpath.endswith('tsv') or fpath.endswith('csv'):
            return TabReader

        elif fpath.endswith('conll'):
            return CONLLReader

        elif fpath.endswith('conllu'):
            return CONLLUReader

        elif fpath.endswith('xml'):
            return TEIReader

        else:
            raise ValueError("Unknown file format: {}".format(fpath))

    def reset(self):
        """
        Called after a full run over `readsents`
        """
        if self.shuffle:
            random.shuffle(self.readers)

    def check_tasks(self, expected=None):
        """
        Check tasks over files
        """
        tasks = set()

        for reader in self.readers:
            tasks.update(reader.check_tasks(expected=expected))

        return tuple(tasks)

    def readsents(self, silent=True, only_tokens=False):
        """
        Read sents over files
        """
        self.reset()
        total = 0
        for reader in self.readers:
            for data in reader.readsents(silent=silent, only_tokens=only_tokens):
                # check # lines processed
                if total >= self.max_sents:
                    break
                total += 1

                yield data
        self.nsents = total

    def get_nsents(self):
        """
        Number of sents in Reader
        """
        if self.nsents is not None:
            return self.nsents
        nsents = 0
        for _ in self.readsents():
            nsents += 1
        self.nsents = nsents
        return nsents

    def get_token_iterator(self):
        """
        Get an iterator over sentences of plain tokens
        """
        return TokenIterator(self)


class TokenIterator():
    def __init__(self, reader):
        self.reader = reader

    def __iter__(self):
        yield from self.reader.readsents(only_tokens=True)
