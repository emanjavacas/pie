
import os
import glob
import logging
import random

from .tabreader import TabReader
from .conll_reader import CONLLReader


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
    """
    def __init__(self, settings, input_path):
        input_path = input_path or settings.input_path

        if os.path.isdir(input_path):
            filenames = [os.path.join(input_path, f)
                         for f in os.listdir(input_path)
                         if not f.startswith('.')]
        elif os.path.isfile(input_path):
            filenames = [input_path]
        else:
            filenames = glob.glob(input_path)

        if len(filenames) == 0:
            raise ValueError("Couldn't find files in: \"{}\"".format(input_path))

        self.readers = [self.get_reader(fpath)(settings, fpath) for fpath in filenames]

        # settings
        self.shuffle = settings.shuffle
        self.max_sents = settings.max_sents

    def get_reader(self, fpath):
        """
        Decide on reader type based on filename
        """
        # imports

        if fpath.endswith('tab') or fpath.endswith('tsv') or fpath.endswith('csv'):
           return TabReader

        elif fpath.endswith('conll'):
            return CONLLReader

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

    def readsents(self):
        """
        Read sents over files
        """
        self.reset()
        total = 0
        for reader in self.readers:
            for data in reader.readsents():
                # check # lines processed
                if total >= self.max_sents:
                    break
                total += 1

                yield data
