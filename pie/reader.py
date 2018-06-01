
import os
import logging
import random


class LineParseException(Exception):
    pass


class BaseReader(object):
    """
    Abstract reader class
    """
    def __init__(self, settings, fpath):
        self.fpath = fpath
        self.tasks = tuple(self.get_tasks())

    def readsents(self):
        """
        Generator over dataset sentences. Each output is a tuple of (Input, Tasks)
        objects with, where each entry is a list of strings.
        """
        current_sent, lines = 0, self.parselines()

        while True:
            try:
                yield (self.fpath, current_sent), next(lines)
                current_sent += 1

            except LineParseException as e:
                logging.warning(
                    "Parse error at [{}:sent={}]\n  => {}"
                    .format(self.fpath, current_sent + 1, str(e)))
                continue

            except StopIteration:
                break

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
        if expected is not None:
            diff = set(expected).difference(set(self.tasks))
            if diff:
                raise ValueError("Following expected tasks are missing "
                                 "from at least one file: '{}'"
                                 .format('"'.join(diff)))

        return tuple(expected or self.tasks)

    def parselines(self):
        """
        Generator over tuples of (Input, Tasks) holding input and target data
        as lists of strings. Some tasks might be missing from a file, in which
        case the corresponding value is None.

        ParseError can be thrown if an issue is encountered during processing.
        The issue can be documented by passing an error message as second argument
        to ParseError
        """
        raise NotImplementedError

    def get_tasks(self):
        """
        Reader is responsible for extracting the expected tasks in the file.

        Returns
        =========
        Set of tasks in a file
        """
        raise NotImplementedError


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
            raise ValueError("Couldn't find files in {}".format(input_path))

        self.readers = [self.get_reader(fpath)(settings, fpath) for fpath in filenames]

        # settings
        self.shuffle = settings.shuffle

    def get_reader(self, fpath):
        """
        Decide on reader type based on filename
        """
        # imports

        if fpath.endswith('tab') or fpath.endswith('tsv') or fpath.endswith('csv'):
            from pie.tabreader import TabReader
            return TabReader

        elif fpath.endswith('conll'):
            from pie.conll_reader import CONLLReader
            return CONLLReader

        else:
            raise ValueError("Unknown file format {}".format(fpath))

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

        for reader in self.readers:

            yield from reader.readsents()
