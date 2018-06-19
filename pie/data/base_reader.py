
import logging


class LineParseException(Exception):
    pass


class MissingDefaultException(Exception):
    def __init__(self, task):
        self.task = task


class BaseReader(object):
    """
    Abstract reader class
    """
    def __init__(self, settings, fpath):
        self.fpath = fpath
        self.tasks = tuple(self.get_tasks())
        self.tasks_defaults = {task['name']: task.get("default")
                               for task in settings.tasks if task['name'] in self.tasks}

    def get_default(self, task, value):
        """
        Get default value for task if given
        """
        default = self.tasks_defaults.get(task)
        if default is not None:
            if default.lower() == 'copy':
                return value
            return default

        raise MissingDefaultException(task)

    def readsents(self, silent=True, only_tokens=False):
        """
        Generator over dataset sentences. Each output is a tuple of (Input, Tasks)
        objects with, where each entry is a list of strings.
        """
        current_sent, lines = 0, self.parselines()

        while True:
            try:
                line = next(lines)
                current_sent += 1
                # check if parse exception
                if isinstance(line, LineParseException):
                    if not silent:
                        logging.warning(
                            "Parse error at [{}:sent={}]\n  => {}"
                            .format(self.fpath, current_sent + 1, str(line)))
                    continue

                inp, tasks = line
                if only_tokens:
                    yield inp
                else:
                    yield (self.fpath, current_sent), (inp, tasks)

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
