import re
from .base_reader import BaseReader, LineParseException, MissingDefaultException


class LineParser(object):
    """
    Inner class to handle sentence breaks
    """
    def __init__(self, tasks, sep, breakline_ref, breakline_data, reader):
        self.reader = reader
        self.sep = sep
        # breakline info
        self.breakline_ref = breakline_ref
        self.breakline_data = re.compile(breakline_data)
        # data
        self.inp = []
        self.tasks = {task: [] for task in tasks}

    def add(self, line, linenum):
        """
        Adds line to current sentence.
        """
        inp, *tasks = line.split(self.sep)

        if len(tasks) < len(self.tasks):
            try:
                tasks = [self.reader.get_default(task, inp) for task in self.tasks]
            except MissingDefaultException:
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
        if len(self.inp) == 0:
            return False

        if self.breakline_ref:
            ref = self.inp
            if self.breakline_ref != 'input':
                ref = self.tasks[self.breakline_ref]
            return self.breakline_data.match(ref[-1])

    def reset(self, nitems=None):
        """
        Reset sentence data
        """
        if nitems is None:
            self.tasks, self.inp = {task: [] for task in self.tasks}, []
        else:
            self.inp = self.inp[nitems:]
            self.tasks = {task: tdata[nitems:] for task, tdata in self.tasks.items()}


class TabReader(BaseReader):
    """
    Reader for files in tab format where each line has annotations for a given word
    and each annotation is located in a column separated by tabs.

    ...
    italiae	italia	NE
    eo	is	PRO
    tasko	taskus	NN
    ...

    Settings
    ===========
    header : bool, whether file has header.
    tasks_order : list, in case of missing header, the expected order of tasks in
        the file (there might be less)
    breakline_data : str
    breakline_ref : str, tab used to decide on line breaks
    max_sent_len : int, break lines to this length if they'd become longer
    tasks[task].default : str, method to use to fill in missing values
    """
    def __init__(self, settings, fpath, line_parser=LineParser):
        self.header = settings.header  # needed for get_tasks
        self.tasks_order = settings.tasks_order
        self.sep = settings.sep
        super(TabReader, self).__init__(settings, fpath)
        self.line_parser = line_parser
        self.breakline_ref = settings.breakline_ref
        self.breakline_data = settings.breakline_data
        self.max_sent_len = settings.max_sent_len

    def parselines(self):
        """
        Generator over sentences in a single file
        """
        with open(self.fpath, 'r+') as f:
            if self.header:
                next(f)

            parser = self.line_parser(
                self.tasks, self.sep, self.breakline_ref, self.breakline_data, self)

            for line_num, line in enumerate(f):
                line = line.strip()

                # Empty line as chunk break
                if not line and len(parser.inp) > 0:
                        yield parser.inp, parser.tasks
                        parser.reset()
                        continue
                # brealine configuration as chunk break
                elif parser.check_breakline():
                    yield parser.inp, parser.tasks
                    parser.reset()
                # max size as chunk break
                elif len(parser.inp) > self.max_sent_len:
                    inp = parser.inp[:self.max_sent_len]
                    tasks = {}
                    for t in parser.tasks:
                        tasks[t] = parser.tasks[t][:self.max_sent_len]
                    yield inp, tasks
                    parser.reset(self.max_sent_len)

                if line:
                    try:
                        parser.add(line, line_num)
                    except LineParseException as e:
                        yield e

            if len(parser.inp) > 0:
                yield parser.inp, parser.tasks

    def get_tasks(self):
        """
        Guess tasks from file assuming expected order
        """
        with open(self.fpath, 'r+') as f:

            if self.header:
                # TODO: this assumes text is first field
                _, *header = next(f).strip().split(self.sep)
                return tuple(header)

            else:
                # move to first non empty line
                line = next(f).strip()
                while not line:
                    line = next(f).strip()

                _, *tasks = line.split(self.sep)
                if len(tasks) == 0:
                    raise ValueError("Not enough input tasks: [{}]".format(self.fpath))

                return tuple(self.tasks_order[:len(tasks)])
