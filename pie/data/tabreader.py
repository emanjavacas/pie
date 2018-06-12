
from .base_reader import BaseReader, LineParseException, MissingDefaultException


class LineParser(object):
    """
    Inner class to handle sentence breaks
    """
    def __init__(self, tasks, breakline_type, breakline_ref, breakline_data, reader):
        self.reader = reader
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
    breakline_type : str, one of "LENGTH" or "FULLSTOP".
    breakline_data : str or int, if breakline_type is LENGTH it will be assumed
        to be an integer defining the number of words per sentence, and the
        dataset will be break into equally sized chunks. If breakline_type is
        FULLSTOP it will be assumed to be a POS tag to use as criterion to
        split sentences.
    breakline_ref : str, tab used to decide on line breaks
    max_sent_len : int, break lines to this length if they'd become longer
    tasks[task].default : str, method to use to fill in missing values
    """
    def __init__(self, settings, fpath, line_parser=LineParser):
        self.header = settings.header  # needed for get_tasks
        self.tasks_order = settings.tasks_order
        super(TabReader, self).__init__(settings, fpath)

        self.line_parser = line_parser
        self.breakline_type = settings.breakline_type
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
                self.tasks, self.breakline_type,
                self.breakline_ref, self.breakline_data, self)

            for line_num, line in enumerate(f):
                line = line.strip()

                if not line:    # avoid empty line
                    continue

                parser.add(line, line_num)

                if parser.check_breakline() or len(parser.inp) >= self.max_sent_len:
                    yield parser.inp, parser.tasks
                    parser.reset()

    def get_tasks(self):
        """
        Guess tasks from file assuming expected order
        """
        with open(self.fpath, 'r+') as f:

            if self.header:
                # TODO: this assumes text is first field
                _, *header = next(f).strip().split()
                return tuple(header)

            else:
                # move to first non empty line
                line = next(f).strip()
                while not line:
                    line = next(f).strip()

                _, *tasks = line.split()
                if len(tasks) == 0:
                    raise ValueError("Not enough input tasks: [{}]".format(self.fpath))

                return tuple(self.tasks_order[:len(tasks)])
