
import collections

from pie.reader import BaseReader, LineParseException


MORPHMAPPER = {
    "CASE": "case",
    "DEGR": "degree",
    "GEND": "gender",
    "INFL": None,               # only has one possible value (n)
    "MOOD": "mood",
    "NUMB": "number",
    "PERS": "person",
    "STRE": None,               # no idea what this is
    "TENS": "tense",
    "VOIC": "voice"
}


def parse_morph(morph):
    """
    Parse morphology field into a dictionary
    """
    vals = morph.split('|')
    output = {}
    for val in vals:
        key, val = val[:-1], val[-1]
        if MORPHMAPPER.get(key, None) is not None:
            output[MORPHMAPPER[key]] = val

    return output


def get_lines(fpath):
    """
    Get lines from a conll file
    """
    with open(fpath) as f:
        for line in f:
            line, tasks = line.strip().split(), {}

            if len(line) == 0:
                yield None

            else:
                _, form, lemma, pos, ppos, morph, head, dep, *_ = line
                tasks['lemma'] = lemma
                tasks['pos'] = pos
                for key, val in parse_morph(morph).items():
                    tasks[key] = val
                tasks['head'] = head
                tasks['dep'] = dep

                yield form, tasks


class CONLLReader(BaseReader):
    """
    CONLLReader format as found on the PROIEL files

    6	David	David	N	Ne	INFLn	5	atr	_	_

    which corresponds to:

    LINE_NUM	FORM	LEMMA	POS	PPOS	MORPH	HEAD	DEP	_	_
    """
    def __init__(self, settings, fpath):
        super(CONLLReader, self).__init__(settings, fpath)

        self.max_sent_len = settings.max_sent_len
        self.fields = ('lemma', 'pos', 'ppos', 'morph', 'head', 'dep')

    def parselines(self):
        """
        Generator over sentences in a single file
        """
        inp, tasks_data = [], collections.defaultdict(list)

        for line_num, line in enumerate(get_lines(self.fpath)):

            if line is None:
                if len(inp) > 0:
                    # TODO: parse dependency graph
                    yield inp, dict(tasks_data)
                    inp, tasks_data = [], collections.defaultdict(list)
            else:
                form, tasks = line
                inp.append(form)
                for task in self.tasks:
                    tasks_data[task].append(tasks.get(task, '_'))
                tasks_data['head'].append(tasks['head'])
                tasks_data['dep'].append(tasks['dep'])

                if len(inp) >= self.max_sent_len:
                    # TODO: parse dependency graph
                    yield inp, dict(tasks_data)
                    inp, tasks_data = [], collections.defaultdict(list)

        if len(inp) > 0:
            # TODO: parse dependency graph
            yield inp, dict(tasks_data)
            inp, tasks_data = [], collections.defaultdict(list)

    def get_tasks(self):
        """
        All conll tasks (as in proiel files) in expected order
        """
        output = set()
        for line in get_lines(self.fpath):
            if line is not None:
                _, tasks = line
                for task in tasks:
                    if task not in ('head', 'dep'):
                        output.add(task)

        return tuple(output)
