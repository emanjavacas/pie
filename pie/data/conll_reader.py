
import collections

from .base_reader import BaseReader, LineParseException


PROIELMORPH = {
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


def get_lines(fpath, _parse_morph):
    with open(fpath) as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            elif line.startswith('#'):
                continue
            else:
                num, form, lemma, pos, ppos, morph, _, dep, *_ = line.split('\t')
                tasks = {'lemma': lemma, 'pos': pos, 'ppos': ppos, 'dep': dep}
                for task, val in _parse_morph(morph).items():
                    tasks[task] = val
                yield form, tasks


def get_sents(fpath, parse_morph_):
    lines = 0
    with open(fpath) as f:
        sent, prev, tasks = [], 0, collections.defaultdict(list)
        for line in f:
            line = line.strip()

            if not line:
                # new line
                yield sent, dict(tasks)
                sent, prev, tasks = [], 0, collections.defaultdict(list)
                lines += 1

            elif line.startswith('#'):
                # skip metadata
                continue

            else:
                num, form, lemma, pos, ppos, morph, _, dep, *_ = line.split('\t')
                try:
                    num = int(num)
                    assert num == prev + 1, (num, prev)
                    prev = num

                    sent.append(form)
                    tasks['lemma'].append(lemma)
                    tasks['pos'].append(pos)
                    tasks['ppos'].append(ppos)
                    tasks['dep'].append(dep)
                    for key, val in parse_morph_(morph).items():
                        tasks[key].append(val)

                except ValueError:
                    # 20-22	"průběžná_inventarizace"
                    continue


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

    def _parse_morph(self, morph):
        if morph == '_':
            return morph
        output = {}
        for val in morph.split("|"):
            task, val = val[:-1], val[-1]
            if PROIELMORPH.get(task):
                output[PROIELMORPH[task]] = val
        return output

    def parse_morph_(self, morph):
        morph = self._parse_morph(morph)
        output = {}
        for task in self.tasks:
            if task not in ('lemma', 'pos', 'ppos', 'dep'):
                output[task] = morph.get(task, '_')
        return output

    def parselines(self):
        """
        Generator over sentences in a single file
        """
        inp = []

        for inp, tasks in get_sents(self.fpath, self.parse_morph_):
            while len(inp) > self.max_sent_len:
                inp_ = inp[:self.max_sent_len]
                tasks_ = {}
                for task in tasks:
                    tasks_[task] = tasks[task][:self.max_sent_len]
                yield inp_, tasks_
                inp = inp[self.max_sent_len:]
                for task in tasks:
                    tasks[task] = tasks[task][self.max_sent_len:]
            yield inp, tasks

    def get_tasks(self):
        """
        All conll tasks (as in proiel files) in expected order
        """
        output = set()
        for _, tasks in get_lines(self.fpath, self._parse_morph):
            for task in tasks:
                output.add(task)

        return tuple(output)


class CONLLUReader(CONLLReader):
    def _parse_morph(self, morph):
        if morph == '_':
            return {}
        output = {}
        for val in morph.split("|"):
            key, val = val.split("=")
            output[key] = val
        return output
