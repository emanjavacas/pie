
import os
import unittest
import copy

from pie.data.tabreader import TabReader


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class TestTabReader(unittest.TestCase):

    TASKS = [{"name": "lemma"}, {"name": "pos"}]
    FILENAME = '/tmp/test-data.tab'
    HEADER = {
        "header": False,
        "tasks_order": ["lemma", "pos", "ignore"],
        "sep": "\t",
        "breakline_ref": "input",
        "breakline_data": r"^(\.|''''|seg)$",
        "max_sent_len": 120,
        "tasks": TASKS
    }

    DEFAULT_INPUT = """SOIGNORS	seignor	NOMcom	NOMB.=p|GENRE=m|CAS=n
or	or4	ADVgen	DEGRE=-
escoutez	escouter	VERcjg	MODE=imp|PERS.=2|NOMB.=p
que	que4	CONsub	MORPH=empty
Dés	Dieu	NOMpro	NOMB.=s|GENRE=m|CAS=n
vos	vos	PROper	PERS.=2|NOMB.=p|GENRE=m|CAS=i
soit	estre1	VERcjg	MODE=sub|TEMPS=pst|PERS.=3|NOMB.=s
amis	ami	NOMcom	NOMB.=s|GENRE=m|CAS=n
{separator}
III	trois	DETcar	NOMB.=p|GENRE=m|CAS=r
vers	vers1	NOMcom	NOMB.=p|GENRE=m|CAS=r
de	de	PRE	MORPH=empty
bone	bon	ADJqua	NOMB.=s|GENRE=f|CAS=r|DEGRE=p
estoire	estoire1	NOMcom	NOMB.=s|GENRE=f|CAS=r
se	si	ADVgen	DEGRE=-
je	je	PROper	PERS.=1|NOMB.=s|GENRE=m|CAS=n
les	il	PROper	PERS.=3|NOMB.=p|GENRE=m|CAS=r
vos	vos	PROper	PERS.=2|NOMB.=p|GENRE=m|CAS=i
devis	deviser	VERcjg	MODE=ind|TEMPS=pst|PERS.=1|NOMB.=s
{separator}
Dou	de+le	PRE.DETdef	NOMB.=s|GENRE=m|CAS=r
premier	premier	ADJord	NOMB.=s|GENRE=m|CAS=r
roi	roi2	NOMcom	NOMB.=s|GENRE=m|CAS=r
de	de	PRE	MORPH=empty
France	France	NOMpro	NOMB.=s|GENRE=f|CAS=r
qui	qui	PROrel	NOMB.=s|GENRE=m|CAS=n
crestïens	crestiien	NOMcom	NOMB.=s|GENRE=m|CAS=n
devint	devenir	VERcjg	MODE=ind|TEMPS=psp|PERS.=3|NOMB.=s"""

    DIFF_INPUT = """SOIGNORS	seignor	NOMcom	NOMB.=p|GENRE=m|CAS=n
or	or4	ADVgen	DEGRE=-
escoutez	escouter	VERcjg	MODE=imp|PERS.=2|NOMB.=p
que	que4	CONsub	MORPH=empty
Dés	Dieu	NOMpro	NOMB.=s|GENRE=m|CAS=n
vos	vos	PROper	PERS.=2|NOMB.=p|GENRE=m|CAS=i
soit	estre1	VERcjg	MODE=sub|TEMPS=pst|PERS.=3|NOMB.=s
amis	ami	NOMcom	NOMB.=s|GENRE=m|CAS=n
{sep1}
III	trois	DETcar	NOMB.=p|GENRE=m|CAS=r
vers	vers1	NOMcom	NOMB.=p|GENRE=m|CAS=r
de	de	PRE	MORPH=empty
bone	bon	ADJqua	NOMB.=s|GENRE=f|CAS=r|DEGRE=p
estoire	estoire1	NOMcom	NOMB.=s|GENRE=f|CAS=r
se	si	ADVgen	DEGRE=-
je	je	PROper	PERS.=1|NOMB.=s|GENRE=m|CAS=n
les	il	PROper	PERS.=3|NOMB.=p|GENRE=m|CAS=r
vos	vos	PROper	PERS.=2|NOMB.=p|GENRE=m|CAS=i
devis	deviser	VERcjg	MODE=ind|TEMPS=pst|PERS.=1|NOMB.=s
{sep2}
Dou	de+le	PRE.DETdef	NOMB.=s|GENRE=m|CAS=r
premier	premier	ADJord	NOMB.=s|GENRE=m|CAS=r
roi	roi2	NOMcom	NOMB.=s|GENRE=m|CAS=r
de	de	PRE	MORPH=empty
France	France	NOMpro	NOMB.=s|GENRE=f|CAS=r
qui	qui	PROrel	NOMB.=s|GENRE=m|CAS=n
crestïens	crestiien	NOMcom	NOMB.=s|GENRE=m|CAS=n
devint	devenir	VERcjg	MODE=ind|TEMPS=psp|PERS.=3|NOMB.=s"""

    tokens = (
        ['SOIGNORS', 'or', 'escoutez', 'que', 'Dés', 'vos', 'soit', 'amis'],
        ['III', 'vers', 'de', 'bone', 'estoire', 'se', 'je', 'les', 'vos', 'devis'],
        ['Dou', 'premier', 'roi', 'de', 'France', 'qui', 'crestïens', 'devint']
    )

    def settings(self, **kwargs):
        x = {k+"": v for k, v in self.HEADER.items()}
        x.update(kwargs)
        return AttributeDict(x)

    def write(self, inp):
        with open(self.FILENAME, "w") as f:
            f.write(inp)

    def tearDown(self):
        if os.path.isfile(self.FILENAME):
            os.remove(self.FILENAME)

    def test_breakline_input(self):
        """ Regular expression should be taken into account as breaking line """
        self.write(self.DEFAULT_INPUT.format(separator="''''\tx\ty\tz"))

        tab_reader = TabReader(settings=self.settings(), fpath=self.FILENAME)

        seen = 0
        expected_tokens = copy.deepcopy(self.tokens)
        expected_tokens[0].append("''''")
        expected_tokens[1].append("''''")

        for (tokens, tasks), expected in zip(tab_reader.parselines(), expected_tokens):
            self.assertEqual(expected, tokens, "Sentence should be correctly cut")
            seen += 1

        self.assertEqual(seen, 3, "There should have been 3 sentences read")

    def test_breakline_input_empty_combination(self):
        """ Regular expression and empty lines should be taken into account as
        breaking line """

        self.write(self.DIFF_INPUT.format(sep1="''''\tx\ty\tz", sep2=""))

        tab_reader = TabReader(settings=self.settings(), fpath=self.FILENAME)

        seen = 0
        expected_tokens = copy.deepcopy(self.tokens)
        expected_tokens[0].append("''''")
        for (tokens, tasks), expected in zip(tab_reader.parselines(), expected_tokens):
            self.assertEqual(expected, tokens, "Sentence should be correctly cut")
            seen += 1

        self.assertEqual(seen, 3, "There should have been 3 sentences read")

    def test_breakline_empty_line(self):
        """ Empty lines should be taken into account as breaking line """
        self.write(self.DEFAULT_INPUT.format(separator=""))
        tab_reader = TabReader(settings=self.settings(), fpath=self.FILENAME)

        seen = 0
        for (tokens, tasks), expected in zip(tab_reader.parselines(), self.tokens):
            self.assertEqual(expected, tokens, "Sentence should be correctly cut")
            seen += 1

        self.assertEqual(seen, 3, "There should have been 3 sentences read")
