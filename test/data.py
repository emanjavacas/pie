
import os
import unittest

from pie.data import Dataset, Reader, MultiLabelEncoder
from pie.settings import settings_from_file


testpath = os.path.join(os.path.dirname(__file__), 'testconfig.json')


class TestLabelEncoderSerialization(unittest.TestCase):
    def setUp(self):
        settings = settings_from_file(testpath)
        reader = Reader(settings, settings.input_path)
        label_encoder = MultiLabelEncoder.from_settings(settings)
        label_encoder.fit_reader(reader)
        self.data = Dataset(settings, reader, label_encoder)

    def test_serialization(self):
        le = self.data.label_encoder
        le.save('/tmp/encoder.json')
        le2 = MultiLabelEncoder.load_from_file('/tmp/encoder.json')

        self.assertEqual(len(le.tasks), len(le2.tasks),
                         "Unequal number of Modality encoders")

        self.assertEqual(le.word, le2.word, "word encoder")
        self.assertEqual(le.char, le2.char, "char encoder")

        for task in le.tasks:
            self.assertTrue(
                le.tasks[task] == le2.tasks[task],
                "Unequal serialized label encoder for task {}".format(task))


class TestWordCharEncoding(unittest.TestCase):
    def setUp(self):
        settings = settings_from_file(testpath)
        reader = Reader(settings, settings.input_path)
        label_encoder = MultiLabelEncoder.from_settings(settings)
        label_encoder.fit_reader(reader)
        self.data = Dataset(settings, reader, label_encoder)

    def test_lengths(self):
        ((word, wlen), (char, clen)), _ = next(self.data.batch_generator())

        for c, cl in zip(char.t(), clen):
            self.assertEqual(c[0].item(), self.data.label_encoder.char.get_bos())
            self.assertEqual(c[cl-1].item(), self.data.label_encoder.char.get_eos())

    def test_word_char(self):
        for ((word, wlen), (char, clen)), _ in self.data.batch_generator():
            idx = 0
            total_words = 0
            for sent, nwords in zip(word.t(), wlen):
                for word in sent[:nwords]:
                    # get word
                    word = self.data.label_encoder.word.inverse_table[word]
                    # get chars
                    chars = char.t()[idx][1:clen[idx]-1].tolist()  # remove <eos>,<bos>
                    chars = ''.join(self.data.label_encoder.char.inverse_transform(chars))
                    self.assertEqual(word, chars)
                    idx += 1
                total_words += nwords
            self.assertEqual(idx, total_words, "Checked all words")


def _test_conversion(settings, level='token'):
    reader = Reader(settings, settings.input_path)
    label_encoder = MultiLabelEncoder.from_settings(settings)
    label_encoder.fit_reader(reader)
    data = Dataset(settings, reader, label_encoder)

    le = label_encoder.tasks['lemma']
    for (inp, tasks), (rinp, rtasks) in data.batch_generator(return_raw=True):
        # preds
        tinp, tlen = tasks['lemma']
        preds = [le.stringify(t, l) for t, l in zip(tinp.t().tolist(), tlen.tolist())]
        if level == 'token':
            preds = [w for line in preds for w in line]
        # tokens
        tokens = [tok for line in rinp for tok in line]
        # trues
        trues = [w for line in rtasks for w in line['lemma']]

        # check
        for pred, token, true in zip(preds, tokens, trues):
            rec = le.preprocessor_fn.inverse_transform(pred, token)
            assert rec == true, (pred, token, true, rec)


class TestGreedyScripts(unittest.TestCase):
    def setUp(self):
        settings = settings_from_file(testpath)
        settings.tasks = [
            {"name": "lemma", "level": "char", "decoder": "linear",
             "settings": {"preprocessor": "greedy_scripts", "target": "lemma"}}
        ]
        settings.char_eos = False
        settings.char_bos = False
        settings.char_max_size = 10000
        self.settings = settings

    def testConversion(self):
        _test_conversion(self.settings, level="char")


class TestEditTrees(unittest.TestCase):
    def setUp(self):
        settings = settings_from_file(testpath)
        settings.tasks = [
            {"name": "lemma", "level": "token", "decoder": "linear",
             "settings": {"preprocessor": "edit_trees", "target": "lemma"}}
        ]
        self.settings = settings

    def testConversion(self):
        _test_conversion(self.settings, level="token")


class TestEditTreeTuples(unittest.TestCase):
    def setUp(self):
        settings = settings_from_file(testpath)
        settings.tasks = [
            {"name": "lemma", "level": "char", "decoder": "attentional",
             "settings": {"preprocessor": "edit_tree_tuples", "target": "lemma"}}
        ]
        self.settings = settings

    def testConversion(self):
        _test_conversion(self.settings, level="char")
