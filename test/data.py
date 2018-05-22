
import os
import unittest

from pie.data import Dataset, MultiLabelEncoder
from pie.settings import settings_from_file


testpath = os.path.join(os.path.dirname(__file__), 'testconfig.json')


class TestLabelEncoderSerialization(unittest.TestCase):
    def setUp(self):
        settings = settings_from_file(testpath)
        self.data = Dataset(settings)

    def test_serialization(self):
        le = self.data.label_encoder
        le.save('/tmp/encoder.json')
        le2 = MultiLabelEncoder.load('/tmp/encoder.json')

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
        self.data = Dataset(settings)

    def test_lengths(self):
        ((word, wlen), (char, clen)), _ = next(self.data.batch_generator())

        for w, wl in zip(word.t(), wlen):
            self.assertEqual(w[wl-1].item(), self.data.label_encoder.word.get_eos())

        for c, cl in zip(char.t(), clen):
            self.assertEqual(c[cl-1].item(), self.data.label_encoder.char.get_eos())

    def test_word_char(self):
        for ((word, wlen), (char, clen)), _ in self.data.batch_generator():
            idx = 0
            total_words = 0
            for sent, nwords in zip(word.t(), wlen):
                for word in sent[:nwords-1]:
                    # get word
                    word = self.data.label_encoder.word.inverse_table[word]
                    # get chars
                    chars = char.t()[idx][:clen[idx]-1].tolist()  # remove <eos>
                    chars = ''.join(self.data.label_encoder.char.inverse_transform(chars))
                    self.assertEqual(word, chars)
                    idx += 1
                total_words += nwords - 1
            self.assertEqual(idx, total_words, "Checked all words")
