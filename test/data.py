
import os
import unittest

from pie.data import Dataset, LabelEncoder
from pie.settings import settings_from_file


testpath = os.path.join(os.path.dirname(__file__), 'testconfig.json')


class TestLabelEncoderSerialization(unittest.TestCase):
    def setUp(self):
        self.data = Dataset(settings_from_file(testpath))

    def test_serialization(self):
        le = self.data.label_encoder
        le.save('/tmp/encoder.json')
        le2 = LabelEncoder.load('/tmp/encoder.json')

        self.assertEqual(len(le._all_encoders), len(le2._all_encoders),
                         "Unequal number of Modality encoders")

        for i in range(len(le._all_encoders)):
            self.assertTrue(le._all_encoders[i] == le2._all_encoders[i],
                            "Unequal serialized label encoder at position {}"
                            .format(i + 1))


class TestTokenCharEncoding(unittest.TestCase):
    def setUp(self):
        self.data = Dataset(settings_from_file(testpath))

    def test_lengths(self):
        ((token, tlen), (char, clen), _), _ = next(self.data.batch_generator())

        for t, tl in zip(token.t(), tlen):
            self.assertEqual(t[tl-1].item(), self.data.label_encoder.token.get_eos())

        for c, cl in zip(char.t(), clen):
            self.assertEqual(c[cl-1].item(), self.data.label_encoder.token.get_eos())

    def test_token_char(self):
        ((token, tlen), (char, clen), lens), _ = next(self.data.batch_generator())

        idx = 0
        for sent, nwords in zip(token.t(), lens):
            for word in sent[:nwords]:
                # get word
                word = self.data.label_encoder.token.inverse_table[word]
                # get chars
                chars = char.t()[idx][:clen[idx]-1]  # remove <eos>
                chars = ''.join(self.data.label_encoder.char.inverse_table[c.item()]
                                for c in chars)
                self.assertEqual(word, chars)
                idx += 1
