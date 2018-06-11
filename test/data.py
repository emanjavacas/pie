
import os
import unittest

from pie.data import Dataset, Reader, MultiLabelEncoder
from pie.settings import settings_from_file


testpath = os.path.join(os.path.dirname(__file__), 'testconfig.json')
delta = 5                       # FIXME


class TestLabelEncoderSerialization(unittest.TestCase):
    def setUp(self):
        settings = settings_from_file(testpath)
        reader = Reader(settings, settings.input_path)
        label_encoder = MultiLabelEncoder.from_settings(settings)
        label_encoder.fit(line for _, line in reader.readsents())
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
        label_encoder.fit(line for _, line in reader.readsents())
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


class TestDevSplit(unittest.TestCase):
    def setUp(self):
        settings = settings_from_file(testpath)
        settings['batch_size'] = 1
        reader = Reader(settings, settings.input_path)
        label_encoder = MultiLabelEncoder.from_settings(settings)
        insts = label_encoder.fit(line for _, line in reader.readsents())
        self.insts = insts
        self.num_batches = insts // settings.batch_size
        self.data = Dataset(settings, reader, label_encoder)

    def test_split_length(self):
        total_batches = 0
        for batch in self.data.batch_generator():
            total_batches += 1

        dev_batches = 0
        for batch in self.data.get_dev_split(self.insts, split=0.05):
            dev_batches += 1

        self.assertAlmostEqual(dev_batches, total_batches * 0.05, delta=delta)

    def test_remaining(self):
        pre_batches = 0
        for batch in self.data.batch_generator():
            pre_batches += 1

        self.assertEqual(pre_batches, self.insts)  # batch size is 1
        self.assertEqual(pre_batches, self.num_batches)

        devset = self.data.get_dev_split(self.insts, split=0.05)

        post_batches = 0
        for batch in self.data.batch_generator():
            post_batches += 1

        # FIXME
        self.assertAlmostEqual(len(devset) + post_batches, pre_batches, delta=delta*5)
        self.assertAlmostEqual(pre_batches * 0.95, post_batches, delta=delta*5)

    def test_batch_level(self):
        settings = settings_from_file(testpath)
        settings['batch_size'] = 20
        reader = Reader(settings, settings.input_path)
        label_encoder = MultiLabelEncoder.from_settings(settings)
        label_encoder.fit(line for _, line in reader.readsents())
        data = Dataset(settings, reader, label_encoder)

        pre_batches = 0
        for batch in data.batch_generator():
            pre_batches += 1

        self.assertAlmostEqual(pre_batches, self.insts // 20, delta=delta)

        devset = data.get_dev_split(self.insts, split=0.05)

        post_batches = 0
        for batch in data.batch_generator():
            post_batches += 1

        self.assertAlmostEqual(pre_batches * 0.95, post_batches, delta=delta)
