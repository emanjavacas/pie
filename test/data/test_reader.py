
import os
import unittest

from pie.data import Reader
from pie.settings import settings_from_file

testpath = os.path.join(os.path.dirname(__file__), 'testconfig.json')


class TestJackknife(unittest.TestCase):
    def test_jackknife(self):
        settings = settings_from_file(testpath)
        # patch default
        settings.input_path = "datasets/patres/alcuinus.tab"
        reader = Reader(settings, settings.input_path)
        jack_sents = []

        total = 0
        for _ in reader.readsents():
            total += 1

        for i, (train, test) in enumerate(reader.jackknife(5)):
            jack_sents.extend([s for (_, s) in test.readsents()])
            train = [idx for ((_, idx), _) in train.readsents()]
            test = [idx for ((_, idx), _) in test.readsents()]

            self.assertFalse(
                set(train).intersection(set(test)),
                "No overlap between splits {}/{}".format(i, 5))

            self.assertTrue(
                len(train) + len(test) == total,
                "All sentences were read. Train: {}, test: {}, total: {}".format(
                    len(train), len(test), total))

        sents = [s for (_, s) in reader.readsents()]
        self.assertEqual(
            sents, jack_sents,
            "Jackknife test sents equal and in equal order as original reader")
