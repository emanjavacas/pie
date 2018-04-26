
import os
import unittest

from pie.data import Dataset, LabelEncoder
from pie.settings import settings_from_file


class TestLabelEncoderSerialization(unittest.TestCase):
    def setUp(self):
        testpath = os.path.join(os.path.dirname(__file__), 'testconfig.json')
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
