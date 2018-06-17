
import uuid
import torch
import os
import unittest

from pie.models import SimpleModel
from pie.data import MultiLabelEncoder, Reader, Dataset
from pie.settings import settings_from_file


testpath = os.path.join(os.path.dirname(__file__), 'testconfig.json')
settings = settings_from_file(testpath)
label_encoder = MultiLabelEncoder.from_settings(settings)
reader = Reader(settings, settings.input_path)
label_encoder.fit_reader(reader)
dataset = Dataset(settings, label_encoder, reader)


class TestModelSerialization(unittest.TestCase):
    def setUp(self):
        emb_dim, hidden_size, num_layers = 64, 100, 1
        self.model = SimpleModel(
            label_encoder, emb_dim, emb_dim, hidden_size, num_layers)

    def test_serialization(self):
        model = self.model
        fid = '/tmp/{}'.format(str(uuid.uuid1()))
        model.save(fid)
        model2 = SimpleModel.load(fid)
        os.remove('{}.tar'.format(fid))
        self.assertEqual(model.label_encoder, model2.label_encoder)

        m1, m2 = dict(model.named_modules()), dict(model2.named_modules())

        for m in m1:
            if m != '':         # skip parent
                self.assertTrue(m in m2)
                for p1, p2 in zip(m1[m].parameters(), m2[m].parameters()):
                    self.assertTrue(torch.allclose(p1, p2))
