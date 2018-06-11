
import os
import unittest

from pie.models import SimpleModel
from pie.data import MultiLabelEncoder, Reader, Dataset
from pie.settings import settings_from_file

# testpath = os.path.join(os.path.dirname(__file__), 'testconfig.json')
testpath = 'test/testconfig.json'
settings = settings_from_file(testpath)
label_encoder = MultiLabelEncoder.from_settings(settings)
reader = Reader(settings, settings.input_path)
label_encoder.fit_reader(reader)
dataset = Dataset(settings, label_encoder, reader)

emb_dim, hidden_size, num_layers = 64, 100, 1
model = SimpleModel(label_encoder, emb_dim, hidden_size, num_layers)
model.save('test')
model2 = SimpleModel.load('test')
