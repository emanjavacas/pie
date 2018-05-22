
import os

from pie.settings import settings_from_file
from pie.data import Dataset, TabReader
from pie.model import SimpleModel
from pie.trainer import Trainer

if __name__ == '__main__':
    settings = settings_from_file(os.path.abspath('config.json'))
    trainset = Dataset(settings)
    model = SimpleModel(trainset.label_encoder, settings.emb_dim, settings.hidden_size)
    trainer = Trainer(trainset, model, settings)
    devset = None
    if settings.dev_dir is not None:
        devset = Dataset(
            settings, reader=TabReader(settings, input_dir=settings.dev_dir),
            label_encoder=trainset.label_encoder)
    try:
        trainer.train_epochs(settings.epochs, dev=devset)
    except KeyboardInterrupt:
        print("Bye!")
