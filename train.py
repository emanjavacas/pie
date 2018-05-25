
import os

from pie.settings import settings_from_file
from pie.data import Dataset, TabReader
from pie.model import SimpleModel
from pie.trainer import Trainer


if __name__ == '__main__':
    settings = settings_from_file(os.path.abspath('config.json'))
    trainset = Dataset(settings)
    devset = None
    if settings.dev_path is not None:
        devset = Dataset(
            settings, reader=TabReader(settings, input_path=settings.dev_path),
            label_encoder=trainset.label_encoder)
    else:
        devset = trainset.get_dev_split(split=settings.dev_split)

    model = SimpleModel(trainset.label_encoder, settings.emb_dim, settings.hidden_size,
                        settings.num_layers, dropout=settings.dropout)
    trainer = Trainer(trainset, model, settings)
    try:
        trainer.train_model(settings.epochs, dev=devset)
    except KeyboardInterrupt:
        print(model.evaluate(devset))
        print("Bye!")
