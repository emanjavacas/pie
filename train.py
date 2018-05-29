
import os

import yaml

from pie.settings import settings_from_file
from pie.data import Dataset, TabReader
from pie.model import SimpleModel
from pie.trainer import Trainer

# set seeds
import random
import numpy
import torch

seed = 1001
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    settings = settings_from_file(os.path.abspath('config.json'))

    # datasets
    trainset = Dataset(settings)

    devset = None
    if settings.dev_path is not None:
        devset = Dataset(
            settings, reader=TabReader(settings, input_path=settings.dev_path),
            label_encoder=trainset.label_encoder
        ).batch_generator()
    elif settings.dev_split > 0:
        devset = trainset.get_dev_split(split=settings.dev_split)
    else:
        print("No devset, cannot monitor/optimize training")

    testset = None
    if settings.test_path is not None:
        testset = Dataset(
            settings, reader=TabReader(settings, input_path=settings.test_path),
            label_encoder=trainset.label_encoder)

    # model
    model = SimpleModel(trainset.label_encoder, settings.emb_dim, settings.hidden_size,
                        settings.num_layers, dropout=settings.dropout, include_self=False,
                        pos_crf=True)

    # training
    trainer = Trainer(trainset, model, settings)

    try:
        trainer.train_model(settings.epochs, dev=devset)

    except KeyboardInterrupt:
        print("Stopping training")

    finally:
        if testset is not None:
            model.eval()
            test_loss = model.evaluate(testset.batch_generator())
            print("\n::: Test scores :::\n")
            print(yaml.dump(test_loss, default_flow_style=False))
            print()

    print("Bye!")
