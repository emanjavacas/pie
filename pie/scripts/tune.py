# Can be run with python -m pie.scripts.train
import os
from datetime import datetime
import logging
import pie
from pie.settings import settings_from_file
from pie.trainer import Trainer
from pie.settings import Settings
from pie import initialization
from pie.data import Dataset, Reader, MultiLabelEncoder
from pie.models import SimpleModel, get_pretrained_embeddings
from pie import optimize
from typing import Dict, Any

# set seeds
import random
import numpy
import torch
import ray
import ray.tune.schedulers
from ray import tune
from hyperopt import hp


def get_targets(settings):
    return [task['name'] for task in settings.tasks if task.get('target')]


def get_fname_infix(settings):
    # fname
    fname = os.path.join(settings.modelpath, settings.modelname)
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    infix = '+'.join(get_targets(settings)) + '-' + timestamp
    return fname, infix


def env_setup():
    now = datetime.now()
    # set seed
    seed = now.hour * 10000 + now.minute * 100 + now.second
    print("Using seed:", seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if settings.verbose:
        logging.basicConfig(level=logging.INFO)


def affect_settings(target: Dict[str, Any], key: str, value):
    if "/" in key:
        i = key.index("/")
        subkey, key = key[:i], key[i+1:]
        if subkey not in target:
            target[subkey] = {}
        target[subkey] = affect_settings(target[subkey], key, value)
    else:
        target[key] = value
    return target


def get_scheduler(name="AsyncHyperBandScheduler"):
    return getattr(ray.tune.schedulers, name)


# https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/async_hyperband_example.py
class PieTrainable(tune.Trainable):
    def _setup(self, config):
        print(config)
        self.timestep = 0
        self._model: SimpleModel = None
        self._trainer: Trainer = None
        self._training_iterations = None
        self._devset: Dataset = None
        self._settings: Settings = config["settings"]
        self._focus: str = config["focus"]  # Task that we optimized, should really have target search here...
        for key, val in config:
            if key not in {"focus", "settings"}:
                self._settings = affect_settings(self._settings, key, val)
        settings = self._settings

        # datasets
        self._reader = Reader(settings, settings.input_path)
        tasks = self._reader.check_tasks(expected=None)

        # label encoder
        self._label_encoder: MultiLabelEncoder = MultiLabelEncoder.from_settings(settings, tasks=tasks)
        self._label_encoder.fit_reader(self._reader)

        self._trainset = Dataset(settings, self._reader, self._label_encoder)

        self._devset = None
        if settings.dev_path:
            self._devset = Dataset(settings, Reader(settings, settings.dev_path), self._label_encoder)
        else:
            logging.warning("No devset: cannot monitor/optimize training")

        # model
        self._model: SimpleModel = SimpleModel(
            self._label_encoder, settings.tasks,
            settings.wemb_dim, settings.cemb_dim, settings.hidden_size,
            settings.num_layers, cell=settings.cell,
            # dropout
            dropout=settings.dropout, word_dropout=settings.word_dropout,
            # word embeddings
            merge_type=settings.merge_type, cemb_type=settings.cemb_type,
            cemb_layers=settings.cemb_layers, custom_cemb_cell=settings.custom_cemb_cell,
            # lm joint loss
            include_lm=settings.include_lm, lm_shared_softmax=settings.lm_shared_softmax,
            # decoder
            scorer=settings.scorer, linear_layers=settings.linear_layers
        )

        # pretrain(/load pretrained) embeddings
        if self._model.wemb is not None:
            if settings.pretrain_embeddings:
                print("Pretraining word embeddings")
                wemb_reader = Reader(
                    settings, settings.input_path, settings.dev_path, settings.test_path)
                weight = get_pretrained_embeddings(
                    wemb_reader, self._label_encoder, size=settings.wemb_dim,
                    window=5, negative=5, min_count=1)
                self._model.wemb.weight.data = torch.tensor(weight, dtype=torch.float32)

            elif settings.load_pretrained_embeddings:
                print("Loading pretrained embeddings")
                if not os.path.isfile(settings.load_pretrained_embeddings):
                    print("Couldn't find pretrained eembeddings in: {}".format(
                        settings.load_pretrained_embeddings))
                initialization.init_pretrained_embeddings(
                    settings.load_pretrained_embeddings, self._label_encoder.word, self._model.wemb)

        # load pretrained weights
        if settings.load_pretrained_encoder:
            self._model.init_from_encoder(pie.Encoder.load(settings.load_pretrained_encoder))

        # freeze embeddings
        if settings.freeze_embeddings:
            self._model.wemb.weight.requires_grad = False

        self._trainer: Trainer = Trainer(self._settings, self._model, self._trainset, self._reader.get_nsents())
        self._training_iterations = self._trainer.train_epochs(epochs=self._settings.epochs, devset=self._devset)

    def _train(self):
        epoch, score = next(self._training_iterations)
        return score[self._focus]["all"]

    def _save(self, checkpoint_dir):
        # save model
        fpath, infix = get_fname_infix(settings)
        fpath = os.path.join(checkpoint_dir, "checkpoint", fpath)
        fpath = self._model.save(fpath, infix=infix, settings=settings)
        return fpath

    def _restore(self, checkpoint_path):
        self._model.load(checkpoint_path)


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', nargs='?', default='config.json')
    #parser.add_argument('tune_path', help='Path to optimization file (see opt.json)')
    args = parser.parse_args()

    settings = settings_from_file(args.config_path)
    #with open(args.tune_path) as f:
    #    tune_config = json.load(f)
    #    ray_opt = tune_config.get("ray", {})
    #    scheduler = tune_config.get("scheduler", {
    #        "name": "AsyncHyperBandScheduler",
    #        "options": {}
    #    })
    ray_opt = {}
    scheduler = {
        "name": "AsyncHyperBandScheduler",
        "options": {}
    }

    #ray.init(**ray_opt)
    ray.init(num_gpus=1)
    scheduler = get_scheduler(scheduler.get("name"))(**scheduler.get("options"))

    env_setup()

    analysis = tune.run(
        PieTrainable,
        #scheduler=scheduler,
        #config={
        #    "settings": settings,
        #    "hidden_size": tune.grid_search([300, 200, 400, 500]),
        #},
        resources_per_trial={"gpu": 1}
        #resources_per_trial={
        #    "gpu": 1
        #}
    )
