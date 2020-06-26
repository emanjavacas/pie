# Can be run with python -m pie.scripts.train
import os
from datetime import datetime
import logging
import pie
from pie.settings import settings_from_file
from pie.trainer import Trainer, EarlyStopException
from pie.settings import Settings
from pie import initialization
from pie.data import Dataset, Reader, MultiLabelEncoder
from pie.models import SimpleModel, get_pretrained_embeddings
from typing import Dict, Any, Optional, List

# set seeds
import random
import numpy
import torch

import optuna


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
        # Would be cool to be able to filter with `[KEY=Value]` in a list, specifically for target=True
        if subkey not in target:
            target[subkey] = {}
        target[subkey] = affect_settings(target[subkey], key, value)
    else:
        target[key] = value
    return target


def get_pruner(pruner_settings: Dict[str, Any]):
    return getattr(optuna.pruners, pruner_settings["name"])(
        *pruner_settings.get("args", []),
        **pruner_settings.get("kwargs", {})
    )


# https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/async_hyperband_example.py
class PieOptimizedTrainer(object):
    def __init__(self, settings, optimization_settings: List[Dict[str, Any]], gpus: List[int] = []):

        self._gpus: List[int] = gpus  # ToDo(Thibault) : Use multi-gpus https://github.com/optuna/optuna/issues/1365
        self.timestep = 0
        self._model: SimpleModel = None
        self._trainer: Trainer = None
        self._training_iterations = None
        self._devset: Dataset = None
        self._settings: Settings = settings
        self._focus: str = [task["name"] for task in settings.tasks if task.get("target") is True][0]

        self._optuna_settings = optimization_settings

    def create_tuna_optimization(self, trial: optuna.Trial, fn: str, name: str, value: List[Any]):
        """

        Might use self one day, so...

        :param trial:
        :param fn:
        :param name:
        :param value:
        :return:
        """
        return getattr(trial, fn)(name, *value)

    def __call__(self, trial: optuna.Trial):
        env_setup()

        # Adapt settings !
        for opt_set in self._optuna_settings:
            self._settings = affect_settings(
                target=self._settings,
                key=opt_set["path"],
                value=self.create_tuna_optimization(
                    trial=trial,
                    fn=opt_set["type"],
                    name=opt_set["path"].replace("/", "__"),
                    value=opt_set["args"]
                )
            )

        # datasets
        self._reader = Reader(settings, self._settings.input_path)
        tasks = self._reader.check_tasks(expected=None)

        # label encoder
        self._label_encoder: MultiLabelEncoder = MultiLabelEncoder.from_settings(self._settings, tasks=tasks)
        self._label_encoder.fit_reader(self._reader)

        self._trainset = Dataset(self._settings, self._reader, self._label_encoder)

        self._devset = None
        if self._settings.dev_path:
            self._devset = Dataset(self._settings, Reader(self._settings, self._settings.dev_path), self._label_encoder)
        else:
            logging.warning("No devset: cannot monitor/optimize training")

        # model
        self._model: SimpleModel = SimpleModel(
            self._label_encoder, self._settings.tasks,
            self._settings.wemb_dim, self._settings.cemb_dim, self._settings.hidden_size,
            self._settings.num_layers, cell=self._settings.cell,
            # dropout
            dropout=self._settings.dropout, word_dropout=self._settings.word_dropout,
            # word embeddings
            merge_type=self._settings.merge_type, cemb_type=self._settings.cemb_type,
            cemb_layers=self._settings.cemb_layers, custom_cemb_cell=self._settings.custom_cemb_cell,
            # lm joint loss
            include_lm=self._settings.include_lm, lm_shared_softmax=self._settings.lm_shared_softmax,
            # decoder
            scorer=self._settings.scorer, linear_layers=self._settings.linear_layers
        )

        # pretrain(/load pretrained) embeddings
        if self._model.wemb is not None:
            if self._settings.pretrain_embeddings:
                print("Pretraining word embeddings")
                wemb_reader = Reader(
                    self._settings, self._settings.input_path, self._settings.dev_path, self._settings.test_path)
                weight = get_pretrained_embeddings(
                    wemb_reader, self._label_encoder, size=self._settings.wemb_dim,
                    window=5, negative=5, min_count=1)
                self._model.wemb.weight.data = torch.tensor(weight, dtype=torch.float32)

            elif self._settings.load_pretrained_embeddings:
                print("Loading pretrained embeddings")
                if not os.path.isfile(self._settings.load_pretrained_embeddings):
                    print("Couldn't find pretrained eembeddings in: {}".format(
                        self._settings.load_pretrained_embeddings))
                initialization.init_pretrained_embeddings(
                    self._settings.load_pretrained_embeddings, self._label_encoder.word, self._model.wemb)

        # load pretrained weights
        if self._settings.load_pretrained_encoder:
            self._model.init_from_encoder(pie.Encoder.load(self._settings.load_pretrained_encoder))

        # freeze embeddings
        if self._settings.freeze_embeddings:
            self._model.wemb.weight.requires_grad = False

        self._model.to(settings.device)

        self._trainer: Trainer = Trainer(self._settings, self._model, self._trainset, self._reader.get_nsents())

        try:
            for epoch in range(self._settings.epochs):
                scores = self._trainer.train_epoch(devset=self._devset, epoch=epoch)
                target = scores[self._focus]["all"]["accuracy"]

                trial.report(target, epoch)

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.TrialPruned()
        except EarlyStopException as e:
            logging.info("Early stopping training: "
                         "task [{}] with best score {:.4f}".format(e.task, e.loss))

            self._model.load_state_dict(e.best_state_dict)
            scores = {e.task: e.loss}

        self.save(str(trial.number))

        return target

    def save(self, checkpoint_dir):
        # save model
        fpath, infix = get_fname_infix(self._settings)
        fpath = os.path.join(fpath, "checkpoint", checkpoint_dir)
        os.makedirs(fpath, exist_ok=True)
        fpath = self._model.save(fpath, infix=infix, settings=self._settings)
        return fpath

def optimize(settings, opt_settings):
    trial_creator = PieOptimizedTrainer(settings, opt_settings["params"])

    study = optuna.create_study(
        direction='maximize',
        pruner=get_pruner(opt_settings["pruner"])
    )
    study.optimize(trial_creator, n_trials=20)
    df = study.trials_dataframe()
    df.to_csv("study.csv")

    optuna.visualization.plot_intermediate_values(study)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help='Path to optimization file (see default_settings.json)')
    parser.add_argument('optuna_path', help='Path to optimization file (see default_optuna.json)')
    args = parser.parse_args()

    settings = settings_from_file(args.config_path)
    opt_settings = settings_from_file(args.optuna_path)
    """{
        "params": [
            {
                "path": "cemb_type",
                "type": "suggest_categorical",
                "args": [["cnn", "rnn"]]
            },
            {
                "path": "cemb_dim",
                "type": "suggest_int",
                "args": [200, 600]
            }
        ],
        "pruner": {
            "name": "MedianPruner",
            "args": [],
            "kwargs": {}
        }
    }"""

    optimize(settings, opt_settings)
