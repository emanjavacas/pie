# Can be run with python -m pie.scripts.tune
import os
from datetime import datetime
import logging
import random

# set seeds
import numpy
import torch
import optuna


from pie.settings import settings_from_file, OPT_DEFAULT_PATH
from pie.trainer import Trainer
from typing import Dict, Any, Optional, List


def get_targets(settings):
    return [task['name'] for task in settings.tasks if task.get('target')]


def get_fname_infix(settings):
    # fname
    fname = os.path.join(settings.modelpath, settings.modelname)
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    infix = '+'.join(get_targets(settings)) + '-' + timestamp
    return fname, infix


def save(checkpoint_dir, settings, model):
    # save model
    fpath, infix = get_fname_infix(settings)
    fpath = os.path.join(fpath, "checkpoint-"+str(checkpoint_dir))
    os.makedirs(fpath, exist_ok=True)
    fpath = model.save(fpath, infix=infix, settings=settings)
    return fpath


def env_setup(settings):
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

def create_tuna_optimization(trial: optuna.Trial, fn: str, name: str, value: List[Any]):
    """ Generate tuna value generator

    Might use self one day, so...

    :param trial:
    :param fn:
    :param name:
    :param value:
    :return:
    """
    return getattr(trial, fn)(name, *value)


class Optimizer(object):
    def __init__(self, settings, optimization_settings: List[Dict[str, Any]], gpus: List[int] = []):
        self.focus: str = [task["name"] for task in settings.tasks if task.get("target") is True][0]
        self.settings = settings
        self.optimization_settings = optimization_settings
        self.gpus = gpus

    def optimize(self, trial: optuna.Trial):
        env_setup(self.settings)

        settings = self.settings

        for opt_set in self.optimization_settings:
            settings = affect_settings(
                target=settings,
                key=opt_set["path"],
                value=create_tuna_optimization(
                    trial=trial,
                    fn=opt_set["type"],
                    name=opt_set["path"].replace("/", "__"),
                    value=opt_set["args"]
                )
            )

        trainer, model, trainset, devset, label_encoder, reader = Trainer.setup(settings)

        def report(epoch_id, _scores):
            target = _scores[self.focus]["all"]["accuracy"]
            trial.report(target, epoch_id)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()

        scores = trainer.train_epochs(self.settings.epochs, devset=devset, epoch_callback=report)
        save(str(trial.number), self.settings, model)

        return scores[self.focus]["all"]["accuracy"]


def run_optimize(
        settings, opt_settings,
        generate_csv: bool = True, generate_html: bool = True,
        use_sqlite: Optional[str] = None, resume: bool = False):
    """

    :param settings:
    :param opt_settings:
    :param study_name:
    :param generate_csv:
    :param generate_html:
    :param use_sqlite:
    :param resume:
    :return:
    """

    import pprint
    pprint.pprint(opt_settings)
    storage = None

    if use_sqlite:
        storage = 'sqlite:///{}'.format(use_sqlite)

    trial_creator = Optimizer(
        settings,
        opt_settings["params"]
    )

    study = optuna.create_study(
        study_name=opt_settings["study"]["name"],
        direction='maximize',
        pruner=get_pruner(opt_settings["pruner"]),
        storage=storage,
        load_if_exists=resume
    )
    study.optimize(trial_creator.optimize, n_trials=20)

    if generate_csv:
        df = study.trials_dataframe()
        df.to_csv(opt_settings["study"]["name"]+".csv")

    if generate_html:
        optuna.visualization.plot_intermediate_values(study).write_html(opt_settings["study"]["name"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help='Path to optimization file (see default_settings.json)')
    parser.add_argument('optuna_path', help='Path to optimization file (see default_optuna.json)')
    parser.add_argument('--html', action='store_true', default=False, help="Generate a HTML report using"
                                                                           "study name")
    parser.add_argument('--csv', action='store_true', default=False, help="Generate a CSV report using"
                                                                          "study name")
    parser.add_argument('--sqlite', default=None, help="Path to a SQLite DB File (Creates if not exists)")
    parser.add_argument('--resume', action='store_true', default=False, help="Resume a previous study using SQLite"
                                                                             "if it exists")
    args = parser.parse_args()

    run_optimize(
        settings_from_file(args.config_path),
        settings_from_file(args.optuna_path, default_path=OPT_DEFAULT_PATH),
        generate_csv=args.csv,
        generate_html=args.html,
        use_sqlite=args.sqlite,
        resume=args.resume
    )
