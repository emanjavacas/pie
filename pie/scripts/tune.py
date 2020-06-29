# Can be run with python -m pie.scripts.tune
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

import optuna

from pie.settings import settings_from_file, OPT_DEFAULT_PATH
from pie.trainer import Trainer, set_seed, get_targets, get_fname_infix


def save(checkpoint_dir, settings, model):
    # save model
    fpath, infix = get_fname_infix(settings)
    fpath = os.path.join(fpath, "checkpoint-"+str(checkpoint_dir))
    os.makedirs(fpath, exist_ok=True)
    fpath = model.save(fpath, infix=infix, settings=settings)
    return fpath


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


def read_json_path(data: Dict[str, Union[Dict, float]], path: str):
    """ Read a simple JSON path

    >>> read_json_path({"a": {"b": {"c": 1}}, "a/b/c")
    1

    :param data: Nested dictionary
    :param path: Path (keys are separated with "/")
    :return: Value at the path
    """
    split = path.split("/")
    if len(split) > 1:
        current, path = split[0], "/".join(split[1:])
        return read_json_path(data[current], path)
    else:
        return data[path]


class Optimizer(object):
    def __init__(
            self,
            settings, optimization_settings: List[Dict[str, Any]],
            devices: List[int] = None, focus: Optional[str] = None,
            save_complete: bool = True,
            save_pruned: bool = False
    ):
        """

        :param settings:
        :param optimization_settings:
        :param devices: List of cuda devices. Leave empty if you use CPU
        """
        self.settings = settings
        self.optimization_settings = optimization_settings
        self.devices = devices or []
        self.save_pruned: bool = save_pruned
        self.save_complete: bool = save_complete
        # Should we set seed at the optimizer level or at the optimize() level
        if focus:
            self.focus: str = focus
        else:
            self.focus: str = "{}/all/accuracy".format(
                [
                    task["name"]
                    for task in settings.tasks
                    if task.get("target") is True
                ][0]
            )

    def optimize(self, trial: optuna.Trial):
        set_seed(verbose=self.settings.verbose)

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

        trainer, reader, devset = Trainer.setup(settings)

        def report(epoch_id, _scores):
            # Read the target to optimize using JSON path
            target = read_json_path(_scores, self.focus)
            trial.report(target, epoch_id)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                if self.save_pruned:
                    save(str(trial.number), self.settings, trainer.model)
                raise optuna.TrialPruned()

        scores = trainer.train_epochs(self.settings.epochs, devset=devset, epoch_callback=report)

        if self.save_complete:
            save(str(trial.number), self.settings, trainer.model)

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
        opt_settings["params"],
        focus=opt_settings["study"]["optimize_metric"],
        save_complete=opt_settings["study"]["save_complete"],
        save_pruned=opt_settings["study"]["save_pruned"]
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
