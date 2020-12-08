import os
import logging
import copy
from typing import Dict, Any, List, Union, Optional, Tuple
# Third-Party
import optuna
# Pie
from pie.trainer import set_seed, Trainer, get_fname_infix, get_targets
from pie.settings import merge_task_defaults, check_settings


IntSpace = List[int]
FloatSpace = List[float]
CategoricalSpace = List[List[str]]


def get_pruner(pruner_settings: Optional[Dict[str, Any]]) -> Optional[optuna.pruners.BasePruner]:
    """ Initialize a Pruner

    :param pruner_settings: Dict containing a name, an args list (optional), a kwargs dictionary (optional)
        or None
    :return: If not None, the pruner required for the study
    """
    if not pruner_settings or not pruner_settings.get("name"):
        return None
    return getattr(optuna.pruners, pruner_settings["name"])(
        *pruner_settings.get("args", []),
        **pruner_settings.get("kwargs", {})
    )


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


def insert_search_space(classic_settings, optuna_settings, trial: Optional[optuna.Trial] = None,
                        path: Optional[List] = None, path_glue: str = "__",
                        dry_run: bool = False) -> Tuple[Dict, Dict]:
    path = path or []
    search_space: Dict[str, Dict[str, Any]] = {}

    for key, value in classic_settings.items():
        if key in classic_settings and key in optuna_settings:
            # If key is in both dictionary, and the classic settings is already a dict,
            #   we merge things together
            if isinstance(value, dict):
                classic_settings[key], temp_space = insert_search_space(
                    classic_settings[key],
                    optuna_settings[key],
                    trial,
                    path=path + [key],
                    path_glue=path_glue,
                    dry_run=dry_run
                )
                search_space.update({
                    temp_space_key: temp_space_value
                    for temp_space_key, temp_space_value in temp_space.items()
                })
            # If the key in optuna settings is a dict but not in the original settings,
            #   we *de-facto* have an optuna initialization dict.
            elif isinstance(optuna_settings[key], dict):
                variable_name = path_glue.join(path + [key])
                search_space[variable_name] = optuna_settings[key]
                if not dry_run:
                    classic_settings[key] = create_tuna_optimization(
                        trial=trial,
                        fn=optuna_settings[key]["type"],
                        name=variable_name,
                        value=optuna_settings[key]["args"]
                    )
    return classic_settings, search_space


def simplify_search_space(suggest_args: Union[IntSpace, FloatSpace, CategoricalSpace]) -> List[Any]:
    """ Converts search space from suggest_uniform and so on to Sampler search space:

    >>> simplify_search_space([["cnn", "rnn"]]) == ["cnn", "rnn"]
    True
    >>> simplify_search_space([200, 600, 25]) == [200, 600, 25]
    True

    :param suggest_args: Args for suggest_*
    :return: Search Space
    """
    if isinstance(suggest_args[0], list):  # If args[0] is a list, this is a Cetagorical search space
        return suggest_args[0]
    else:
        return suggest_args


class Optimizer(object):
    def __init__(
            self,
            settings, optimization_settings: Dict[str, Any],
            devices: List[int] = None, focus: Optional[str] = None,
            save_complete: bool = True,
            save_pruned: bool = False
    ):
        """

        :param settings:
        :param optimization_settings:
        :param devices: List of cuda devices. Leave empty if you use CPU
        """
        self.settings: Optional[Dict] = None
        self._base_settings = settings
        self.optimization_settings = optimization_settings
        self.devices = devices or []
        self.save_pruned: bool = save_pruned
        self.save_complete: bool = save_complete
        self.search_space: Dict = {}
        # Should we set seed at the optimizer level or at the optimize() level
        if focus:
            self.focus: str = focus
        else:
            self.focus: str = "{}/all/accuracy".format(
                get_targets(settings)[0]
            )

    def get_sampler(self, sampler_settings: Optional[Dict[str, Any]]) -> Optional[optuna.samplers.BaseSampler]:
        """ Initialize a Sampler

        :param sampler_settings: Dict containing a name, an args list \
            (optional), a kwargs dictionary (optional) or None
        :return: If not None, the samp[ler required for the study

        """
        if not sampler_settings or not sampler_settings.get("name"):
            return None

        kwargs = sampler_settings.get("kwargs", {})
        args = sampler_settings.get("args", [])

        # Checking that the sampler does not need to be fed the search space
        #       (Using set for potential evolutions)
        if sampler_settings["name"] in {"GridSampler"} and "search_space" not in kwargs:
            _, search_space = insert_search_space(
                classic_settings=self.settings,
                optuna_settings=self.optimization_settings,
                trial=None, dry_run=True
            )
            kwargs["search_space"] = {
                key: simplify_search_space(value["args"])
                for key, value in search_space.items()
            }

        return getattr(optuna.samplers, sampler_settings["name"])(
            *args,
            **kwargs
        )

    @property
    def base_settings(self):
        return self._base_settings

    def reset_settings(self):
        self.settings = copy.deepcopy(self.base_settings)

    def initialize_optimize(self, trial: optuna.Trial) -> None:
        """ Initialize the search space and merging of settings.

        :param trial: Current trial
        """
        self.reset_settings()
        if self.settings.verbose:
            logging.info("Initializing search space")
        self.settings, self.search_space = insert_search_space(
            classic_settings=self.settings,
            optuna_settings=self.optimization_settings,
            trial=trial
        )
        self.settings = merge_task_defaults(self.settings)

    def optimize(self, trial: optuna.Trial):
        self.initialize_optimize(trial)

        set_seed(verbose=self.settings.verbose)
        settings = self.settings

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

        return read_json_path(scores, self.focus)


def save(checkpoint_dir, settings, model):
    # save model
    fpath, infix = get_fname_infix(settings)
    fpath = os.path.join(fpath, "checkpoint-"+str(checkpoint_dir))
    os.makedirs(fpath, exist_ok=True)
    fpath = model.save(fpath, infix=infix, settings=settings)
    return fpath
