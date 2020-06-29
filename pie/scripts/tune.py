# Can be run with python -m pie.scripts.tune
import logging
from typing import Optional

import optuna

from pie.contrib.optuna_adapter import get_pruner, Optimizer
from pie.settings import settings_from_file, OPT_DEFAULT_PATH


def run_optimize(
        settings, opt_settings,
        generate_csv: bool = True, generate_html: bool = True,
        use_sqlite: Optional[str] = None, resume: bool = False):
    """ Run an Optuna-based optimization

    :param settings: Settings for classic training
    :param opt_settings: Settings for Optuna
    :param generate_csv: Produces a CSV output
    :param generate_html: Procudes a HTML output
    :param use_sqlite: Save/Store/Read a database for resuming operation or  \
                        multiprocessing
    :param resume: Resume training if it exists
    """

    storage = None

    if settings.verbose:
        logging.basicConfig(level=logging.INFO)

    if use_sqlite:
        storage = 'sqlite:///{}'.format(use_sqlite)

    trial_creator = Optimizer(
        settings,
        opt_settings["params"],
        focus=opt_settings["study"]["optimize_metric"],
        save_complete=opt_settings["study"]["save_completed"],
        save_pruned=opt_settings["study"]["save_pruned"]
    )

    study = optuna.create_study(
        study_name=opt_settings["study"]["name"],
        direction='maximize',
        pruner=get_pruner(opt_settings.get("pruner")),
        storage=storage,
        load_if_exists=resume,
        sampler=trial_creator.get_sampler(opt_settings.get("sampler"))
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
        settings_from_file(args.config_path, apply_task_default=False),
        settings_from_file(args.optuna_path, default_path=OPT_DEFAULT_PATH, apply_task_default=False),
        generate_csv=args.csv,
        generate_html=args.html,
        use_sqlite=args.sqlite,
        resume=args.resume
    )
