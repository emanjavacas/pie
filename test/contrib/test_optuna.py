from unittest import TestCase
import logging
import random
import os
import optuna

from pie.contrib.optuna_adapter import Optimizer
from pie.settings import load_default_settings
from pie.utils import shutup

opt_config = {
    "cemb_type": {
        "type": "suggest_categorical",
        "args": [
            ["cnn", "rnn"]
        ]
    },
    "cemb_dim": {
        "type": "suggest_int",
        "args": [200, 600]
    },
    "task_defaults": {
        "decoder": {
            "type": "suggest_categorical",
            "args": [["linear", "crf"]]
        }
    }
}
base_config = load_default_settings()
base_config.tasks = [{"name": "lemma", "target": True}]
base_config.verbose = False


class TestOptuna(TestCase):
    def tearDown(self):
        if os.path.isfile("./test_db.db"):
            os.remove("./test_db.db")

    def test_config(self):
        opt = Optimizer(settings=base_config, optimization_settings=opt_config)
        study = optuna.create_study()
        opt.used_settings = []

        # Prevent optuna from logging
        optuna.logging.disable_default_handler()

        def test(trial):
            opt.initialize_optimize(trial=trial)
            opt.used_settings.append(opt.settings)
            return random.randint(0, 100) / 100

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.CRITICAL)
        study.optimize(test, n_trials=40)

        self.assertCountEqual(
            list(set([sett["cemb_type"] for sett in opt.used_settings])),
            ["cnn", "rnn"],
            "Categorical should be used"
        )

        self.assertTrue(
            len(set([sett["cemb_dim"] for sett in opt.used_settings])) > 2,
            "CEMB Dim should evolve"
        )

        self.assertCountEqual(
            list(set([sett["tasks"][0]["decoder"] for sett in opt.used_settings])),
            ["linear", "crf"],
            "Target tasks should be applied"
        )
