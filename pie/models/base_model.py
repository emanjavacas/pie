import os
import json
import tarfile
import logging

import tqdm
import torch
import torch.nn as nn

from pie import utils
from pie.data import MultiLabelEncoder
from pie.settings import Settings

from .scorer import Scorer, get_known_and_ambigous_tokens


class BaseModel(nn.Module):
    """
    Abstract model class defining the model interface
    """
    def __init__(self, label_encoder, tasks, *args, **kwargs):
        self.label_encoder = label_encoder
        # prepare input task data from task settings
        if isinstance(tasks, list):
            tasks = {task['name']: task for task in tasks}
        self.tasks = tasks
        self.known = set()
        self.ambs = {task: set() for task in tasks}
        self._fitted_trainset_scorer = False
        super().__init__()

    def get_scorer(self, task, trainset=None):
        """ Given a task, gets a scorer. Trainset can be used for computing
        unknown and ambiguous tokens.

        :param task: Taskname (str)
        :param trainset: Dataset for training
        :return: Scorer
        """
        scorer = Scorer(self.label_encoder.tasks[task])
        if not self._fitted_trainset_scorer and trainset:
            self.known, self.ambs = get_known_and_ambigous_tokens(
                trainset, list(self.label_encoder.tasks.values()))
            self._fitted_trainset_scorer = True
        scorer.set_known_and_amb(self.known, self.ambs[task])
        return scorer

    def loss(self, batch_data):
        """
        """
        raise NotImplementedError

    def predict(self, inp, *tasks, **kwargs):
        """
        Compute predictions based on already processed input
        """
        raise NotImplementedError

    def get_args_and_kwargs(self):
        """
        Return a dictionary of {'args': tuple, 'kwargs': dict} that were used
        to instantiate the model (excluding the label_encoder and tasks)
        """
        raise NotImplementedError

    def evaluate(self, dataset, trainset=None, **kwargs):
        """
        Get scores per task

        dataset: pie.data.Dataset, dataset to evaluate on (your dev or test set)
        trainset: pie.data.Dataset (optional), if passed scores for unknown and ambiguous
            tokens can be computed
        **kwargs: any other arguments to Model.predict
        """
        assert not self.training, "Ooops! Inference in training mode. Call model.eval()"

        scorers = {task: self.get_scorer(task, trainset) for task in self.tasks}

        with torch.no_grad():
            for (inp, tasks), (rinp, rtasks) in tqdm.tqdm(
                    dataset.batch_generator(return_raw=True)):

                preds = self.predict(inp, **kwargs)

                # - get input tokens
                tokens = [w for line in rinp for w in line]

                # - get trues
                trues = {}
                for task in preds:
                    le = self.label_encoder.tasks[task]
                    # - transform targets
                    trues[task] = le.preprocess(
                        [t for line in rtasks for t in line[le.target]], tokens)

                    # - flatten token level predictions
                    if le.level == 'token':
                        preds[task] = [pred for batch in preds[task] for pred in batch]

                # accumulate
                for task, scorer in scorers.items():
                    scorer.register_batch(preds[task], trues[task], tokens)

        return scorers

    def save(self, fpath, infix=None, settings=None):
        """
        Serialize model to path
        """
        import pie
        fpath = utils.ensure_ext(fpath, 'tar', infix)

        # create dir if necessary
        dirname = os.path.dirname(fpath)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        with tarfile.open(fpath, 'w') as tar:
            # serialize label_encoder
            string = json.dumps(self.label_encoder.jsonify())
            path = 'label_encoder.zip'
            utils.add_gzip_to_tar(string, path, tar)

            # serialize tasks
            string, path = json.dumps(self.tasks), 'tasks.zip'
            utils.add_gzip_to_tar(string, path, tar)

            # serialize model class
            string, path = str(type(self).__name__), 'class.zip'
            utils.add_gzip_to_tar(string, path, tar)

            # serialize parameters
            string, path = json.dumps(self.get_args_and_kwargs()), 'parameters.zip'
            utils.add_gzip_to_tar(string, path, tar)

            # serialize weights
            utils.add_weights_to_tar(self.state_dict(), 'state_dict.pt', tar)

            # serialize current pie commit
            if pie.__commit__ is not None:
                string, path = pie.__commit__, 'pie-commit.zip'
                utils.add_gzip_to_tar(string, path, tar)

            # if passed, serialize settings
            if settings is not None:
                string, path = json.dumps(settings), 'settings.zip'
                utils.add_gzip_to_tar(string, path, tar)

        return fpath

    @staticmethod
    def load_settings(fpath):
        """
        Load settings from path
        """
        with tarfile.open(utils.ensure_ext(fpath, 'tar'), 'r') as tar:
            return Settings(json.loads(utils.get_gzip_from_tar(tar, 'settings.zip')))

    @staticmethod
    def load(fpath):
        """
        Load model from path
        """
        import pie

        with tarfile.open(utils.ensure_ext(fpath, 'tar'), 'r') as tar:
            # check commit
            try:
                commit = utils.get_gzip_from_tar(tar, 'pie-commit.zip')
            except Exception:
                commit = None
            if (pie.__commit__ and commit) and pie.__commit__ != commit:
                logging.warn(
                    ("Model {} was serialized with a previous "
                     "version of `pie`. This might result in issues. "
                     "Model commit is {}, whereas current `pie` commit is {}.").format(
                         fpath, commit, pie.__commit__))

            # load label encoder
            le = MultiLabelEncoder.load_from_string(
                utils.get_gzip_from_tar(tar, 'label_encoder.zip'))

            # load tasks
            tasks = json.loads(utils.get_gzip_from_tar(tar, 'tasks.zip'))

            # load model parameters
            params = json.loads(utils.get_gzip_from_tar(tar, 'parameters.zip'))

            # instantiate model
            model_type = getattr(pie.models, utils.get_gzip_from_tar(tar, 'class.zip'))
            with utils.shutup():
                model = model_type(le, tasks, *params['args'], **params['kwargs'])

            # load settings
            try:
                settings = Settings(
                    json.loads(utils.get_gzip_from_tar(tar, 'settings.zip')))
                model._settings = settings
            except Exception:
                logging.warn("Couldn't load settings for model {}!".format(fpath))

            # load state_dict
            model.load_state_dict(torch.load(tar.extractfile('state_dict.pt'), map_location='cpu'))
        model.eval()

        return model
