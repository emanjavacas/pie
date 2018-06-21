
import os
import shutil
import uuid
import json
import tarfile
import gzip
import logging

import tqdm
import torch
import torch.nn as nn

from pie import utils
from pie.data import MultiLabelEncoder
from pie.settings import Settings

from .scorer import Scorer


def add_gzip_to_tar(string, subpath, tar):
    fid = str(uuid.uuid1())
    tmppath = '/tmp/{}'.format(fid)

    with gzip.GzipFile(tmppath, 'w') as f:
        f.write(string.encode())

    tar.add(tmppath, arcname=subpath)
    os.remove(tmppath)


def get_gzip_from_tar(tar, fpath):
    return gzip.open(tar.extractfile(fpath)).read().decode().strip()


class BaseModel(nn.Module):
    """
    Abstract model class defining the model interface
    """
    def __init__(self, label_encoder, *args, **kwargs):
        self.label_encoder = label_encoder
        super().__init__()

    def loss(self, batch_data):
        """
        """
        raise NotImplementedError

    def predict(self, inp, *tasks):
        """
        Compute predictions based on already processed input
        """
        raise NotImplementedError

    def get_args_and_kwargs(self):
        """
        Return a dictionary of {'args': tuple, 'kwargs': dict} that were used
        to instantiate the model (excluding the label_encoder)
        """
        raise NotImplementedError

    def evaluate(self, dataset, total=None):
        """
        Get scores per task
        """
        scorers = {}
        for task, le in self.label_encoder.tasks.items():
            scorers[task] = Scorer(le, compute_unknown=le.level == 'char')

        with torch.no_grad():
            for inp, tasks in tqdm.tqdm(dataset, total=total):
                # get preds
                preds = self.predict(inp)

                # get trues
                trues = {}
                for task, le in self.label_encoder.tasks.items():
                    tinp, tlen = tasks[task]
                    tinp, tlen = tinp.t().tolist(), tlen.tolist()
                    if le.level == 'char':
                        trues[task] = [''.join(le.stringify(t)) for t in tinp]
                    else:
                        trues[task] = [le.stringify(t, l) for t, l in zip(tinp, tlen)]

                # accumulate
                for task, scorer in scorers.items():
                    scorer.register_batch(preds[task], trues[task])

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
            string, path = json.dumps(self.label_encoder.jsonify()), 'label_encoder.zip'
            add_gzip_to_tar(string, path, tar)

            # serialize model class
            string, path = str(type(self).__name__), 'class.zip'
            add_gzip_to_tar(string, path, tar)

            # serialize parameters
            string, path = json.dumps(self.get_args_and_kwargs()), 'parameters.zip'
            add_gzip_to_tar(string, path, tar)

            # serialize weights
            tmppath = '/tmp/{}.'.format(str(uuid.uuid1()))
            torch.save(self.state_dict(), tmppath)
            tar.add(tmppath, arcname='state_dict.pt')
            os.remove(tmppath)

            # serialize current pie commit
            if pie.__commit__ is not None:
                string, path = pie.__commit__, 'pie-commit.zip'
                add_gzip_to_tar(string, path, tar)

            # if passed, serialize settings
            if settings is not None:
                string, path = json.dumps(settings), 'settings.zip'
                add_gzip_to_tar(string, path, tar)

        return fpath

    @staticmethod
    def load(fpath):
        """
        Load model from path
        """
        import pie

        with tarfile.open(utils.ensure_ext(fpath, 'tar'), 'r') as tar:
            # check commit
            try:
                commit = get_gzip_from_tar(tar, 'pie-commit.zip')
            except Exception:
                # no commit in file
                commit = None
            if pie.__commit__ is not None and commit is not None \
               and pie.__commit__ != commit:
                logging.warn(
                    ("Model {} was serialized with a previous "
                     "version of `pie`. This might result in issues. "
                     "Model commit is {}, whereas current `pie` commit is {}."
                    ).format(fpath, commit, pie.__commit__))
            # load label encoder
            le = MultiLabelEncoder.load_from_string(
                get_gzip_from_tar(tar, 'label_encoder.zip'))
            # load model parameters
            params = json.loads(get_gzip_from_tar(tar, 'parameters.zip'))
            # instantiate model
            model_type = getattr(pie.models, get_gzip_from_tar(tar, 'class.zip'))
            with utils.shutup():
                model = model_type(le, *params['args'], **params['kwargs'])
            # (optional) load settings
            try:
                settings = Settings(json.loads(get_gzip_from_tar(tar, 'settings.zip')))
                model._settings = settings
            except:
                pass
            # load state_dict
            tmppath = '/tmp/{}'.format(str(uuid.uuid1()))
            tar.extract('state_dict.pt', path=tmppath)
            model.load_state_dict(torch.load(os.path.join(tmppath, 'state_dict.pt')))
            shutil.rmtree(tmppath)

        model.eval()

        return model
