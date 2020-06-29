import os
import uuid
import logging
import time
import collections
import random
from datetime import datetime

import tqdm

import numpy
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from typing import Tuple, Generator, Dict, Optional, Callable, List

from pie.models import SimpleModel, get_pretrained_embeddings
from pie.data.dataset import Dataset, MultiLabelEncoder
from pie.data.reader import Reader
import pie.initialization
import pie.pretrain_encoder

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

StepCallback = Callable[[int, Dict], None]  # Takes epoch ID + Score Dict


def get_targets(settings: Dict) -> List[str]:
    """ List targets tasks from settings

    :param settings: Settings
    :return: Task's name which were marked as target.
    """
    return [task['name'] for task in settings.tasks if task.get('target')]


def get_fname_infix(settings: Dict) -> Tuple[str, str]:
    """ Based on settings, build the filename and the path

    :param settings: Settings
    :return: Tuple(GenericPath, Unique identifier based on date/time/target tasks) where \
             GenericPath contains the directory path (based on modelpath) \
             and the file contains the modelname
    """
    # fname
    fname = os.path.join(settings.modelpath, settings.modelname)
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    infix = '+'.join(get_targets(settings)) + '-' + timestamp
    return fname, infix


def get_batch_task(tasks, **kwargs):
    tnames, target = [], None
    for task in tasks:
        tnames.append(task['name'])
        if task.get('target'):
            target = task['name']
    return sample_task(target, tnames, **kwargs)


def sample_task(target, tasks, factor=2):
    # sample target task factor times as many as any other task
    aux = (1 / (len(tasks) - 1)) / factor if len(tasks) > 1 else 0
    trg = 1 - (aux * (len(tasks) - 1))
    weights = [aux if task != target else trg for task in tasks]
    return random.choices(tasks, weights)[0]


def get_target_task(settings):
    for task in settings.tasks:
        if task.get('target'):
            return task['name']
    raise ValueError("No target task?")


def set_seed(seed=None, verbose=False) -> int:
    if not seed:
        now = datetime.now()
        # set seed
        seed = now.hour * 10000 + now.minute * 100 + now.second
        if verbose:
            print("Using seed:", seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    return seed


class EarlyStopException(Exception):
    def __init__(self, task, loss, state_dict):
        self.task = task
        self.loss = loss
        self.best_state_dict = state_dict


class TaskScheduler(object):
    """
    Track scores
    """
    def __init__(self, settings):
        tasks = {}
        # preprocess tasks
        for task in settings.tasks:
            # add schedule and target
            tasks[task['name']] = task.get('schedule', {})
            tasks[task['name']]['target'] = task.get('target', False)
            # add task data for lm loss
            if settings.include_lm:
                tasks['lm_fwd'] = dict(settings.lm_schedule)
                tasks['lm_bwd'] = dict(settings.lm_schedule)

        for task, tdata in tasks.items():
            # set step counter
            tdata['steps'] = 0
            # set default task mode
            tdata['mode'] = tdata.get('mode', 'max')
            # set initial weight
            tdata['weight'] = tdata.get('weight', 1.0)
            # set initial best
            tdata['best'] = -float('inf') if tdata['mode'] == 'max' else float('inf')

        self.tasks = tasks

        # task schedule
        self.patience = settings.patience
        self.factor = settings.factor
        self.threshold = settings.threshold
        self.min_weight = settings.min_weight
        self.fid = '/tmp/{}'.format(str(uuid.uuid1()))

    def __repr__(self):
        # task scheduler
        output = (
            '<TaskScheduler patience="{}" factor="{}" ' +
            'threshold="{}" min_weight="{}">').format(
                self.patience, self.factor, self.threshold, self.min_weight)

        for task, values in self.tasks.items():
            output += '\n    <Task name="{}" '.format(task)
            output += ' '.join(
                '{}="{}"'.format(key, val) for key, val in values.items())
            output += '/>'
        output += '\n</TaskScheduler>'

        return output

    def is_best(self, task, value):
        threshold = self.tasks[task].get('threshold', self.threshold)
        mode = self.tasks[task]['mode']
        if mode.lower() == 'max':
            return value > (self.tasks[task]['best'] + threshold)
        elif mode.lower() == 'min':
            return value < (self.tasks[task]['best'] - threshold)
        else:
            raise ValueError("Wrong mode value [{}] for task: {}".format(mode, task))

    def step(self, scores, model):
        """
        Advance schedule step based on dev scores
        """
        for task, score in scores.items():
            if task not in self.tasks:
                # ignore
                continue

            is_target = self.tasks[task].get('target', False)

            # check if we improve
            if self.is_best(task, score):
                self.tasks[task]['best'] = score
                self.tasks[task]['steps'] = 0
                if is_target:
                    # serialize model params
                    torch.save(model.state_dict(), self.fid)
            else:
                self.tasks[task]['steps'] += 1

            # check if we need to stop globally or downweight a task loss
            patience = self.tasks[task].get('patience', self.patience)
            if self.tasks[task]['steps'] >= patience:
                # maybe stop entire training
                if is_target:
                    state_dict = torch.load(self.fid)
                    os.remove(self.fid)
                    raise EarlyStopException(task, self.tasks[task]['best'], state_dict)
                # update task weight
                else:
                    factor = self.tasks[task].get('factor', self.factor)
                    new_weight = self.tasks[task]['weight'] * factor
                    min_weight = self.tasks[task].get('min_weight', self.min_weight)
                    self.tasks[task]['weight'] = max(new_weight, min_weight)

    def get_weights(self):
        return {task: self.tasks[task]['weight'] for task in self.tasks}


class LRScheduler(object):
    def __init__(self, optimizer, threshold=0.0, **kwargs):
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', threshold=threshold, **kwargs)

    def step(self, score):
        self.lr_scheduler.step(score)

    def __repr__(self):
        return '<LrScheduler lr="{:g}" steps="{}" patience="{}" threshold="{}"/>' \
            .format(self.lr_scheduler.optimizer.param_groups[0]['lr'],
                    self.lr_scheduler.num_bad_epochs,
                    self.lr_scheduler.patience,
                    self.lr_scheduler.threshold)


class Trainer(object):
    """
    Trainer

    Settings
    ========
    optim
    lr (lr_factor, lr_patience, min_lr)
    clip_norm
    weights
    report_freq
    checks_per_epoch
    """
    def __init__(self, settings, model, dataset, num_instances):
        self.target_task = get_target_task(settings)
        self.verbose = settings.verbose
        self.dataset = dataset
        self.model: SimpleModel = model
        self.optimizer = getattr(optim, settings.optimizer)(
            model.parameters(), lr=settings.lr)
        self.clip_norm = settings.clip_norm

        self.report_freq = settings.report_freq
        self.num_batches = num_instances // dataset.batch_size
        if settings.checks_per_epoch == 1:
            self.check_freq = self.num_batches - 1  # check after last batch
        elif settings.checks_per_epoch > self.num_batches:
            self.check_freq = 1  # check after each batch
        elif settings.checks_per_epoch > 1:
            self.check_freq = self.num_batches // settings.checks_per_epoch  # check just
        else:
            self.check_freq = 0  # no checks

        self.task_scheduler = TaskScheduler(settings)
        self.lr_scheduler = LRScheduler(
            self.optimizer, factor=settings.lr_factor,
            patience=settings.lr_patience, min_lr=settings.min_lr)

        if settings.verbose:
            print()
            print("Evaluation check every {}/{} batches".format(
                self.check_freq, self.num_batches))
            print()
            print("::: Task schedules :::")
            print()
            print(self.task_scheduler)
            print()
            print("::: LR schedule :::")
            print()
            print(self.lr_scheduler)
            print()

    @classmethod
    def setup(cls, settings) -> Tuple["Trainer", Reader, Optional[Dataset]]:
        """ Setup the trainer instance through settings

        You can access the model, the encoder through
        >>> trainer.model, trainer.model.label_encoder

        You can access the trainset through
        >>> trainer.dataset

        :param settings: Settings to setup the trainer
        :return:  (Trainer instance, Reader, Devset)
        """
        # datasets
        reader = Reader(settings, settings.input_path)
        tasks = reader.check_tasks(expected=None)

        # label encoder
        label_encoder: MultiLabelEncoder = MultiLabelEncoder.from_settings(settings, tasks=tasks)
        label_encoder.fit_reader(reader)

        trainset = Dataset(settings, reader, label_encoder)

        devset = None
        if settings.dev_path:
            devset = Dataset(settings, Reader(settings, settings.dev_path), label_encoder)
        else:
            logging.warning("No devset: cannot monitor/optimize training")

        # model
        model: SimpleModel = SimpleModel(
            label_encoder, settings.tasks,
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
        if model.wemb is not None:
            if settings.pretrain_embeddings:
                print("Pretraining word embeddings")
                wemb_reader = Reader(
                    settings, settings.input_path, settings.dev_path, settings.test_path)
                weight = get_pretrained_embeddings(
                    wemb_reader, label_encoder, size=settings.wemb_dim,
                    window=5, negative=5, min_count=1)
                model.wemb.weight.data = torch.tensor(weight, dtype=torch.float32)

            elif settings.load_pretrained_embeddings:
                print("Loading pretrained embeddings")
                if not os.path.isfile(settings.load_pretrained_embeddings):
                    print("Couldn't find pretrained eembeddings in: {}".format(
                        settings.load_pretrained_embeddings))
                pie.initialization.init_pretrained_embeddings(
                    settings.load_pretrained_embeddings, label_encoder.word, model.wemb)

        # load pretrained weights
        if settings.load_pretrained_encoder:
            model.init_from_encoder(pie.pretrain_encoder.Encoder.load(settings.load_pretrained_encoder))

        # freeze embeddings
        if settings.freeze_embeddings:
            model.wemb.weight.requires_grad = False

        model.to(settings.device)

        trainer = cls(settings, model, trainset, reader.get_nsents())

        #"Trainer", Optional[Dataset], Reader
        return trainer, reader, devset

    def weight_loss(self, loss):
        """
        Apply weights to losses and return a single loss number
        """
        weights = self.task_scheduler.get_weights()

        return sum(weights.get(k, 1) * loss[k] for k in loss)

    def evaluate(self, dataset):
        """
        Evaluate objective on held-out data
        """
        total_losses, total_batches = collections.defaultdict(float), 0

        # get all tasks
        tasks = list(self.model.tasks)

        for batch in tqdm.tqdm(dataset.batch_generator()):
            total_batches += 1
            for k, v in self.model.loss(batch, *tasks).items():
                total_losses[k] += v.item()

        for k, v in total_losses.items():
            total_losses[k] = v / total_batches

        return dict(total_losses)

    def run_check(self, devset, adapt=True) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Monitor dev loss and eventually early-stop training


        :param devset: Dataset
        :param adapt: Use scores to adapt schedulers
        :returns: Dev scores
        """
        print()
        print("Evaluating model on dev set...")
        print()

        self.model.eval()

        # This score dict holds all metrics (F1, Acc, Prec, etc.) for all different categories
        #   like Known tokens, Unknown Targets, All, etc. for each task
        # Eg. {"lemma": {"all": {"accuracy": 0.90, "precision": 0.70, ...}, "known-tokens": {..}...}
        dev_scores = {}

        # This score dict holds metrics used for adapting the task_scheduler and the
        #    lr_scheduler.
        target_scores = {}

        with torch.no_grad():
            dev_loss = self.evaluate(devset)
            print()
            print("::: Dev losses :::")
            print()
            print('\n'.join('{}: {:.4f}'.format(k, v) for k, v in dev_loss.items()))
            print()
            summary = self.model.evaluate(devset, self.dataset)
            for task_name, scorer in summary.items():
                dev_scores[task_name] = scorer.get_scores()
                scorer.print_summary(scores=dev_scores[task_name])

        if not adapt:
            return dev_scores

        self.model.train()
        for task, scored in dev_scores.items():
            target_scores[task] = scored['all']['accuracy']
        # add lm scores
        if 'lm_fwd' in dev_loss or 'lm_bwd' in dev_loss:
            target_scores['lm_fwd'] = dev_loss['lm_fwd']
            target_scores['lm_bwd'] = dev_loss['lm_bwd']

        self.task_scheduler.step(target_scores, self.model)
        self.lr_scheduler.step(target_scores[self.target_task])

        if self.verbose:
            print(self.task_scheduler)
            print()
            print(self.lr_scheduler)
            print()

        return dev_scores

    def train_epoch(self, devset, epoch):
        rep_loss = collections.defaultdict(float)
        rep_batches = collections.defaultdict(int)
        rep_items, rep_start = 0, time.time()
        scores = None

        for b, batch in enumerate(self.dataset.batch_generator()):
            # get loss
            loss = self.model.loss(batch, get_batch_task(self.model.tasks.values()))

            if not loss:
                raise ValueError("Got empty loss, no tasks defined?")

            # optimize
            self.optimizer.zero_grad()
            self.weight_loss(loss).backward()
            if self.clip_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()

            # accumulate
            rep_items += type(self.dataset).get_nelement(batch)
            for k, v in loss.items():
                rep_batches[k] += 1
                rep_loss[k] += v.item()

            # report
            if b > 0 and b % self.report_freq == 0:
                rep = ""
                for t in sorted(rep_loss):
                    rep += '{}:{:.4f}  '.format(t, rep_loss[t] / rep_batches[t])
                logging.info("Batch [{}/{}] || {} || {:.0f} w/s".format(
                    b, self.num_batches, rep, rep_items / (time.time() - rep_start)))
                rep_loss = collections.defaultdict(float)
                rep_batches = collections.defaultdict(int)
                rep_items, rep_start = 0, time.time()

            if self.check_freq > 0 and b > 0 and b % self.check_freq == 0:
                if devset is not None:
                    rep_start = time.time()
                    scores = self.run_check(devset)
                    logging.info("Evaluation time: {:.0f} sec".format(
                        time.time() - rep_start))
                    rep_start = time.time()

        return scores

    def train_epochs(self, epochs, devset=None, epoch_callback: Optional[StepCallback] = None) -> Dict:
        """ Train the model for a number of epochs

        Yields the batch id as well as the accuracy of each task (1, {"taskname": accuracy})
        """
        start = time.time()
        scores: Dict = {}

        try:
            for epoch in range(1, epochs + 1):
                # train epoch
                epoch_start = time.time()
                logging.info("Starting epoch [{}]".format(epoch))
                scores = self.train_epoch(devset, epoch)
                # ToDo: Use this score to dump a json somewhere {epoch_id: epoch_score} ?
                epoch_callback(epoch, scores)

                epoch_total = time.time() - epoch_start
                logging.info("Finished epoch [{}] in [{:.0f}] secs".format(
                    epoch, epoch_total))

        except EarlyStopException as e:
            logging.info("Early stopping training: "
                         "task [{}] with best score {:.4f}".format(e.task, e.loss))

            self.model.load_state_dict(e.best_state_dict)
            # Re-evaluate to get best devscore
            scores = self.run_check(devset, adapt=False)

        logging.info("Finished training in [{:.0f}] secs".format(time.time() - start))

        # will be None if no dev test was provided or the model failed to converge
        return scores
