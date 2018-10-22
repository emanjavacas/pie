
import os
import uuid
import logging
import time
import collections

import tqdm

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


class EarlyStopException(Exception):
    def __init__(self, task, loss, state_dict):
        self.task = task
        self.loss = loss
        self.best_state_dict = state_dict


class TaskScheduler(object):
    """
    Track scores
    """
    def __init__(self, tasks, patience, factor, threshold, min_weight):
        for task, values in tasks.items():
            tasks[task] = {'steps': 0, **values}
            # set task mode
            if 'mode' not in tasks[task]:
                tasks[task]['mode'] = 'max'
            # set initial weight
            if 'weight' not in tasks[task]:
                tasks[task]['weight'] = 1.0
            # set initial best
            if tasks[task]['mode'] == 'max':
                tasks[task]['best'] = -float('inf')
            else:
                tasks[task]['best'] = float('inf')

        self.tasks = tasks
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.min_weight = min_weight
        self.fid = '/tmp/{}'.format(str(uuid.uuid1()))

    def __repr__(self):
        output = (
            "<TaskScheduler patience=\"{}\" factor=\"{}\" " +
            "threshold=\"{}\" min_weight=\"{}\">").format(
                self.patience, self.factor, self.threshold, self.min_weight)

        for task, values in self.tasks.items():
            output += '\n    <Task name="{}" '.format(task)
            output += ' '.join('{}="{}"'.format(key, val) for key, val in values.items())
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

            # check if we improve
            if self.is_best(task, score):
                self.tasks[task]['best'] = score
                self.tasks[task]['steps'] = 0
                if self.tasks[task].get('target', False):
                    # serialize model params
                    torch.save(model.state_dict(), self.fid)
            else:
                self.tasks[task]['steps'] += 1

            # check if we need to stop globally or downweight a task loss
            patience = self.tasks[task].get('patience', self.patience)
            if self.tasks[task]['steps'] >= patience:
                # maybe stop entire training
                if self.tasks[task].get('target', False):
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


class Trainer(object):
    """
    Trainer

    Settings
    ========
    optim
    lr
    clip_norm
    weights
    report_freq
    checks_per_epoch
    """
    def __init__(self, settings, model, dataset, num_instances):

        self.verbose = settings.verbose
        self.dataset = dataset
        self.model = model
        self.optim = getattr(optim, settings.optimizer)(
            model.parameters(), lr=settings.lr)
        self.lr_scheduler = None
        if settings.lr_factor < 1.0:
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optim, patience=settings.lr_patience, factor=settings.lr_factor,
                verbose=settings.verbose)
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

        tasks = {}
        for task in settings.tasks:
            tasks[task['name']] = task.get('schedule', {})
            tasks[task['name']]['target'] = task.get('target', False)
        if settings.include_lm:
            tasks['fwd_lm'] = settings.lm_schedule
            tasks['bwd_lm'] = settings.lm_schedule
        self.task_scheduler = TaskScheduler(
            tasks, settings.patience, settings.factor, settings.threshold,
            settings.min_weight)

        if settings.verbose:
            print()
            print("Evaluation check every {}/{} batches".format(
                self.check_freq, self.num_batches))
            print()
            print("::: Task schedules :::")
            print()
            print(self.task_scheduler)
            print()

    def weight_loss(self, loss):
        """
        Apply weights to losses and return a single loss number
        """
        weights = self.task_scheduler.get_weights()

        return sum(weights.get(k, 1) * loss[k] for k in loss)

    def evaluate(self, dataset, num_batches=None):
        """
        Evaluate objective on held-out data
        """
        total_losses, total_batches = collections.defaultdict(float), 0

        # get total number of batches
        if isinstance(dataset, collections.Sized):
            total = len(dataset)
        elif num_batches is not None:
            total = num_batches

        for batch in tqdm.tqdm(dataset, total=total):
            total_batches += 1
            for k, v in self.model.loss(batch).items():
                total_losses[k] += v.item()

        for k, v in total_losses.items():
            total_losses[k] = v / total_batches

        return dict(total_losses)

    def monitor_batch(self, batch, items, start, nbatches, loss, sep=' '*3):
        """
        Print the report for monitoring
        """
        rep = sep.join('{}:{:.3f}'.format(k, v / nbatches) for k, v in loss.items())
        speed = items / (time.time() - start)
        formatter = "Batch [{}/{}] || {} || {:.0f} w/s"
        logging.info(formatter.format(batch, self.num_batches, rep, speed))

    def run_check(self, dev):
        """
        Monitor dev loss and eventually early-stop training
        """
        print()
        print("Evaluating model on dev set...")
        print()

        self.model.eval()

        with torch.no_grad():
            dev_loss = self.evaluate(dev)
            print()
            print("::: Dev losses :::")
            print()
            print('\n'.join('{}: {:.3f}'.format(k, v) for k, v in dev_loss.items()))
            print()
            summary = self.model.evaluate(dev)
            for task in summary.values():
                task.print_summary()

        self.model.train()
        dev_scores = {t: scorer.get_scores()['accuracy'] for t, scorer in summary.items()}
        # add lm scores
        if 'fwd_lm' in dev_loss or 'bwd_lm' in dev_loss:
            dev_scores['fwd_lm'] = dev_loss['fwd_lm']
            dev_scores['bwd_lm'] = dev_loss['bwd_lm']

        self.task_scheduler.step(dev_scores, self.model)

        if self.verbose:
            print(self.task_scheduler)
            print()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.weight_loss(dev_loss))

        return dev_scores

    def train_epoch(self, dev):
        rep_loss, rep_items, rep_batches = collections.defaultdict(float), 0, 0
        rep_start = time.time()
        scores = None

        for b, batch in enumerate(self.dataset.batch_generator()):
            # get loss
            loss = self.model.loss(batch)

            if not loss:
                raise ValueError("Got empty loss, no tasks defined?")

            # optimize
            self.optim.zero_grad()
            self.weight_loss(loss).backward()
            if self.clip_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optim.step()

            # accumulate
            rep_items += type(self.dataset).get_nelement(batch)
            rep_batches += 1
            for k, v in loss.items():
                rep_loss[k] += v.item()

            # report
            if b > 0 and b % self.report_freq == 0:
                self.monitor_batch(b, rep_items, rep_start, rep_batches, rep_loss)
                rep_loss, rep_items, rep_batches = collections.defaultdict(float), 0, 0
                rep_start = time.time()

            if self.check_freq > 0 and b > 0 and b % self.check_freq == 0:
                if dev is not None:
                    scores = self.run_check(dev)
                    return scores

    def train_epochs(self, epochs, dev=None):
        """
        Train the model for a number of epochs
        """
        start = time.time()
        scores = None

        try:
            for e in range(1, epochs + 1):
                # train epoch
                epoch_start = time.time()
                logging.info("Starting epoch [{}]".format(e))
                self.train_epoch(dev)
                epoch_total = time.time() - epoch_start
                logging.info("Finished epoch [{}] in [{:g}] secs".format(e, epoch_total))

        except EarlyStopException as e:
            logging.info("Early stopping training: "
                         "task [{}] with best score {:.3f}".format(e.task, e.loss))

            self.model.load_state_dict(e.best_state_dict)
            scores = {e.task: e.loss}

        logging.info("Finished training in [{:g}]".format(time.time() - start))

        # will be None if no dev test was provided or the model failed to converge
        return scores
