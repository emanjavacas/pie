
import os
import uuid
import logging
import time
import collections
import random

import tqdm

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


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


class EarlyStopException(Exception):
    def __init__(self, task, loss, state_dict):
        self.task = task
        self.loss = loss
        self.best_state_dict = state_dict


class TaskScheduler(object):
    """
    Track scores
    """
    def __init__(self, tasks, patience, factor, threshold, min_weight,
                 optimizer=None, lr_factor=1, lr_patience=100):
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

        # lr schedule
        self.optimizer = optimizer
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_steps = 0

        # task schedule
        self.tasks = tasks
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.min_weight = min_weight
        self.fid = '/tmp/{}'.format(str(uuid.uuid1()))

    def get_lr(self):
        # assumes single param group
        return float(self.optimizer.param_groups[0]['lr'])

    def set_lr(self, new_lr):
        self.optimizer.param_groups[0]['lr'] = new_lr

    def __repr__(self):
        # task scheduler
        output = (
            '<TaskScheduler patience="{}" factor="{}" ' +
            'threshold="{}" min_weight="{}">').format(
                self.patience, self.factor, self.threshold, self.min_weight)

        for task, values in self.tasks.items():
            output += '\n    <Task name="{}" '.format(task)
            output += ' '.join('{}="{}"'.format(key, val) for key, val in values.items())
            output += '/>'
        output += '\n</TaskScheduler>'

        # lr scheduler
        if self.optimizer is not None:
            output += '\n'
            output += '<LrScheduler lr="{}" lr_steps="{}" lr_patience="{}"/>'.format(
                round(self.get_lr(), 5), self.lr_steps, self.lr_patience)

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
                    # lr schedule
                    self.lr_steps = 0
            else:
                self.tasks[task]['steps'] += 1
                # lr schedule
                if is_target:
                    self.lr_steps += 1

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

            # lr schedule
            if is_target and self.lr_steps >= self.lr_patience:
                self.set_lr(self.get_lr() * self.lr_factor)

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
        self.tasks = settings.tasks
        self.verbose = settings.verbose
        self.dataset = dataset
        self.model = model
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

        tasks = {}
        for task in settings.tasks:
            tasks[task['name']] = task.get('schedule', {})
            tasks[task['name']]['target'] = task.get('target', False)
        if settings.include_lm:
            tasks['lm_fwd'] = settings.lm_schedule
            tasks['lm_bwd'] = settings.lm_schedule
        self.task_scheduler = TaskScheduler(
            # task schedule
            tasks, settings.patience, settings.factor, settings.threshold,
            settings.min_weight,
            # lr schedule
            optimizer=self.optimizer,
            lr_factor=settings.lr_factor, lr_patience=settings.lr_patience)

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

    def evaluate(self, dataset):
        """
        Evaluate objective on held-out data
        """
        total_losses, total_batches = collections.defaultdict(float), 0

        # get all tasks
        tasks = list(self.model.label_encoder.tasks)

        for batch in tqdm.tqdm(dataset.batch_generator()):
            total_batches += 1
            for k, v in self.model.loss(batch, *tasks).items():
                total_losses[k] += v.item()

        for k, v in total_losses.items():
            total_losses[k] = v / total_batches

        return dict(total_losses)

    def run_check(self, devset):
        """
        Monitor dev loss and eventually early-stop training
        """
        print()
        print("Evaluating model on dev set...")
        print()

        self.model.eval()

        with torch.no_grad():
            dev_loss = self.evaluate(devset)
            print()
            print("::: Dev losses :::")
            print()
            print('\n'.join('{}: {:.3f}'.format(k, v) for k, v in dev_loss.items()))
            print()
            summary = self.model.evaluate(devset, self.dataset)
            for task in summary.values():
                task.print_summary()

        self.model.train()
        dev_scores = {}
        for task, scorer in summary.items():
            dev_scores[task] = scorer.get_scores()['all']['accuracy']
        # add lm scores
        if 'lm_fwd' in dev_loss or 'lm_bwd' in dev_loss:
            dev_scores['lm_fwd'] = dev_loss['lm_fwd']
            dev_scores['lm_bwd'] = dev_loss['lm_bwd']

        self.task_scheduler.step(dev_scores, self.model)

        if self.verbose:
            print(self.task_scheduler)
            print()

        return dev_scores

    def train_epoch(self, devset, epoch):
        rep_loss = collections.defaultdict(float)
        rep_batches = collections.defaultdict(int)
        rep_items, rep_start = 0, time.time()
        scores = None

        for b, batch in enumerate(self.dataset.batch_generator()):
            # get loss
            loss = self.model.loss(batch, get_batch_task(self.tasks))

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
                    rep += '{}:{:.3f}  '.format(t, rep_loss[t] / rep_batches[t])
                logging.info("Batch [{}/{}] || {} || {:.0f} w/s".format(
                    b, self.num_batches, rep, rep_items / (time.time() - rep_start)))
                rep_loss = collections.defaultdict(float)
                rep_batches = collections.defaultdict(int)
                rep_items, rep_start = 0, time.time()

            if self.check_freq > 0 and b > 0 and b % self.check_freq == 0:
                if devset is not None:
                    scores = self.run_check(devset)

        return scores

    def train_epochs(self, epochs, devset=None):
        """
        Train the model for a number of epochs
        """
        start = time.time()
        scores = None

        try:
            for epoch in range(1, epochs + 1):
                # train epoch
                epoch_start = time.time()
                logging.info("Starting epoch [{}]".format(epoch))
                self.train_epoch(devset, epoch)
                epoch_total = time.time() - epoch_start
                logging.info("Finished epoch [{}] in [{:g}] secs".format(
                    epoch, epoch_total))

        except EarlyStopException as e:
            logging.info("Early stopping training: "
                         "task [{}] with best score {:.5f}".format(e.task, e.loss))

            self.model.load_state_dict(e.best_state_dict)
            scores = {e.task: e.loss}

        logging.info("Finished training in [{:g}]".format(time.time() - start))

        # will be None if no dev test was provided or the model failed to converge
        return scores
