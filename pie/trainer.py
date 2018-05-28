
import logging
import yaml
import time
from collections import defaultdict

import tqdm

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


class Trainer(object):
    """
    Trainer
    """
    def __init__(self, dataset, model, settings):

        self.dataset = dataset
        self.model = model
        self.optim = getattr(optim, settings.optim)(model.parameters(), lr=settings.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, patience=3, verbose=settings.verbose, factor=0.25)
        self.clip_norm = settings.clip_norm
        self.weights = settings.weights

        self.report_freq = settings.report_freq
        num_batches = len(dataset) - 1
        if settings.checks_per_epoch == 1:
            self.check_freq = num_batches
        elif settings.checks_per_epoch > 1:
            self.check_freq = num_batches // settings.checks_per_epoch
        elif settings.checks_per_epoch > num_batches:
            self.check_freq = 1
        else:
            self.check_freq = 0

    def model_report(self):
        """
        Show model report
        """
        nparams = sum(p.nelement() for p in self.model.parameters())
        logging.info("::: Model :::")
        logging.info('')
        logging.info('\n' + str(self.model))
        logging.info('')
        logging.info("::: Model parameters :::")
        logging.info('')
        logging.info(nparams)
        logging.info('')

    def weight_loss(self, loss):
        """
        Apply weights to losses and return a single loss number
        """
        if self.weights is not None:
            loss = sum(self.weights[k] * loss[k] for k in loss)
        else:
            loss = sum(loss.values())

        return loss

    def evaluate(self, dataset):
        """
        Evaluate objective on held-out data
        """
        total_losses, total_batches = defaultdict(float), 0

        for batch in tqdm.tqdm(dataset, total=len(dataset)):
            total_batches += 1
            for k, v in self.model.loss(batch).items():
                total_losses[k] += v.item()

        for k, v in total_losses.items():
            total_losses[k] = v / total_batches

        return dict(total_losses)

    def monitor_batch(self, batch, items, start, nbatches, loss, sep='   '):
        """
        Logging.Info the report for monitoring
        """
        total = len(self.dataset)
        rep = sep.join('{}:{:.3f}'.format(k, v / nbatches) for k, v in loss.items())
        speed = items / (time.time() - start)
        formatter = "Batch [{}/{}] || {} || {:.0f} words/sec"
        logging.info(formatter.format(batch, total, rep, speed))

    def run_check(self, dev):
        """
        Monitor dev loss and eventually early-stop training
        """
        with torch.no_grad():
            self.model.eval()
            dev_loss = self.evaluate(dev)
            self.model.train()
            # scores
            dev_scores = self.model.evaluate(dev)

        logging.info("Dev losses")
        logging.info('')
        logging.info('\n'.join('{}: {:.3f}'.format(k, v) for k, v in dev_loss.items()))
        logging.info('')
        self.scheduler.step(sum(dev_loss[k] for k in ('pos', 'lemma')))

        logging.info("Dev scores")
        logging.info('')
        logging.info(yaml.dump(dev_scores, default_flow_style=False))
        logging.info('')

    def train_epoch(self, dev):
        rep_loss, rep_items, rep_batches = defaultdict(float), 0, 0
        rep_start = time.time()

        for b, batch in enumerate(self.dataset.batch_generator()):
            # get loss
            loss = self.model.loss(batch)

            # optimize
            self.optim.zero_grad()
            self.weight_loss(loss).backward()
            if self.clip_norm is not None:
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
                rep_loss, rep_items, rep_batches = defaultdict(float), 0, 0
                rep_start = time.time()

            if self.check_freq > 0 and b > 0 and b % self.check_freq == 0:
                if dev is not None:
                    logging.info("Evaluating model on dev set...")
                    self.run_check(dev)
                    logging.info('')

    def train_epochs(self, epochs, dev):
        """
        Train the model for a number of epochs
        """
        start = time.time()

        for e in range(1, epochs + 1):
            # train epoch
            epoch_start = time.time()
            logging.info("Starting epoch [{}]".format(e))
            self.train_epoch(dev)
            epoch_total = time.time() - epoch_start
            logging.info("Finished epoch [{}] in [{:g}] secs\n".format(e, epoch_total))

        logging.info("Finished training in [{:g}]".format(time.time() - start))

    def train_model(self, epochs, dev=None):
        self.model_report()

        self.model.train()
        self.model.to(self.dataset.device)  # put on same device as dataset

        self.train_epochs(epochs, dev)
