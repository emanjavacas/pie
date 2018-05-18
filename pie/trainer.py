
import time
from collections import defaultdict

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_


class Trainer(object):
    """
    Trainer
    """
    def __init__(self, dataset, model, settings):

        self.dataset = dataset
        self.model = model
        self.optim = getattr(optim, settings.optim)(model.parameters(), lr=settings.lr)
        self.clip_norm = settings.clip_norm
        self.report_freq = settings.report_freq
        self.weights = settings.weights

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
        total_losses = defaultdict(float)

        for batch in dataset:
            for k, v in self.model.loss(batch).items():
                total_losses[k] += v.item()

        for k, v in total_losses:
            total_losses[k] = v / len(dataset)

        return total_losses

    def train_epochs(self, epochs, dev=None):
        """
        Train the model for a number of epochs
        """
        start = time.time()
        self.model.train()
        self.model.to(self.dataset.device)

        for e in range(epochs):
            # reset variables
            print("Starting epoch [{}] ...".format(e))
            epoch_start, report_start = time.time(), time.time()
            report_loss, report_items = defaultdict(float), 0

            for batch, batch_data in enumerate(self.dataset.batch_generator()):
                # get loss
                loss = self.model.loss(batch_data)

                # optimize
                self.optim.zero_grad()
                self.weight_loss(loss).backward()
                if self.clip_norm is not None:
                    clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.optim.step()

                # accumulate
                report_items += self.dataset.get_nelement(batch_data)
                for k, v in loss.items():
                    report_loss[k] += v.item()

                # report
                if batch > 0 and batch % self.report_freq == 0:
                    speed = report_items / (time.time() - report_start)
                    report = ";".join(
                        '{}:{:g}'.format(k, v) for k, v in report_loss.items())
                    print("Batch: {} || Losses: {} || Speed: {:g} words/sec".format(
                        batch, report, speed))
                    report_loss, report_items = defaultdict(float), 0

            print("Finished epoch [{}] in [{:g}] secs ...".format(
                e, time.time() - epoch_start))

            # evaluation
            if dev is not None:
                print("Evaluating model on dev set...")
                with torch.no_grad():
                    self.model.eval()
                    dev_loss = self.evaluate(dev)
                    report = ";".join('{}:{:g}'.format(k, v) for k, v in dev_loss.items())
                    self.model.train()
                print("Evaluation loss: {}".format(report))

        print("Finished training in [{:g}]".format(time.time() - start))
