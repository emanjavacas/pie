
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

    def print_report(self):
        """
        Print training report
        """
        nparams = sum(p.nelement() for p in self.model.parameters())
        print("::: Model :::")
        print("\n\t" + "\n\t".join(str(self.model).split('\n')))
        print()
        print("::: Model parameters :::")
        print("\t{}".format(nparams))
        print()

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

        for batch in dataset:
            total_batches += 1
            for k, v in self.model.loss(batch).items():
                total_losses[k] += v.item()

        for k, v in total_losses.items():
            total_losses[k] = v / total_batches

        return total_losses

    def report(self, batch, items, start, nbatches, loss, sep='   '):
        """
        Print the report for monitoring
        """
        speed = items / (time.time() - start)
        rep = sep.join('{}:{:.3f}'.format(k, v / nbatches) for k, v in loss.items())
        print("Batch: {} || {} || {:.3f} words/sec".format(batch, rep, speed))

    def train_epochs(self, epochs, dev=None):
        """
        Train the model for a number of epochs
        """
        self.print_report()

        start = time.time()
        self.model.train()
        self.model.to(self.dataset.device)  # put on same device as dataset

        for e in range(1, epochs + 1):
            # reset variables
            print("Starting epoch [{}]".format(e))
            epoch_start, rep_start = time.time(), time.time()
            rep_loss, rep_items, rep_batches = defaultdict(float), 0, 0

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
                rep_items += self.dataset.get_nelement(batch)
                rep_batches += 1
                for k, v in loss.items():
                    rep_loss[k] += v.item()

                # report
                if b > 0 and b % self.report_freq == 0:
                    self.report(b, rep_items, rep_start, rep_batches, rep_loss)
                    rep_loss, rep_items, rep_batches = defaultdict(float), 0, 0
                    rep_start = time.time()

            epoch_total = time.time() - epoch_start
            print("Finished epoch [{}] in [{:g}] secs".format(e, epoch_total))

            # evaluation
            if dev is not None:
                print("Evaluating model on dev set")
                with torch.no_grad():
                    self.model.eval()
                    dev_loss = self.evaluate(dev)
                    rep = ";".join('{}:{:g}'.format(k, v) for k, v in dev_loss.items())
                    self.model.train()
                print("Evaluation loss: {}".format(rep))

        print("Finished training in [{:g}]".format(time.time() - start))
