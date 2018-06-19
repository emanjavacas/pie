
import yaml
import difflib
from termcolor import colored
from collections import Counter, defaultdict

from sklearn.metrics import precision_score, recall_score, accuracy_score
from pie import utils


def compute_scores(trues, preds):

    def format_score(score):
        return round(float(score), 4)

    with utils.shutup():
        p = format_score(precision_score(trues, preds, average='macro'))
        r = format_score(recall_score(trues, preds, average='macro'))
        a = format_score(accuracy_score(trues, preds))

    return {'accuracy': a, 'precision': p, 'recall': r, 'support': len(trues)}


class Scorer(object):
    """
    Accumulate predictions over batches and compute evaluation scores
    """
    def __init__(self, label_encoder, compute_unknown=False):
        self.label_encoder = label_encoder
        self.compute_unknown = compute_unknown
        self.preds = []
        self.trues = []

    def register_batch(self, hyps, targets):
        """
        hyps : list(batch, seq_len)
        targets : list(batch, seq_len)
        """
        for hyp, target in zip(hyps, targets):
            if isinstance(hyp, (list, tuple)):
                if len(hyp) != len(target):
                    raise ValueError("Unequal hyp {} and target {} lengths"
                                     .format(len(hyp), len(target)))
                self.preds.extend(hyp)
                self.trues.extend(target)
            else:
                self.preds.append(hyp)
                self.trues.append(target)

    def get_scores(self):
        """
        Return a dictionary of scores
        """
        output = compute_scores(self.trues, self.preds)

        # compute scores for unknown tokens
        if self.compute_unknown:
            unk_preds, unk_trues = [], []
            for i, true in enumerate(self.trues):
                if true not in self.label_encoder.known_tokens:
                    unk_trues.append(true)
                    unk_preds.append(self.preds[i])

            support = len(unk_trues)
            if support > 0:
                output['unknown'] = compute_scores(unk_trues, unk_preds)

        return output

    def get_most_common(self, errors, most_common):
        # sort by number of target errors
        def key(item): target, errors = item; return sum(errors.values())

        errors = dict(sorted(errors.items(), key=key, reverse=True))

        output = []
        for true, preds in errors.items():
            if len(output) >= most_common:
                break
            output.append((true, sum(preds.values()), preds))

        return output

    def get_classification_summary(self, most_common=200):
        """
        Get a printable summary for classification errors
        """
        errors = defaultdict(Counter)
        for true, pred in zip(self.trues, self.preds):
            if true != pred:
                errors[true][pred] += 1

        output = ''
        for true, counts, preds in self.get_most_common(errors, most_common):
            true = '{}(#{})'.format(colored(true, 'green'), counts)
            true = '{}(#{})'.format('green', counts)
            preds = Counter(preds).most_common(10)
            preds = ''.join('{:<10}'.format('{}(#{})'.format(p, c)) for p, c in preds)
            output += '{:<10}{}\n'.format(true, preds)

        return output

    def get_transduction_summary(self, most_common=100):
        """
        Get a printable summary of string transduction errors
        """
        def get_diff(true, pred):
            diff = ''
            for action, *_, char in difflib.Differ().compare(true, pred):
                color = {' ': 'white', '+': 'green', '-': 'red'}[action]
                diff += colored(char, color)
            return diff

        def error_summary(true, count, preds):
            summary = '{}(#{})\n'.format(true, count)
            summary += ''.join(
                '\t{:<20}\t{}\n'.format(
                    '{}(#{})'.format(pred, pcount), get_diff(true, pred))
                for pred, pcount in preds.items())
            return summary

        known, unknown = defaultdict(Counter), defaultdict(Counter)
        for true, pred in zip(self.trues, self.preds):
            if true != pred:
                if self.compute_unknown and true not in self.label_encoder.known_tokens:
                    unknown[true][pred] += 1
                else:
                    known[true][pred] += 1

        known_summary = '::: Known tokens :::\n\n'
        for true, count, preds in self.get_most_common(known, most_common):
            known_summary += '{}\n'.format(error_summary(true, count, preds))
        unknown_summary = '::: Unknown tokens :::\n\n'
        for true, count, preds in self.get_most_common(unknown, most_common):
            unknown_summary += '{}\n'.format(error_summary(true, count, preds))

        return '{}\n\n{}'.format(known_summary, unknown_summary)

    def print_summary(self, full=False, most_common=100):
        """
        Get evaluation summary
        """
        print()
        print("::: Evaluation report for task: {} :::".format(self.label_encoder.name))
        print()

        # print scores
        print(yaml.dump(self.get_scores(), default_flow_style=False))

        if full:
            print()
            print("::: Error summary for task: {} :::".format(self.label_encoder.name))
            print()
            if self.label_encoder.level == 'char':
                print(self.get_transduction_summary(most_common=most_common))
            else:
                print(self.get_classification_summary(most_common=most_common))

