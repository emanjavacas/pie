
import itertools
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
    def __init__(self, label_encoder, known_tokens):
        self.label_encoder = label_encoder
        self.known_tokens = known_tokens
        self.preds = []
        self.trues = []
        self.tokens = []

    def serialize_preds(self, path):
        """
        write predictions to file
        """
        with open(path, 'w') as f:
            for tok, true, pred in zip(self.tokens, self.trues, self.preds):
                f.write('{}\t{}\t{}\n'.format(tok, true, pred))

    def register_batch(self, hyps, targets, tokens):
        """
        hyps : list
        targets : list
        tokens : list
        """
        if len(hyps) != len(targets) or len(targets) != len(tokens):
            raise ValueError("Unequal input lengths. Hyps {}, targets {}, tokens {}"
                             .format(len(hyps), len(targets), len(tokens)))

        for pred, true, token in zip(hyps, targets, tokens):
            self.preds.append(pred)
            self.trues.append(true)
            self.tokens.append(token)

    def get_scores(self):
        """
        Return a dictionary of scores
        """
        output = compute_scores(self.trues, self.preds)

        # compute scores for ambiguous tokens
        tok2class = defaultdict(Counter)
        for tok, true in zip(self.tokens, self.trues):
            tok2class[tok][true] += 1

        # compute scores for unknown input tokens
        unk_trues, unk_preds, amb_trues, amb_preds = [], [], [], []
        for true, pred, token in zip(self.trues, self.preds, self.tokens):
            if token not in self.known_tokens:
                unk_trues.append(true)
                unk_preds.append(pred)
            if len(tok2class[token]) > 1:
                amb_trues.append(true)
                amb_preds.append(pred)
        support = len(unk_trues)
        if support > 0:
            output['unknown-tokens'] = compute_scores(unk_trues, unk_preds)
        support = len(amb_trues)
        if support > 0:
            output['ambiguous-tokens'] = compute_scores(amb_trues, amb_preds)

        # compute scores for unknown targets
        if self.label_encoder.known_tokens:
            # token-level encoding doesn't have unknown targets (only OOV)
            unk_trues, unk_preds = [], []
            for true, pred in zip(self.trues, self.preds):
                if true not in self.label_encoder.known_tokens:
                    unk_trues.append(true)
                    unk_preds.append(pred)
            support = len(unk_trues)
            if support > 0:
                output['unknown-targets'] = compute_scores(unk_trues, unk_preds)

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

        known = defaultdict(Counter)
        unk_targets = defaultdict(Counter)
        unk_tokens = defaultdict(Counter),
        tokens = self.tokens or itertools.repeat(None)
        for true, pred, token in zip(self.trues, self.preds, tokens):
            if true == pred:
                continue

            unk_target = true not in self.label_encoder.known_tokens
            unk_token = token and self.known_tokens and token not in self.known_tokens
            if not unk_target and not unk_token:
                known[true][pred] += 1
            else:
                if unk_target:
                    unk_targets[true][pred] += 1
                if unk_token:
                    unk_tokens[true][pred] += 1

        known_summary = '::: Known tokens :::\n\n'
        for true, count, preds in self.get_most_common(known, most_common):
            known_summary += '{}\n'.format(error_summary(true, count, preds))
        unk_target_summary = '::: Unknown targets :::\n\n'
        for true, count, preds in self.get_most_common(unk_targets, most_common):
            unk_target_summary += '{}\n'.format(error_summary(true, count, preds))
        unk_tokens_summary = '::: Unknown tokens :::\n\n'
        for true, count, preds in self.get_most_common(unk_tokens, most_common):
            unk_tokens_summary += '{}\n'.format(error_summary(true, count, preds))

        return '{}\n{}\n{}'.format(known_summary, unk_target_summary, unk_tokens_summary)

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
