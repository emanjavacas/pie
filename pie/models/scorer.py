
import yaml
import difflib
from termcolor import colored
from terminaltables import github_table
from collections import Counter, defaultdict

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
import numpy as np
from pie import utils
from pie import constants


def get_known_and_ambigous_tokens(trainset, label_encoders):
    """ Retrieve known and ambiguous token for all label encoders, for one dataset

    :param trainset: Trainset
    :param label_encoders: List of label encoders
    :return: Known set of tokens, ambiguous token par task (Dict[task, Set[str])
    """
    known = set()
    ambs = defaultdict(lambda: defaultdict(Counter))
    targets = [le.target for le in label_encoders]
    for _, (inp, tasks) in trainset.reader.readsents():
        known.update(inp)
        for le in label_encoders:
            for tok, true in zip(inp, le.preprocess(tasks[le.target], inp)):
                ambs[le.target][tok][true] += 1
    ambs = {t: set(tok for tok in ambs[t] if len(ambs[t][tok]) > 1) for t in ambs}
    return known, ambs


def compute_scores(trues, preds):

    def format_score(score):
        return round(float(score), 4)

    with utils.shutup():
        p, r, f1, _ = precision_recall_fscore_support(trues, preds, average="macro")
        p = format_score(p)
        r = format_score(r)
        a = format_score(accuracy_score(trues, preds))

    return {'accuracy': a, 'precision': p, 'recall': r, 'support': len(trues)}


class Scorer(object):
    """
    Accumulate predictions over batches and compute evaluation scores
    """
    def __init__(self, label_encoder):
        self.label_encoder = label_encoder
        self.known_tokens = self.amb_tokens = None
        self.preds = []
        self.trues = []
        self.tokens = []

    def set_known_and_amb(self, known_tokens, amb_tokens):
        """ Set known tokens as well as ambiguous tokens """
        self.known_tokens = known_tokens
        self.amb_tokens = amb_tokens

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
            if self.label_encoder.preprocessor_fn:
                true = self.label_encoder.preprocessor_fn.inverse_transform(true, token)
                try:
                    pred = self.label_encoder.preprocessor_fn.inverse_transform(
                        pred, token)
                except:
                    pred = constants.INVALID

            self.preds.append(pred)
            self.trues.append(true)
            self.tokens.append(token)

    def get_scores(self):
        """
        Return a dictionary of scores
        """
        output = {}
        output['all'] = compute_scores(self.trues, self.preds)

        # apply text transformations to known tokens
        known_targets = None
        if self.label_encoder.known_tokens:
            known_targets = set(self.label_encoder.preprocess_text(
                list(self.label_encoder.known_tokens)))

        # compute scores for unknown input tokens
        unk_trues, unk_preds, amb_trues, amb_preds = [], [], [], []
        knw_trues, knw_preds, unk_trg_trues, unk_trg_preds = [], [], [], []
        for true, pred, token in zip(self.trues, self.preds, self.tokens):
            if self.known_tokens and token in self.known_tokens:
                knw_trues.append(true)
                knw_preds.append(pred)
            if self.known_tokens and token not in self.known_tokens:
                unk_trues.append(true)
                unk_preds.append(pred)
            if self.amb_tokens and token in self.amb_tokens:
                amb_trues.append(true)
                amb_preds.append(pred)
            # token-level encoding doesn't have unknown targets (only OOV)
            if known_targets and true not in known_targets:
                unk_trg_trues.append(true)
                unk_trg_preds.append(pred)

        output['known-tokens'] = compute_scores(knw_trues, knw_preds)
        support = len(unk_trues)
        if support > 0:
            output['unknown-tokens'] = compute_scores(unk_trues, unk_preds)
        support = len(amb_trues)
        if support > 0:
            output['ambiguous-tokens'] = compute_scores(amb_trues, amb_preds)
        support = len(unk_trg_trues)
        if support > 0:
            output['unknown-targets'] = compute_scores(unk_trg_trues, unk_trg_preds)

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

    def get_confusion_matrix(self):
        errors = defaultdict(Counter)
        for true, pred in zip(self.trues, self.preds):
            if true != pred:
                errors[true][pred] += 1

        return errors

    def get_confusion_matrix_table(self) -> list:
        """
        Returns a table formated confusion matrix
        """
        matrix = self.get_confusion_matrix()
        table = []
        # Retrieve each true prediction and its dictionary of errors
        for expected, pred_counter in matrix.items():
            counts = [(word, counter) for word, counter in sorted(
                pred_counter.items(), key=lambda tup: tup[1], reverse=True)]
            total = sum(pred_counter.values())
            table.append((expected, total, counts))
        # Sort by error sum
        table = sorted(table, reverse=True, key=lambda tup: tup[1])
        # Then, we expand lines
        output = []
        for word, total, errors in table:
            for index, (prediction, counter) in enumerate(errors):
                row = ["", ""]
                if index == 0:
                    row = [word, total]
                row += [prediction, counter]
                output.append(row)
        return [["Expected", "Total Errors", "Predictions", "Predicted times"]] + output

    def get_classification_summary(self, most_common=200):
        """
        Get a printable summary for classification errors
        """
        errors = self.get_confusion_matrix()

        output = ''
        for true, counts, preds in self.get_most_common(errors, most_common):
            true = '{}(#{})'.format(colored(true, 'green'), counts)
            preds = Counter(preds).most_common(10)
            preds = '\t'.join('{}'.format('{}(#{})'.format(p, c)) for p, c in preds)
            output += '{}\t{}\n'.format(true, preds)

        return output

    def get_transduction_summary(self, most_common=100):
        """
        Get a printable summary of string transduction errors
        """
        COLORS = {' ': 'white', '+': 'green', '-': 'red'}

        def get_diff(true, pred):
            diff = ''
            for action, *_, char in difflib.Differ().compare(true, pred):
                diff += colored(char, COLORS[action])
            return diff

        def error_summary(true, count, preds):
            summary = '{}(#{})\n'.format(true, count)
            summary += ''.join(
                '\t{:<20}\t{}\n'.format(
                    '{}(#{})'.format(pred, pcount), get_diff(true, pred))
                for pred, pcount in preds.items())
            return summary

        known_targets = self.label_encoder.known_tokens
        known, unk_trg = defaultdict(Counter), defaultdict(Counter)
        unk_tok, amb_tok = defaultdict(Counter), defaultdict(Counter)
        for true, pred, token in zip(self.trues, self.preds, self.tokens):
            if true == pred:
                continue
            if self.known_tokens and token in self.known_tokens:
                known[true][pred] += 1
            if known_targets and true not in known_targets:
                unk_trg[true][pred] += 1
            if self.known_tokens and token not in self.known_tokens:
                unk_tok[true][pred] += 1
            if self.amb_tokens and token in self.amb_tokens:
                amb_tok[true][pred] += 1

        summary = []
        summary_ = '::: Known tokens :::\n\n'
        for true, count, preds in self.get_most_common(known, most_common):
            summary_ += '{}\n'.format(error_summary(true, count, preds))
        summary += [summary_]
        if unk_trg:
            summary_ = '::: Unknown targets :::\n\n'
            for true, count, preds in self.get_most_common(unk_trg, most_common):
                summary_ += '{}\n'.format(error_summary(true, count, preds))
            summary += [summary_]
        if amb_tok:
            summary_ = '::: Ambiguous tokens :::\n\n'
            for true, count, preds in self.get_most_common(amb_tok, most_common):
                summary_ += '{}\n'.format(error_summary(true, count, preds))
            summary += [summary_]
        if unk_tok:
            summary_ = '::: Unknown tokens :::\n\n'
            for true, count, preds in self.get_most_common(unk_tok, most_common):
                summary_ += '{}\n'.format(error_summary(true, count, preds))
            summary += [summary_]

        return '\n'.join(summary)

    def print_summary(self, full=False, most_common=100, confusion_matrix=False,
                      scores=None, report=False, markdown=True):
        """
        Get evaluation summary

        :param full: Get full report with error summary
        :param confusion_matrix: Get a confusion matrix
        :param most_common: Limit the full report to the number indicated
        :param scores: If scores are already computed, get passed here
        """

        print()
        if markdown:
            print("## " + self.label_encoder.name)
        else:
            print("::: Evaluation report for task: {} :::".format(
                self.label_encoder.name))
        print()

        if scores is None:
            scores = self.get_scores()

        # print scores
        if markdown:
            print(self.scores_in_markdown(scores) + '\n')
        else:
            print(yaml.dump(scores, default_flow_style=False))

        if full:
            print()
            if markdown:
                print("### Error summary for task {}".format(self.label_encoder.name))
            else:
                print("::: Error summary for task: {} :::".format(
                    self.label_encoder.name))
            print()
            if self.label_encoder.level == 'char':
                print(self.get_transduction_summary(most_common=most_common))
            else:
                print(self.get_classification_summary(most_common=most_common))

        if report:
            print()
            if markdown:
                print("### {} Classification report".format(self.label_encoder.name))
            else:
                print("::: Classification report :::")
            print()
            print(self.get_classification_report())

        if confusion_matrix:
            print()
            if markdown:
                print("### {} Confusion Matrix".format(self.label_encoder.name))
            else:
                print("::: Confusion Matrix :::")
            print()
            print(github_table.GithubFlavoredMarkdownTable(
                self.get_confusion_matrix_table()).table)

    def get_classification_report(self):
        return classification_report(
            y_true=self.trues,
            y_pred=self.preds)

    @staticmethod
    def scores_in_markdown(scores):
        measures = ["accuracy", "precision", "recall", "support"]
        table = [[""] + measures]
        for key in scores:
            table.append([key, *[scores[key][meas] for meas in measures]])

        return (github_table.GithubFlavoredMarkdownTable(table)).table


def classification_report(y_true, y_pred, digits=2):
    """ Generate a classification report similar to
    sklearn.metrics.classification_report but in markdown

    :param y_true: List of GT values
    :param y_pred: List of predictions
    :param digits: Number of float digits
    :return: Github Markdown Table
    """
    floatfmt = "{0:" + '.{:}f'.format(digits) + "}"

    labels = unique_labels(y_true, y_pred)
    target_names = [str(key) for key in labels]

    last_line_heading = 'avg / total'
    headers = ["target", "precision", "recall", "f1-score", "support"]

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, average=None)

    formatted = []
    for nb_list in [p, r, f1]:
        formatted.append([floatfmt.format(x) for x in nb_list.tolist()])
    support = [[str(x) for x in s.tolist()]]

    tbl_rows = list(zip(target_names, *formatted, *support))

    # compute averages
    last_row = [last_line_heading,
                floatfmt.format(np.average(p)),
                floatfmt.format(np.average(r)),
                floatfmt.format(np.average(f1)),
                str(np.sum(s))]
    tbl_rows.append(last_row)

    return github_table.GithubFlavoredMarkdownTable([headers] + tbl_rows).table
