
from sklearn.metrics import precision_score, recall_score, accuracy_score


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
        output = {
            'precision': float(precision_score(self.trues, self.preds, average='macro')),
            'recall': float(recall_score(self.trues, self.preds, average='macro')),
            'accuracy': float(accuracy_score(self.trues, self.preds))}

        if self.compute_unknown:
            unk_preds, unk_trues = [], []
            for i, true in enumerate(self.trues):
                if true not in self.label_encoder.known_tokens:
                    unk_trues.append(true)
                    unk_preds.append(self.preds[i])

            support = len(unk_trues)
            if support > 0:
                output['unknown_support'] = support
                output['unknown_precision'] = float(precision_score(
                    unk_trues, unk_preds, average='macro'))
                output['unknown_recall'] = float(recall_score(
                    unk_trues, unk_preds, average='macro'))
                output['unknown_accuracy'] = float(accuracy_score(
                    unk_trues, unk_preds))

        return output

