import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = round(self._data.total[key] / self._data.counts[key], 6)

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        return torch.sum(pred == target).item() / len(target)


def macro_f(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        score = f1_score(target.cpu(), pred.cpu(), average="macro")
        return score


def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.
        k

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.
        k

    Returns:
        np.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def group_auc(labels, pred):
    return round(np.mean([roc_auc_score(label, p) for label, p in zip(labels, pred)]).item(), 4)


def mean_mrr(labels, pred):
    return round(np.mean([mrr_score(label, p) for label, p in zip(labels, pred)]).item(), 4)


def ndcg(labels, pred, k):
    return round(np.mean([ndcg_score(label, p, k) for label, p in zip(labels, pred)]).item(), 4)


def ndcg_5(labels, pred):
    return ndcg(labels, pred, 5)


def ndcg_10(labels, pred):
    return ndcg(labels, pred, 10)
