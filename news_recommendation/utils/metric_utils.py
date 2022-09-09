from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score
from .auc_utils import roc_auc_score


class MetricTracker:
    def __init__(self, *funcs, writer=None):
        self.writer = writer
        self.funcs = funcs
        keys = [m.__name__ for m in funcs]
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        if key not in self._data.index:
            self._data.loc[key] = [0, 0, 0]
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = round(self._data.total[key] / self._data.counts[key], 6)

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def average(self):
        df = pd.DataFrame.from_dict(self.result_dict, orient="index")
        return dict(np.round(df.mean(), 4))


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
        y_true (np.ndarray): ground-truth label.
        y_score (np.ndarray): predicted label.

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
        y_true (np.ndarray): ground-truth label.
        y_score (np.ndarray): predicted label.
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
        y_true (np.ndarray): ground-truth label.
        y_score (np.ndarray): predicted label.
        k

    Returns:
        np.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def group_auc(label, pred):
    """
    Compute the area under the ROC curve
    :param label: List[np.ndarray] or np.ndarray
    :param pred: List[np.ndarray] or np.ndarray
    :return: roc auc score
    """
    if isinstance(label, list) or len(label.shape) > 1:
        return np.round(np.mean([roc_auc_score(l, p) for l, p in zip(label, pred)]).item(), 4)
    else:
        return roc_auc_score(label, pred)


def mean_mrr(label, pred):
    """
    Compute the mean reciprocal rank
    :param label: List[np.ndarray] or np.ndarray
    :param pred: List[np.ndarray] or np.ndarray
    :return: MRR score
    """
    if isinstance(label, list) or len(label.shape) > 1:
        return np.round(np.mean([mrr_score(l, p) for l, p in zip(label, pred)]).item(), 4)
    else:
        return mrr_score(label, pred)


def ndcg(label, pred, k):
    """
    Compute the normalized discounted cumulative gain
    :param label: List[np.ndarray] or np.ndarray
    :param pred: List[np.ndarray] or np.ndarray
    :param k: the number of evaluated items
    :return: NDCG score
    """
    if isinstance(label, list) or len(label.shape) > 1:
        return np.round(np.mean([ndcg_score(l, p, k) for l, p in zip(label, pred)]).item(), 4)
    else:
        return ndcg_score(label, pred, k)


def ndcg_5(label, pred):
    return ndcg(label, pred, 5)


def ndcg_10(label, pred):
    return ndcg(label, pred, 10)
