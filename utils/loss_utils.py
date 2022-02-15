import torch
import torch.nn as nn
import torch.nn.functional as F


def nll_loss(predict, target):
    return F.nll_loss(predict, target)


def cross_entropy(predict, target):
    return F.cross_entropy(predict, target)


def categorical_loss(predict, target, epsilon=1e-12):
    """
    Computes cross entropy between target (encoded as one-hot vectors) and predict.
    Input: predict (N, k) ndarray
           target (N, k) ndarray
    Returns: scalar
    """
    predict, target = predict.float(), target.float()
    predict = torch.clamp(predict, epsilon, 1. - epsilon)
    return -torch.sum(target * torch.log(predict + 1e-9)) / predict.shape[0]


def bce_loss(predict, target, smooth_lambda=10):
    predict, target = predict.float(), target.float()
    return nn.BCELoss()(nn.Softmax(dim=-1)(smooth_lambda * predict), target)
