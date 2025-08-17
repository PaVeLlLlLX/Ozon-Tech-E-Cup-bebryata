from torch import nn


def binary_cross_entropy_loss(output, target):
    return nn.BCELoss(output, target)
