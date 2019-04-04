import torch
import torch.nn as nn
import numpy as np

def relu_evidence(logits):
    return torch.nn.ReLU(logits)


def KL(alpha):
    beta = torch.from_numpy(np.ones((1, alpha.shape()[1])))
    S_alpha = alpha.sum(axis=1, keep_dims=True)
    S_beta = beta.sum(axis=1, keep_dims=True)
    lnB = torch.lgamma(S_alpha, axis=1, keep_dims=True)
    lnB_uni = torch.lgamma(beta, axis=1, keep_dims=True).sum(axis=1, keep_dims=True) - torch.lgamma(S_beta, axis=1, keep_dims=True)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = ((alpha - beta) * (dg1 - dg0)).sum(axis=1, keep_dims=True) + lnB + lnB_uni

    return kl

class AdaptedMSELoss(nn._Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(AdaptedMSELoss, self).__init__(weight, size_average, reduce, reduction)
    def forward(self, p, alpha, global_step, annealing_step):
        S = alpha.sum(dim=1, keep_dims=True)
        E = alpha - 1
        m = alpha / S

        A = ((p - m) ** 2).sum(dim=1, keep_dims=True)
        B = (alpha * (S - alpha) / (S * S * (S + 1))).sum(dim=1, keep_dims=True)

        annealing_coef = torch.min(1.0, (global_step / annealing_step).float())
        alp = E * (1 - p) + 1
        C = annealing_coef * KL(alp)
        return (A + B) + C