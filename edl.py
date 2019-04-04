import torch

def relu_evidence(logits):
    return torch.nn.ReLU(logits)

def mse_loss(p, alpha, global_step, annealing_step):
    S = alpha.sum(dim=1, keep_dims=True)
    E = alpha - 1
    m = alpha/ S

    A = ((p-m)**2).sum(dim=1, keep_dims=True)
    B = (alpha*(S-alpha)/(S*S*(S+1))).sum(dim=1, keep_dims=True)

    annealing_coef = torch.min(1.0, (global_step/annealing_step).float())
    alp = E*(1-p) + 1
    C = annealing_coef * KL(alp)
    return (A + B) + C
