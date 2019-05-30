import torch


def log_sum_exp(x, dim=None):
    """
    Sum probabilities in the log-space.
    """
    xmax, _ = x.max(dim=dim, keepdim=True)
    xmax_, _ = x.max(dim=dim)
    # return xmax_
    return xmax_ + torch.log(torch.exp(x - xmax).sum(dim=dim))