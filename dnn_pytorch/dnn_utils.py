import numpy as np
import torch
from dnn_pytorch import FloatTensor


def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def init_variable(shape, zero=False):
    if zero:
        value = np.zeros(shape)  # bias are initialized with zeros
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    return value


def init_param(layer):
    """
    randomly initialize parameters of the given layer
    """
    for p in layer.parameters():
        p.data = torch.from_numpy(init_variable(p.size())).type(FloatTensor)

    return layer


def log_sum_exp(x, dim=None):
    """
    Sum probabilities in the log-space.
    """
    xmax, _ = x.max(dim=dim, keepdim=True)
    xmax_, _ = x.max(dim=dim)
    # return xmax_
    return xmax_ + torch.log(torch.exp(x - xmax).sum(dim=dim))


def sequence_mask(batch_len, max_len=None):
    if not max_len:
        max_len = np.max(batch_len)

    mask = np.zeros((len(batch_len), max_len))
    for i in range(len(batch_len)):
        mask[i, range(batch_len[i])] = 1

    return mask
