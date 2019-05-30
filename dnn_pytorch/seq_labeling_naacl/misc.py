import torch
from torch.autograd import Variable


def get_mask(batch_len, max_len=None):
    if not max_len:
        max_len = torch.max(batch_len)

    mask = Variable(torch.zeros((len(batch_len), max_len.item())))
    if batch_len.is_cuda:
        mask = mask.cuda()
    for i in range(len(batch_len)):
        mask[i, :batch_len.data[i]] = 1

    return mask