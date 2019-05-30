import logging

import torch.optim as optim
import inspect
import re


def get_optimizer(model, lr_method, **kwargs):
    """
    parse optimization method parameters, and initialize optimizer function
    """
    if "-" in lr_method:
        lr_method_name = lr_method[:lr_method.find('-')]
        lr_method_parameters = {}
        for x in lr_method[lr_method.find('-') + 1:].split('-'):
            split = x.split('=')
            assert len(split) == 2
            lr_method_parameters[split[0]] = float(split[1])
    else:
        lr_method_name = lr_method
        lr_method_parameters = {}

    # initialize optimizer function
    if lr_method_name == 'sgd':
        optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif lr_method_name == 'adagrad':
        optimizer_ft = optim.Adagrad(model.parameters(), lr=0.01)
    else:
        raise Exception('unknown optimization method.')

    return lr_method_parameters, optimizer_ft


def get_optimizer_new(model_parameters, lr_method, **kwargs):
    """
    parse optimization method parameters, and initialize optimizer function
    """
    if "-" in lr_method:
        lr_method_name = lr_method[:lr_method.find('-')]
        lr_method_parameters = {}
        for x in lr_method[lr_method.find('-') + 1:].split('-'):
            split = x.split('=')
            assert len(split) == 2
            lr_method_parameters[split[0]] = float(split[1])
    else:
        lr_method_name = lr_method
        lr_method_parameters = {}

    # initialize optimizer function
    if lr_method_name == 'sgd':
        optimizer_ft = optim.SGD(model_parameters, lr=0.01, momentum=0.9)
    elif lr_method_name == 'adagrad':
        optimizer_ft = optim.Adagrad(model_parameters, lr=0.01)
    else:
        raise Exception('unknown optimization method.')

    return lr_method_parameters, optimizer_ft


def get_optimizer_bk(model, lr_method, **kwargs):
    """
    parse optimization method parameters, and initialize optimizer function
    """
    if "-" in lr_method:
        lr_method_name = lr_method[:lr_method.find('-')]
        lr_method_parameters = {}
        for x in lr_method[lr_method.find('-') + 1:].split('-'):
            split = x.split('=')
            assert len(split) == 2
            lr_method_parameters[split[0]] = float(split[1])
    else:
        lr_method_name = lr_method
        lr_method_parameters = {}

    # initialize optimizer function
    if lr_method_name == 'sgd':
        optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif lr_method_name == 'adagrad':
        optimizer_ft = optim.Adagrad(model.parameters(), lr=0.01)
    else:
        raise Exception('unknown optimization method.')

    return lr_method_parameters, optimizer_ft


def get_word_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


def clip_parameters(model, clip):
    """
    Clip model weights.
    """
    if clip > 0:
        for x in model.parameters():
            x.data.clamp_(-clip, clip)


def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=7):
    """
    Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
    """
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer