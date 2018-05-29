import torch

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def device():
    return _device


def use_cpu():
    global _device
    _device = torch.device('cpu')


def use_cuda():
    global _device
    _device = torch.device('cuda')
