import sys
import torch


def get_object_size(obj):
    if isinstance(obj, torch.Tensor):
        return sys.getsizeof(obj.storage())
    if isinstance(obj, torch.nn.Module):
        size = 0
        for name, param in obj.named_parameters():
            size += sys.getsizeof(param.storage())
        return size
    return sys.getsizeof(obj)
