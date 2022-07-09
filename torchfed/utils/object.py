import sys
import torch
import numpy as np


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError as te:
        return False


# TODO: this function can be greatly improved
def get_object_size(obj):
    if is_iterable(obj):
        if isinstance(obj, str) or isinstance(obj, np.ndarray):
            return sys.getsizeof(obj)

        if isinstance(obj, torch.Tensor):
            return sys.getsizeof(obj.storage())

        size = 0
        if isinstance(obj, dict):
            for inner_obj_k, inner_obj_v in obj.items():
                size += (get_object_size(inner_obj_k) +
                         get_object_size(inner_obj_v))
        else:
            for inner_obj in obj:
                size += get_object_size(inner_obj)
        return size

    if isinstance(obj, torch.nn.Module):
        size = 0
        for name, param in obj.named_parameters():
            size += get_object_size(param)
        return size
    return sys.getsizeof(obj)


if __name__ == '__main__':
    test = (132, "3213", [3, 32, 11, 3], torch.Tensor([52.3, 11.3, 44.1, 32]))
    print(get_object_size(test))
