import torch
from typing import List
import sys


def get_available_gpus():
    """
    Get a list of available GPU devices.
    """
    return [x.name for x in list(torch.cuda.device_count().keys())]


def get_eligible_gpus(gpus: List[int]) -> List[int]:
    if not torch.cuda.is_available():
        return []
    return list(set(gpus) & set(get_available_gpus()))


def recommend_gpu(gpus: List[int]) -> int:
    eligible_gpus = get_eligible_gpus(gpus)
    if len(eligible_gpus) == 0:
        return -1

    min_allocated_memory = sys.maxsize
    min_allocated_gpu = -1
    for gpu in gpus:
        allocated = torch.cuda.memory_allocated(gpu)
        if allocated < min_allocated_memory:
            min_allocated_memory = allocated
            min_allocated_gpu = gpu
    return min_allocated_gpu
