import torch

from torchfed.modules.module import Module
from torchfed.utils.decorator import exposed


class WeightedDataDistributing(Module):
    def __init__(self, name, router, debug=False):
        super(WeightedDataDistributing, self).__init__(name, router, debug)
        self.total_weight = 0
        self.storage = {}
        self.shared = None

    def execute(self):
        yield False

    @exposed
    def upload(self, from_, weight, data):
        self.total_weight += weight
        self.storage[from_] = [weight, data]
        return True

    @exposed
    def download(self):
        return self.shared

    def update(self, data):
        self.shared = data

    def aggregate(self):
        ret = None
        if len(self.storage) == 0:
            return ret
        for data in self.storage.values():
            [weight, data] = data
            if isinstance(data, dict):
                if ret is None:
                    ret = {k: v * (weight / self.total_weight)
                           for k, v in data.items()}
                else:
                    ret = {k: ret[k] + v * (weight / self.total_weight)
                           for k, v in data.items()}
            else:
                if ret is None:
                    ret = data * (weight / self.total_weight)
                else:
                    ret += data * (weight / self.total_weight)
        self.total_weight = 0
        self.storage.clear()
        return ret
