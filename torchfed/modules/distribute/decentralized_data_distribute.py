import torch

from torchfed.modules.module import Module
from torchfed.utils.decorator import exposed


class DecentralizedDataDistributing(Module):
    def __init__(
            self,
            router,
            alias=None,
            visualizer=False,
            writer=None):
        super(
            DecentralizedDataDistributing,
            self).__init__(
            router,
            alias=alias,
            visualizer=visualizer,
            writer=writer)
        self.total_weight = 0
        self.storage = {}
        self.shared = None

    @exposed
    def upload(self, from_, weight, num_peers_to, data):
        self.total_weight += (weight / min(num_peers_to, 1.))
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
