import copy

import torch

from torchfed.modules.module import Module
from torchfed.utils.decorator import exposed


class WeightedDataDistributing(Module):
    def __init__(
            self,
            router,
            alias=None,
            visualizer=False,
            writer=None):
        super(
            WeightedDataDistributing,
            self).__init__(
            router,
            alias=alias,
            visualizer=visualizer,
            writer=writer)
        self.total_weight = 0
        self.storage = {}
        self.shared = None

    @exposed
    def download(self):
        return self.shared

    def update(self, data, weight):
        self.shared = (copy.deepcopy(data), weight)

    def fetch(self, download_path):
        self.storage.clear()
        for peer in self.router.get_peers(self):
            data, weight = self.send(peer, download_path, ())[0].data
            self.total_weight += weight
            self.storage[peer] = [data, weight]

    def aggregate(self):
        ret = None
        if len(self.storage) == 0:
            return ret
        for data in self.storage.values():
            [data, weight] = data
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
        return ret
