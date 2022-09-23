import torch

from torchfed.modules.module import Module
from torchfed.utils.decorator import exposed


class DataDistributing(Module):
    def __init__(
            self,
            router,
            alias=None,
            visualizer=False,
            writer=None):
        super(
            DataDistributing,
            self).__init__(
            router,
            alias=alias,
            visualizer=visualizer,
            writer=writer)
        self.storage = {}
        self.shared = None

    @exposed
    def upload(self, from_, data):
        self.storage[from_] = data
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
            if isinstance(data, dict):
                if ret is None:
                    ret = {k: v / len(self.storage) for k, v in data.items()}
                else:
                    ret = {k: ret[k] + v / len(self.storage)
                           for k, v in data.items()}
            else:
                if ret is None:
                    ret = data / len(self.storage)
                else:
                    ret += data / len(self.storage)
        self.storage.clear()
        return ret
