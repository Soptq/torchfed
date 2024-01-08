import copy

from torchfed.modules.module import Module
from torchfed.utils.decorator import exposed
from torchfed.utils.helper import interface_join


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
    def download(self):
        return self.shared

    def update(self, data):
        self.shared = (copy.deepcopy(data), )

    def fetch(self, download_path):
        self.storage.clear()
        for peer in self.router.get_peers(self):
            self.storage[peer] = self.send(peer, download_path, ())[0].data

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
        return ret
