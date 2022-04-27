from typing import Callable

from torchfed.base.backend import BaseBackend


class LocalBackend(BaseBackend):
    def __init__(self, node_id):
        super().__init__(node_id)
        self.index = {}

    def set_peers(self, nodes):
        for node in nodes:
            self.index[node.node_id] = node.backend

    def call(self, to, func, *args, **kwargs):
        if to not in self.index:
            raise Exception("Node {} not registered".format(to))
        if hasattr(func, '__name__'):
            return self.index[to].on_call(
                self.node_id, func.__name__, *args, **kwargs)
        else:
            return self.index[to].on_call(self.node_id, func, *args, **kwargs)

    def broadcast(self, func, *args, **kwargs):
        for backend in self.index.values():
            backend.on_call(self.node_id, func, *args, **kwargs)

    def on_call(self, _from, func, *args, **kwargs):
        return self.callback(_from, func, *args, **kwargs)
