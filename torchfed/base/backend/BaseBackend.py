from __future__ import annotations
from abc import abstractmethod, ABC

import sys

from torchfed.utils.object import get_object_size


class BaseBackend(ABC):
    def __init__(self, node_id):
        self.communication_send_size = 0
        self.communication_recv_size = 0
        self.node_id = node_id
        self.callback = None

    @abstractmethod
    def set_peers(self, nodes):
        pass

    def add_listener(self, callback):
        self.callback = callback

    @abstractmethod
    def _call(self, to, func, *args, **kwargs):
        pass

    def call(self, to, func, *args, **kwargs):
        self.communication_send_size += get_object_size(args)
        self.communication_send_size += get_object_size(kwargs)
        result = self._call(to, func, *args, **kwargs)
        self.communication_recv_size += get_object_size(result)
        return result

    def _on_call(self, _from, func, *args, **kwargs):
        return self.callback(_from, func, *args, **kwargs)

    def on_call(self, _from, func, *args, **kwargs):
        self.communication_recv_size += get_object_size(args)
        self.communication_recv_size += get_object_size(kwargs)
        result = self._on_call(_from, func, *args, **kwargs)
        self.communication_send_size += get_object_size(result)
        return result

    def broadcast(self, func, *args, **kwargs):
        results = []
        for backend in self.index.values():
            results.append(backend.on_call(self.node_id, func, *args, **kwargs))
        return results
