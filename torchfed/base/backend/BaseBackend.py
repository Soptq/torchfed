from __future__ import annotations
from abc import abstractmethod, ABC


class BaseBackend(ABC):
    def __init__(self, node_id):
        self.node_id = node_id
        self.callback = None

    @abstractmethod
    def set_peers(self, nodes):
        pass

    def add_listener(self, callback):
        self.callback = callback

    @abstractmethod
    def call(self, to, func, *args, **kwargs):
        pass

    @abstractmethod
    def broadcast(self, func, *args, **kwargs):
        pass

    @abstractmethod
    def on_call(self, _from, func, *args, **kwargs):
        pass
