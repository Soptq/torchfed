from abc import abstractmethod, ABC
from typing import List

from torchfed.base.backend.BaseBackend import BaseBackend
from torchfed.base.node import BaseNode


class BaseTrainer(ABC):
    def __init__(self, world_size: int, *args, **kwargs):
        self.world_size = world_size
        self.args = args
        self.kwargs = kwargs

        if 'params' in kwargs:
            self.params = self.kwargs['params']
            if len(self.params) > 0:
                self._process_params()

        self.backend: BaseBackend = self.generate_backend()
        # Initializing Nodes
        self.nodes = self.generate_nodes()
        for node in self.nodes:
            self.backend.register_node(node)
        self.backend.post_register_node()

    def _process_params(self):
        for param in self.params.keys():
            setattr(self, param, self.params[param])

    def train(self, epochs: int):
        self.pre_train()
        for epoch in range(epochs):
            for node in self.backend.get_nodes():
                node.pre_train()
            for node in self.backend.get_nodes():
                node.train()
            for node in self.backend.get_nodes():
                node.post_train()
        self.post_train()

    @abstractmethod
    def generate_backend(self) -> BaseBackend:
        pass

    @abstractmethod
    def generate_nodes(self) -> List[BaseNode]:
        pass

    @abstractmethod
    def pre_train(self):
        pass

    @abstractmethod
    def post_train(self):
        pass
