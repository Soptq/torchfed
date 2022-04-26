import sys
from abc import abstractmethod, ABC
import datetime
from typing import List

from torchfed.base.backend.BaseBackend import BaseBackend
from torchfed.base.node import BaseNode
from torchfed.logging import get_logger
from torchfed.utils.hash import hex_hash

from tqdm import tqdm


class BaseTrainer(ABC):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        if 'params' in kwargs:
            self.params = self.kwargs['params']
            if len(self.params) > 0:
                self._process_params()
        self.logger = None
        self._setup_logger()

        self.backend: BaseBackend = self.generate_backend()
        # Initializing Nodes
        self.nodes = self.generate_nodes()

        self.backend.pre_register_node()
        for node in self.nodes:
            self.backend.register_node(node)
        self.backend.post_register_node()

    def _process_params(self):
        for param in self.params.keys():
            setattr(self, param, self.params[param])

    def _setup_logger(self):
        formatted_params = {}
        for param, value in self.params.items():
            if isinstance(value, str):
                formatted_params[param] = value
            elif isinstance(value, int) or isinstance(value, float):
                formatted_params[param] = f"{value:.5f}"
            elif hasattr(value, 'name'):
                formatted_params[param] = f"{value.name}"
        self.logger = get_logger(f"{hex_hash(str(formatted_params))}-{datetime.datetime.now()}")
        self.logger.info(f"Trainer Parameters: {formatted_params}")

    def train(self, epochs: int):
        self.pre_train()
        for epoch in tqdm(range(epochs), file=sys.stdout, leave=False, desc="Global Training"):
            ready_nodes = [node for node in self.backend.get_nodes() if node.will_train(epoch)]
            for node in ready_nodes:
                node.pre_train(epoch)
            for node in ready_nodes:
                node.train(epoch)
            for node in ready_nodes:
                node.post_train(epoch)
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
