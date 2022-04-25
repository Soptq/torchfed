from __future__ import annotations

from abc import abstractmethod, ABC
from typing import List

from torchfed.base.backend.BaseBackend import BaseBackend
from torchfed.base.component.BaseComponent import BaseComponent


class BaseNode(ABC):
    def __init__(self, node_id: str, *args, **kwargs):
        self.id = node_id
        if 'params' in kwargs:
            self.params = kwargs['params']
            self.params = kwargs['params']
            if len(self.params) > 0:
                self._process_params()

        self.components = {}
        self._process_components()
        self.backend: BaseBackend | None = None
        self.logger = None

    def _process_params(self):
        for param in self.params.keys():
            setattr(self, param, self.params[param])

    def _process_components(self):
        for component in self.generate_components():
            if component.id in self.components:
                raise Exception(
                    f'Component {component.id} already exists in node {self.id}')
            self.components[component.id] = component
            component.bind(self)

    def bind(self, backend: BaseBackend):
        self.backend = backend
        self.logger = self.backend.logger

    @abstractmethod
    def generate_components(self) -> List[BaseComponent]:
        pass

    @abstractmethod
    def pre_train(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def post_train(self):
        pass
