from __future__ import annotations

from abc import abstractmethod, ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchfed.base.node.BaseNode import BaseNode


class BaseComponent(ABC):
    def __init__(self, component_id, *args, **kwargs):
        self.id = component_id
        if 'params' in kwargs:
            self.params = kwargs['params']
            if len(self.params) > 0:
                self._process_params()
        self.node: "BaseNode" | None = None

    def _process_params(self):
        for param in self.params.keys():
            setattr(self, param, self.params[param])

    def bind(self, node: "BaseNode"):
        self.node = node
