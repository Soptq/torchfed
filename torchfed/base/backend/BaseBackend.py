from __future__ import annotations
from abc import abstractmethod, ABC

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from torchfed.base.node.BaseNode import BaseNode


class BaseBackend(ABC):
    def __init__(self, logger):
        self.logger = logger
        pass

    @abstractmethod
    def register_node(self, node: "BaseNode"):
        pass

    @abstractmethod
    def post_register_node(self):
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> "BaseNode":
        pass

    @abstractmethod
    def get_nodes(self) -> List["BaseNode"]:
        pass
