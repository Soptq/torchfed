from __future__ import annotations

from abc import abstractmethod, ABC


class BaseComponent(ABC):
    def __init__(self, component_id, stage):
        self.stage = stage
        self.component_id = component_id
        self.node_id = None
        self.backend = None
        self.logger = None

    def bind_context(self, node_id, backend, logger):
        self.node_id = node_id
        self.backend = backend
        self.logger = logger

    @abstractmethod
    def execute(self, epoch: int):
        pass
