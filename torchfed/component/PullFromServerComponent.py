from abc import ABC, abstractmethod

from torchfed.base.component import BaseComponent
from torchfed.base.backend.BaseBackend import BaseBackend
from torchfed.base.node.BaseNode import BaseNode


class PullFromServerComponentCompatible(ABC):
    @abstractmethod
    def get_model(self, _from):
        pass


class PullFromServerComponent(BaseComponent):
    def __init__(self, component_id, stage, model, server_id):
        super().__init__(component_id, stage)
        self.model = model
        self.server_id = server_id

    def execute(self, epoch: int):
        global_model = self.backend.call(self.server_id, "get_model")
        self.model.load_state_dict(global_model.state_dict())
