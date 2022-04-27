from abc import ABC, abstractmethod

from torch import optim

from torchfed.base.component import BaseComponent
from torchfed.base.backend.BaseBackend import BaseBackend
from torchfed.base.node.BaseNode import BaseNode

import torch
from copy import deepcopy


class PushToPeersComponentCompatible(ABC):
    @abstractmethod
    def update_model(self, _from, model, dataset_size):
        pass


class PushToPeersComponent(BaseComponent):
    def __init__(self, component_id, stage, model, dataset_size):
        super().__init__(component_id, stage)
        self.model = model
        self.dataset_size = dataset_size

    def execute(self, epoch: int):
        self.backend.broadcast("update_model", self.model, self.dataset_size)
