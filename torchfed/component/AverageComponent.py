import random
from typing import List

from torchfed.base.component import BaseComponent


class AverageComponent(BaseComponent):
    def __init__(self, component_id, *args, **kwargs):
        super().__init__(component_id, *args, **kwargs)
        self.total_size = 0
        self.storage = {}

    def _reset(self):
        for name, param in self.storage.items():
            param.data.zero_()
        self.total_size = 0

    def update_model(self, model, dataset_size):
        self.total_size += dataset_size
        for name, param in model.named_parameters():
            if name not in self.storage:
                self.storage[name] = param.data.clone() * dataset_size
            else:
                self.storage[name] += param.data.clone() * dataset_size

    def pre_train(self, epoch: int):
        pass

    def train(self, epoch: int):
        if self.total_size == 0:
            return
        for name, param in self.node.model.named_parameters():
            param.data = self.storage[name] / self.total_size
        self._reset()

    def post_train(self, epoch: int):
        pass
