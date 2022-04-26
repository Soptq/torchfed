from torch import optim

from torchfed.base.component import BaseComponent
from torchfed.base.backend.BaseBackend import BaseBackend
from torchfed.base.node.BaseNode import BaseNode

import torch
from copy import deepcopy


class PushToOthersComponent(BaseComponent):
    def __init__(self, component_id, *args, **kwargs):
        super().__init__(component_id, *args, **kwargs)

    def pre_train(self, epoch: int):
        pass

    def train(self, epoch: int):
        pass

    def post_train(self, epoch: int):
        for peer in self.node.peers:
            peer.update_model(
                self.node.id, self.node.model, self.node.dataset_size
            )
