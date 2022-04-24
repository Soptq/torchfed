from torch import optim

from torchfed.base.component import BaseComponent
from torchfed.base.backend.BaseBackend import BaseBackend
from torchfed.base.node.BaseNode import BaseNode

import torch
from copy import deepcopy


class PushToOthersComponent(BaseComponent):
    def __init__(self, component_id, *args, **kwargs):
        super().__init__(component_id, *args, **kwargs)

    def push_model(self, target_id, model, dataset_size):
        self.node.backend.get_node(target_id).update_model(model, dataset_size)
