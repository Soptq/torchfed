from typing import List

import torch.nn

from torchfed.base.component import BaseComponent
from torchfed.centralized.component import AverageComponent
from torchfed.base.node import BaseNode


class ServerNode(BaseNode):
    def __init__(self, node_id: str, model: torch.nn.Module, device: str, *args, **kwargs):
        self.device = device
        self.model = model.to(device)
        super().__init__(node_id, model, device, *args, **kwargs)
        super().__init__(node_id, *args, **kwargs)

    def generate_components(self) -> List[BaseComponent]:
        return [
            AverageComponent("comp_avg", self.model)
        ]

    def update_model(self, model, dataset_size):
        self.components["comp_avg"].update_model(model, dataset_size)

    def pre_train(self):
        pass

    def train(self):
        self.model = self.components["comp_avg"].average()

    def post_train(self):
        pass
