import random
from typing import List

from torchfed.base.component import BaseComponent
from torchfed.component import AverageComponent
from torchfed.base.node import BaseNode


class ServerNode(BaseNode):
    def __init__(
            self,
            node_id: str,
            *args,
            **kwargs):
        super().__init__(node_id, *args, **kwargs)
        self.model = self.model.to(self.device)
        self.selected_nodes = []

    def _select_peers(self, num_samples=-1):
        if num_samples < 0:
            num_samples = len(self.peers)
        selected_nodes = random.sample(list(self.peers), num_samples)
        return [node.id for node in selected_nodes]

    def generate_components(self) -> List[BaseComponent]:
        return [
            AverageComponent("comp_avg")
        ]

    def update_model(self, node_id, model, dataset_size):
        if node_id not in self.selected_nodes:
            return
        self.components["comp_avg"].update_model(model, dataset_size)

    def get_peers(self, nodes: List[BaseNode]) -> List[BaseNode]:
        return nodes

    def epoch_init(self, epoch: int):
        self.selected_nodes = self._select_peers(self.sample_size)

    def will_train(self, epoch: int) -> bool:
        return True
