import random
from typing import List

import torch
import torch.optim as optim

from torchfed.base.component import BaseComponent
from torchfed.component import TrainComponent, PushToOthersComponent, AverageComponent, TestComponent
from torchfed.base.node import BaseNode


class FedAvgNode(BaseNode):
    def __init__(
            self,
            node_id: str,
            *args,
            **kwargs):
        super().__init__(node_id, *args, **kwargs)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.dataset_size = len(self.train_dataset)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=True)
        self.selected_nodes = []

    def _select_peers(self, num_samples=-1):
        if num_samples < 0:
            num_samples = len(self.peers)
        selected_nodes = random.sample(list(self.peers), num_samples)
        return [node.id for node in selected_nodes]

    def generate_components(self) -> List[BaseComponent]:
        return [
            AverageComponent("comp_avg"),
            TrainComponent('comp_train'),
            PushToOthersComponent('comp_push'),
            TestComponent("comp_test")
        ]

    def update_model(self, node_id, model, dataset_size):
        if node_id not in self.selected_nodes:
            return
        self.components["comp_avg"].update_model(model, dataset_size)

    def get_peers(self, nodes: List[BaseNode]) -> List[BaseNode]:
        return random.sample(list(nodes), self.peer_size)

    def epoch_init(self, epoch: int):
        self.selected_nodes = self._select_peers(self.sample_size)

    def will_train(self, epoch: int) -> bool:
        will_train = False
        for peer in self.peers:
            will_train |= (self.id in peer.selected_nodes)
        return will_train
