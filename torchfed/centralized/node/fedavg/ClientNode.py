from typing import List

import torch.nn
import torch.optim as optim

from torchfed.base.component import BaseComponent
from torchfed.component import TrainComponent, TestComponent, PullFromOthersComponent, PushToOthersComponent
from torchfed.base.node import BaseNode


class ClientNode(BaseNode):
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

    def generate_components(self) -> List[BaseComponent]:
        return [
            PullFromOthersComponent("comp_pull"),
            TrainComponent("comp_train"),
            PushToOthersComponent("comp_push"),
            TestComponent("comp_test")
        ]

    def update_model(self, model):
        self.model = model

    def get_peers(self, nodes: List[BaseNode]) -> List[BaseNode]:
        return [node for node in nodes if node.id == self.server_id]

    def epoch_init(self, epoch: int):
        pass

    def will_train(self, epoch: int) -> bool:
        will_train = False
        for peer in self.peers:
            will_train |= (self.id in peer.selected_nodes)
        return will_train
