from typing import List

import torch.nn
import torch.optim as optim

from torchfed.base.component import BaseComponent
from torchfed.centralized.component import TrainComponent, TestComponent, PullFromOthersComponent, PushToOthersComponent
from torchfed.base.node import BaseNode

from torchfed.utils.datasets import UserDataset


class ClientNode(BaseNode):
    def __init__(
            self,
            node_id: str,
            server_id: str,
            model: torch.nn.Module,
            train_dataset: UserDataset,
            test_dataset: UserDataset,
            device: str,
            *args,
            **kwargs):
        self.server_id = server_id
        super().__init__(node_id, *args, **kwargs)
        self.device = device
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.dataset_size = len(train_dataset)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

    def generate_components(self) -> List[BaseComponent]:
        return [
            PullFromOthersComponent("comp_pull"),
            TrainComponent("comp_train"),
            PushToOthersComponent("comp_push"),
            TestComponent("comp_test")
        ]

    def update_model(self, model):
        self.model = model

    def pre_train(self):
        self.components["comp_pull"].pull_model(self.model, self.server_id)

    def train(self):
        for i in range(self.local_iterations):
            self.components["comp_train"].local_train(self.model, self.optimizer, self.loss_fn, self.train_loader)

    def post_train(self):
        self.components["comp_push"].push_model(self.server_id, self.model, self.dataset_size)
        self.components["comp_test"].local_test(self.model, self.test_loader)
