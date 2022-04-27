from typing import List

import torch.nn
import torch.optim as optim

from torchfed.base.component import BaseComponent
from torchfed.component import TrainComponent, TestComponent, PullFromServerComponent, PushToPeersComponent
from torchfed.base.node import BaseNode, ComponentStage


class ClientNode(BaseNode):
    def __init__(
            self,
            trainer_id: str,
            node_id: str,
            *args,
            **kwargs):
        super().__init__(trainer_id, node_id, *args, **kwargs)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.dataset_size = len(self.train_dataset)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=True)

        self.add_component(PullFromServerComponent("comp_pull",
                                                   ComponentStage.PRE_TRAIN,
                                                   self.model,
                                                   self.server_id))
        self.add_component(TrainComponent("comp_train",
                                          ComponentStage.TRAIN,
                                          self.model,
                                          self.train_loader,
                                          self.local_iterations,
                                          self.optimizer,
                                          self.loss_fn,
                                          self.device))
        self.add_component(PushToPeersComponent("comp_push",
                                                ComponentStage.POST_TRAIN,
                                                self.model,
                                                self.dataset_size))
        self.add_component(TestComponent("comp_test",
                                         ComponentStage.POST_TRAIN,
                                         self.model,
                                         self.test_loader,
                                         self.device))
