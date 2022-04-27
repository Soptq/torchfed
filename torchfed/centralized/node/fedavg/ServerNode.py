import random
from typing import List

from torchfed.component import AverageComponent, PullFromServerComponentCompatible, PushToPeersComponentCompatible
from torchfed.base.node import BaseNode, ComponentStage


class ServerNode(
        BaseNode,
        PullFromServerComponentCompatible,
        PushToPeersComponentCompatible):
    def __init__(
            self,
            trainer_id: str,
            node_id: str,
            *args,
            **kwargs):
        super().__init__(trainer_id, node_id, *args, **kwargs)
        self.model = self.model.to(self.device)
        self.add_component(AverageComponent("comp_avg",
                                            ComponentStage.PRE_TRAIN,
                                            self.model,
                                            self.sample_size))

    def get_model(self, _from):
        return self.model

    def update_model(self, _from, model, dataset_size):
        self.components["comp_avg"].update_model(model, dataset_size)
