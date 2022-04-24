from typing import List
import copy

import torch.nn

from torchfed.base.backend import LocalBackend
from torchfed.base.node import BaseNode
from torchfed.base.trainer import BaseTrainer

from torchfed.centralized.node.fedavg import ServerNode, ClientNode
from torchfed.utils.cuda import recommend_gpu, get_eligible_gpus
from torchfed.utils.datasets import BundleSplitDataset
from torchfed.datasets import Dataset


class Trainer(BaseTrainer):
    def __init__(
            self,
            world_size: int,
            model: torch.nn.Module,
            train_dataset: Dataset,
            test_dataset: Dataset,
            *args,
            **kwargs):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.server_id = "server"
        self.client_id = "client"
        self.cuda = kwargs["cuda"] if "cuda" in kwargs else False
        if self.cuda:
            assert "gpus" in kwargs, "gpus must be specified when cuda is True"
            assert isinstance(kwargs["gpus"], list), "gpus must be a list"
            self.available_gpus = get_eligible_gpus(kwargs["gpus"])
            self.cuda &= len(self.available_gpus) > 0
        super().__init__(world_size, *args, **kwargs)

    def _process_params(self):
        for param in self.params.keys():
            setattr(self, param, self.params[param])

    def generate_backend(self) -> LocalBackend:
        return LocalBackend()

    def generate_nodes(self) -> List[BaseNode]:
        nodes = []
        server_node = ServerNode(f"{self.server_id}_0",
                                 copy.deepcopy(self.model),
                                 f"cuda:{recommend_gpu(self.available_gpus)}" if self.cuda else "cpu")
        nodes.append(server_node)

        for index in range(self.world_size):
            client_node = ClientNode(f"{self.client_id}_{index}",
                                     f"{self.server_id}_0",
                                     copy.deepcopy(self.model),
                                     self.train_dataset.get_user_dataset(index),
                                     self.test_dataset.get_user_dataset(index),
                                     f"cuda:{recommend_gpu(self.available_gpus)}" if self.cuda else "cpu",
                                     params={
                                         "lr": self.lr,
                                         "batch_size": self.batch_size,
                                     })
            nodes.append(client_node)

        return nodes

    def pre_train(self):
        pass

    def post_train(self):
        pass
