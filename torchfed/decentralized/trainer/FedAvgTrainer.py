import random
from typing import List
import copy

import torch.nn

from torchfed.base.backend import LocalBackend
from torchfed.base.node import BaseNode
from torchfed.base.trainer import BaseTrainer

from torchfed.decentralized.node import FedAvgNode
from torchfed.utils.cuda import recommend_gpu, get_eligible_gpus


class FedAvgTrainer(BaseTrainer):
    def __init__(
            self,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self._process_cuda_params()
        # add nodes
        for index in range(self.world_size):
            self.add_node(
                FedAvgNode,
                True,
                params={
                    "sample_size": self.sample_size,
                    "peer_size": self.peer_size,
                    "model": copy.deepcopy(
                        self.model),
                    "train_dataset": self.dataset.get_user_dataset(index)[0],
                    "test_dataset": self.dataset.get_user_dataset(index)[1],
                    "device": f"cuda:{recommend_gpu(self.available_gpus)}" if self.cuda else "cpu",
                    "lr": self.lr,
                    "batch_size": self.batch_size,
                    "local_iterations": self.local_iterations,
                })

    def _process_cuda_params(self):
        if not hasattr(self, 'cuda'):
            setattr(self, 'cuda', False)
        if self.cuda and hasattr(self, 'gpus'):
            assert isinstance(self.gpus, list), "gpus must be a list"
            self.available_gpus = get_eligible_gpus(self.gpus)
            self.cuda &= len(self.available_gpus) > 0
        setattr(self, 'device', "cuda:{}".format(recommend_gpu(
            self.available_gpus)) if self.cuda else "cpu")

    def build_graph(self, epoch, size):
        connections = []
        for i in range(size):
            default = [True for _ in range(size)]
            connected = random.sample(list(range(size)), self.peer_size)
            for j in connected:
                default[j] = True
            connections.append(default)
        return connections

    def post_train(self):
        for node in self.nodes:
            self.logger.info(
                f"{node.node_id} communication size: {node.backend.communication_size} bytes")
            model = node.model
            model.eval()
            test_dataset = self.dataset.get_bundle_dataset()[1]
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=True)
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(test_loader, 0):
                    data, targets = data.to(
                        self.device), targets.to(
                        self.device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            print(f'Final Test Accuracy: {100 * correct // total} %')
