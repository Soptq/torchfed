from typing import List
import copy

import torch.nn

from torchfed.base.backend import LocalBackend
from torchfed.base.node import BaseNode
from torchfed.base.trainer import BaseTrainer

from torchfed.centralized.node.fedavg import ServerNode, ClientNode
from torchfed.utils.cuda import recommend_gpu, get_eligible_gpus


class Trainer(BaseTrainer):
    def __init__(
            self,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

    def _process_params(self):
        setattr(self, "cuda", False)
        for param in self.params.keys():
            setattr(self, param, self.params[param])
        if self.cuda and self.gpus:
            assert isinstance(self.gpus, list), "gpus must be a list"
            self.available_gpus = get_eligible_gpus(self.gpus)
            self.cuda &= len(self.available_gpus) > 0
        self.device = "cuda:{}".format(recommend_gpu(
            self.available_gpus)) if self.cuda else "cpu"
        return 0

    def generate_backend(self) -> LocalBackend:
        return LocalBackend(self.logger)

    def generate_nodes(self) -> List[BaseNode]:
        nodes = []
        server_node = ServerNode(
            f"{self.server_id}_0",
            params={
                "sample_size": self.sample_size,
                "model": copy.deepcopy(self.model),
                "device": f"cuda:{recommend_gpu(self.available_gpus)}" if self.cuda else "cpu",
            })
        nodes.append(server_node)

        for index in range(self.world_size):
            client_node = ClientNode(
                f"{self.client_id}_{index}",
                params={
                    "server_id": f"{self.server_id}_0",
                    "model": copy.deepcopy(self.model),
                    "train_dataset": self.dataset.get_user_dataset(index)[0],
                    "test_dataset": self.dataset.get_user_dataset(index)[1],
                    "device": f"cuda:{recommend_gpu(self.available_gpus)}" if self.cuda else "cpu",
                    "lr": self.lr,
                    "batch_size": self.batch_size,
                    "local_iterations": self.local_iterations,
                })
            nodes.append(client_node)

        return nodes

    def pre_train(self):
        pass

    def post_train(self):
        for node in self.nodes:
            if not node.id.startswith(self.server_id):
                continue
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
