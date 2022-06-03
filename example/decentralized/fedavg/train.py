import os
import random

import torch
import torch.optim as optim

from torchfed.routers.router import Router
from torchfed.modules.module import Module
from torchfed.modules.compute.trainer import Trainer
from torchfed.modules.compute.tester import Tester
from torchfed.modules.distribute.weighted_data_distribute import WeightedDataDistributing

from torchvision.transforms import transforms
from torchfed.datasets.CIFAR10 import CIFAR10
from torchfed.models.CIFARNet import CIFARNet
from torchfed.managers.dataset_manager import DatasetManager

import config


class FedAvgNode(Module):
    def __init__(
            self,
            name,
            router,
            rank,
            peers,
            bootstrap_from,
            dataset_manager,
            visualizer=False,
            debug=False):
        super(
            FedAvgNode,
            self).__init__(
            name,
            router,
            visualizer=visualizer,
            debug=debug)
        self.model = CIFARNet()

        self.dataset_manager = dataset_manager
        [self.train_dataset,
         self.test_dataset] = self.dataset_manager.get_user_dataset(rank)
        self.global_test_dataset = self.dataset_manager.get_dataset()[1]
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=config.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=config.batch_size, shuffle=True)
        self.global_test_loader = torch.utils.data.DataLoader(
            self.global_test_dataset, batch_size=config.batch_size, shuffle=True)

        self.dataset_size = len(self.train_dataset)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.distributor = self.register_submodule(
            WeightedDataDistributing, "distributor", router)
        self.trainer = self.register_submodule(
            Trainer,
            "trainer",
            router,
            self.model,
            self.train_loader,
            self.optimizer,
            self.loss_fn)
        self.tester = self.register_submodule(
            Tester, "tester", router, self.model, self.test_loader)
        self.global_tester = self.register_submodule(
            Tester, "global_tester", router, self.model, self.global_test_loader)

        if bootstrap_from is not None:
            global_model = self.send(
                bootstrap_from, "distributor/download", ())[0].data
            self.model.load_state_dict(global_model)

        self.distributor.update(self.model.state_dict())

        router.connect(self, peers)

    def log_hp(self):
        return {
            "lr": config.lr,
            "dataset": self.dataset_manager.dataset.name,
            "model": self.model.name,
            "batch_size": config.batch_size,
            "dataset_size": self.dataset_size,
            "optimizer": "adam",
            "loss_fn": "cross entropy loss",
            "local_iterations": config.local_iterations,
        }

    def execute(self):
        # generate latest local model
        aggregated = self.distributor.aggregate()
        if aggregated is None:
            aggregated = self.model.state_dict()
        else:
            self.model.load_state_dict(aggregated)
        self.distributor.update(aggregated)
        yield True
        # train and tests
        self.global_tester()
        self.tester()
        for i in range(config.local_iterations):
            self.trainer()
        yield True
        # upload to peers
        for peer in router.get_peers(self):
            self.send(
                peer,
                "distributor/upload",
                (self.name,
                 self.dataset_size,
                 self.model.state_dict()))
        yield False


if __name__ == '__main__':
    # init
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    router = Router(0, 1, visualizer=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset_manager = DatasetManager("cifar10_manager",
                                     CIFAR10(
                                         "../../data",
                                         config.num_users,
                                         config.num_labels,
                                         download=True,
                                         transform=transform)
                                     )

    nodes = []
    for rank in range(config.num_users):
        connected_peers = [
            f"node_{peer_rank}" for peer_rank in random.sample(
                range(
                    config.num_users), 5)]
        print(f"node {rank} will connect to {connected_peers}")
        nodes.append(FedAvgNode(f"node_{rank}", router, rank,
                                connected_peers,
                                f"node_{rank - 1}" if rank > 0 else None,
                                dataset_manager,
                                visualizer=True))

    # train
    for epoch in range(config.num_epochs):
        print(f"---------- Epoch {epoch} ----------")
        while True:
            _continue = False
            for node in nodes:
                _continue |= node()
            if not _continue:
                break

    for node in nodes:
        node.release()
    router.release()
