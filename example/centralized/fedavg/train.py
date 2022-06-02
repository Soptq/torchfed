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

import config


class FedAvgServer(Module):
    def __init__(self, name, router, visualizer=False, debug=False):
        super(
            FedAvgServer,
            self).__init__(
            name,
            router,
            visualizer=visualizer,
            debug=debug)
        self.model = CIFARNet()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = CIFAR10(
            "../../data",
            config.num_users,
            config.num_labels,
            download=True,
            transform=transform)
        test_dataset = self.dataset.get_bundle_dataset()[1]
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=True)

        self.distributor = self.register_submodule(
            WeightedDataDistributing, "distributor", router)
        self.tester = self.register_submodule(
            Tester, "tester", router, self.model, self.test_loader)

        router.connect(
            self, [
                f"client_{_rank}" for _rank in range(
                    config.num_users)])

        self.distributor.update(self.model.state_dict())

    def execute(self):
        self.tester()

        aggregated = self.distributor.aggregate()
        if aggregated is None:
            aggregated = self.model.state_dict()
        else:
            self.model.load_state_dict(aggregated)
        self.distributor.update(aggregated)
        yield False


class FedAvgClient(Module):
    def __init__(self, name, router, rank, visualizer=False, debug=False):
        super(
            FedAvgClient,
            self).__init__(
            name,
            router,
            visualizer=visualizer,
            debug=debug)
        self.model = CIFARNet()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = CIFAR10(
            "../../data",
            config.num_users,
            config.num_labels,
            download=True,
            transform=transform)
        [self.train_dataset,
         self.test_dataset] = dataset.get_user_dataset(rank)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=config.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=config.batch_size, shuffle=True)

        self.dataset_size = len(self.train_dataset)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

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

        router.connect(self, ["server"])

    def execute(self):
        global_model = self.send(
            router.get_peers(self)[0],
            "distributor/download",
            ())[0].data
        self.model.load_state_dict(global_model)

        self.tester()
        for i in range(config.local_iterations):
            self.trainer()

        self.send(
            router.get_peers(self)[0],
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

    server = FedAvgServer("server", router, visualizer=True)
    clients = []
    for rank in range(config.num_users):
        clients.append(
            FedAvgClient(
                f"client_{rank}",
                router,
                rank,
                visualizer=True))

    # train
    for epoch in range(config.num_epochs):
        print(f"---------- Epoch {epoch} ----------")
        while server():
            pass
        for client in random.sample(clients, 5):
            while client():
                pass
