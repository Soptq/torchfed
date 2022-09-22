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


class FedAvgServer(Module):
    def __init__(
            self,
            name,
            router,
            dataset_manager,
            visualizer=False,
            debug=False):
        super(
            FedAvgServer,
            self).__init__(
            name,
            router,
            visualizer=visualizer,
            debug=debug)
        self.model = CIFARNet()

        self.dataset_manager = dataset_manager
        test_dataset = self.dataset_manager.get_dataset()[1]
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.hparams["batch_size"], shuffle=True)

        self.distributor = self.register_submodule(
            WeightedDataDistributing, "distributor", router)
        self.global_tester = self.register_submodule(
            Tester, "global_tester", router, self.model, self.test_loader)

        router.connect(
            self, [
                f"client_{_rank}" for _rank in range(
                    config.num_users)])

        self.distributor.update(self.model.state_dict())

    def set_hparams(self):
        return {
            "batch_size": config.batch_size,
        }

    def run(self):
        self.global_tester.test()

        aggregated = self.distributor.aggregate()
        if aggregated is None:
            aggregated = self.model.state_dict()
        else:
            self.model.load_state_dict(aggregated)
        self.distributor.update(aggregated)


class FedAvgClient(Module):
    def __init__(
            self,
            name,
            router,
            rank,
            dataset_manager,
            visualizer=False,
            debug=False):
        super(
            FedAvgClient,
            self).__init__(
            name,
            router,
            visualizer=visualizer,
            debug=debug)
        self.model = CIFARNet()

        self.dataset_manager = dataset_manager
        [self.train_dataset,
         self.test_dataset] = self.dataset_manager.get_user_dataset(rank)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams["batch_size"], shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.hparams["batch_size"], shuffle=True)

        self.dataset_size = len(self.train_dataset)
        self.optimizer = getattr(
            optim, self.hparams["optimizer"])(
            self.model.parameters(), lr=self.hparams["lr"])
        self.loss_fn = getattr(torch.nn, self.hparams["loss_fn"])()

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

    def set_hparams(self):
        return {
            "lr": config.lr,
            "batch_size": config.batch_size,
            "optimizer": "Adam",
            "loss_fn": "CrossEntropyLoss",
            "local_iterations": config.local_iterations,
        }

    def run(self):
        global_model = self.send(
            router.get_peers(self)[0],
            "distributor/download",
            ())[0].data
        self.model.load_state_dict(global_model)

        self.tester.test()
        for i in range(self.hparams["local_iterations"]):
            self.trainer.train()

        self.send(
            router.get_peers(self)[0],
            "distributor/upload",
            (self.name,
             self.dataset_size,
             self.model.state_dict()))


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

    server = FedAvgServer("server", router, dataset_manager, visualizer=True)
    clients = []
    for rank in range(config.num_users):
        clients.append(
            FedAvgClient(
                f"client_{rank}",
                router,
                rank,
                dataset_manager,
                visualizer=True))

    # train
    for epoch in range(config.num_epochs):
        print(f"---------- Epoch {epoch} ----------")
        server.run()
        for client in random.sample(clients, 5):
            client.run()

    for client in clients:
        client.release()
    server.release()
    router.release()
