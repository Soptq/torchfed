import os
import random

import torch
import torch.optim as optim

from torchfed.routers import TorchDistributedRPCRouter
from torchfed.modules.module import Module
from torchfed.modules.compute.trainer import Trainer
from torchfed.modules.compute.tester import Tester
from torchfed.modules.distribute.weighted_data_distribute import WeightedDataDistributing

from torchvision.transforms import transforms
from torchfed.datasets.CIFAR10 import CIFAR10
from torchfed.models.CIFARNet import CIFARNet
from torchfed.managers.dataset_manager import DatasetManager

import config
import argparse


class FedAvgServer(Module):
    def __init__(
            self,
            router,
            dataset_manager,
            visualizer=False):
        super(
            FedAvgServer,
            self).__init__(
            router,
            alias="server",
            visualizer=visualizer)
        self.model = CIFARNet()

        self.dataset_manager = dataset_manager
        test_dataset = self.dataset_manager.get_dataset()[1]
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.hparams["batch_size"], shuffle=True)

        self.distributor = self.register_submodule(
            WeightedDataDistributing, "distributor", router)
        self.global_tester = self.register_submodule(
            Tester, "global_tester", router, self.model, self.test_loader)

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
            router,
            rank,
            dataset_manager,
            visualizer=False):
        super(
            FedAvgClient,
            self).__init__(
            router,
            alias="client-{}".format(rank),
            visualizer=visualizer)
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
    parser = argparse.ArgumentParser(description='Simulation of FedAvg')
    parser.add_argument('--world_size', type=int, help='number of nodes in the world')
    parser.add_argument('--rank', type=int, help='the rank of the node')
    parser.add_argument('--master_addr', type=str, help='the address of the master node')
    parser.add_argument('--master_port', type=str, help='the port of the master node')
    parser.add_argument('--interface', type=str, default='eth0', help='the interface of the network (e.g. eth0)')

    args = parser.parse_args()
    print(args)
    # init
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["GLOO_SOCKET_IFNAME"] = args.interface
    os.environ["TP_SOCKET_IFNAME"] = args.interface
    router = TorchDistributedRPCRouter(args.rank, args.world_size, visualizer=True)

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

    if args.rank == 0:
        node = FedAvgServer(router, dataset_manager, visualizer=True)
        router.connect(node, [f"client-{i}" for i in range(1, args.world_size - 1)])
    else:
        node = FedAvgClient(router, args.rank - 1, dataset_manager, visualizer=True)
        router.connect(node, ["server"])

    # train
    for epoch in range(config.num_epochs):
        print(f"---------- Epoch {epoch} ----------")
        node.run()

    node.release()
    router.release()
