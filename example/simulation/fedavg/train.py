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


class FedAvgNode(Module):
    def __init__(
            self,
            router,
            rank,
            dataset_manager,
            visualizer=False):
        super(
            FedAvgNode,
            self).__init__(
            router,
            alias="node_{}".format(rank),
            visualizer=visualizer)
        self.model = CIFARNet()

        self.dataset_manager = dataset_manager
        [self.train_dataset,
         self.test_dataset] = self.dataset_manager.get_user_dataset(rank)
        self.global_test_dataset = self.dataset_manager.get_dataset()[1]
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams["batch_size"], shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.hparams["batch_size"], shuffle=True)
        self.global_test_loader = torch.utils.data.DataLoader(
            self.global_test_dataset, batch_size=self.hparams["batch_size"], shuffle=True)

        self.dataset_size = len(self.train_dataset)
        self.optimizer = getattr(
            optim, self.hparams["optimizer"])(
            self.model.parameters(), lr=self.hparams["lr"])
        self.loss_fn = getattr(torch.nn, self.hparams["loss_fn"])()

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

        self.distributor.update(self.model.state_dict())

    def set_hparams(self):
        return {
            "lr": config.lr,
            "batch_size": config.batch_size,
            "optimizer": "Adam",
            "loss_fn": "CrossEntropyLoss",
            "local_iterations": config.local_iterations,
        }

    def bootstrap(self, bootstrap_from):
        if bootstrap_from is not None:
            global_model = self.send(
                bootstrap_from, "distributor/download", ())[0].data
            self.model.load_state_dict(global_model)

        self.distributor.update(self.model.state_dict())

    def aggregate(self):
        # generate latest local model
        aggregated = self.distributor.aggregate()
        if aggregated is None:
            aggregated = self.model.state_dict()
        else:
            self.model.load_state_dict(aggregated)
        self.distributor.update(aggregated)

    def train_and_test(self):
        # train and tests
        self.global_tester.test()
        self.tester.test()
        for i in range(self.hparams["local_iterations"]):
            self.trainer.train()

    def upload(self):
        # upload to peers
        for peer in router.get_peers(self):
            self.send(
                peer,
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

    args = parser.parse_args()
    
    # init
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
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

    node = FedAvgNode(router, args.rank, dataset_manager, visualizer=True)

    # bootstrap
    router.connect(node, ["node_0"])
    node.bootstrap("node_0")
    router.disconnect(node, ["node_0"])

    # connect
    current_node_name = "node_{}".format(args.rank)
    other_nodes_name = ["node_{}".format(i) for i in range(args.world_size) if i != args.rank]
    connected_peers = random.sample(other_nodes_name, 5) + [current_node_name]
    print(f"node {current_node_name} will connect to {connected_peers}")
    router.connect(node, connected_peers)

    # train
    for epoch in range(config.num_epochs):
        print(f"---------- Epoch {epoch} ----------")
        node.aggregate()
        node.train_and_test()
        node.upload()

    node.release()
    router.release()
