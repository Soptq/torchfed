import os
import random

import torch
import torch.optim as optim

from torchfed.routers import TorchDistributedRPCRouter
from torchfed.modules.module import Module
from torchfed.modules.compute.trainer import Trainer
from torchfed.modules.compute.tester import Tester
from torchfed.modules.distribute.decentralized_data_distribute import DecentralizedDataDistributing
from torchfed.utils.helper import interface_join

from torchvision.transforms import transforms
from torchfed.datasets.CIFAR10 import TorchCIFAR10
from torchfed.models.CIFARNet import CIFAR10Net
from torchfed.managers.dataset_manager import DatasetManager

import config


class DeFTANode(Module):
    def __init__(
            self,
            router,
            rank,
            dataset_manager,
            visualizer=False):
        super(
            DeFTANode,
            self).__init__(
            router,
            visualizer=visualizer)
        self.model = CIFAR10Net()

        self.dataset_manager = dataset_manager
        [self.train_dataset,
         self.test_dataset] = self.dataset_manager.get_user_dataset(rank)
        self.global_test_dataset = self.dataset_manager.get_global_dataset()[1]
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
            DecentralizedDataDistributing, "distributor", router)
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

    def get_default_hparams(self):
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
                bootstrap_from,
                interface_join(
                    "distributor",
                    DecentralizedDataDistributing.download),
                ())[0].data
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
                interface_join("distributor", DecentralizedDataDistributing.upload),
                (self.name,
                 self.dataset_size,
                 len(router.get_peers(self)),
                 self.model.state_dict()))


if __name__ == '__main__':
    # init
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    router = TorchDistributedRPCRouter(0, 1, visualizer=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset_manager = DatasetManager("cifar10_manager",
                                     TorchCIFAR10(
                                         "../../data",
                                         config.num_users,
                                         config.num_labels,
                                         download=True,
                                         transform=transform)
                                     )

    nodes = []
    for rank in range(config.num_users):
        nodes.append(DeFTANode(router, rank,
                               dataset_manager,
                               visualizer=True))

    # bootstrap
    boostrap_node = nodes[0].get_node_name()
    for node in nodes:
        router.connect(node, [boostrap_node])
        node.bootstrap(boostrap_node)
        router.disconnect(node, [boostrap_node])

    # connect
    for node in nodes:
        current_node_name = node.get_node_name()
        other_nodes_names = [
            n.get_node_name() for n in nodes if n.get_node_name() != current_node_name]
        connected_peers = random.sample(
            other_nodes_names,
            5) + [current_node_name]  # self connect
        print(f"node {current_node_name} will connect to {connected_peers}")
        router.connect(node, connected_peers)

    # train
    for epoch in range(config.num_epochs):
        print(f"---------- Epoch {epoch} ----------")
        for node in nodes:
            node.aggregate()
        for node in nodes:
            node.train_and_test()
        for node in nodes:
            node.upload()

    for node in nodes:
        node.release()
    router.release()
