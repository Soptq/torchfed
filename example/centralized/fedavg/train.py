import os
import random

from torchfed.routers import TorchDistributedRPCRouter
from torchfed.modules.node import CentralizedFedAvgServer, CentralizedFedAvgClient

from torchvision.transforms import transforms
from torchfed.datasets.CIFAR10 import TorchCIFAR10
from torchfed.managers.dataset_manager import DatasetManager

import config


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

    server = CentralizedFedAvgServer(router, dataset_manager, visualizer=True)
    clients = []
    for rank in range(config.num_users):
        clients.append(
            CentralizedFedAvgClient(
                router,
                rank,
                dataset_manager,
                visualizer=True))

    router.connect(server, [client.get_node_name() for client in clients])
    for client in clients:
        router.connect(client, [server.get_node_name()])

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
