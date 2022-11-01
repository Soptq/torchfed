import os
import random

from torchfed.routers import TorchDistributedRPCRouter
from torchfed.modules.node import DecentralizedFedAvgNode

from torchvision.transforms import transforms
from torchfed.datasets.CIFAR10 import TorchCIFAR10
from torchfed.managers.dataset_manager import DatasetManager

import config
import optuna


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

    def objective(trial):
        hparams = {
            "lr": trial.suggest_categorical("lr", [1e-1, 1e-2, 1e-3]),
            "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        }
        router.refresh_exp_id()
        nodes = []
        for rank in range(config.num_users):
            nodes.append(DecentralizedFedAvgNode(router, rank,
                                    dataset_manager,
                                    visualizer=True,
                                    override_hparams=hparams
                                    )
                         )

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
                other_nodes_names, 5) + [current_node_name]  # self connect
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

        metrics = min([node.get_metrics() for node in nodes])
        print(f"Metrics: {metrics}")

        for node in nodes:
            node.release()

        return metrics

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    router.release()
