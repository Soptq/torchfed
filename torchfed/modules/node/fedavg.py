import copy

import torch
import torch.optim as optim

from torchfed.modules.module import Module
from torchfed.modules.compute.trainer import Trainer
from torchfed.modules.compute.tester import Tester
from torchfed.modules.distribute.weighted_data_distribute import WeightedDataDistributing
from torchfed.utils.helper import interface_join
import torchfed.models as models


class DecentralizedFedAvgNode(Module):
    def __init__(
            self,
            router,
            rank,
            dataset_manager,
            device="cpu",
            alias=None,
            visualizer=False,
            writer=None,
            override_hparams=None):
        super(
            DecentralizedFedAvgNode,
            self).__init__(
            router,
            alias=alias,
            visualizer=visualizer,
            writer=writer,
            override_hparams=override_hparams)

        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = getattr(
            models, self.hparams["model"])().to(self.device)
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
            WeightedDataDistributing, "distributor", router)
        self.trainer = self.register_submodule(
            Trainer,
            "trainer",
            router,
            self.model,
            self.train_loader,
            self.optimizer,
            self.loss_fn,
            device=self.device
        )
        self.tester = self.register_submodule(
            Tester, "tester", router, self.model, self.test_loader, device=self.device)
        self.global_tester = self.register_submodule(
            Tester, "global_tester", router, self.model, self.global_test_loader, device=self.device)

        self.distributor.update(copy.deepcopy(self.model).cpu().state_dict(), self.dataset_size)

    def get_default_hparams(self):
        return {
            "lr": 1e-3,
            "batch_size": 32,
            "model": "CIFAR10Net",
            "optimizer": "SGD",
            "loss_fn": "CrossEntropyLoss",
            "local_iterations": 10,
        }

    def before_bootstrap(self, bootstrap_from):
        pass

    def after_bootstrap(self, bootstrap_from):
        pass

    def bootstrap(self, bootstrap_from):
        self.before_bootstrap(bootstrap_from)
        if bootstrap_from is not None:
            global_model, _ = self.send(
                bootstrap_from,
                interface_join(
                    "distributor",
                    WeightedDataDistributing.download),
                ())[0].data
            self.model.load_state_dict(global_model)

        self.distributor.update(copy.deepcopy(self.model).cpu().state_dict(), self.dataset_size)
        self.after_bootstrap(bootstrap_from)

    def before_aggregate(self):
        pass

    def after_aggregate(self):
        pass

    def aggregate(self):
        self.before_aggregate()
        # generate latest local model
        aggregated = self.distributor.aggregate()
        if aggregated is not None:
            self.model.load_state_dict(aggregated)
        self.distributor.update(copy.deepcopy(self.model).cpu().state_dict(), self.dataset_size)
        self.after_aggregate()

    def before_train_and_test(self):
        pass

    def after_train_and_test(self):
        pass

    def train_and_test(self):
        self.before_train_and_test()
        # train and tests
        for i in range(self.hparams["local_iterations"]):
            self.trainer.train()
        self.tester.test()
        self.global_tester.test()
        self.distributor.update(copy.deepcopy(self.model).cpu().state_dict(), self.dataset_size)
        self.after_train_and_test()

    def before_fetch(self):
        pass

    def after_fetch(self):
        pass

    def fetch(self):
        self.before_fetch()
        self.distributor.fetch(interface_join("distributor", WeightedDataDistributing.download))
        self.after_fetch()


class CentralizedFedAvgServer(Module):
    def __init__(
            self,
            router,
            dataset_manager,
            alias=None,
            visualizer=False,
            writer=None,
            override_hparams=None):
        super(
            CentralizedFedAvgServer,
            self).__init__(
            router,
            alias=alias,
            visualizer=visualizer,
            writer=writer,
            override_hparams=override_hparams)
        self.model = getattr(
            models, self.hparams["model"])()

        self.dataset_manager = dataset_manager
        test_dataset = self.dataset_manager.get_global_dataset()[1]
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.hparams["batch_size"], shuffle=True)

        self.distributor = self.register_submodule(
            WeightedDataDistributing, "distributor", router)
        self.global_tester = self.register_submodule(
            Tester, "global_tester", router, self.model, self.test_loader)

        self.distributor.update(copy.deepcopy(self.model).cpu().state_dict(), 1)

    def get_default_hparams(self):
        return {
            "model": "CIFAR10Net",
            "batch_size": 32,
        }

    def before_run(self):
        pass

    def after_run(self):
        pass

    def run(self):
        self.before_run()
        aggregated = self.distributor.aggregate()
        if aggregated is not None:
            self.model.load_state_dict(aggregated)
        self.global_tester.test()
        self.distributor.update(copy.deepcopy(self.model).cpu().state_dict(), 1)
        self.distributor.fetch(interface_join("distributor", WeightedDataDistributing.download))
        self.after_run()


class CentralizedFedAvgClient(Module):
    def __init__(
            self,
            router,
            rank,
            dataset_manager,
            alias=None,
            visualizer=False,
            writer=None,
            override_hparams=None):
        super(
            CentralizedFedAvgClient,
            self).__init__(
            router,
            alias=alias,
            visualizer=visualizer,
            writer=writer,
            override_hparams=override_hparams)
        self.model = getattr(
            models, self.hparams["model"])()

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

        self.distributor.update(copy.deepcopy(self.model).cpu().state_dict(), self.dataset_size)

    def get_default_hparams(self):
        return {
            "lr": 1e-3,
            "batch_size": 32,
            "model": "CIFAR10Net",
            "optimizer": "SGD",
            "loss_fn": "CrossEntropyLoss",
            "local_iterations": 10,
        }

    def before_run(self):
        pass

    def after_run(self):
        pass

    def run(self):
        self.before_run()
        aggregated = self.distributor.aggregate()
        self.model.load_state_dict(aggregated)

        for i in range(self.hparams["local_iterations"]):
            self.trainer.train()
        self.tester.test()

        self.distributor.update(copy.deepcopy(self.model).cpu().state_dict(), self.dataset_size)
        self.distributor.fetch(interface_join("distributor", WeightedDataDistributing.download))
        self.after_run()
