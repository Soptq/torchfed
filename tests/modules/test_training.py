import os
import torch
import torch.optim as optim

from torchfed.routers import Router
from torchfed.modules.module import Module
from torchfed.modules.compute.trainer import Trainer
from torchfed.modules.compute.tester import Tester
from torchfed.utils.decorator import exposed

from torchvision.transforms import transforms
from torchfed.datasets.CIFAR10 import CIFAR10
from torchfed.models.CIFARNet import CIFARNet

DEBUG = True


class MainModule(Module):
    def __init__(self, name, router, debug=False):
        super(MainModule, self).__init__(name, router, debug)
        self.model = CIFARNet()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = CIFAR10(
            "../../example/data",
            20,
            3,
            download=True,
            transform=transform)
        [self.train_dataset, self.test_dataset] = dataset.get_user_dataset(0)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=32, shuffle=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
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

    @exposed
    def execute(self):
        self.tester()
        for i in range(5):
            self.trainer()
        self.tester()


def test_training():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    router_a = Router(0, 1, debug=DEBUG)

    main = MainModule("main", router_a, debug=DEBUG)
    main()
