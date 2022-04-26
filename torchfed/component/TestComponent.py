import sys
from torchfed.base.component import BaseComponent
from tqdm import tqdm

import torch


class TestComponent(BaseComponent):
    def __init__(self, component_id, *args, **kwargs):
        super().__init__(component_id, *args, **kwargs)

    def pre_train(self, epoch: int):
        pass

    def train(self, epoch: int):
        pass

    def post_train(self, epoch: int):
        self.node.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(
                    self.node.test_loader, 0):
                data, targets = data.to(
                    self.node.device), targets.to(
                    self.node.device)
                outputs = self.node.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        self.node.logger.info(
            f'[{self.node.id}] Test Accuracy: {100 * correct // total} %')
