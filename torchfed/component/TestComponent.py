import sys
from torchfed.base.component import BaseComponent
from tqdm import tqdm

import torch


class TestComponent(BaseComponent):
    def __init__(self, component_id, stage, model, data_loader, device):
        super().__init__(component_id, stage)
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def execute(self, epoch: int):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(
                    self.data_loader, 0):
                data, targets = data.to(
                    self.device), targets.to(
                    self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        self.logger.info(
            f'[{self.node_id}] Test Accuracy: {100 * correct // total} %')
