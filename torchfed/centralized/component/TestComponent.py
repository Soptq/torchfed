from torchfed.base.component import BaseComponent
from torchfed.base.backend.BaseBackend import BaseBackend
from torchfed.base.node.BaseNode import BaseNode

import torch
from copy import deepcopy


class TestComponent(BaseComponent):
    def __init__(self, component_id, *args, **kwargs):
        super().__init__(component_id, *args, **kwargs)

    def local_test(self, model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader, 0):
                data, targets = data.to(
                    self.node.device), targets.to(
                    self.node.device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print(f'[{self.node.id}] Test Accuracy: {100 * correct // total} %')
