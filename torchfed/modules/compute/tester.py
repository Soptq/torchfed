import torch
from torchfed.modules.module import Module


class Tester(Module):
    def __init__(self, name, router, model, dataloader, debug=False):
        super(Tester, self).__init__(name, router, debug)
        self.model = model
        self.dataloader = dataloader

    def execute(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(
                    self.dataloader, 0):
                data, targets = data, targets
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print(f'[{self.name}] Test Accuracy: {100 * correct // total} %')
        yield False
