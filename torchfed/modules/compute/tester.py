import torch
from torchfed.modules.module import Module


class Tester(Module):
    def __init__(
            self,
            name,
            router,
            model,
            dataloader,
            visualizer=False,
            debug=False):
        super(
            Tester,
            self).__init__(
            name,
            router,
            visualizer=visualizer,
            debug=debug)
        self.model = model
        self.dataloader = dataloader
        self.tester_visualizer_step = 0

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
        self.logger.info(
            f'[{self.name}] Test Accuracy: {100 * correct / total:.3f} %')
        if self.visualizer:
            self.writer.line([100 * correct / total], X=[self.tester_visualizer_step], name=self.name, update="append", win=f"Accuracy/Test/{self.name}")
            self.tester_visualizer_step += 1
        yield False
