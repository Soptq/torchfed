from prettytable import PrettyTable
import torch
from torchfed.modules.module import Module
from torchfed.third_party.aim_extension.distribution import Distribution

import aim


class Tester(Module):
    def __init__(
            self,
            name,
            router,
            model,
            dataloader,
            visualizer=False,
            writer=None,
            debug=False):
        super(
            Tester,
            self).__init__(
            name,
            router,
            visualizer=visualizer,
            writer=writer,
            debug=debug)
        self.model = model
        self.dataloader = dataloader

        self._log_dataset_distribution()

    def _log_dataset_distribution(self):
        num_classes = self.dataloader.dataset.num_classes
        labels = []
        dist = {k: 0 for k in range(num_classes)}
        for data in self.dataloader:
            labels.extend(data["labels"].tolist())
        for label in labels:
            dist[label] += 1
        dist_table = PrettyTable()
        dist_table.field_names = dist.keys()
        dist_table.add_row(dist.values())
        self.logger.info(f"[{self.name}] Dataset distribution:")
        for row in dist_table.get_string().split("\n"):
            self.logger.info(row)

        if self.visualizer:
            dist = Distribution(
                distribution=labels,
                bin_count=num_classes
            )
            self.writer.track(dist, name=f"Dataset Distribution/{self.get_path()}")

    def execute(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(
                    self.dataloader, 0):
                inputs, labels = data["inputs"], data["labels"]
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        self.logger.info(
            f'[{self.name}] Test Accuracy: {100 * correct / total:.3f} %')
        if self.visualizer:
            self.writer.track(
                100 * correct / total,
                name=f"Accuracy/Test/{self.get_path()}")
        yield False
