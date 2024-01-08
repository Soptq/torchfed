from prettytable import PrettyTable
import torch
from torchfed.modules.module import Module
from torchfed.third_party.aim_extension.distribution import Distribution


class Tester(Module):
    def __init__(
            self,
            router,
            model,
            dataloader,
            device="cpu",
            alias=None,
            visualizer=False,
            writer=None):
        super(
            Tester,
            self).__init__(
            router,
            alias=alias,
            visualizer=visualizer,
            writer=writer)
        self.dataloader = dataloader
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.metrics = None

        self._log_dataset_distribution()

    def get_metrics(self):
        return self.metrics

    def _log_dataset_distribution(self):
        # TODO: Combined dataset label shift
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
            self.writer.track(
                dist, name=f"Dataset Distribution/{self.get_path()}")

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(
                    self.dataloader, 0):
                inputs, labels = data["inputs"].to(self.device), data["labels"].to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        self.metrics = 100 * correct / total
        self.logger.info(
            f'[{self.name}] Test Accuracy: {self.metrics:.3f} %')
        if self.visualizer:
            self.writer.track(
                self.metrics,
                name=f"Accuracy/Test/{self.get_path()}")
