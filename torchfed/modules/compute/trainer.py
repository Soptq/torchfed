import torch
from prettytable import PrettyTable

from torchfed.modules.module import Module
from torchfed.third_party.aim_extension.distribution import Distribution


class Trainer(Module):
    def __init__(
            self,
            router,
            model,
            dataloader,
            optimizer,
            loss_fn,
            device="cpu",
            alias=None,
            visualizer=False,
            writer=None):
        super(
            Trainer,
            self).__init__(
            router,
            alias=alias,
            visualizer=visualizer,
            writer=writer)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.metrics = None

        self._log_dataset_distribution()

        # if self.visualizer:
        #     # graph_writer = self.get_tensorboard_writer()
        #     inputs, _ = next(iter(self.dataloader))
        #     self.writer.add_graph(self.model, inputs)
        #     # graph_writer.close()

    def get_metrics(self):
        return self.metrics

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
            self.writer.track(
                dist, name=f"Dataset Distribution/{self.get_path()}")

    def train(self):
        self.model.train()
        counter = 0
        running_loss = 0.0
        for batch_idx, data in enumerate(self.dataloader, 0):
            inputs, labels = data["inputs"].to(self.device), data["labels"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.cpu().item()
            counter += 1
        self.metrics = running_loss / counter
        self.logger.info(
            f'[{self.name}] Training Loss: {self.metrics:.3f}')
        if self.visualizer:
            self.writer.track(
                self.metrics,
                name=f"Loss/Train/{self.get_path()}")
