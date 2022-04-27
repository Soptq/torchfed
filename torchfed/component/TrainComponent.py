from torchfed.base.component import BaseComponent


class TrainComponent(BaseComponent):
    def __init__(
            self,
            component_id,
            stage,
            model,
            data_loader,
            iterations,
            optimizer,
            loss_fn,
            device):
        super().__init__(component_id, stage)
        self.model = model
        self.data_loader = data_loader
        self.iterations = iterations
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def execute(self, epoch: int):
        self.model.train()
        for i in range(self.iterations):
            counter = 0
            running_loss = 0.0
            for batch_idx, (data, targets) in enumerate(
                    self.data_loader, 0):
                data, targets = data.to(
                    self.device), targets.to(
                    self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                counter += 1
            self.logger.info(
                f'[{self.node_id}] Loss: {running_loss / counter:.3f}')
