from torchfed.base.component import BaseComponent


class TrainComponent(BaseComponent):
    def __init__(self, component_id, *args, **kwargs):
        super().__init__(component_id, *args, **kwargs)

    def pre_train(self, epoch: int):
        pass

    def train(self, epoch: int):
        self.node.model.train()
        for i in range(self.node.local_iterations):
            counter = 0
            running_loss = 0.0
            for batch_idx, (data, targets) in enumerate(
                    self.node.train_loader, 0):
                data, targets = data.to(
                    self.node.device), targets.to(
                    self.node.device)
                self.node.optimizer.zero_grad()
                outputs = self.node.model(data)
                loss = self.node.loss_fn(outputs, targets)
                loss.backward()
                self.node.optimizer.step()
                running_loss += loss.item()
                counter += 1
            self.node.logger.info(
                f'[{self.node.id}] Loss: {running_loss / counter:.3f}')

    def post_train(self, epoch: int):
        pass
