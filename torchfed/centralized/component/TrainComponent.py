from torchfed.base.component import BaseComponent


class TrainComponent(BaseComponent):
    def __init__(self, component_id, *args, **kwargs):
        super().__init__(component_id, *args, **kwargs)

    def local_train(self, model, optimizer, loss_fn, train_loader):
        model.train()
        counter = 0
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader, 0):
            data, targets = data.to(
                self.node.device), targets.to(
                self.node.device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            counter += 1
        self.node.logger.info(
            f'[{self.node.id}] Loss: {running_loss / counter:.3f}')
