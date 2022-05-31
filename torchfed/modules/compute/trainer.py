from torchfed.modules.module import Module


class Trainer(Module):
    def __init__(
            self,
            name,
            router,
            model,
            dataloader,
            optimizer,
            loss_fn,
            debug=False):
        super(Trainer, self).__init__(name, router, debug)
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def execute(self):
        self.model.train()
        counter = 0
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(self.dataloader, 0):
            data, targets = data, targets
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            counter += 1
        self.logger.info(f'[{self.name}] Training Loss: {running_loss / counter:.3f}')
        yield False
