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
            visualizer=False,
            writer=None,
            debug=False):
        super(
            Trainer,
            self).__init__(
            name,
            router,
            visualizer=visualizer,
            writer=writer,
            debug=debug)
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        # if self.visualizer:
        #     # graph_writer = self.get_tensorboard_writer()
        #     inputs, _ = next(iter(self.dataloader))
        #     self.writer.add_graph(self.model, inputs)
        #     # graph_writer.close()

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
        self.logger.info(
            f'[{self.name}] Training Loss: {running_loss / counter:.3f}')
        if self.visualizer:
            self.writer.track(running_loss / counter, name=f"Loss/Train/{self.get_path()}")
        yield False
