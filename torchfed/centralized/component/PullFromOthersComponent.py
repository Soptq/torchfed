from torchfed.base.component import BaseComponent
from torchfed.base.backend.BaseBackend import BaseBackend
from torchfed.base.node.BaseNode import BaseNode


class PullFromOthersComponent(BaseComponent):
    def __init__(self, component_id, *args, **kwargs):
        super().__init__(component_id, *args, **kwargs)

    def pull_model(self, model, target_id):
        assert isinstance(self.node, BaseNode)
        assert isinstance(self.node.backend, BaseBackend)
        global_model = self.node.backend.get_node(target_id).model
        model.load_state_dict(global_model.state_dict())

    def push_model(self, target_id, model):
        self.node.backend.get_node(target_id).update_model(model)

    def local_train(self, model, optimizer, loss_fn, train_loader):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader, 0):
            data, targets = data.to(self.node.device), targets.to(self.node.device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f'[{self.node.id}, {batch_idx + 1:5d}] Loss: {running_loss / 10:.3f}')
                running_loss = 0.0
