from torchfed.base.component import BaseComponent
from torchfed.base.backend.BaseBackend import BaseBackend
from torchfed.base.node.BaseNode import BaseNode


class PullFromOthersComponent(BaseComponent):
    def __init__(self, component_id, *args, **kwargs):
        super().__init__(component_id, *args, **kwargs)

    def pre_train(self, epoch: int):
        global_model = self.node.backend.get_node(self.node.server_id).model
        self.node.model.load_state_dict(global_model.state_dict())

    def train(self, epoch: int):
        pass

    def post_train(self, epoch: int):
        pass
