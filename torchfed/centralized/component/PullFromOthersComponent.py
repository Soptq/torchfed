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
