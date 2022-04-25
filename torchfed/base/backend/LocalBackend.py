from torchfed.base.node.BaseNode import BaseNode
from torchfed.base.backend import BaseBackend


class LocalBackend(BaseBackend):
    def __init__(self, logger):
        super().__init__(logger)
        self.nodes = {}

    def register_node(self, node: BaseNode):
        if node.id in self.nodes:
            raise Exception(f"Node with id {node.id} already registered")
        self.nodes[node.id] = node

    def post_register_node(self):
        for node in self.get_nodes():
            node.bind(self)

    def get_node(self, node_id: str):
        if node_id not in self.nodes:
            raise Exception(f"Node with id {node_id} not registered")
        return self.nodes[node_id]

    def get_nodes(self):
        return self.nodes.values()
