import random
import sys
from abc import abstractmethod, ABC
import datetime

from torchfed.logging import get_logger
from torchfed.utils.hash import hex_hash

from tqdm import tqdm


class BaseTrainer(ABC):
    def __init__(self, *args, **kwargs):
        # bind params
        self.args = args
        self.kwargs = kwargs
        self._bind_params()

        # initialize logger
        formatted_params = {}
        for param, value in self.params.items():
            if isinstance(value, str):
                formatted_params[param] = value
            elif isinstance(value, int) or isinstance(value, float):
                formatted_params[param] = f"{value:.5f}"
            elif hasattr(value, 'name'):
                formatted_params[param] = f"{value.name}"
        self.trainer_id = hex_hash(
            f"{str(formatted_params)}-{datetime.datetime.now()}")
        self.logger = get_logger(self.trainer_id, "Trainer")
        self.logger.info(f"Trainer Parameters: {formatted_params}")

        # initialize Nodes
        self.name_resolver = {}
        self.nodes = []
        self.shuffle_nodes = {}

    def _bind_params(self):
        if 'params' not in self.kwargs:
            return
        self.params = self.kwargs['params']
        if len(self.params) == 0:
            return
        for param in self.params.keys():
            setattr(self, param, self.params[param])

    def add_node(self, node, shuffle, *args, **kwargs):
        if node.__name__ in self.name_resolver:
            self.name_resolver[node.__name__] += 1
        else:
            self.name_resolver[node.__name__] = 0
        node_id = f"{node.__name__}-{self.name_resolver[node.__name__]}"
        self.nodes.append(node(self.trainer_id, node_id, *args, **kwargs))
        self.shuffle_nodes[node_id] = shuffle
        return node_id

    def build_graph(self, epoch, size):
        # by default all nodes are densely connected
        return [[True for _ in range(size)] for _ in range(size)]

    def sort_nodes(self, nodes):
        random.shuffle(nodes)
        return nodes

    def pre_train(self):
        pass

    def train(self, epochs: int):
        self.pre_train()
        for epoch in tqdm(
                range(epochs),
                file=sys.stdout,
                leave=False,
                desc="Global Training"):
            # build connection graph
            connections = self.build_graph(epoch, len(self.nodes))
            for node, connections in zip(self.nodes, connections):
                node.backend.set_peers(
                    [node for index, node in enumerate(self.nodes) if connections[index]])
            # shuffle nodes list for randomness
            exec_nodes = []
            shuffle_nodes = [
                node for node in self.nodes if self.shuffle_nodes[node.node_id]]
            shuffle_nodes = self.sort_nodes(shuffle_nodes)
            for node in self.nodes:
                if self.shuffle_nodes[node.node_id]:
                    exec_nodes.append(shuffle_nodes.pop())
                else:
                    exec_nodes.append(node)

            # train
            for order, node in enumerate(exec_nodes):
                node.pre_train(epoch, order)
            for order, node in enumerate(exec_nodes):
                node.train(epoch, order)
            for order, node in enumerate(exec_nodes):
                node.post_train(epoch, order)
        self.post_train()

    def post_train(self):
        pass
