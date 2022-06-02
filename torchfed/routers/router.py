import abc
import time

import visdom

import torch
import torch.distributed.rpc as rpc

from .router_msg import RouterMsg
from torchfed.logging import get_logger
from torchfed.utils.hash import hex_hash


class Router(abc.ABC):
    context = None

    def __init__(
            self,
            rank,
            world_size,
            backend=None,
            rpc_backend_options=None,
            visualizer=False,
            debug=False):
        if backend is None:
            backend = rpc.BackendType.TENSORPIPE

        if rpc_backend_options is None:
            rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
                init_method="env://",
                rpc_timeout=0
            )
        Router.context = self
        self.name = f"router_{rank}"
        self.rank = rank
        self.world_size = world_size
        self.visualizer = visualizer
        self.debug = debug
        self.ident = hex_hash(f"{time.time_ns()}")
        self.logger = get_logger(self.ident, self.name)

        if self.visualizer:
            self.logger.info(
                f"[{self.name}] Visualizer enabled. Run `visdom -env_path=./runs` to start.")
            self.writer = self.get_visualizer()

        self.owned_nodes = {}
        self.peers_table = {}

        self.network_edges = set()
        torch.distributed.rpc.init_rpc(
            self.name, backend, rank, world_size, rpc_backend_options)

        self.logger.info(f"[{self.name}] Initialized completed: {self.ident}")

    def register(self, module):
        if not module.is_root():
            return
        if module.name not in self.owned_nodes.keys():
            self.owned_nodes[module.name] = module.receive

    def connect(self, module, peers: list):
        if not module.is_root():
            return
        peers = [peer.split("/")[0] for peer in peers]
        if hasattr(self.peers_table, module.name):
            self.peers_table[module.name] += peers
        else:
            self.peers_table[module.name] = peers

        for peer in peers:
            self.network_edges.add((module.name, peer))

        if self.visualizer:
            node_labels = {}
            edges = []
            for edge in self.network_edges:
                (from_node, to_node) = edge
                if from_node not in node_labels:
                    node_labels[from_node] = len(node_labels)
                if to_node not in node_labels:
                    node_labels[to_node] = len(node_labels)
                edges.append((node_labels[from_node], node_labels[to_node]))
            node_labels = list(dict(sorted(node_labels.items(), key=lambda item: item[1])).keys())
            print(node_labels)
            self.writer.graph(edges,
                              nodeLabels=node_labels,
                              opts={"showEdgeLabels": True,
                                    "showVertexLabels": True,
                                    "scheme": "different",
                                    "directed": True,
                                    "height": 335,
                                    "width": 371},
                              win="Connection Graph")

    def get_peers(self, module):
        name = module.get_root_name()
        return self.peers_table[name]

    def unregister(self, worker):
        if worker.name in self.owned_nodes.keys():
            del self.owned_nodes[worker.name]

    def broadcast(self, router_msg: RouterMsg):
        if self.debug:
            self.logger.debug(
                f"[{self.name}] broadcasting message {router_msg}")
        futs, rets = [], []
        for rank in range(self.world_size):
            futs.append(
                rpc.rpc_async(
                    f"router_{rank}",
                    Router.receive,
                    args=(
                        router_msg,
                    )))
        for fut in futs:
            rets.append(fut.wait())
        return rets

    @staticmethod
    def receive(router_msg: RouterMsg):
        if Router.context.debug:
            print(
                f"[{Router.context.name}] receiving message {router_msg}")
        if router_msg.to in Router.context.owned_nodes.keys():
            return Router.context.owned_nodes[router_msg.to](router_msg)
        return None

    def get_visualizer(self):
        v = visdom.Visdom(env=self.ident, log_to_filename=f"runs/{self.ident}/{self.name}.vis")
        if not v.check_connection():
            self.logger.warning("Visualizer server has to be started ahead of time")
            self.logger.warning(
                f"Using offline mode, visualizer logs to runs/{self.ident}/{self.name}.vis")
        return v

    def __del__(self):
        rpc.shutdown()
