import abc
import time

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.distributed.rpc as rpc

from .router_msg import RouterMsg
from torchfed.logging import get_logger
from torchfed.utils.hash import hex_hash
from torchfed.utils.plotter import NetworkConnectionsPlotter


class Router(abc.ABC):
    context = None

    def __init__(
            self,
            rank,
            world_size,
            backend=None,
            rpc_backend_options=None,
            tensorboard=False,
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
        self.tensorboard = tensorboard
        self.debug = debug
        self.ident = hex_hash(f"{time.time_ns()}")
        self.logger = get_logger(self.ident, self.name)

        if self.tensorboard:
            self.logger.info(
                f"[{self.name}] Tensorboard enabled. Run `tensorboard --logdir=runs/{self.ident}` to start.")
            self.writer = self.get_tensorboard_writer()

        self.owned_workers = {}
        self.peers_table = {}
        torch.distributed.rpc.init_rpc(
            self.name, backend, rank, world_size, rpc_backend_options)

        self.logger.info(f"[{self.name}] Initialized completed: {self.ident}")

    def register(self, module):
        if not module.is_root():
            return
        if module.name not in self.owned_workers.keys():
            self.owned_workers[module.name] = module.receive

    def connect(self, module, peers: list):
        if not module.is_root():
            return
        peers = [peer.split("/")[0] for peer in peers]
        if hasattr(self.peers_table, module.name):
            self.peers_table[module.name] += peers
        else:
            self.peers_table[module.name] = peers

    def get_peers(self, module):
        name = module.get_root_name()
        return self.peers_table[name]

    def unregister(self, worker):
        if worker.name in self.owned_workers.keys():
            del self.owned_workers[worker.name]

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
        if router_msg.to in Router.context.owned_workers.keys():
            return Router.context.owned_workers[router_msg.to](router_msg)
        return None

    def get_tensorboard_writer(self):
        return SummaryWriter(
            log_dir=f"runs/{self.ident}/{self.name}")

    def __del__(self):
        rpc.shutdown()
