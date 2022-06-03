import os
import time
from typing import List

import aim

import torch
import torch.distributed.rpc as rpc

from .router_msg import RouterMsg, RouterMsgResponse
from torchfed.logging import get_logger
from torchfed.utils.hash import hex_hash
from torchfed.utils.helper import NetworkConnectionsPlotter, DataTransmitted


class Router:
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
                f"[{self.name}] Visualizer enabled. Run `aim up` to start.")
            self.writer = self.get_visualizer()

        self.owned_nodes = {}
        self.peers_table = {}

        self.network_plotter = NetworkConnectionsPlotter()
        self.data_transmitted = DataTransmitted()

        torch.distributed.rpc.init_rpc(
            self.name, backend, rank, world_size, rpc_backend_options)

        self.logger.info(f"[{self.name}] Initialized completed: {self.ident}")

    def register(self, module):
        if not module.is_root():
            return
        if module.name not in self.owned_nodes.keys():
            self.owned_nodes[module.name] = module.receive

    def unregister(self, worker):
        if worker.name in self.owned_nodes.keys():
            del self.owned_nodes[worker.name]

    def connect(self, module, peers: list):
        if not module.is_root():
            return
        peers = [self.get_root_name(peer) for peer in peers]
        if hasattr(self.peers_table, module.name):
            for peer in peers:
                if peer not in self.peers_table[module.name]:
                    self.peers_table[module.name].append(peer)
                    self.network_plotter.add_edge(module.name, peer)
        else:
            self.peers_table[module.name] = peers

        if self.visualizer:
            fig = self.network_plotter.get_figure()
            self.writer.track(aim.Figure(fig), name="Network Graph")

    def disconnect(self, module, peers: list):
        if not module.is_root():
            return
        peers = [self.get_root_name(peer) for peer in peers]
        if hasattr(self.peers_table, module.name):
            for peer in peers:
                if peer in self.peers_table[module.name]:
                    self.peers_table[module.name].remove(peer)
                    self.network_plotter.remove_edge(module.name, peer)

        if self.visualizer:
            fig = self.network_plotter.get_figure()
            self.writer.track(aim.Figure(fig), name="Network Graph")

    def get_peers(self, module):
        name = module.get_root_name()
        return self.peers_table[name]

    def broadcast(self, router_msg: RouterMsg) -> List[RouterMsgResponse]:
        if self.debug:
            self.logger.debug(
                f"[{self.name}] broadcasting message {router_msg}")

        self.data_transmitted.add(
            self.get_root_name(router_msg.from_),
            self.get_root_name(router_msg.to),
            router_msg.size
        )

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
            resp = fut.wait()
            if resp is not None:
                rets.append(resp)

        for ret in rets:
            self.data_transmitted.add(
                self.get_root_name(ret.from_),
                self.get_root_name(ret.to),
                ret.size
            )
        return rets

    @staticmethod
    def receive(router_msg: RouterMsg):
        if Router.context.debug:
            print(
                f"[{Router.context.name}] receiving message {router_msg}")

        Router.context.data_transmitted.add(
            Router.get_root_name(router_msg.from_),
            Router.get_root_name(router_msg.to),
            router_msg.size
        )

        if router_msg.to in Router.context.owned_nodes.keys():
            resp_msg = Router.context.owned_nodes[router_msg.to](router_msg)
            Router.context.data_transmitted.add(
                Router.get_root_name(resp_msg.from_),
                Router.get_root_name(resp_msg.to),
                resp_msg.size
            )
            return resp_msg
        return None

    @staticmethod
    def get_root_name(name):
        return name.split("/")[0]

    def get_visualizer(self):
        return aim.Run(
            run_hash=self.name,
            experiment=self.ident
        )

    def release(self):
        self.logger.info(f"[{self.name}] Data transmission matrix:")
        for row in self.data_transmitted.get_transmission_matrix_str().split("\n"):
            self.logger.info(row)
        if self.visualizer:
            fig = self.data_transmitted.get_figure()
            self.writer.track(aim.Figure(fig), name="Data Transmission")
        rpc.shutdown()
        self.writer.close()
        self.logger.info(f"[{self.name}] Terminated")
