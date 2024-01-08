import time
from typing import List

import aim

from .router_msg import RouterMsg, RouterMsgResponse
from torchfed.logging import get_logger
from torchfed.utils.hash import hex_hash
from torchfed.utils.plotter import NetworkConnectionsPlotter, DataTransmitted


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, mode="singleton", ident=None, **kwargs):
        if mode == "singleton":
            if cls not in cls._instances:
                cls._instances[cls] = super(
                    Singleton, cls).__call__(
                    *args, **kwargs)
            return cls._instances[cls]
        elif mode == "simulate":
            if ident is None:
                raise ValueError(
                    "ident must be provided when mode is simulate")
            if ident not in cls._instances:
                cls._instances[ident] = super(
                    Singleton, cls).__call__(
                    *args, **kwargs)
            return cls._instances[ident]


class Router(metaclass=Singleton):
    context = None

    def __init__(
            self,
            alias=None,
            visualizer=False):
        Router.context = self
        self.ident = hex_hash(f"{time.time_ns()}")
        self.exp_id = hex_hash(f"{time.time_ns()}")
        self.alias = alias
        self.name = self.get_router_name()
        self.visualizer = visualizer
        self.logger = get_logger(self.exp_id, self.name)
        self.released = False

        if self.visualizer:
            self.logger.info(
                f"[{self.name}] Visualizer enabled. Run `aim up` to start.")
            self.writer = self.get_visualizer()

        self.owned_nodes = {}
        self.peers_table = {}

        self.network_plotter = NetworkConnectionsPlotter()
        self.data_transmitted = DataTransmitted()

        self.logger.info(
            f"[{self.name}] Initialized completed. Router ID: {self.ident}. Experiment ID: {self.exp_id}")

    def register(self, module):
        if module.name not in self.owned_nodes.keys():
            self.owned_nodes[module.name] = module.receive

    def unregister(self, worker):
        if worker.name in self.owned_nodes.keys():
            del self.owned_nodes[worker.name]

    def connect(self, module, peers: list):
        if not module.is_root():
            return
        peers = [self.get_root_name(peer) for peer in peers]
        if module.name in self.peers_table:
            for peer in peers:
                if peer not in self.peers_table[module.name]:
                    self.peers_table[module.name].append(peer)
        else:
            self.peers_table[module.name] = peers

        for peer in peers:
            self.network_plotter.add_edge(module.name, peer)

        if self.visualizer:
            fig = self.network_plotter.get_figure()
            self.writer.track(aim.Figure(fig), name="Network Graph")

    def disconnect(self, module, peers: list):
        if not module.is_root():
            return
        peers = [self.get_root_name(peer) for peer in peers]
        if module.name in self.peers_table:
            for peer in peers:
                if peer in self.peers_table[module.name]:
                    self.peers_table[module.name].remove(peer)
                    self.network_plotter.remove_edge(module.name, peer)

        if self.visualizer:
            fig = self.network_plotter.get_figure()
            self.writer.track(aim.Figure(fig), name="Network Graph")

    def disconnect_all(self, module):
        peers = self.get_peers(module)
        self.disconnect(module, peers)

    def get_peers(self, module):
        name = module.get_root_name()
        return self.peers_table[name]

    def n_peers(self, module):
        name = module.get_root_name()
        return len(self.peers_table[name]) if name in self.peers_table else 1

    def broadcast(
            self,
            router_msg: List[RouterMsg]) -> List[RouterMsgResponse]:
        for msg in router_msg:
            self.logger.debug(
                f"[{self.name}] broadcasting message {msg}")

            self.data_transmitted.add(
                self.get_root_name(msg.from_),
                self.get_root_name(msg.to),
                msg.size
            )
        return self.impl_broadcast(router_msg)

    def impl_broadcast(
            self,
            router_msg: List[RouterMsg]) -> List[RouterMsgResponse]:
        raise NotImplementedError

    @staticmethod
    def receive(router_msg: List[RouterMsg]):
        responses = []
        for msg in router_msg:
            if msg.to not in Router.context.owned_nodes.keys():
                continue

            Router.context.logger.debug(
                f"[{Router.context.name}] receiving message {msg}")

            Router.context.data_transmitted.add(
                Router.get_root_name(msg.from_),
                Router.get_root_name(msg.to),
                msg.size
            )
            resp_msg = Router.context.owned_nodes[msg.to](msg)
            Router.context.data_transmitted.add(
                Router.get_root_name(resp_msg.from_),
                Router.get_root_name(resp_msg.to),
                resp_msg.size
            )
            responses.append(resp_msg)
        return responses

    @staticmethod
    def get_root_name(name):
        return name.split("/")[0]

    def get_visualizer(self):
        return aim.Run(
            run_hash=self.name,
            experiment=self.exp_id
        )

    def refresh_exp_id(self):
        self.exp_id = hex_hash(f"{time.time_ns()}")
        self.logger.info(
            f"[{self.name}] Experiment ID refreshed: {self.exp_id}")

    def release(self):
        if self.released:
            return
        self.impl_release()
        self.logger.info(f"[{self.name}] Data transmission matrix:")
        for row in self.data_transmitted.get_transmission_matrix_str().split("\n"):
            self.logger.info(row)
        if self.visualizer:
            fig = self.data_transmitted.get_figure()
            self.writer.track(aim.Figure(fig), name="Data Transmission")
            self.writer.close()
        self.released = True
        self.logger.info(f"[{self.name}] Terminated")

    def impl_release(self):
        raise NotImplementedError

    def get_router_name(self):
        if self.alias is not None:
            return self.alias
        return f"{self.__class__.__name__}_{self.ident[:4]}_{self.exp_id[:4]}"
