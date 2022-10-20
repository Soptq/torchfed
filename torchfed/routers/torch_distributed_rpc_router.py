import os
from typing import List

import torch
import torch.distributed.rpc as rpc

from .router import Router
from .router_msg import RouterMsg, RouterMsgResponse


class TorchDistributedRPCRouter(Router):
    def __init__(
            self,
            rank,
            world_size,
            backend=None,
            rpc_backend_options=None,
            alias=None,
            visualizer=False):
        super().__init__(alias=alias, visualizer=visualizer)
        self.rank = rank
        self.world_size = world_size
        if backend is None:
            backend = rpc.BackendType.TENSORPIPE

        if rpc_backend_options is None:
            rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
                init_method="env://",
                rpc_timeout=0
            )
        torch.distributed.rpc.init_rpc(
            self.name, backend, rank, world_size, rpc_backend_options)

    def impl_broadcast(
            self,
            router_msg: List[RouterMsg]) -> List[RouterMsgResponse]:
        futs, rets = [], []
        for rank in range(self.world_size):
            futs.append(
                rpc.rpc_async(
                    rank,
                    Router.receive,
                    args=(
                        router_msg,
                    )))
        for fut in futs:
            resp = fut.wait()
            if resp is not None:
                rets.extend(fut.wait())

        for ret in rets:
            self.data_transmitted.add(
                self.get_root_name(ret.from_),
                self.get_root_name(ret.to),
                ret.size
            )
        return rets

    def impl_release(self):
        rpc.shutdown()
