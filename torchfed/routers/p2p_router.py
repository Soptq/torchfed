import time
from typing import List

from torchfed.routers.p2p.p2p_node import P2PNode, P2PQueryType, construct_query

from .router import Router
from .router_msg import RouterMsg, RouterMsgResponse


class P2PRouter(Router):
    response_buffer = None

    def __init__(
            self,
            host: str,
            port: int,
            alias=None,
            visualizer=False):
        super().__init__(alias=alias, visualizer=visualizer)
        self.host = host
        self.port = port
        self.response_buffer = None
        self.timeout = 60 * 100
        self.p2p_node = P2PNode(
            host,
            port,
            id=self.name,
            receive_callback=Router.receive,
            response_callback=P2PRouter.response_callback)
        self.p2p_node.start()
        # TODO: Connect to peers

    @staticmethod
    def response_callback(data):
        P2PRouter.response_buffer = data

    def impl_broadcast(self, router_msg: RouterMsg) -> List[RouterMsgResponse]:
        # self response
        rets = [Router.receive(router_msg)]

        P2PRouter.response_buffer = None
        self.p2p_node.send_to_nodes(
            construct_query(
                P2PQueryType.REGULAR_MSG,
                router_msg.serialize()))

        timeout_counter = 0
        while True:
            if timeout_counter > self.timeout:
                raise Exception("Broadcast timeout")
            if P2PRouter.response_buffer is not None:
                rets.append(P2PRouter.response_buffer)
                break

            # TODO: Wait response. Maybe we will async this process in the
            # future
            timeout_counter += 1
            time.sleep(0.01)

        for ret in rets:
            self.data_transmitted.add(
                self.get_root_name(ret.from_),
                self.get_root_name(ret.to),
                ret.size
            )
        return rets

    def impl_release(self):
        self.p2p_node.stop()
