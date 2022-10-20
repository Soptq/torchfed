import json
from enum import Enum

from torchfed.routers.p2p.node import Node
from torchfed.routers.router_msg import RouterMsg, RouterMsgResponse


class P2PQueryType(str, Enum):
    GET_PEERS_QUERY = 'GET_PEERS_QUERY'
    GET_PEERS_RESP = 'GET_PEERS_RESP'
    REGULAR_MSG = 'REGULAR_MSG'
    REGULAR_MSG_RESP = 'REGULAR_MSG_RESP'


def construct_query(query_type: P2PQueryType, msg):
    return {
        'type': query_type,
        'msg': msg
    }


class P2PNode(Node):
    # Python class constructor
    def __init__(
            self,
            host,
            port,
            id=None,
            receive_callback=None,
            response_callback=None,
            max_connections=0):
        super(
            P2PNode,
            self).__init__(
            host,
            port,
            id,
            callback=None,
            max_connections=max_connections)
        self.receive_callback = receive_callback
        self.response_callback = response_callback

    def bootstrap(self, host: str, port: int, reconnect: bool = False) -> bool:
        bootstrap_server_connected = self.connect_with_node(
            host, port, reconnect=reconnect)
        if not bootstrap_server_connected:
            raise Exception("Bootstrap server is not connected")

    def outbound_node_connected(self, connected_node):
        pass

    def outbound_node_connection_error(self, exception: Exception):
        pass

    def inbound_node_connected(self, connected_node):
        pass

    def inbound_node_connection_error(self, exception: Exception):
        pass

    def inbound_node_disconnected(self, connected_node):
        print("inbound_node_disconnected: " + connected_node.id)

    def outbound_node_disconnected(self, connected_node):
        print("outbound_node_disconnected: " + connected_node.id)

    def node_message(self, connected_node, data):
        print("node_message from " + connected_node.id + ": " + str(data))

        if data['type'] == P2PQueryType.GET_PEERS_QUERY:
            all_nodes_info = [{"host": node.host,
                               "port": node.port,
                               "id": node.id} for node in self.all_nodes]
            connected_node.send(
                construct_query(
                    P2PQueryType.GET_PEERS_RESP,
                    all_nodes_info))

        # TODO
        if data['type'] == P2PQueryType.GET_PEERS_RESP:
            print(data)

        if data['type'] == P2PQueryType.REGULAR_MSG:
            router_msg = RouterMsg.deserialize(data['msg'])
            response = self.receive_callback(router_msg)
            if response is not None:
                connected_node.send(
                    construct_query(
                        P2PQueryType.REGULAR_MSG_RESP,
                        response.serialize()))

        if data['type'] == P2PQueryType.REGULAR_MSG_RESP:
            router_msg_response = RouterMsgResponse.deserialize(data['msg'])
            self.response_callback(router_msg_response)

    def node_disconnect_with_outbound_node(self, connected_node):
        print(
            "node wants to disconnect with oher outbound node: " +
            connected_node.id)

    def node_request_to_stop(self):
        print("node is requested to stop!")
