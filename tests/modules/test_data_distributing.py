import os
from torchfed.routers import TorchDistributedRPCRouter
from torchfed.modules.module import Module
from torchfed.modules.distribute.data_distribute import DataDistributing
from torchfed.utils.decorator import exposed


class Server(Module):
    def __init__(self, router):
        super(Server, self).__init__(router)
        self.distributor = self.register_submodule(
            DataDistributing, "distributor", router)

    @exposed
    def execute(self):
        return "Executing"


class Client(Module):
    @exposed
    def execute(self, *args):
        return "Executing"


def test_data_distributing():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    router_a = TorchDistributedRPCRouter(0, 1)

    server = Server(router_a)
    clients = []

    for i in range(5):
        clients.append(Client(router_a))

    for index, client in enumerate(clients):
        client.send(
            server.get_node_name(),
            server.distributor.upload,
            (client.name,
             index))

    aggregation = server.entry(
        server.distributor.aggregate, (), check_exposed=False)
    assert aggregation == 2.0
    server.entry(server.distributor.update,
                 (aggregation,), check_exposed=False)

    resp = clients[0].send(
        server.get_node_name(),
        server.distributor.download,
        ())[0]
    assert resp.from_ == server.get_node_name()
    assert resp.to == clients[0].get_node_name()
    assert resp.data == 2.0
