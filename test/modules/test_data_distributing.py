import os
from torchfed.routers import Router
from torchfed.modules.module import Module
from torchfed.modules.distribute.data_distribute import DataDistributing
from torchfed.utils.decorator import exposed

DEBUG = True


class Server(Module):
    def __init__(self, name, router, debug=False):
        super(Server, self).__init__(name, router, debug)
        self.distributor = self.register_submodule(DataDistributing, "distributor", router)

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
    router_a = Router(0, 1, debug=DEBUG)

    server = Server("server", router_a, debug=DEBUG)
    clients = []

    for i in range(5):
        clients.append(Client(f"client_{i}", router_a, debug=DEBUG))

    for index, client in enumerate(clients):
        client.send("server", server.distributor.upload, (client.name, index))

    aggregation = server.manual_call(server.distributor.aggregate, (), check_exposed=False)
    assert aggregation == 2.0
    server.manual_call(server.distributor.update, (aggregation,), check_exposed=False)

    resp = clients[0].send("server", server.distributor.download, ())[0]
    assert resp.from_ == "server"
    assert resp.to == "client_0"
    assert resp.data == 2.0
