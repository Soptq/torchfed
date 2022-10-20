import os
from torchfed.routers import TorchDistributedRPCRouter
from torchfed.modules.module import Module
from torchfed.utils.decorator import exposed


class TestSubSubModule(Module):
    @exposed
    def execute(self):
        return "SubSubModule Executing"


class TestSubModule(Module):
    def __init__(self,
                 router,
                 alias=None,
                 visualizer=False,
                 writer=None,
                 override_hparams=None):
        super(
            TestSubModule,
            self).__init__(
            router,
            alias=alias,
            visualizer=visualizer,
            writer=writer,
            override_hparams=override_hparams)
        self.submodule = self.register_submodule(
            TestSubSubModule, "submodule", router)

    @exposed
    def execute(self):
        return "SubModule Executing"


class TestMainModule(Module):
    def __init__(self, router):
        super(TestMainModule, self).__init__(router)
        self.submodule = self.register_submodule(
            TestSubModule, "submodule", router)

    @exposed
    def execute(self):
        return "Executing"


def test_connectivity_local():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    router_a = TorchDistributedRPCRouter(0, 1)

    alice = TestMainModule(router_a)
    bob = TestMainModule(router_a)

    resp_a = alice.send(to=bob.get_node_name(), path="execute", args=())[0]
    assert resp_a.from_ == bob.get_node_name()
    assert resp_a.to == alice.get_node_name()
    assert resp_a.data == "Executing"

    resp_b = bob.send(to=alice.get_node_name(), path="execute", args=())[0]
    assert resp_b.from_ == alice.get_node_name()
    assert resp_b.to == bob.get_node_name()
    assert resp_b.data == "Executing"
    del router_a


def test_connectivity_submodule_local():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    router_a = TorchDistributedRPCRouter(0, 1)

    alice = TestMainModule(router_a)
    bob = TestMainModule(router_a)

    resp_a = alice.send(
        to=bob.get_node_name(),
        path="submodule/execute",
        args=())[0]
    assert resp_a.from_ == bob.get_node_name()
    assert resp_a.to == alice.get_node_name()
    assert resp_a.data == "SubModule Executing"
    del router_a


def test_connectivity_subsubmodule_local():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    router_a = TorchDistributedRPCRouter(0, 1)

    alice = TestMainModule(router_a)
    bob = TestMainModule(router_a)

    resp_a = alice.send(
        to=bob.get_node_name(),
        path="submodule/submodule/execute",
        args=())[0]
    assert resp_a.from_ == bob.get_node_name()
    assert resp_a.to == alice.get_node_name()
    assert resp_a.data == "SubSubModule Executing"
    del router_a
