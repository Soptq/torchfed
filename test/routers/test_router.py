import os
from torchfed.routers import Router
from torchfed.modules.module import Module
from torchfed.utils.decorator import exposed

DEBUG = True


class TestSubSubModule(Module):
    @exposed
    def execute(self):
        return "SubSubModule Executing"


class TestSubModule(Module):
    def __init__(self, name, router, debug=False):
        super(TestSubModule, self).__init__(name, router, debug)
        self.submodule = self.register_submodule(TestSubSubModule, "submodule", router)

    @exposed
    def execute(self):
        return "SubModule Executing"


class TestMainModule(Module):
    def __init__(self, name, router, debug=False):
        super(TestMainModule, self).__init__(name, router, debug)
        self.submodule = self.register_submodule(TestSubModule, "submodule", router)

    @exposed
    def execute(self):
        return "Executing"


def test_connectivity_local():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    router_a = Router(0, 1, debug=DEBUG)

    module_a = TestMainModule("alice", router_a, debug=DEBUG)
    module_b = TestMainModule("bob", router_a, debug=DEBUG)

    resp_a = module_a.send(to="bob", path="execute", args=())[0]
    assert resp_a.from_ == "bob"
    assert resp_a.to == "alice"
    assert resp_a.data == "Executing"

    resp_b = module_b.send(to="alice", path="execute", args=())[0]
    assert resp_b.from_ == "alice"
    assert resp_b.to == "bob"
    assert resp_b.data == "Executing"
    del router_a


def test_connectivity_submodule_local():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    router_a = Router(0, 1, debug=DEBUG)

    module_a = TestMainModule("alice", router_a, debug=DEBUG)
    module_b = TestMainModule("bob", router_a, debug=DEBUG)

    resp_a = module_a.send(to="bob", path="submodule/execute", args=())[0]
    assert resp_a.from_ == "bob"
    assert resp_a.to == "alice"
    assert resp_a.data == "SubModule Executing"
    del router_a


def test_connectivity_subsubmodule_local():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    router_a = Router(0, 1, debug=DEBUG)

    module_a = TestMainModule("alice", router_a, debug=DEBUG)
    module_b = TestMainModule("bob", router_a, debug=DEBUG)

    resp_a = module_a.send(to="bob", path="submodule/submodule/execute", args=())[0]
    assert resp_a.from_ == "bob"
    assert resp_a.to == "alice"
    assert resp_a.data == "SubSubModule Executing"
    del router_a
