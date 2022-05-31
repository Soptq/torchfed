import abc

from torchfed.routers.router_msg import RouterMsg, RouterMsgResponse
from typing import TypeVar, Generic

T = TypeVar('T')


class Module(abc.ABC):
    def __init__(self, name, router, debug=False):
        self.name = name
        self.debug = debug
        self.router = router
        self.routing_table = {}

        router.register(self)

        self.execute_gen = self.execute()

    def __call__(self, *args, **kwargs):
        _continue = next(self.execute_gen)
        if not _continue:
            self.execute_gen = self.execute()
        return _continue

    @abc.abstractmethod
    def execute(self):
        yield False

    def register_submodule(self, module: Generic[T], name, router, *args) -> T:
        submodule_name = f"{self.name}/{name}"
        if submodule_name in self.routing_table:
            raise Exception("Cannot register modules with the same name")
        module_obj = module(submodule_name, router, *args, self.debug)
        self.routing_table[name] = module_obj
        return module_obj

    def send(self, to, path, args):
        if callable(path):
            path = f"{'/'.join(path.__self__.name.split('/')[1:])}/{path.__name__}"
        router_msg = RouterMsg(from_=self.name, to=to, path=path, args=args)
        return self.router.broadcast(router_msg)

    def receive(self, router_msg: RouterMsg) -> RouterMsgResponse:
        if self.debug:
            print(f"Module {self.name} receiving data {router_msg}")
        if self.debug:
            print(f"Module {self.name} is calling path {router_msg.path} with args {router_msg.args}")

        ret = RouterMsgResponse(from_=self.name, to=router_msg.from_, data=self.manual_call(router_msg.path, router_msg.args))

        if self.debug:
            if ret.data is None:
                print(f"Module {self.name} does not have path {router_msg.path}")
            print(f"Module {self.name} responses with data {ret}")
        return ret

    def manual_call(self, path, args, check_exposed=True):
        if callable(path):
            path = f"{'/'.join(path.__self__.name.split('/')[1:])}/{path.__name__}"

        paths = path.split("/")
        target = paths.pop(0)

        if target in self.routing_table:
            return self.routing_table[target].manual_call("/".join(paths), args, check_exposed=check_exposed)
        elif hasattr(self, target):
            entrance = getattr(self, target)
            if callable(entrance) and (not check_exposed or (hasattr(entrance, "exposed") and entrance.exposed)):
                return entrance(*args)
        return None

