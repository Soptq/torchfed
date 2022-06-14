import aim

from torchfed.routers.router_msg import RouterMsg, RouterMsgResponse
from typing import TypeVar, Generic
from torchfed.logging import get_logger
from torchfed.types.meta import PostInitCaller

from prettytable import PrettyTable

T = TypeVar('T')


class Module(metaclass=PostInitCaller):
    def __init__(
            self,
            name,
            router,
            visualizer=False,
            writer=None,
            override_hparams=None,
            debug=False):
        self.name = name
        self.debug = debug
        self.router = router
        self.logger = get_logger(router.ident, self.get_root_name())
        self.override_hparams = override_hparams
        self.hparams = None

        self.visualizer = visualizer
        self.writer = writer
        if self.visualizer:
            if writer is None:
                self.logger.info(
                    f"[{self.name}] Visualizer enabled. Run `aim up` to start.")
                self.writer = self.get_visualizer()

        self.data_sent, self.data_received = 0, 0

        self.routing_table = {}
        router.register(self)

        self.execute_gen = self.execute()

        if self.is_root():
            self.hparams = self.set_hparams()

            if self.override_hparams is not None and isinstance(self.override_hparams, dict):
                for key, value in self.override_hparams.items():
                    self.hparams[key] = value

            self.hparams["name"] = self.name
            self.hparams["visualizer"] = self.visualizer
            self.hparams["debug"] = self.debug
            hp_table = PrettyTable()
            hp_table.field_names = self.hparams.keys()
            hp_table.add_row(self.hparams.values())
            self.logger.info(f"[{self.name}] Hyper-parameters:")
            for row in hp_table.get_string().split("\n"):
                self.logger.info(row)
            if self.visualizer:
                self.writer['hparams'] = self.hparams

    def __post__init__(self):
        pass

    def __call__(self, *args, **kwargs):
        _continue = next(self.execute_gen)
        if not _continue:
            self.execute_gen = self.execute()
        return _continue

    def set_hparams(self):
        return {}

    def execute(self):
        yield False

    def get_metrics(self):
        return None

    def register_submodule(self, module: Generic[T], name, router, *args) -> T:
        submodule_name = f"{self.name}/{name}"
        if submodule_name in self.routing_table:
            self.logger.error("Cannot register modules with the same name")
            raise Exception("Cannot register modules with the same name")
        module_obj = module(
            submodule_name,
            router,
            *args,
            visualizer=self.visualizer,
            writer=self.writer,
            debug=self.debug)
        self.routing_table[name] = module_obj
        return module_obj

    def send(self, to, path, args):
        if callable(path):
            path = f"{'/'.join(path.__self__.name.split('/')[1:])}/{path.__name__}"
        router_msg = RouterMsg(from_=self.name, to=to, path=path, args=args)
        self.data_sent += router_msg.size
        if self.visualizer:
            self.writer.track(self.data_sent, name="Data Sent (bytes)")
        responses = self.router.broadcast(router_msg)
        resp_size = 0
        for response in responses:
            resp_size += response.size
        self.data_received += resp_size
        if self.visualizer:
            self.writer.track(self.data_received, name="Data Received (bytes)")
        return responses

    def receive(self, router_msg: RouterMsg) -> RouterMsgResponse:
        if self.debug:
            self.logger.debug(
                f"Module {self.name} receiving data {router_msg}")
        if self.debug:
            self.logger.debug(
                f"Module {self.name} is calling path {router_msg.path} with args {router_msg.args}")

        self.data_received += router_msg.size
        if self.visualizer:
            self.writer.track(self.data_received, name="Data Received (bytes)")

        ret = RouterMsgResponse(
            from_=self.name,
            to=router_msg.from_,
            data=self.manual_call(
                router_msg.path,
                router_msg.args))

        self.data_sent += ret.size
        if self.visualizer:
            self.writer.track(self.data_sent, name="Data Sent (bytes)")

        if ret.data is None:
            self.logger.warning(
                f"Module {self.name} does not have path {router_msg.path}")
        if self.debug:
            self.logger.debug(f"Module {self.name} responses with data {ret}")
        return ret

    def manual_call(self, path, args, check_exposed=True):
        if callable(path):
            path = f"{'/'.join(path.__self__.name.split('/')[1:])}/{path.__name__}"

        paths = path.split("/")
        target = paths.pop(0)

        if target in self.routing_table:
            return self.routing_table[target].manual_call(
                "/".join(paths), args, check_exposed=check_exposed)
        elif hasattr(self, target):
            entrance = getattr(self, target)
            if callable(entrance) and (
                not check_exposed or (
                    hasattr(
                        entrance,
                        "exposed") and entrance.exposed)):
                return entrance(*args)
        return None

    def get_visualizer(self):
        return aim.Run(
            run_hash=f"{self.get_root_name()}-{self.router.ident[:4]}",
            experiment=self.router.ident
        )

    def is_root(self):
        return "/" not in self.name

    def get_root_name(self):
        return self.name.split("/")[0]

    def get_path(self):
        return "/".join(self.name.split("/")[1:])

    def release(self):
        for module in self.routing_table.values():
            module.release()
        if self.visualizer:
            self.writer.close()
        self.logger.info(f"[{self.name}] Terminated")
