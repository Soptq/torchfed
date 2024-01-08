import aim
import time
import types

from torchfed.routers.router_msg import RouterMsg, RouterMsgResponse
from typing import TypeVar, Type
from torchfed.logging import get_logger
from torchfed.types.meta import PostInitCaller

from prettytable import PrettyTable

from torchfed.utils.hash import hex_hash

T = TypeVar('T')


class Module(metaclass=PostInitCaller):
    def __init__(
            self,
            router,
            alias=None,
            visualizer=False,
            writer=None,
            override_hparams=None):
        self.ident = hex_hash(f"{time.time_ns()}")
        self.router = router
        self.alias = alias
        self.name = self.get_node_name()
        self.logger = get_logger(router.exp_id, self.get_root_name())
        self.override_hparams = override_hparams
        self.hparams = None
        self.released = False

        self.visualizer = visualizer
        self.writer = writer
        if self.visualizer:
            if writer is None:
                self.logger.info(
                    f"[{self.name}] Visualizer enabled. Run `aim up` to start.")
                self.writer = self.get_visualizer()

        self.data_sent, self.data_received = 0, 0

        self.routing_table = {}
        if self.is_root():
            router.register(self)

        if self.is_root():
            self.hparams = self.get_default_hparams()

            if self.override_hparams is not None and isinstance(
                    self.override_hparams, dict):
                for key, value in self.override_hparams.items():
                    self.hparams[key] = value

            self.hparams["name"] = self.name
            self.hparams["visualizer"] = self.visualizer
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

    def get_default_hparams(self):
        return {}

    def get_metrics(self):
        return None

    def register_submodule(self, module: Type[T], name, router, *args, **kwargs) -> T:
        submodule_name = f"{self.name}/{name}"
        if submodule_name in self.routing_table:
            self.logger.error("Cannot register modules with the same name")
            raise Exception("Cannot register modules with the same name")

        # use kwargs to override default
        if "alias" not in kwargs:
            kwargs["alias"] = submodule_name
        if "visualizer" not in kwargs:
            kwargs["visualizer"] = self.visualizer
        if "writer" not in kwargs:
            kwargs["writer"] = self.writer

        module_obj = module(
            router,
            *args,
            **kwargs)
        self.routing_table[name] = module_obj
        return module_obj

    def send(self, to, path, args):
        path = path.__name__ if callable(path) else path
        if not isinstance(to, list):
            to = [to]
        router_msgs = []
        for t in to:
            router_msg = RouterMsg(from_=self.name, to=t, path=path, args=args)
            self.data_sent += router_msg.size
            if self.visualizer:
                self.writer.track(self.data_sent, name="Data Sent (bytes)")
            router_msgs.append(router_msg)
        responses = self.router.broadcast(router_msgs)
        if len(responses) == 0:
            self.logger.warning(f"No response received for msgs {router_msgs}")
        resp_size = 0
        for response in responses:
            resp_size += response.size
        self.data_received += resp_size
        if self.visualizer:
            self.writer.track(self.data_received, name="Data Received (bytes)")
        return responses

    def receive(self, router_msg: RouterMsg) -> RouterMsgResponse:
        self.logger.debug(
            f"Module {self.name} receiving data {router_msg}")
        self.logger.debug(
            f"Module {self.name} is calling path {router_msg.path} with args {router_msg.args}")

        self.data_received += router_msg.size
        if self.visualizer:
            self.writer.track(self.data_received, name="Data Received (bytes)")

        # infinite loop until there is some return value
        while True:
            try:
                ret = RouterMsgResponse(
                    from_=self.name,
                    to=router_msg.from_,
                    data=self.entry(
                        router_msg.path,
                        router_msg.args))
                break
            except Exception as e:
                self.logger.warning(
                    f"Error in {self.name} when calling {router_msg.path} from {router_msg.from_} "
                    f"with args {router_msg.args}: {e}")
                self.logger.warning(f"Will try again in 1 second")
                time.sleep(1)

        self.data_sent += ret.size
        if self.visualizer:
            self.writer.track(self.data_sent, name="Data Sent (bytes)")

        if ret.data is None:
            self.logger.warning(
                f"Module {self.name} does not have path {router_msg.path}")
        self.logger.debug(f"Module {self.name} responses with data {ret}")
        return ret

    def entry(self, path, args, check_exposed=True):
        if isinstance(
                path,
                types.MethodType) and isinstance(
                path.__self__,
                Module):
            path = f"{'/'.join(path.__self__.name.split('/')[1:])}/{path.__name__}"

        paths = path.split("/")
        target = paths.pop(0)

        if target in self.routing_table:
            return self.routing_table[target].entry(
                "/".join(paths), args, check_exposed=check_exposed)
        elif hasattr(self, target):
            entrance = getattr(self, target)
            if callable(entrance) and (
                    not check_exposed or (
                    hasattr(
                        entrance,
                        "exposed") and entrance.exposed)):
                return entrance(*args)

    def release(self):
        if self.released:
            return
        for module in self.routing_table.values():
            module.release()
        if self.visualizer:
            self.writer.close()
        self.released = True
        self.logger.info(f"[{self.name}] Terminated")

    def get_node_name(self):
        if self.alias is not None:
            return self.alias
        return f"{self.__class__.__name__}_{self.ident[:4]}_{self.router.exp_id[:4]}"

    def get_visualizer(self):
        return aim.Run(
            run_hash=self.get_node_name(),
            experiment=self.router.exp_id
        )

    def is_root(self):
        return "/" not in self.name

    def get_root_name(self):
        return self.name.split("/")[0]

    def get_path(self):
        return "/".join(self.name.split("/")[1:])
