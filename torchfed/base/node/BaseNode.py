from __future__ import annotations

from abc import abstractmethod, ABC
from typing import List

from torchfed.base.backend import LocalBackend
from torchfed.base.backend.BaseBackend import BaseBackend
from torchfed.base.component.BaseComponent import BaseComponent
from torchfed.logging import get_logger
from torchfed.utils.hash import hex_hash
from enum import Enum


class ComponentStage(Enum):
    PRE_TRAIN = 0
    TRAIN = 1
    POST_TRAIN = 2


class BaseNode(ABC):
    def __init__(self, trainer_id: str, node_id: str, *args, **kwargs):
        # bind params
        self.trainer_id = trainer_id
        self.node_id = node_id
        self.args = args
        self.kwargs = kwargs
        self._bind_params()
        # initialize logger
        self.logger = get_logger(trainer_id, node_id)
        # initialize backends
        self.backend = LocalBackend(self.node_id)
        self.backend.add_listener(self.recv_msg)
        # initialize components
        self.components = {}
        self.logger.info(f'{self.node_id} is initialized')

    def _bind_params(self):
        if 'params' not in self.kwargs:
            return
        self.params = self.kwargs['params']
        if len(self.params) == 0:
            return
        for param in self.params.keys():
            setattr(self, param, self.params[param])

    def add_component(self, component):
        component.bind_context(self.node_id, self.backend, self.logger)
        self.components[component.component_id] = component

    def pre_train(self, epoch: int):
        for component in self.components.values():
            if component.stage == ComponentStage.PRE_TRAIN:
                component.execute(epoch)

    def train(self, epoch: int):
        for component in self.components.values():
            if component.stage == ComponentStage.TRAIN:
                component.execute(epoch)

    def post_train(self, epoch: int):
        for component in self.components.values():
            if component.stage == ComponentStage.POST_TRAIN:
                component.execute(epoch)

    def recv_msg(self, _from, func, *args, **kwargs):
        if hasattr(self, func):
            self.logger.info(f'{self.node_id} is calling {func} by {_from}')
            return getattr(self, func)(_from, *args, **kwargs)
