from typing import Callable

from torchfed.base.component import BaseComponent


class SimpleCustomComponent(BaseComponent):
    def __init__(self, component_id, stage, func: Callable[[int], None]):
        super().__init__(component_id, stage)
        self.exec = func

    def execute(self, epoch: int):
        self.exec(epoch)
