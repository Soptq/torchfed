from torchfed.base.component import BaseComponent


class AverageComponent(BaseComponent):
    def __init__(self, component_id, stage, model, sample_size=-1):
        super().__init__(component_id, stage)
        self.model = model
        self.sample_size = sample_size
        self.total_size = 0
        self.count = 0
        self.storage = {}

    def _reset(self):
        for name, param in self.storage.items():
            param.data.zero_()
        self.total_size = 0
        self.count = 0

    def update_model(self, model, dataset_size):
        if 0 < self.sample_size <= self.count:
            return
        self.count += 1
        self.total_size += dataset_size
        for name, param in model.named_parameters():
            if name not in self.storage:
                self.storage[name] = param.data.clone() * dataset_size
            else:
                self.storage[name] += param.data.clone() * dataset_size

    def execute(self, epoch: int):
        if self.total_size == 0:
            return
        for name, param in self.model.named_parameters():
            param.data = self.storage[name] / self.total_size
        self._reset()
