import abc
from torchfed.datasets import Dataset


class DatasetManager(abc.ABC):
    def __init__(self, name, dataset: Dataset):
        self.name = name
        self.dataset = dataset

    def get_dataset(self):
        return self.dataset.get_bundle_dataset()

    def get_user_dataset(self, idx):
        return self.dataset.get_user_dataset(idx)
