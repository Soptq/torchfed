from abc import abstractmethod, ABC

from torchfed.types.datasets import UserDataset
from torchfed.types.named import Named


class Dataset(Named):
    @abstractmethod
    def get_user_dataset(self, user_idx) -> UserDataset:
        pass

    @abstractmethod
    def get_bundle_dataset(self):
        pass
