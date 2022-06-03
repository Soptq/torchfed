from abc import abstractmethod, ABC
from typing import List

from torchfed.types.datasets import UserDataset, GlobalDataset
from torchfed.types.named import Named


class Dataset(Named):
    @abstractmethod
    def get_user_dataset(self, user_idx) -> UserDataset:
        pass

    @abstractmethod
    def get_dataset(self) -> List[GlobalDataset]:
        pass
