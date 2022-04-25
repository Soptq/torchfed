from abc import abstractmethod, ABC

from torchfed.utils.datasets import UserDataset


class Dataset(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_user_dataset(self, user_idx) -> UserDataset:
        pass

    @abstractmethod
    def get_bundle_dataset(self):
        pass
