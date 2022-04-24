from abc import abstractmethod, ABC

from torchfed.utils.datasets import UserDataset


class Dataset(ABC):
    @abstractmethod
    def get_user_dataset(self, user_idx) -> UserDataset:
        pass
