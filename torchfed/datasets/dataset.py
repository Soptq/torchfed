import os
from itertools import accumulate
from operator import add
import torch

from torch.utils.data import Dataset

from torchvision.transforms import transforms

from abc import abstractmethod, ABC
from typing import Optional, Callable, List


class TorchGlobalDataset(Dataset):
    """
    GlobalDataset, as its name suggests, is a global dataset wrapper that contains all data.
    """

    def __init__(self, inputs: list, labels: list, num_classes: int):
        self.inputs: list = inputs
        self.labels: list = labels
        self.num_classes: int = num_classes

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "inputs": self.inputs[idx],
            "labels": self.labels[idx],
        }


class TorchUserDataset(Dataset):
    """UserDataset, as its name suggests, is a dataset wrapper for a specific user"""

    def __init__(self, user_id, inputs, labels, num_classes):
        self.user_id = user_id
        self.inputs = inputs
        self.labels = labels
        self.num_classes = num_classes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "inputs": self.inputs[idx],
            "labels": self.labels[idx]
        }


class BaseTorchDataset:
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_global_dataset(self) -> List[TorchGlobalDataset]:
        raise NotImplementedError

    @abstractmethod
    def get_user_dataset(self, user_idx) -> List[TorchUserDataset]:
        raise NotImplementedError


class TorchDataset(BaseTorchDataset):
    def __init__(
            self,
            root: str,
            num_classes: int,
            num_users: int,
            num_labels_for_users: int,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            rebuild: bool = False,
            cache_salt: int = 0,
    ) -> None:
        self.root = root
        self.num_classes = num_classes
        self.num_users = num_users
        self.num_labels_for_users = num_labels_for_users
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.identifier = f"{self.name}-{num_users}-{num_labels_for_users}-{cache_salt}"

        self.global_dataset: Optional[List[TorchGlobalDataset]] = None
        self.user_dataset: Optional[List[List[TorchUserDataset]]] = None

        if not rebuild:
            try:
                # load global dataset
                with open(os.path.join(root, f"{self.identifier}.global.fds"), 'rb') as f:
                    self.global_dataset = torch.load(f)
                with open(os.path.join(root, f"{self.identifier}.user.fds"), 'rb') as f:
                    self.user_dataset = torch.load(f)
                return
            except Exception:
                pass

        self.global_dataset = self.load_global_dataset()
        self.user_dataset = self.load_user_dataset()
        assert len(self.user_dataset) == self.num_users
        with open(os.path.join(root, f"{self.identifier}.global.fds"), 'wb') as f:
            torch.save(self.global_dataset, f)
        with open(os.path.join(root, f"{self.identifier}.user.fds"), 'wb') as f:
            torch.save(self.user_dataset, f)

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def load_global_dataset(self) -> List[TorchGlobalDataset]:
        raise NotImplementedError

    @abstractmethod
    def load_user_dataset(self) -> List[List[TorchUserDataset]]:
        raise NotImplementedError

    def get_global_dataset(self) -> List[TorchGlobalDataset]:
        return self.global_dataset

    def get_user_dataset(self, user_idx) -> List[TorchUserDataset]:
        return self.user_dataset[user_idx]


class ComposedTorchDataset(BaseTorchDataset):
    def __init__(
            self,
            datasets: List[BaseTorchDataset],
            user_dist: List[int],
            num_classes: int,
    ) -> None:
        self.datasets = datasets
        self.user_dist = user_dist
        self.accu_user_dist = list(accumulate(user_dist, func=add))
        self.num_classes = num_classes

    @property
    def name(self) -> str:
        return "-".join([d.name for d in self.datasets])

    def get_global_dataset(self) -> List[TorchGlobalDataset]:
        inner_global_datasets = [d.get_global_dataset() for d in self.datasets]
        train_inputs, train_labels = [], []
        test_inputs, test_labels = [], []
        for [train_dataset, test_dataset] in inner_global_datasets:
            train_inputs.extend(train_dataset.inputs)
            train_labels.extend(train_dataset.labels)
            test_inputs.extend(test_dataset.inputs)
            test_labels.extend(test_dataset.labels)
        return [
            TorchGlobalDataset(train_inputs, train_labels, self.num_classes),
            TorchGlobalDataset(test_inputs, test_labels, self.num_classes),
        ]

    def get_user_dataset(self, user_idx) -> List[TorchUserDataset]:
        for idx, dist in enumerate(self.accu_user_dist):
            if user_idx < dist:
                return self.datasets[idx].get_user_dataset(user_idx - (0 if idx == 0 else self.accu_user_dist[idx - 1]))
        raise ValueError(f"Invalid user_idx: {user_idx}")
