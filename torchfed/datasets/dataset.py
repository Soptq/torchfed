import os
import torch

from torch.utils.data import Dataset

from torchvision.transforms import transforms

from abc import abstractmethod, ABC
from typing import Optional, Callable, List

from torchfed.types.named import Named


class TorchGlobalDataset(Dataset):
    """
    GlobalDataset, as its name suggests, is a global dataset wrapper that contains all data.
    """

    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            "inputs": self.dataset[idx][0],
            "labels": self.dataset[idx][1],
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


class TorchDataset(Named):
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
