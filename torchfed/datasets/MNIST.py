import sys
import os.path
import random
from typing import Optional, Callable, List

import numpy as np

import torch
import torchvision
from tqdm import trange, tqdm

from torchfed.datasets import TorchDataset, TorchUserDataset, TorchGlobalDataset


class TorchMNIST(TorchDataset):
    num_classes = 10

    def __init__(
            self,
            root: str,
            num_users: int,
            num_labels_for_users: int,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            rebuild: bool = False,
            cache_salt: int = 0,
    ) -> None:
        super().__init__(
            root,
            TorchMNIST.num_classes,
            num_users,
            num_labels_for_users,
            transform,
            target_transform,
            download,
            rebuild,
            cache_salt
        )

    @property
    def name(self) -> str:
        return "MNIST"

    def load_user_dataset(self) -> List[List[TorchUserDataset]]:
        train_dataset = torchvision.datasets.MNIST(
            self.root, True, self.transform, self.target_transform, self.download)
        test_dataset = torchvision.datasets.MNIST(
            self.root, False, self.transform, self.target_transform, self.download)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=len(
                train_dataset.data), shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=len(
                test_dataset.data), shuffle=False)

        tensor_train_dataset, tensor_test_dataset = {}, {}
        for _, data in tqdm(enumerate(train_dataloader, 0), file=sys.stdout):
            tensor_train_dataset["data"], tensor_train_dataset["targets"] = data
        for _, data in tqdm(enumerate(test_dataloader, 0), file=sys.stdout):
            tensor_test_dataset["data"], tensor_test_dataset["targets"] = data

        inputs, split_inputs, labels = [], [], []
        inputs.extend(tensor_train_dataset["data"].cpu().detach().numpy())
        inputs.extend(tensor_test_dataset["data"].cpu().detach().numpy())
        labels.extend(tensor_train_dataset["targets"].cpu().detach().numpy())
        labels.extend(tensor_test_dataset["targets"].cpu().detach().numpy())
        inputs = np.array(inputs)
        labels = np.array(labels)

        for label in trange(self.num_classes):
            split_inputs.append(inputs[labels == label])
        _, num_channels, num_height, num_width = split_inputs[0].shape

        user_x = [[] for _ in range(self.num_users)]
        user_y = [[] for _ in range(self.num_users)]
        idx = np.zeros(self.num_classes, dtype=np.int64)
        for user_idx in trange(self.num_users):
            for label_idx in range(self.num_labels_for_users):
                assigned_label = (user_idx + label_idx) % self.num_classes
                user_x[user_idx] += split_inputs[assigned_label][idx[assigned_label]                                                                 : idx[assigned_label] + 10].tolist()
                user_y[user_idx] += (assigned_label * np.ones(10)).tolist()
                idx[assigned_label] += 10

        props = np.random.lognormal(
            0, 2., (10, self.num_users, self.num_labels_for_users)
        )
        props = np.array([[[len(v) - self.num_users]] for v in split_inputs]
                         ) * props / np.sum(props, (1, 2), keepdims=True)
        for user_idx in trange(self.num_users):
            for label_idx in range(self.num_labels_for_users):
                assigned_label = (user_idx + label_idx) % self.num_classes
                num_samples = int(
                    props[assigned_label, user_idx // int(self.num_users / 10), label_idx])
                num_samples += random.randint(300, 600)
                if self.num_users <= 20:
                    num_samples *= 2
                if idx[assigned_label] + \
                        num_samples < len(split_inputs[assigned_label]):
                    user_x[user_idx] += split_inputs[assigned_label][idx[assigned_label]                                                                     : idx[assigned_label] + num_samples].tolist()
                    user_y[user_idx] += (assigned_label *
                                         np.ones(num_samples)).tolist()
                    idx[assigned_label] += num_samples
            user_x[user_idx] = torch.Tensor(user_x[user_idx]) \
                .view(-1, num_channels, num_width, num_height) \
                .type(torch.float32)
            user_y[user_idx] = torch.Tensor(user_y[user_idx]) \
                .type(torch.int64)

        user_dataset = []
        for user_idx in trange(self.num_users):
            combined = list(zip(user_x[user_idx], user_y[user_idx]))
            random.shuffle(combined)
            user_x[user_idx], user_y[user_idx] = zip(*combined)

            num_samples = len(user_x[user_idx])
            train_len = int(num_samples * 0.75)
            train_user_data = TorchUserDataset(user_idx,
                                               user_x[user_idx][:train_len],
                                               user_y[user_idx][:train_len],
                                               self.num_classes)
            test_user_data = TorchUserDataset(user_idx,
                                              user_x[user_idx][train_len:],
                                              user_y[user_idx][train_len:],
                                              self.num_classes)
            user_dataset.append([train_user_data, test_user_data])

        return user_dataset

    def load_global_dataset(self) -> List[TorchGlobalDataset]:
        train_dataset = torchvision.datasets.MNIST(
            self.root, True, self.transform, self.target_transform, self.download)
        test_dataset = torchvision.datasets.MNIST(
            self.root, False, self.transform, self.target_transform, self.download)
        train_inputs, train_labels = [], []
        for train_idx in range(len(train_dataset)):
            train_inputs.append(train_dataset[train_idx][0])
            train_labels.append(train_dataset[train_idx][1])
        test_inputs, test_labels = [], []
        for test_idx in range(len(test_dataset)):
            test_inputs.append(test_dataset[test_idx][0])
            test_labels.append(test_dataset[test_idx][1])
        return [
            TorchGlobalDataset(train_inputs, train_labels, self.num_classes),
            TorchGlobalDataset(test_inputs, test_labels, self.num_classes)
        ]
