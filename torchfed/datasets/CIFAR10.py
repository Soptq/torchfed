import sys
import os.path
import random
from typing import Optional, Callable

import numpy as np

import torch
import torchvision
from torchvision.transforms import transforms
from tqdm import trange, tqdm

from torchfed.datasets import Dataset
from torchfed.utils.datasets import BundleSplitDataset, UserDataset
from torchfed.utils.hash import hex_hash


class CIFAR10(Dataset):
    split_base_folder = "cifar-10-batches-py-split"
    split_dataset_name = "cifar-10-data"

    num_classes = 10

    def __init__(
            self,
            root: str,
            num_users: int,
            num_labels_for_users: int,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            seed: int = 0,
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.identifier = hex_hash(
            f"{root}-{num_users}-{num_labels_for_users}-{seed}")

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

        self.train_dataset = torchvision.datasets.CIFAR10(
            root, True, transform, target_transform, download)
        self.test_dataset = torchvision.datasets.CIFAR10(
            root, False, transform, target_transform, download)

        dataset_path = os.path.join(root, self.split_base_folder)
        data_file_name = f"{self.split_dataset_name}.{self.identifier}.pkl"

        if os.path.exists(dataset_path):
            pass
            try:
                with open(os.path.join(dataset_path, data_file_name), 'rb') as f:
                    self.split_dataset = torch.load(f)
                return
            except Exception:
                pass
        else:
            os.makedirs(dataset_path)

        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=len(
                self.train_dataset.data), shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=len(
                self.test_dataset.data), shuffle=False)

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

        user_x = [[] for _ in range(num_users)]
        user_y = [[] for _ in range(num_users)]
        idx = np.zeros(self.num_classes, dtype=np.int64)
        for user_idx in trange(num_users):
            for label_idx in range(num_labels_for_users):
                assigned_label = (user_idx + label_idx) % self.num_classes
                user_x[user_idx] += split_inputs[assigned_label][idx[assigned_label]: idx[assigned_label] + 10].tolist()
                user_y[user_idx] += (assigned_label * np.ones(10)).tolist()
                idx[assigned_label] += 10

        props = np.random.lognormal(
            0, 2., (10, num_users, num_labels_for_users)
        )
        props = np.array([[[len(v) - num_users]] for v in split_inputs]) * \
            props / np.sum(props, (1, 2), keepdims=True)
        for user_idx in trange(num_users):
            for label_idx in range(num_labels_for_users):
                assigned_label = (user_idx + label_idx) % self.num_classes
                num_samples = int(
                    props[assigned_label, user_idx // int(num_users / 10), label_idx])
                num_samples += random.randint(300, 600)
                if num_users <= 20:
                    num_samples *= 2
                if idx[assigned_label] + \
                        num_samples < len(split_inputs[assigned_label]):
                    user_x[user_idx] += split_inputs[assigned_label][idx[assigned_label]: idx[assigned_label] + num_samples].tolist()
                    user_y[user_idx] += (assigned_label *
                                         np.ones(num_samples)).tolist()
                    idx[assigned_label] += num_samples
            user_x[user_idx] = torch.Tensor(user_x[user_idx]) \
                .view(-1, num_channels, num_width, num_height) \
                .type(torch.float32)
            user_y[user_idx] = torch.Tensor(user_y[user_idx]) \
                .type(torch.int64)

        split_dataset = BundleSplitDataset()
        for user_idx in trange(num_users):
            combined = list(zip(user_x[user_idx], user_y[user_idx]))
            random.shuffle(combined)
            user_x[user_idx], user_y[user_idx] = zip(*combined)

            num_samples = len(user_x[user_idx])
            train_len = int(num_samples * 0.75)
            train_user_data = UserDataset(
                user_idx, user_x[user_idx][:train_len], user_y[user_idx][:train_len])
            test_user_data = UserDataset(user_idx,
                                         user_x[user_idx][train_len:],
                                         user_y[user_idx][train_len:])
            split_dataset.add_user_dataset(train_user_data, test_user_data)
        with open(os.path.join(dataset_path, data_file_name), 'wb') as f:
            torch.save(split_dataset, f)

        self.split_dataset = split_dataset

    @property
    def name(self) -> str:
        return "CIFAR10"

    def get_user_dataset(self, user_idx) -> UserDataset:
        return self.split_dataset[user_idx]

    def get_bundle_dataset(self):
        return self.train_dataset, self.test_dataset
