from torch.utils.data import Dataset
from .named import Named


class GlobalDataset(Dataset):
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


class UserDataset(Dataset):
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


class BundleSplitDataset(object):
    def __init__(self):
        self.train_dataset = []
        self.test_dataset = []

    def add_user_dataset(
            self,
            train_dataset: UserDataset,
            test_dataset: UserDataset):
        self.train_dataset.append(train_dataset)
        self.test_dataset.append(test_dataset)

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, index):
        return self.train_dataset[index], self.test_dataset[index]
