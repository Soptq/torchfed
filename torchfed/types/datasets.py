from torch.utils.data import Dataset
from .named import Named


class UserDataset(Dataset):
    def __init__(self, user_id, inputs, labels):
        self.user_id = user_id
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


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
