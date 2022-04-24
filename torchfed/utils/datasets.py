from torch.utils.data import Dataset


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
        self.dataset = []

    def add_user_dataset(self, dataset: UserDataset):
        self.dataset.append(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
