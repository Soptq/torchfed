from torchfed.datasets import BaseTorchDataset
from torchvision.transforms import transforms
from torchfed.datasets.CIFAR10 import TorchCIFAR10


class DatasetManager:
    def __init__(self, name, dataset: BaseTorchDataset):
        self.name = name
        self.dataset = dataset

    def get_global_dataset(self):
        return self.dataset.get_global_dataset()

    def get_user_dataset(self, idx):
        return self.dataset.get_user_dataset(idx)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset_manager = DatasetManager("cifar10_manager",
                                     TorchCIFAR10(
                                         "../../example/data",
                                         20,
                                         10,
                                         download=True,
                                         transform=transform)
                                     )

    train, test = dataset_manager.get_user_dataset(0)
    import torch
    train_loader = torch.utils.data.DataLoader(
        train, 32, shuffle=True)
    print(train_loader.dataset.num_classes)
    # for data in train_loader:
    #     print(data["labels"].tolist())
