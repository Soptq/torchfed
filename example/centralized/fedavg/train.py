from torchvision.transforms import transforms

from torchfed.centralized.trainer import FedAvgTrainer
from torchfed.datasets.CIFAR10 import CIFAR10

from torchfed.models.CIFARNet import CIFARNet

if __name__ == '__main__':
    num_users = 20
    num_labels = 3
    num_epochs = 10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(
        "../../data",
        num_users,
        num_labels,
        download=True,
        transform=transform)

    model = CIFARNet()

    trainer = FedAvgTrainer(params={
        "world_size": num_users,
        "sample_size": 5,
        "model": model,
        "dataset": dataset,
        "lr": 1e-3,
        "batch_size": 32,
        "local_iterations": 10,
    })
    trainer.train(10)
