from torchfed.centralized.trainer.fedavg import Trainer
from torchfed.datasets.CIFAR10 import CIFAR10

from torchfed.models.CIFARNet import CIFARNet

if __name__ == '__main__':
    num_users = 20
    num_labels = 3
    num_epochs = 10

    train_dataset = CIFAR10("./", num_users, num_labels, train=True, download=True)
    test_dataset = CIFAR10("./", num_users, num_labels, train=False, download=True)

    model = CIFARNet()

    trainer = Trainer(num_users, model, train_dataset, test_dataset, params={
        "lr": 0.01,
        "batch_size": 32
    })
    trainer.train(10)
