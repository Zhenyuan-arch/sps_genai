import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loader(
    data_dir="./data",
    batch_size=32,
    train=True,
    dataset_name="CIFAR10"
):

    if dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )

    elif dataset_name == "FashionMNIST":
        transform = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor()
        ])

        dataset = datasets.FashionMNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )

    elif dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = datasets.MNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )


    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )

    return loader