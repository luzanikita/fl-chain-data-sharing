import random
from functools import partial
from typing import List, Optional

import torch
from fl.tmp.task import (
    DEVICE,
    Net,
    get_weights,
    load_data,
    set_weights,
    test,
    train,
)
from flwr.client import ClientApp, NumPyClient
from torch.utils.data import DataLoader, Dataset

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()


class AugmentedDataset(Dataset):
    def __init__(self, original_dataset, augment_fn):
        self.original_dataset = original_dataset
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        augmented_label = self.augment_fn(label)
        return image, augmented_label


def flip_label(label: int, class_num: int, victim_label: Optional[int] = None, target_label: Optional[int] = None):
    if victim_label is not None and victim_label != label:
        return label

    if target_label is not None:
        return target_label

    return random.randint(0, class_num - 1)


def get_label_flipping_loader(
    dataloader: DataLoader, class_num: int, victim_label: Optional[int] = None, target_label: Optional[int] = None
):
    augment_fn = partial(
        flip_label,
        class_num=class_num,
        victim_label=victim_label,
        target_label=target_label,
    )
    augmented_dataset = AugmentedDataset(dataloader.dataset, augment_fn)
    augmented_dataloader = DataLoader(augmented_dataset, batch_size=dataloader.batch_size)

    return augmented_dataloader


# Define FlowerClient and client_fn
class LabelFlippingFlowerClient(NumPyClient):
    def __init__(
        self, class_num: int, victim_label: Optional[int] = None, target_label: Optional[int] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_num = class_num
        self.victim_label = victim_label
        self.target_label = target_label

    def fit(self, parameters, config):
        set_weights(net, parameters)
        # augmented_trainloader = get_label_flipping_loader(
        #     trainloader, self.class_num, self.victim_label, self.target_label
        # )
        augmented_trainloader = trainloader
        results = train(net, augmented_trainloader, testloader, epochs=1, device=DEVICE)
        return get_weights(net), len(trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(net, parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return LabelFlippingFlowerClient(class_num=10, victim_label=None).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=LabelFlippingFlowerClient().to_client(),
    )
