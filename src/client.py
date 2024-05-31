from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch.nn as nn
from flwr.common import Scalar
from torch.utils.data import DataLoader

from src.centralized_train import test, train
from src.model import Net, get_parameters, set_parameters
from src.settings import DEVICE


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: int, net: nn.Module, trainloader: DataLoader, valloader: DataLoader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        return get_parameters(self.net)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        local_epochs = config.get("local_epochs", 1)
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=local_epochs)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(
    cid: str,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(int(cid), net, trainloader, valloader).to_client()


# Flower ClientApp
app = fl.client.ClientApp(
    client_fn=client_fn,
)
