from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import Metrics, Scalar
from torch.utils.data import DataLoader

from src.centralized_train import test, train
from src.dataset import load_datasets
from src.model import Net
from src.settings import DEVICE, NUM_CLIENTS


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


def set_parameters(net: nn.Module, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def server_evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
    testloader: DataLoader,
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = Net().to(DEVICE)
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, testloader)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


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


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1 if server_round < 2 else 2,  #
    }
    return config


if __name__ == "__main__":
    print(f"Training on {DEVICE} using PyTorch {torch.__version__}")

    # Specify the resources each of your clients need. By default, each
    # client will be allocated 1x CPU and 0x GPUs
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}
    if DEVICE.type == "cuda":
        # here we are assigning an entire GPU for each client.
        client_resources = {"num_cpus": 1, "num_gpus": 1.0}
        # Refer to our documentation for more details about Flower Simulations
        # and how to setup these `client_resources`.

    # by default, initializes the global model by asking one random client for the initial parameters
    # override the initial params
    init_params = get_parameters(Net())

    trainloaders, valloaders, testloader = load_datasets()

    parametrized_server_evaluate = partial(server_evaluate, testloader=testloader)
    parametrized_client_fn = partial(client_fn, trainloaders=trainloaders, valloaders=valloaders)

    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,  # distributed evaluation
        evaluate_fn=parametrized_server_evaluate,  # centralized evaluation
        initial_parameters=fl.common.ndarrays_to_parameters(init_params),
        on_fit_config_fn=fit_config,  # pass config values from server to clients
    )
    # Start simulation
    fl.simulation.start_simulation(
        client_fn=parametrized_client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_resources=client_resources,
    )
