from functools import partial
from typing import Dict, Optional, Tuple

import flwr as fl
from torch.utils.data import DataLoader

from src.centralized_train import test
from src.client import client_fn
from src.dataset import load_datasets
from src.model import Net, get_parameters, set_parameters
from src.settings import DEVICE, NUM_CLIENTS


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


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


def setup_server():
    # by default, initializes the global model by asking one random client for the initial parameters
    # override the initial params
    init_params = get_parameters(Net())

    trainloaders, valloaders, testloader = load_datasets()

    parametrized_server_evaluate = partial(server_evaluate, testloader=testloader)

    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,  # distributed evaluation
        evaluate_fn=parametrized_server_evaluate,  # centralized evaluation
        initial_parameters=fl.common.ndarrays_to_parameters(init_params),
        on_fit_config_fn=fit_config,  # pass config values from server to clients
    )

    app = fl.server.ServerApp(
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
    return app, strategy, trainloaders, valloaders


# app will be used from CLI
app, strategy, trainloaders, valloaders = setup_server()


if __name__ == "__main__":
    # Start simulation
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}
    if DEVICE.type == "cuda":
        client_resources = {"num_cpus": 1, "num_gpus": 1.0}

    parametrized_client_fn = partial(client_fn, trainloaders=trainloaders, valloaders=valloaders)

    fl.simulation.start_simulation(
        client_fn=parametrized_client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_resources=client_resources,
    )
