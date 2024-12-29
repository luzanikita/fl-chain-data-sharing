"""src: An authenticated Flower / PyTorch app."""

from typing import Any, List, Tuple

from flwr.common import Context, Metrics, log, ndarrays_to_parameters
from flwr.common.typing import Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from src.blockchain_api import BlockchainAPI
from src.strategy import BlockchainStrategy
from src.task import INFO, Net
from src.zkp import compute_hash


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def query_model(context: Context) -> Parameters:
    # TODO: query from private data collection instead of empty model init
    ndarrays = Net()._get_weights()
    parameters = ndarrays_to_parameters(ndarrays)
    model_hash = compute_hash(ndarrays, 0)

    log(INFO, f"[CUSTOM] Global model is loaded {model_hash}")

    api = BlockchainAPI()
    create_result = api.invoke(
        channel_id="mychannel",
        chaincode_id="global_model_chaincode",
        function_name="CreateGlobalModel",
        args=[
            model_hash,
            "",  # previous_global_model_hash
            "",  # local_model_hashes
            "",  # zkp_hash
            "genesis_run",  # run_id
            1,  # round_id
        ],
        transient=parameters.tensors[0],  # TODO: should save all layers
        endorsing_orgs=["Org1MSP"],
    )
    log(INFO, f"[Custom] Create result: {create_result}")

    return parameters


def fit_config(server_round: int) -> dict[str, Any]:
    """Generate training configuration for each round."""

    config = {
        "run_id": "test_run",
        "current_round": server_round,
        "channel_id": "mychannel",
        "local_chaincode_id": "local_model_chaincode",
        "global_chaincode_id": "global_model_chaincode",
    }
    return config


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize model parameters
    parameters = query_model(context)

    # Define the strategy
    strategy = BlockchainStrategy(
        fraction_fit=1.0,
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
        on_fit_config_fn=fit_config,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
