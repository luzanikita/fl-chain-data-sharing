"""src: An authenticated Flower / PyTorch app."""

import asyncio
import os
from typing import Any

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, Parameters, log
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from datasets import load_from_disk
from src.blockchain_api import BlockchainAPI
from src.mods import blockchain_mod
from src.task import INFO, Net
from src.zkp import AggregationVerifier, compute_hash, verify_proof


class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate, context):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.client_state = context.state
        self.api = BlockchainAPI("http://localhost:3001")  # Org2 Gateway

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        verification_result, root_global_model_hash = self._verify_inputs(parameters, config)
        assert verification_result

        self.net._set_weights(parameters)
        results = self.net._train_local(
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )

        weights, num_examples = self.net._get_weights(), len(self.trainloader.dataset)
        self._submit_client_data(weights, num_examples, root_global_model_hash, config)

        return weights, num_examples, results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        self.net._set_weights(parameters)
        loss, accuracy = self.net._test(self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

    def _verify_inputs(self, parameters: Parameters, config: dict[str, Any]) -> tuple[bool, str]:
        # Verify integrity of incoming parameters
        aggregated_model_hash = compute_hash(parameters, 0)
        log(INFO, f"[Custom] Incoming model hash: {aggregated_model_hash}")

        query_result = self.api.query(
            channel_id=config.get("channel_id"),
            chaincode_id=config.get("global_chaincode_id"),
            function_name="ReadGlobalModel",
            args=aggregated_model_hash,
        )
        # TODO: Verify query result: previous local model hash is in the query_result["local_model_hashes"]
        log(INFO, f"[Custom] Query result: {query_result}")

        # Verify ZKP of correct aggregation before training
        if query_result.get("previous_global_model_hash") == "":
            return True, aggregated_model_hash

        proof_path = query_result.get("zkp_hash")
        if not proof_path:
            return False, aggregated_model_hash

        log(INFO, f"[Custom] Start ZKP verification for aggregation: {proof_path}")
        # verification_result = False
        verification_result = asyncio.run(self._verify_zkp(proof_path))
        log(INFO, f"[Custom] Verified ZKP for aggregation. Proof is valid: {verification_result}")

        return verification_result, aggregated_model_hash

    # In production, we should verify the proof by other endorsment peers
    # Or we can share the proof with the clients, but this may be too heavy
    async def _verify_zkp(
        self,
        proof_path: str,
        settings_path: str = os.path.join("experiments", "aggregation", "faulty", "settings.json"),
        vk_path: str = os.path.join("experiments", "aggregation", "faulty", "test.vk"),
    ) -> bool:
        verifier = AggregationVerifier(settings_path, vk_path)
        verification_result = await verify_proof(verifier, proof_path)

        return verification_result

    def _submit_client_data(
        self,
        weights: list[np.ndarray],
        num_examples: int,
        root_global_model_hash: str,
        config: dict[str, Any],
    ):
        client_model_hash = compute_hash(weights, num_examples)

        create_result = self.api.invoke(
            channel_id=config.get("channel_id"),
            chaincode_id=config.get("local_chaincode_id"),
            function_name="CreateLocalModel",
            args=[
                client_model_hash,
                num_examples,
                root_global_model_hash,
                config.get("run_id"),
                config.get("current_round"),
            ],
        )
        log(INFO, f"[Custom] Create result: {create_result}")


def load_data_from_disk(path: str, batch_size: int):
    partition_train_test = load_from_disk(path)
    pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to get the path to the dataset the SuperNode running
    # this ClientApp has access to
    dataset_path: str = context.node_config["dataset-path"]

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size: int = context.run_config["batch-size"]
    trainloader, valloader = load_data_from_disk(dataset_path, batch_size)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(trainloader, valloader, local_epochs, learning_rate, context).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[blockchain_mod],
)
