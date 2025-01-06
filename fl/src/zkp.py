import json
import os
from typing import List, Optional, Tuple, Union

import ezkl
import numpy as np
import torch
from flwr.common import Parameters, parameters_to_ndarrays
from torch import nn


class AggregateModel(nn.Module):
    def __init__(self):
        super(AggregateModel, self).__init__()

    def forward(self, weights: torch.Tensor, num_examples: torch.Tensor) -> torch.Tensor:
        """Weighted average of model params"""
        num_examples_total = torch.sum(num_examples)

        num_examples = num_examples.view(num_examples.size(0), 1)

        weighted_sum = torch.sum(weights * num_examples, dim=0)
        weights_prime = weighted_sum / num_examples_total

        return weights_prime


class FaultyAggregateModel(nn.Module):
    def __init__(self):
        super(FaultyAggregateModel, self).__init__()

    def forward(self, weights: torch.Tensor, num_examples: torch.Tensor) -> torch.Tensor:
        """Weighted average of model params"""
        num_examples_total = torch.sum(num_examples)

        num_examples = num_examples.view(num_examples.size(0), 1)

        weighted_sum = torch.sum(weights * num_examples, dim=0)
        weights_prime = weighted_sum / num_examples_total

        return weights_prime * 0.9  # tamper the weights


class AggregationSetup:
    def __init__(self, circuit: nn.Module, root_dir: str, model_prefix: str = ""):
        self.circuit = circuit
        self.circuit.eval()

        os.makedirs(root_dir, exist_ok=True)

        self.model_path = os.path.join(root_dir, f"{model_prefix}network.onnx")
        self.compiled_model_path = os.path.join(root_dir, f"{model_prefix}network.compiled")
        self.pk_path = os.path.join(root_dir, f"{model_prefix}test.pk")
        self.vk_path = os.path.join(root_dir, f"{model_prefix}test.vk")
        self.settings_path = os.path.join(root_dir, f"{model_prefix}settings.json")

    def _export_model(self, sample_input):
        """Export the ONNX model using preprocessed sample input"""
        torch.onnx.export(
            self.circuit,
            sample_input,
            self.model_path,
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )

    def _setup_run_args(self, logrows: int = 10) -> ezkl.PyRunArgs:
        """Configure run arguments"""
        run_args = ezkl.PyRunArgs()
        run_args.input_visibility = "private"
        run_args.param_visibility = "fixed"
        run_args.output_visibility = "public"
        run_args.scale_rebase_multiplier = 1000
        run_args.logrows = logrows
        return run_args

    async def setup(self, sample_input, witness_path: Optional[str] = None, logrows: int = 10) -> bool:
        """Perform complete setup including proving and verification keys"""
        try:
            # Export model and generate settings
            self._export_model(sample_input)
            run_args = self._setup_run_args(logrows)

            if not ezkl.gen_settings(self.model_path, self.settings_path, py_run_args=run_args):
                print("Failed to generate settings")
                return False

            if not ezkl.compile_circuit(self.model_path, self.compiled_model_path, self.settings_path):
                print("Failed to compile circuit")
                return False

            # Get SRS
            await ezkl.get_srs(self.settings_path)

            # Setup proving and verification keys
            success = ezkl.setup(self.compiled_model_path, self.vk_path, self.pk_path, witness_path=witness_path)

            if not success:
                print("Failed to setup keys")
                return False

            return os.path.isfile(self.vk_path) and os.path.isfile(self.pk_path) and os.path.isfile(self.settings_path)

        except Exception as e:
            print(f"Setup failed with error: {str(e)}")
            return False


class AggregationProver:
    def __init__(self, compiled_model_path: str, pk_path: str):
        self.compiled_model_path = compiled_model_path
        self.pk_path = pk_path

    async def generate_witness(
        self, preprocessed_input: Tuple[torch.Tensor, torch.Tensor], output_path: str
    ) -> Tuple[str, str]:
        """Generate witness for aggregation using preprocessed input"""
        data_path = f"{output_path}_data.json"
        witness_path = f"{output_path}_witness.json"

        weights_tensor, num_examples_tensor = preprocessed_input
        weights = weights_tensor.detach().numpy().reshape([-1]).tolist()
        num_examples = num_examples_tensor.detach().numpy().reshape([-1]).tolist()

        data = {"input_data": (weights, num_examples)}

        with open(data_path, "w") as f:
            json.dump(data, f)

        await ezkl.gen_witness(data_path, self.compiled_model_path, witness_path)
        return data_path, witness_path

    async def prove(self, witness_path: str, proof_path: str) -> bool:
        """Generate proof"""
        return ezkl.prove(witness_path, self.compiled_model_path, self.pk_path, proof_path, "single")


class AggregationVerifier:
    def __init__(self, settings_path: str, vk_path: str):
        self.settings_path = settings_path
        self.vk_path = vk_path

    def verify(self, proof_path: str) -> bool:
        """Verify a proof for specific compiled model (circuit) and verifiation key"""
        return ezkl.verify(proof_path, self.settings_path, self.vk_path)


def hash_model(weights: torch.Tensor, num_examples: torch.Tensor, precision: int = 7) -> Tuple[str, int]:
    """Compute hash value of a flattened ML model weights with Poseidon hashing function."""
    field_elements = [ezkl.float_to_felt(w, precision) for w in weights] + [ezkl.float_to_felt(num_examples, precision)]
    hash_value = ezkl.poseidon_hash(field_elements)[0]
    hash_value_int = ezkl.felt_to_int(hash_value)

    return hash_value, hash_value_int


def compute_hash(
    parameters: Union[Parameters, List[np.ndarray]],
    num_examples: int,
) -> str:
    """Compute hash value of ML model params."""

    # First, convert inputs to list of vectors
    if isinstance(parameters, Parameters):
        weights_ndarrays: list[np.ndarray] = parameters_to_ndarrays(parameters)
    else:
        weights_ndarrays = parameters

    # Flatten the weights
    preprocessed_input = preprocess([[weights_ndarrays, num_examples]])
    weights, num_examples = preprocessed_input

    # Compute Poseidon hash
    model_hash, _ = hash_model(weights[0], num_examples[0])

    return model_hash


def preprocess(results: List[List]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert nested list input format to tensors"""
    client_tensors = []
    num_examples = []

    for client_data in results:
        # Get layers and num_examples
        layers, n_examples = client_data

        # Flatten all layers into a single 1D array
        flattened = np.concatenate([layer.flatten() for layer in layers])

        client_tensors.append(flattened)
        num_examples.append(n_examples)

    # Stack all clients' flattened weights into a single tensor
    weights_tensor = torch.tensor(np.stack(client_tensors), dtype=torch.float32)
    num_examples_tensor = torch.tensor(num_examples, dtype=torch.float32)

    return weights_tensor, num_examples_tensor


async def perform_setup(
    circuit: nn.Module,
    save_dir: str,
    model_prefix: str = "",
    preprocessed_input: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    logrows: int = 10,
) -> Optional[AggregationSetup]:
    """Create setup with preprocessed sample input"""

    setup = AggregationSetup(circuit, save_dir, model_prefix)

    if preprocessed_input is not None:
        # Perform complete setup
        success = await setup.setup(preprocessed_input, logrows=logrows)
        if not success:
            print("Setup failed")
            return None

    return setup


async def generate_proof(
    prover: AggregationProver,
    preprocessed_input: Tuple[torch.Tensor, torch.Tensor],
    save_dir: str,
    model_prefix: str = "",
) -> Optional[str]:
    """Generate proof"""
    # Generate initial witness for setup
    _, witness_path = await prover.generate_witness(
        preprocessed_input, os.path.join(save_dir, f"{model_prefix}setup_witness")
    )
    # Generate proof
    proof_path = os.path.join(save_dir, f"{model_prefix}aggregation_proof.pf")
    is_proved = await prover.prove(witness_path, proof_path)

    return proof_path if is_proved else None


async def verify_proof(verifier: AggregationVerifier, proof_path: str) -> bool:
    """Verify proof"""
    try:
        verification_result = verifier.verify(proof_path)
        print(f"Aggregation proof verification result: {verification_result}")
        return verification_result
    except RuntimeError as e:
        print(f"Verification failed with error: {str(e)}")
        return False


def load_data(server_round: int, weights_dir: str = "experiments/aggregation/models"):
    """Load data from the saved ML model checkpoints."""

    # Load aggregated model
    with np.load(os.path.join(weights_dir, f"round-{server_round}-aggregated.npz")) as data:
        aggregated_weights = [data[name] for name in data.files if name != "num_examples"]
        total_num_examples = data["num_examples"][0]
        aggregated = [aggregated_weights, total_num_examples]

    # Load clients' models used for aggregation
    clients = []
    for i in range(2):
        with np.load(os.path.join(weights_dir, f"round-{server_round}-client-{i}.npz")) as data:
            input_weights = [data[name] for name in data.files if name != "num_examples"]
            num_examples = data["num_examples"][0]
            clients.append([input_weights, num_examples])

    return clients, aggregated


def get_test_data():
    # Example usage
    test_clients = [
        [[np.array([1, 2, 3, 4]), np.array([5, 6])], 10],
        [[np.array([9, 8, 7, 6]), np.array([5, 4])], 5],
    ]
    return test_clients


async def faulty_experiment():
    """Run full ZKP experiment with setup, aggregatrion prove generation and validation. Faulty model was precomputed."""
    clients, _ = load_data(server_round=1)
    preprocessed_input = preprocess(clients)

    circuit = AggregateModel()
    faulty_circuit = FaultyAggregateModel()

    save_dir = os.path.join("experiments", "aggregation", "faulty")

    setup = await perform_setup(circuit, save_dir, model_prefix="", preprocessed_input=preprocessed_input)
    faulty_setup = await perform_setup(
        faulty_circuit, save_dir, model_prefix="faulty-", preprocessed_input=preprocessed_input
    )

    assert setup is not None
    assert faulty_setup is not None

    prover = AggregationProver(setup.compiled_model_path, setup.pk_path)
    faulty_prover = AggregationProver(faulty_setup.compiled_model_path, setup.pk_path)

    proof_path = await generate_proof(prover, preprocessed_input, save_dir, model_prefix="")
    faulty_proof_path = await generate_proof(faulty_prover, preprocessed_input, save_dir, model_prefix="faulty-")

    assert proof_path is not None
    assert faulty_proof_path is not None

    verifier = AggregationVerifier(setup.settings_path, setup.vk_path)

    verification_result = await verify_proof(verifier, proof_path)
    faulty_verification_result = await verify_proof(verifier, faulty_proof_path)

    assert verification_result
    assert not faulty_verification_result


async def simple_experiment():
    """Run ZKP experiment on the toy model with small number of params."""
    clients = get_test_data()
    # clients, aggregated = load_data(1)
    preprocessed_input = preprocess(clients)

    circuit = AggregateModel()
    save_dir = os.path.join("experiments", "aggregation", "simple")

    # Increase logrows to handle larger values
    setup = await perform_setup(
        circuit,
        save_dir,
        model_prefix="",
        preprocessed_input=preprocessed_input,
        logrows=20,
    )

    assert setup is not None

    prover = AggregationProver(setup.compiled_model_path, setup.pk_path)
    proof_path = await generate_proof(prover, preprocessed_input, save_dir, model_prefix="")
    assert proof_path is not None

    verifier = AggregationVerifier(setup.settings_path, setup.vk_path)
    verification_result = await verify_proof(verifier, proof_path)
    assert verification_result


async def main():
    clients, aggregated = load_data(server_round=1)
    # test_clients = get_test_data()

    preprocessed_input = preprocess(clients)
    # preprocessed_input = preprocess(test_clients)

    aggregated_weights, hashes = AggregateModel().forward(*preprocessed_input)
    print(aggregated_weights, hashes)

    await simple_experiment()
    # await faulty_experiment()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
