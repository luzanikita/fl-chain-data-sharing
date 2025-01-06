import asyncio
import os
from typing import Optional, Union

import flwr as fl
import numpy as np
from flwr.common import FitRes, Parameters, Scalar, log, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from src.blockchain_api import BlockchainAPI
from src.task import INFO
from src.zkp import AggregationProver, compute_hash, generate_proof, preprocess


class BlockchainStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Perform aggregation by the server."""
        api = BlockchainAPI()

        # Collect and verify the integrity of local models with the local model chaincode queries
        previous_global_model_hash, local_model_hashes = self._verify_local_models(api, results)

        # Run weighted federated average
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # Submit new model to the ledger via the global model chaincode invokation
        assert aggregated_parameters is not None
        self._submit_aggregated_model(
            api,
            results,
            aggregated_parameters,
            "test_run",
            server_round,
            local_model_hashes,
            previous_global_model_hash,
        )

        return aggregated_parameters, aggregated_metrics

    def _verify_local_models(
        self,
        api: BlockchainAPI,
        results: list[tuple[ClientProxy, FitRes]],
    ) -> tuple[str, list[str]]:
        local_model_hashes = []
        previous_global_model_hash = None
        for _, (_, fit_res) in enumerate(results):
            client_num_examples = fit_res.num_examples
            client_model_hash = compute_hash(fit_res.parameters, client_num_examples)

            read_result = api.query(
                channel_id="mychannel",
                chaincode_id="local_model_chaincode",
                function_name="ReadLocalModel",
                args=client_model_hash,
            )
            log(INFO, str(read_result))

            if previous_global_model_hash is None:
                previous_global_model_hash = read_result.get("root_global_model_hash")
            else:
                assert previous_global_model_hash == read_result.get("root_global_model_hash")

            local_model_hashes.append(client_model_hash)

        log(INFO, f"[Custom] Local models hashes: {local_model_hashes}")

        return previous_global_model_hash, local_model_hashes

    def _submit_aggregated_model(
        self,
        api: BlockchainAPI,
        results: list[tuple[ClientProxy, FitRes]],
        aggregated_parameters: Parameters,
        run_id: str,
        server_round: int,
        local_model_hashes: list[str],
        previous_global_model_hash: str,
    ):
        aggregated_model_hash = compute_hash(aggregated_parameters, 0)
        log(INFO, f"[Custom] Aggregated model hash: {aggregated_model_hash}")

        log(INFO, "[Custom] Start computing ZKP for aggregation...")
        # Let's simulate a faulty aggregation from round to round
        if server_round % 2 == 1:
            zkp_result = asyncio.run(self._compute_zkp(results))
            # zkp_result = os.path.join("experiments", "aggregation", "faulty", "aggregation_proof.pf")
            assert zkp_result is not None
            proof_path = zkp_result
        else:
            proof_path = os.path.join("experiments", "aggregation", "faulty", "faulty-aggregation_proof.pf")

        log(INFO, f"[Custom] Computed ZKP for aggregation: {proof_path is not None}")

        # verification_result = asyncio.run(verify_zkp(proof_path))
        # log(INFO, f"[Custom] Verified ZKP for aggregation. Proof is valid: {verification_result}")

        create_result = api.invoke(
            channel_id="mychannel",
            chaincode_id="global_model_chaincode",
            function_name="CreateGlobalModel",
            args=[
                aggregated_model_hash,
                previous_global_model_hash,
                ";".join(local_model_hashes),
                proof_path,  # zkp_hash,
                run_id,
                server_round,
            ],
            transient=aggregated_parameters.tensors[0],  # TODO: should save all layers
            endorsing_orgs=["Org1MSP"],
        )
        log(INFO, f"[Custom] Create result: {create_result}")

    async def _compute_zkp(
        self,
        results: list[tuple[ClientProxy, FitRes]],
        compiled_model_path: str = os.path.join("experiments", "aggregation", "faulty", "network.compiled"),
        pk_path: str = os.path.join("experiments", "aggregation", "faulty", "test.pk"),
        save_dir: str = os.path.join("experiments", "aggregation", "faulty"),
    ) -> Optional[str]:
        clients = []
        for _, (_, fit_res) in enumerate(results):
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            client_num_examples = fit_res.num_examples
            clients.append([client_weights, client_num_examples])

        preprocessed_input = preprocess(clients)
        prover = AggregationProver(compiled_model_path, pk_path)
        proof_path = await generate_proof(prover, preprocessed_input, save_dir, model_prefix="")

        return proof_path

    def _save_client_data(
        self,
        results: list[tuple[ClientProxy, FitRes]],
        server_round: int,
        save_dir: str = "experiments/aggregation/models",
    ):
        # Save each client's weights and num_examples separately
        for client_idx, (_, fit_res) in enumerate(results):
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            client_num_examples = fit_res.num_examples

            # Save individual client data - each layer separately
            np.savez(
                os.path.join(save_dir, f"round-{server_round}-client-{client_idx}.npz"),
                *client_weights,
                num_examples=np.array([client_num_examples]),
            )

    def _save_aggregated_weights(
        self,
        aggregated_ndarrays: list[np.ndarray],
        results: list[tuple[ClientProxy, FitRes]],
        server_round: int,
        save_dir: str = "experiments/aggregation/models",
    ):
        # Save aggregated weights - each layer separately
        np.savez(
            os.path.join(save_dir, f"round-{server_round}-aggregated.npz"),
            *aggregated_ndarrays,
            num_examples=np.array([sum(fit_res.num_examples for _, fit_res in results)]),
        )
