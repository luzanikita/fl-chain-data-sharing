import json
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
from flwr.common.parameter import ndarray_to_bytes


class BlockchainAPI:
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url

    def invoke(
        self,
        channel_id: str,
        chaincode_id: str,
        function_name: str,
        args: Union[List[str], str],
        transient: Optional[list[bytes]] = None,
        endorsing_orgs: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/invoke"

        payload = {
            "channelid": channel_id,
            "chaincodeid": chaincode_id,
            "function": function_name,
            "args": args,
            "transient": transient if transient is not None else [],
            "endorsing_orgs": endorsing_orgs if endorsing_orgs is not None else [],
        }

        try:
            response = requests.post(
                endpoint, data=payload, headers={"content-type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()

            if not response.text:
                return {"status": "success", "message": "Function invoked successfully"}

            return {"status": "success", "message": response.text}
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}

    def query(
        self,
        channel_id: str,
        chaincode_id: str,
        function_name: str,
        args: Union[List[str], str] = "",
    ) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/query"

        # Handle both string and list arguments
        if isinstance(args, list):
            formatted_args = ";".join(args)
        else:
            formatted_args = args

        params = {
            "channelid": channel_id,
            "chaincodeid": chaincode_id,
            "function": function_name,
            "args": formatted_args,
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()

            if not response.text:
                return {"error": "No data found"}

            response_text = response.text
            if response_text.startswith("Response: "):
                response_text = response_text[len("Response: ") :]

            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                if "error" in response_text.lower():
                    return {"error": response_text}
                return {"error": f"Invalid response format: {response_text}"}

        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}


if __name__ == "__main__":
    org1_api = BlockchainAPI("http://localhost:3000")
    org2_api = BlockchainAPI("http://localhost:3001")

    # Initial global model
    create_result = org1_api.invoke(
        channel_id="mychannel",
        chaincode_id="global_model_chaincode",
        function_name="CreateGlobalModel",
        args=[
            "genesis_global_model",
            "",  # previous_global_model_hash
            "",  # local_model_hashes
            "",  # zkp_hash
            "genesis",  # run_id
            1,  # round_id
        ],
        transient=ndarray_to_bytes(np.array([1, 2, 3, 4, 5])),
    )
    print("Genesis global model creation:", create_result)

    time.sleep(1)

    create_result = org2_api.invoke(
        channel_id="mychannel",
        chaincode_id="local_model_chaincode",
        function_name="CreateLocalModel",
        args=[
            "local_model_1",
            100,  # num_examples
            "genesis_global_model",  # root_global_model_hash
            "fl_run_test",  # run_id
            1,  # round_id
        ],
    )
    print("Local model 1 creation:", create_result)

    # Local model with fake run_id
    create_result = org2_api.invoke(
        channel_id="mychannel",
        chaincode_id="local_model_chaincode",
        function_name="CreateLocalModel",
        args=[
            "local_model_2",
            100,  # num_examples
            "genesis_global_model",  # root_global_model_hash
            "fl_run_fake",  # run_id
            1,  # round_id
        ],
    )
    print("Local model 2 (fake run) creation:", create_result)

    create_result = org2_api.invoke(
        channel_id="mychannel",
        chaincode_id="local_model_chaincode",
        function_name="CreateLocalModel",
        args=[
            "local_model_3",
            100,  # num_examples
            "genesis_global_model",  # root_global_model_hash
            "fl_run_test",  # run_id
            1,  # round_id
        ],
    )
    print("Local model 3 creation:", create_result)

    time.sleep(1)

    # Create new aggregated global model
    create_result = org1_api.invoke(
        channel_id="mychannel",
        chaincode_id="global_model_chaincode",
        function_name="CreateGlobalModel",
        args=[
            "new_global_model",
            "genesis_global_model",  # previous_global_model_hash
            "local_model_1;local_model_2;local_model_3",  # local_model_hashes
            "some_zkp",  # zkp_hash
            "fl_run_test",  # run_id
            1,  # round_id
        ],
        transient=ndarray_to_bytes(np.array([1, 2, 3, 4, 5])),
        endorsing_orgs=["Org1MSP"],
    )
    print("New global model creation with fake local models:", create_result)

    # Create new aggregated global model
    create_result = org1_api.invoke(
        channel_id="mychannel",
        chaincode_id="global_model_chaincode",
        function_name="CreateGlobalModel",
        args=[
            "new_global_model",
            "genesis_global_model",  # previous_global_model_hash
            "local_model_1;local_model_3",  # local_model_hashes
            "some_zkp",  # zkp_hash
            "fl_run_test",  # run_id
            1,  # round_id
        ],
        transient=ndarray_to_bytes(np.array([1, 2, 3, 4, 5])),
        endorsing_orgs=["Org1MSP"],
    )
    print("New global model creation:", create_result)

    time.sleep(1)

    read_result = org1_api.query(
        channel_id="mychannel",
        chaincode_id="local_model_chaincode",
        function_name="ReadLocalModel",
        args="7f3e3d290db412c391f23a9a38472bb2fd0076fdd962600248a36f8dd303f218",
    )
    print("Query local model:", read_result)

    # Create new aggregated global model as Org2 member
    create_result = org2_api.invoke(
        channel_id="mychannel",
        chaincode_id="global_model_chaincode",
        function_name="CreateGlobalModel",
        args=[
            "another_global_model",
            "genesis_global_model",  # previous_global_model_hash
            "local_model_1;local_model_3",  # local_model_hashes
            "some_zkp",  # zkp_hash
            "fl_run_test",  # run_id
            1,  # round_id
        ],
        transient=ndarray_to_bytes(np.array([1, 2, 3, 4, 5])),
        endorsing_orgs=["Org1MSP"],
    )
    print("New global model creation by Org2:", create_result)

    time.sleep(1)

    read_result = org2_api.query(
        channel_id="mychannel",
        chaincode_id="global_model_chaincode",
        function_name="ReadGlobalModel",
        args="another_global_model",
    )
    print("Query global model:", read_result)
