# Secure Data Sharing in the Internet of Vehicles Using Blockchain-based Federated Learning

This project implements the proof-of-concept of blockchain-based federated learning system with zero-knowledge proofs for model aggregation verification.

We used:
- Hyperledger Fabric for private blockchain network
- EZKL for zero-knowledge proofs
- Flower framework for federated learning

Full Thesis: [Link](https://thesis.cs.ut.ee/ed21d512-9750-4463-aaa0-87018b4fd0e3) (NB! Implementation details on page 70)

DEMO Video: [Link](https://www.youtube.com/watch?v=RxbJgq6MiQE)

# Project Structure

## Blockchain Component ([/blockchain](/blockchain))

Class diagram can be found in [`figs/blockchain_uml.txt`](/figs/blockchain_uml.txt)

![Blockchain Class Diagram](/figs/blockchain_uml.svg)


The blockchain component is built using Hyperledger Fabric and consists of:

### Smart Contracts

- Local Model Chaincode ([`asset-transfer-basic/chaincode-go/chaincode/local_model_chaincode.go`](/blockchain/asset-transfer-basic/chaincode-go/chaincode/local_model_chaincode.go))
  - Handles metadata for local ML models (no weights)
  - Tracks model hashes, number of examples, round IDs etc.
  - Uses `LocalModelSmartContract` to manage `LocalModel` assets
- Global Model Chaincode ([`asset-transfer-private-data/chaincode-go/chaincode/global_model_chaincode.go`](/blockchain/asset-transfer-private-data/chaincode-go/chaincode/global_model_chaincode.go))
  - Manages aggregated global models
  - Stores model weights in Private Data Collections
  - Uses `GlobalModelSmartContract` to manage `GlobalModelAsset` and `GlobalModelPrivateDetails`


### REST API Gateway

Golang gateway server ([`/rest-api-go`](/blockchain/asset-transfer-basic/rest-api-go))
  - Provides HTTP endpoints for organizations to interact with the blockchain
  - Implements `OrgSetup` for managing organization credentials and connections
  - Exposes methods to invoke and query chaincode

## Federated Learning Component ([/fl](/fl))

Class diagram can be found in [`figs/fl_uml.txt`](/figs/fl_uml.txt)

![Federated Learning Class Diagram](/figs/fl_uml.svg)

The FL system is built using Flower framework and consists of:

### Core Components

- Client ([`src/client_app.py`](/fl/src/client_app.py))
  - FlowerClient class for training local models
  - Integrates with blockchain API to verify and submit local models
  - Uses PyTorch for model training
- Server ([`src/server_app.py`](/fl/src/server_app.py))
  - Implements federated averaging strategy
  - Verifies local models via blockchain before aggregation
  - Submits aggregated models to blockchain
- Strategy ([`src/strategy.py`](/fl/src/strategy.py))
  - `BlockchainStrategy` extends Flower's `FedAvg`
  - Handles model verification and blockchain integration
  - Generates zero-knowledge proofs for aggregation
- Task ([`src/task.py`](/fl/src/task.py))
  - Defines the ML model architecture
- ZKP ([`src/zkp.py`](/fl/src/zkp.py))
  - Zero-knowledge proof setup, generation and verification logic
- Blockchain API ([`src/blockchain_api.py`](/fl/src/blockchain_api.py))
  - Interface to interact with the blockchain network


# Setup and Usage
## Blockchain Network Setup

0. Installation:

Follow the instructions in https://hyperledger-fabric.readthedocs.io/en/release-2.5/getting_started.html

1. Start the Fabric test network:
```bash
cd blockchain/test-network
./network.sh up createChannel -ca
```

2. Deploy local model chaincode:
```bash
./network.sh deployCC -ccn local_model_chaincode -ccp ../asset-transfer-basic/chaincode-go -ccl go -ccep "OR('Org2MSP.peer')" -ccv 1.0
```

3. Deploy global model chaincode with private data collections:
```bash
./network.sh deployCC -ccn global_model_chaincode -ccp ../asset-transfer-private-data/chaincode-go/ -ccl go -ccep "OR('Org1MSP.member')" -cccg ../asset-transfer-private-data/chaincode-go/collections_config.json -ccv 2.0
```

4. Start Org1 (aggregation server) gateway REST API:
```bash
cd blockchain/asset-transfer-basic/rest-api-go
go run main.go
```

5. Start Org2 (client) gateway REST API:
```bash
go run main.go -orgName Org2 -mspID Org2MSP -cryptoPath ../../test-network/organizations/peerOrganizations/org2.example.com -peerEndpoint localhost:9051 -gatewayPeer peer0.org2.example.com -port 3001
```

## Federated Learning Setup

0. Follow the detailed instructions in [`fl/README.md`](/fl/README.md) to:
- Set up the Python environment
- Generate certificates for secure communication
- Prepare the dataset


1. Start the Flower server with superlink authentication:
```bash
flower-superlink \
    --ssl-ca-certfile certificates/ca.crt \
    --ssl-certfile certificates/server.pem \
    --ssl-keyfile certificates/server.key \
    --auth-list-public-keys keys/client_public_keys.csv \
    --auth-superlink-private-key keys/server_credentials \
    --auth-superlink-public-key keys/server_credentials.pub
```

2. Start the Flower supernode:
```bash 
flower-supernode \
    --root-certificates certificates/ca.crt \
    --auth-supernode-private-key keys/client_credentials_1 \
    --auth-supernode-public-key keys/client_credentials_1.pub \
    --node-config 'dataset-path="datasets/cifar10_part_1"' \
    --clientappio-api-address="0.0.0.0:9094"
```

3. Start the second supernode:
```bash 
flower-supernode \
    --root-certificates certificates/ca.crt \
    --auth-supernode-private-key keys/client_credentials_2 \
    --auth-supernode-public-key keys/client_credentials_2.pub \
    --node-config 'dataset-path="datasets/cifar10_part_2"' \
    --clientappio-api-address="0.0.0.0:9095"
```

4. Initiate the FL run:
```bash
flwr run . my-federation
```

## System Sequence Diagrams

### Federated Learning Data Sharing Flow
<img src="figs/sequence_fl.png" alt="Sequence diagram for FL data sharing" width="100%" style="max-height: 800px; object-fit: contain;">

### Blockchain Operations

#### Reading Models from the Ledger
<img src="figs/sequence_read_model.png" alt="Sequence diagram for chaincode querying" width="100%" style="max-height: 300px; object-fit: contain;">

#### Creating Models on the Ledger
<img src="figs/sequence_create_model.png" alt="Sequence diagram for chaincode invocation" width="100%" style="max-height: 500px; object-fit: contain;">

## License and Notice

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

As required by the Apache License, a [NOTICE](NOTICE) file is included with the distribution.
