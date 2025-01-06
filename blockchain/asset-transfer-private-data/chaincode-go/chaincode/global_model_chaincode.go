package chaincode

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

// GlobalModelSmartContract provides functions for managing global ML models
type GlobalModelSmartContract struct {
	contractapi.Contract
}

// GlobalModelAsset describes the global model details that are visible to all organizations
type GlobalModelAsset struct {
	Type                    string   `json:"objectType"` // Used to distinguish different types of objects in state database
	GlobalModelHash         string   `json:"global_model_hash"`
	PreviousGlobalModelHash string   `json:"previous_global_model_hash"`
	LocalModelHashes        []string `json:"local_model_hashes"` // List of local model hashes used in aggregation
	ZkpHash                 string   `json:"zkp_hash"`           // Hash of the ZKP (in our simplified case can be just a path to the ZKP file)
	RunID                   string   `json:"run_id"`
	RoundID                 uint64   `json:"round_id"`
}

// GlobalModelPrivateDetails describes the private details of the global model
type GlobalModelPrivateDetails struct {
	GlobalModelHash   string `json:"global_model_hash"`
	AggregatedWeights []byte `json:"aggregated_weights"` // Serialized model weights
}

// LocalModel describes a local model contribution
type LocalModel struct {
	Type                string `json:"objectType"`
	LocalModelHash      string `json:"local_model_hash"`
	NumExamples         uint64 `json:"num_examples"`
	RootGlobalModelHash string `json:"root_global_model_hash"`
	RunID               string `json:"run_id"`
	RoundID             uint64 `json:"round_id"`
}

// ReadGlobalModel returns the global model stored with given hash
func (s *GlobalModelSmartContract) ReadGlobalModel(ctx contractapi.TransactionContextInterface, globalModelHash string) (*GlobalModelAsset, error) {
	modelAsBytes, err := ctx.GetStub().GetState(globalModelHash)
	if err != nil {
		return nil, fmt.Errorf("failed to read local model: %v", err)
	}
	if modelAsBytes == nil {
		return nil, fmt.Errorf("global model %s does not exist", globalModelHash)
	}

	var model GlobalModelAsset
	err = json.Unmarshal(modelAsBytes, &model)
	if err != nil {
		return nil, err
	}

	return &model, nil
}

// CreateGlobalModel creates a new global model by placing the main asset details in the public collection
// and the model weights in the organization's private collection
func (s *GlobalModelSmartContract) CreateGlobalModel(
	ctx contractapi.TransactionContextInterface,
	globalModelHash string,
	previousGlobalModelHash string,
	localModelHashes string,
	zkpHash string,
	runID string,
	roundID uint64,
) error {
	// Get the client org id
	clientMSPID, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return fmt.Errorf("failed getting client's orgID: %v", err)
	}

	// Get only the weights from transient map
	transientMap, err := ctx.GetStub().GetTransient()
	if err != nil {
		return fmt.Errorf("error getting transient: %v", err)
	}

	weights, ok := transientMap["weights"]
	if !ok {
		return fmt.Errorf("weights not found in transient map")
	}

	// Check if global model already exists
	exists, err := s.GlobalModelExists(ctx, globalModelHash)
	if err != nil {
		return err
	}
	if exists {
		return fmt.Errorf("the global model %s already exists", globalModelHash)
	}

	// Validate input
	if len(globalModelHash) == 0 {
		return fmt.Errorf("global model hash field must be a non-empty string")
	}
	if len(runID) == 0 {
		return fmt.Errorf("run ID field must be a non-empty string")
	}
	if roundID == 0 {
		return fmt.Errorf("round ID must be greater than 0")
	}
	if len(weights) == 0 {
		return fmt.Errorf("aggregated weights must not be empty")
	}

	var localModelHashesSlice = make([]string, 0)
	if localModelHashes != "" {
		localModelHashesSlice = strings.Split(localModelHashes, ";")
	}

	// Verify all referenced local models exist and are valid
	err = s.VerifyLocalModels(ctx, localModelHashesSlice, previousGlobalModelHash, runID, roundID)
	if err != nil {
		return fmt.Errorf("local model verification failed: %v", err)
	}

	// Create the global model asset
	asset := GlobalModelAsset{
		Type:                    "globalModel",
		GlobalModelHash:         globalModelHash,
		PreviousGlobalModelHash: previousGlobalModelHash,
		LocalModelHashes:        localModelHashesSlice,
		ZkpHash:                 zkpHash,
		RunID:                   runID,
		RoundID:                 roundID,
	}

	assetJSON, err := json.Marshal(asset)
	if err != nil {
		return fmt.Errorf("failed to marshal asset into JSON: %v", err)
	}

	// Save public asset data
	err = ctx.GetStub().PutState(globalModelHash, assetJSON)
	if err != nil {
		return fmt.Errorf("failed to put asset in public state: %v", err)
	}

	// Save the private data (weights)
	privateDetails := GlobalModelPrivateDetails{
		GlobalModelHash:   globalModelHash,
		AggregatedWeights: weights,
	}

	privateDetailsJSON, err := json.Marshal(privateDetails)
	if err != nil {
		return fmt.Errorf("failed to marshal private details: %v", err)
	}

	// Get collection name for this organization
	orgCollection := fmt.Sprintf("%sPrivateCollection", clientMSPID)

	// Put the weights in the organization's private collection
	err = ctx.GetStub().PutPrivateData(orgCollection, globalModelHash, privateDetailsJSON)
	if err != nil {
		return fmt.Errorf("failed to put private details: %v", err)
	}

	return nil
}

// verifyLocalModels verifies local models
func (s *GlobalModelSmartContract) VerifyLocalModels(ctx contractapi.TransactionContextInterface,
	localModelHashes []string,
	previousGlobalModelHash string,
	runID string,
	roundID uint64) error {

	for _, hash := range localModelHashes {
		// Query local model chaincode to verify the model exists
		queryArgs := [][]byte{[]byte("ReadLocalModel"), []byte(hash)}
		response := ctx.GetStub().InvokeChaincode("local_model_chaincode", queryArgs, "mychannel")

		if response.Status != 200 {
			return fmt.Errorf("local model %s verification failed: %s", hash, response.Message)
		}

		var localModel LocalModel
		err := json.Unmarshal(response.Payload, &localModel)
		if err != nil {
			return fmt.Errorf("failed to unmarshal local model %s: %v", hash, err)
		}

		// Verify the local model belongs to the correct run and round
		if localModel.RunID != runID {
			return fmt.Errorf("local model %s belongs to different run", hash)
		}
		if localModel.RoundID != roundID {
			return fmt.Errorf("local model %s belongs to different round", hash)
		}

		// Verify the local model was derived from the previous global model
		if localModel.RootGlobalModelHash != previousGlobalModelHash {
			return fmt.Errorf("local model %s was not derived from the previous global model", hash)
		}
	}
	return nil
}

// GlobalModelExists returns true when global model with given ID exists in world state
func (s *GlobalModelSmartContract) GlobalModelExists(ctx contractapi.TransactionContextInterface, globalModelHash string) (bool, error) {
	modelJSON, err := ctx.GetStub().GetState(globalModelHash)
	if err != nil {
		return false, fmt.Errorf("failed to read from world state: %v", err)
	}

	return modelJSON != nil, nil
}
