package chaincode

import (
	"encoding/json"
	"fmt"

	"github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

// LocalModelSmartContract provides functions for managing local ML models
type LocalModelSmartContract struct {
	contractapi.Contract
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

// CreateLocalModel creates a new local model record
func (s *LocalModelSmartContract) CreateLocalModel(ctx contractapi.TransactionContextInterface, localModelHash string, numExamples uint64, rootGlobalModelHash string, runID string, roundID uint64) error {
	// Verify that only Org2 can submit local models
	clientMSPID, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return fmt.Errorf("failed to get client's MSPID: %v", err)
	}
	if clientMSPID != "Org2MSP" {
		return fmt.Errorf("only Org2 members can submit local models")
	}

	exists, err := s.LocalModelExists(ctx, localModelHash)
	if err != nil {
		return err
	}
	if exists {
		return fmt.Errorf("the local model %s already exists", localModelHash)
	}

	// Validate model data
	if len(localModelHash) == 0 {
		return fmt.Errorf("local model hash must not be empty")
	}
	if len(rootGlobalModelHash) == 0 {
		return fmt.Errorf("root global model hash must not be empty")
	}

	// Verify the referenced global model exists by querying the global model chaincode
	globalModelArgs := [][]byte{[]byte("ReadGlobalModel"), []byte(rootGlobalModelHash)}
	response := ctx.GetStub().InvokeChaincode("global_model_chaincode", globalModelArgs, "mychannel")
	if response.Status != 200 {
		return fmt.Errorf("failed to verify global model existence: %s", response.Message)
	}

	model := LocalModel{
		LocalModelHash:      localModelHash,
		NumExamples:         numExamples,
		RootGlobalModelHash: rootGlobalModelHash,
		RunID:               runID,
		RoundID:             roundID,
	}

	// Store the local model
	modelAsBytes, err := json.Marshal(model)
	if err != nil {
		return fmt.Errorf("failed to marshal model: %v", err)
	}

	return ctx.GetStub().PutState(model.LocalModelHash, modelAsBytes)
}

// ReadLocalModel returns the local model stored with given hash
func (s *LocalModelSmartContract) ReadLocalModel(ctx contractapi.TransactionContextInterface, localModelHash string) (*LocalModel, error) {
	modelAsBytes, err := ctx.GetStub().GetState(localModelHash)
	if err != nil {
		return nil, fmt.Errorf("failed to read local model: %v", err)
	}
	if modelAsBytes == nil {
		return nil, fmt.Errorf("local model %s does not exist", localModelHash)
	}

	var model LocalModel
	err = json.Unmarshal(modelAsBytes, &model)
	if err != nil {
		return nil, err
	}

	return &model, nil
}

// QueryLocalModelsByRound returns all local models for a specific run and round
func (s *LocalModelSmartContract) QueryLocalModelsByRound(ctx contractapi.TransactionContextInterface, runID string, roundID uint64) ([]*LocalModel, error) {
	queryString := fmt.Sprintf(`{"selector":{"objectType":"localModel","run_id":"%s","round_id":%d}}`, runID, roundID)

	resultsIterator, err := ctx.GetStub().GetQueryResult(queryString)
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	var models []*LocalModel
	for resultsIterator.HasNext() {
		queryResult, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}

		var model LocalModel
		err = json.Unmarshal(queryResult.Value, &model)
		if err != nil {
			return nil, err
		}
		models = append(models, &model)
	}

	return models, nil
}

// LocalModelExists returns true when local model with given ID exists in world state
func (s *LocalModelSmartContract) LocalModelExists(ctx contractapi.TransactionContextInterface, localModelHash string) (bool, error) {
	modelJSON, err := ctx.GetStub().GetState(localModelHash)
	if err != nil {
		return false, fmt.Errorf("failed to read from world state: %v", err)
	}

	return modelJSON != nil, nil
}
