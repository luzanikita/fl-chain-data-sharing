package web

import (
	"fmt"
	"net/http"

	"github.com/hyperledger/fabric-gateway/pkg/client"
)

// Invoke handles chaincode invoke requests.
func (setup *OrgSetup) Invoke(w http.ResponseWriter, r *http.Request) {
	fmt.Println("Received Invoke request")
	if err := r.ParseForm(); err != nil {
		fmt.Fprintf(w, "ParseForm() err: %s", err)
		return
	}
	chainCodeName := r.FormValue("chaincodeid")
	channelID := r.FormValue("channelid")
	function := r.FormValue("function")
	args := r.Form["args"]
	transient := r.Form["transient"]
	endorsingOrgs := r.Form["endorsing_orgs"]
	var transientData []byte
	if len(transient) > 0 {
		transientData = []byte(transient[0])
	}
	fmt.Printf("channel: %s, chaincode: %s, function: %s, args: %s\n", channelID, chainCodeName, function, args)
	network := setup.Gateway.GetNetwork(channelID)
	contract := network.GetContract(chainCodeName)

	// For private data transactions, specify the endorsing orgs
	var options []client.ProposalOption
	options = append(options, client.WithArguments(args...))
	options = append(options, client.WithTransient(map[string][]byte{"weights": transientData}))

	// If it's a private data transaction for Org1, specify Org1 as endorser
	if len(endorsingOrgs) > 0 {
		options = append(options, client.WithEndorsingOrganizations(endorsingOrgs[0]))
	}

	txn_proposal, err := contract.NewProposal(function, options...)
	if err != nil {
		fmt.Fprintf(w, "Error creating txn proposal: %s", err)
		return
	}
	txn_endorsed, err := txn_proposal.Endorse()
	if err != nil {
		fmt.Fprintf(w, "Error endorsing txn: %s", err)
		return
	}
	txn_committed, err := txn_endorsed.Submit()
	if err != nil {
		fmt.Fprintf(w, "Error submitting transaction: %s", err)
		return
	}
	fmt.Fprintf(w, "Transaction ID : %s Response: %s", txn_committed.TransactionID(), txn_endorsed.Result())
}
