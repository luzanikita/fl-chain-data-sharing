package main

import (
	"flag"
	"fmt"
	"rest-api-go/web"
)

type Config struct {
	orgName      string
	mspID        string
	cryptoPath   string
	peerEndpoint string
	gatewayPeer  string
	port         string
}

func main() {
	// Define command line flags
	config := Config{}
	flag.StringVar(&config.orgName, "orgName", "Org1", "Organization name")
	flag.StringVar(&config.mspID, "mspID", "Org1MSP", "MSP ID")
	flag.StringVar(&config.cryptoPath, "cryptoPath", "../../test-network/organizations/peerOrganizations/org1.example.com", "Crypto path")
	flag.StringVar(&config.peerEndpoint, "peerEndpoint", "localhost:7051", "Peer endpoint")
	flag.StringVar(&config.gatewayPeer, "gatewayPeer", "peer0.org1.example.com", "Gateway peer")
	flag.StringVar(&config.port, "port", "3000", "Port")
	// Parse command line flags
	flag.Parse()

	// Create OrgSetup using the provided configuration
	orgConfig := web.OrgSetup{
		OrgName:      config.orgName,
		MSPID:        config.mspID,
		CertPath:     config.cryptoPath + "/users/User1@" + config.orgName + ".example.com/msp/signcerts/cert.pem", // User1@" + config.orgName + ".example.com-cert.pem",
		KeyPath:      config.cryptoPath + "/users/User1@" + config.orgName + ".example.com/msp/keystore/",
		TLSCertPath:  config.cryptoPath + "/peers/peer0." + config.orgName + ".example.com/tls/ca.crt",
		PeerEndpoint: "dns:///" + config.peerEndpoint,
		GatewayPeer:  config.gatewayPeer,
	}

	orgSetup, err := web.Initialize(orgConfig)
	if err != nil {
		fmt.Printf("Error initializing setup for %s: %v\n", config.orgName, err)
		return
	}
	web.Serve(web.OrgSetup(*orgSetup), config.port)
}
