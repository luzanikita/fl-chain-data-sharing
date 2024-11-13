# How to start

## Setup Environment

Create virutal env
```sh
python -m venv venv
source venv/bin/activate
```

Install requirements
```sh
make install
```

Prepare security certificates and generate keys for 2 clients
```sh
./generate.sh 2
```

## Run FL with Authentication

Run SuperLink (manager)
```sh
flower-superlink \
    --ssl-ca-certfile certificates/ca.crt \
    --ssl-certfile certificates/server.pem \
    --ssl-keyfile certificates/server.key \
    --auth-list-public-keys keys/client_public_keys.csv \
    --auth-superlink-private-key keys/server_credentials \
    --auth-superlink-public-key keys/server_credentials.pub
```

Run SuperNode (client #1)
```sh
flower-client-app client:app \
    --server 127.0.0.1:9092 \
    --auth-supernode-private-key keys/client_credentials_1 \
    --auth-supernode-public-key keys/client_credentials_1.pub \
    --root-certificates certificates/ca.crt \
    --dir ./tmp
```

Run another SuperNode (client #2)
```sh
flower-client-app client:app \
    --server 127.0.0.1:9092 \
    --auth-supernode-private-key keys/client_credentials_2 \
    --auth-supernode-public-key keys/client_credentials_2.pub \
    --root-certificates certificates/ca.crt \
    --dir ./tmp
```

Run Flower App (aggregation server)
```sh
flower-server-app server:app \
    --server 127.0.0.1:9091 \
    --root-certificates certificates/ca.crt \
    --dir ./tmp
```
