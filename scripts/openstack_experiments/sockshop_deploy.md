
# Sock Shop Deployment on Provider-Representative VMs

Sock Shop was deployed on both `CloudA-openstack` and `CloudB-aws` to represent equivalent pre-deployed application environments for the FCDM decision-engine validation. Process of deploying sockshop app and paste the commands in vm terminal.

## Install dependencies

```bash
sudo apt update
sudo apt install -y docker.io docker-compose git
sudo systemctl enable --now docker

```

## Clone dependencies

```bash
git clone https://github.com/microservices-demo/microservices-demo.git

```
## Start the application
```bash

cd microservices-demo/deploy/docker-compose
sudo docker-compose up -d

```

## Verify

```bash
sudo docker ps
curl -I http://<floating-ip>
