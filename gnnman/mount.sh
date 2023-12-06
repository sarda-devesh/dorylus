#! /bin/bash
sudo apt install -y nfs-common
sudo mkdir -p /filepool
sudo mount 172.31.42.148:/home/ubuntu/nfs1 /filepool