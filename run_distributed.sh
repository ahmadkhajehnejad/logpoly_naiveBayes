#!/usr/bin/env bash

echo starting run_client_nodes
python run_client_nodes.py > clients.out &

sleep 10

echo starting tcpdump
tcpdump -i any -n portrange 20000-30000 > tcpdump.out 2>&1 &

echo starting run_center_node
python run_center_node.py > center.out

echo killing tcpdumpca
pkill tcpdump
