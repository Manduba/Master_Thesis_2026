#!/bin/bash
cd ~/fcdm_inference

python log_openstack_metrics.py &
sleep 600
stress --cpu $(nproc) --timeout 45m
wait
wc -l data/openstack_metrics.csv
head data/openstack_metrics.csv
