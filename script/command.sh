#!/bin/bash


job_name=$1
train_gpu=$2
num_node=$3
command=$4
total_process=$((train_gpu*num_node))

mkdir -p log


port=$(( $RANDOM % 300 + 23450 ))


# nohup
GLOG_vmodule=MemcachedClient=-1 \
srun --partition=VA \
--mpi=pmi2 -n$total_process \
--gres=gpu:$train_gpu \
--ntasks-per-node=$train_gpu \
--job-name=$job_name \
--kill-on-bad-exit=1 \
--cpus-per-task=6 \
-x "BJ-IDC1-10-10-16-[53,98,115,116,117,119,120,121,122,123,124]" \
$command --port $port --job_name $job_name 2>&1|tee -a log/$job_name.log &
