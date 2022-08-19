#!/bin/bash

docker run --rm \
  --runtime=nvidia \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v ${PWD}:/home/sunshine/pc2d \
  -e NVIDIA_VISIBLE_DEVICES=$1 \
  pc2d python train_gcn.py --source=$2 --dataset=$3 --retrieval=$4 --k=$5