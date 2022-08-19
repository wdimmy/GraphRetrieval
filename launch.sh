#!/bin/bash

docker run --rm -it \
  --runtime=nvidia \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v ${PWD}:/home/qiliu/pc2d \
  -e NVIDIA_VISIBLE_DEVICES=7 \
  pc2d