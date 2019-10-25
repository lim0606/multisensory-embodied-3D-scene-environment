#!/bin/bash

# no-cuda
docker build -t no-cuda-mujoco-openaigym-pytorch-tensorflow -f ./no-cuda/Dockerfile ./
