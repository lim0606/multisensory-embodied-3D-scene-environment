#!/bin/bash

# gpu
docker build -t cuda90-cudnn7-mujoco-openaigym-pytorch-tensorflow -f ./cuda-9.0-cudnn7/Dockerfile ./
