#!/bin/bash

PROJ_PATH=${1}
docker run -it --rm \
  -v "${PROJ_PATH}:/app/projs/multisensory-embodied-3D-scene-environment" \
  -u $(id -u) --name "test" no-cuda-mujoco-openaigym-pytorch-tensorflow
