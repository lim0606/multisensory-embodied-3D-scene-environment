# Docker images 
Docker images for Ubuntu 16.04 + Mujoco + OpenAI mujoco-py + OpenAI gym + PyTorch (optionally + CUDA 9.0)  
(Ths code is modified from https://hub.docker.com/r/anibali/pytorch/)

## Requirements

### Mujoco
Copy the mujoco license at the current folder.
```sh
cp <PATH TO LINCENCE>/mjkey.txt .
``` 

### Docker
In order to use this image you must have Docker Engine installed. Instructions
for setting up Docker Engine are
[available on the Docker website](https://docs.docker.com/engine/installation/).


## Build docker images

### Without CUDA
```sh
./build-no-cuda.sh
```

### With CUDA 9.0 + cudnn 7
```sh
./build-cuda-9.0-cudnn7.sh
```

## Usage

This docker image run xvfb by default. See `run.sh`.

```sh
PROJ_PATH=<PATH TO YOUR PYTHON CODE>
docker run -it --rm \
  -v "${PROJ_PATH}:/app/proj" \
  -u $(id -u) \
  --name "example" \
  no-cuda-mujoco-openaigym-pytorch-tensorflow:latest
```

## References
1. https://github.com/anibali/docker-pytorch 
1. https://github.com/pascalwhoop/tf_openailab_gpu_docker/blob/master/Dockerfile 
1. https://github.com/openai/mujoco-py/blob/master/Dockerfile 
1. https://github.com/openai/gym/blob/master/Dockerfile
1. https://github.com/eric-heiden/deep-rl-docker
