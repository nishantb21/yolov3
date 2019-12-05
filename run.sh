#! /bin/bash
docker build --no-cache --build-arg usr="$(id -u -n)" --build-arg usrid="$(id -u)" --build-arg grp="$(id -g -n)" --build-arg grpid="$(id -g)" -f ./dockerfiles/base/Dockerfile -t base_pytorch .
docker build --no-cache -f ./dockerfiles/backbone/train/Dockerfile -t backbone_pytorch .
docker run -d --gpus 1 --shm-size 8G --name backbone_trainer_pytorch --mount type=bind,source="$(pwd)"/save,destination=/code/save --mount type=bind,source="$(pwd)"/datasets/transfer,destination=/code/datasets/transfer backbone_pytorch