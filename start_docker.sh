#!/bin/bash
IMAGE_NAME="robotic-grasping"
TAG="latest"

docker run -it --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" --rm \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="./:/workspace/KIidP-Gruppe-4:rw" \
    --privileged --shm-size=256m --gpus all \
    $IMAGE_NAME:$TAG
