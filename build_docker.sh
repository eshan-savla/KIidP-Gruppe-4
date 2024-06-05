#!/bin/bash

# Set the Docker image name and tag
IMAGE_NAME="robotic-grasping"
TAG="latest"

# Build the Docker image
docker build -t "$IMAGE_NAME:$TAG" .

if [ "$1" == "--push" ]; then
   # Push the Docker image to Docker Hub
   docker push "$IMAGE_NAME:$TAG"
fi