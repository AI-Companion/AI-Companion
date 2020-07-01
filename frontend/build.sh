#!/bin/sh
set -e

# build docker image ( run DockerFile)
docker build -t frontend .

# # run docker image ( on port 8000)
docker run -d -p 8000:8000 --name frontend frontend:latest