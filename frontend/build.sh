#!/bin/sh
set -e

# build docker image ( run DockerFile)
docker build -t frontend .

# # run docker image ( on port 8000)
docker run -d -p 3000:3000 --name frontend frontend:latest