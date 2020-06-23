#!/bin/sh
set -e

# install application 
sudo python3 setup.py install

# build docker image ( run DockerFile)
# docker build -t src .

# # run docker image ( on port 8000)
# docker run -d -p 8000:8000 --name evaluation evaluation:latest

