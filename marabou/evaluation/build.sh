#!/bin/sh
set -e

# install application 
python3 setup.py install

# build docker image ( run DockerFile)
docker build -t marabou-evaluation .

# # run docker image ( on port 8000)
docker run -d -p 5000:5000 --name marabou-evaluation marabou-evaluation:latest

