==build:
docker build -t evaluation .

==run:
runs evaluation on port 8000, available by any address
    docker run -d -p 8000:8000 --name evaluation evaluation:latest

