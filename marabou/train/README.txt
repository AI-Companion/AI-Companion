==backend
api for webapp

see https://sebest.github.io/post/protips-using-gunicorn-inside-a-docker-image/

==goal:
to set up a gunicorn-powered python API, probably with FLASK

==build:
docker build -t backend .

==run:
runs backend on port 8000, available by any address
    docker run -d -p 8000:8000 --name backend backend:latest

