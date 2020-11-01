FROM python:3-alpine

COPY . /srv
WORKDIR ["/srv"]

# This installs gunicorn and Flask, among other things
RUN ["pip", "install", "/srv"]
WORKDIR ["/srv/backend"]

CMD ["gunicorn", "-b", "0.0.0.0:8000", "backend.app"]
