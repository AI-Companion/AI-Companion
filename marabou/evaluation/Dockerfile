FROM python:3-alpine

COPY . /srv
WORKDIR ["/srv"]

RUN apk update
RUN apk add git make automake gcc g++ subversion
RUN pip3 install tb-nightly
RUN ["pip3", "install","/srv/"]
WORKDIR ["/srv/src"]

CMD ["gunicorn", "-b", "0.0.0.0:8000", "src.app"]
