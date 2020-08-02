FROM tensorflow/tensorflow:2.2.0

COPY . /marabou/evaluation
WORKDIR ["/marabou/evaluation/"]

RUN ["apt-get","update", "-y"]
RUN ["apt-get","install", "-y","git"]
RUN ["apt-get","install", "-y","libsm6","libxext6","libxrender-dev"]
RUN ["pip3","install", "opencv-python"]
RUN ["python","/marabou/evaluation/setup.py", "install"]
WORKDIR ["/marabou/evaluation/src"]
ENV MARABOU_HOME=/
ENV PYTHONPATH=/marabou/evaluation
EXPOSE 5000

CMD ["python3","/marabou/evaluation/src/app.py"]
