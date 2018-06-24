#! /bin/bash

docker build -t "hg:v1" .
docker run -it -p 8888:8888 hg:v1
