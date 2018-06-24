#! /bin/bash

docker build -t "hg" .
docker run -it -p 8888:8888 hg
