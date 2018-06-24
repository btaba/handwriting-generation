FROM tensorflow/tensorflow:1.8.0-py3

# RUN apt update &&\
#    apt install --yes libsm6 libxext6 libfontconfig1 libxrender1 python3-tk  python-setuptools libffi-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig git

ADD requirements.txt .
RUN pip3 install -r requirements.txt

ADD setup.py /handwriting-generation/setup.py
ADD handwriting_gen /handwriting-generation/handwriting_gen
ADD notebooks /handwriting-generation/notebooks

ADD data /handwriting-generation/data
ADD models /handwriting-generation/models

WORKDIR /handwriting-generation
RUN python3 setup.py install
RUN pytest handwriting_gen

ENV HANDWRITING_GENERATION_DATA_DIR /handwriting-generation/data/
ENV HANDWRITING_GENERATION_MODEL_DIR /handwriting-generation/models/

WORKDIR /handwriting-generation/notebooks

CMD jupyter notebook --allow-root --ip 0.0.0.0 --no-browser