FROM continuumio/miniconda3:latest
MAINTAINER Mingxun Wang "mwang87@gmail.com"

RUN conda create -n rdkit -c rdkit rdkit=2019.09.3.0
RUN apt-get update
RUN apt-get install -y build-essential
COPY requirements.txt .
RUN /bin/bash -c "source activate rdkit && pip install -r requirements.txt"

COPY . /app
WORKDIR /app

