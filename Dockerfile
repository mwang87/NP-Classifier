FROM continuumio/miniconda3:4.10.3
MAINTAINER Mingxun Wang "mwang87@gmail.com"

RUN apt-get update && apt-get install -y build-essential libarchive-dev

# installing mamba
RUN conda install -c conda-forge mamba

RUN mamba create -n rdkit -c rdkit rdkit=2019.09.3.0

COPY requirements.txt .
RUN /bin/bash -c "source activate rdkit && pip install -r requirements.txt"

COPY . /app
WORKDIR /app

