FROM --platform=linux/amd64 ubuntu:22.04
MAINTAINER Mingxun Wang "mwang87@gmail.com"

RUN apt-get update && apt-get install -y build-essential libarchive-dev wget vim g++ gcc make cmake git libglib2.0-0 libgl1-mesa-glx libpq-dev libsm6 libxext6 libxrender-dev

# Install Mamba
ENV CONDA_DIR /opt/conda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && /bin/bash ~/miniforge.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Adding to bashrc
RUN echo "export PATH=$CONDA_DIR:$PATH" >> ~/.bashrc

RUN mamba create -n rdkit -c rdkit rdkit --yes

COPY requirements.txt .
RUN /bin/bash -c "source activate rdkit && pip install -r requirements.txt"

COPY . /app
WORKDIR /app

