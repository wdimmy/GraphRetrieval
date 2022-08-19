FROM nvidia/cuda:10.2-runtime-ubuntu18.04

RUN mkdir /home/sunshine
RUN chmod -R 777 /home/sunshine

WORKDIR /home/sunshine

ENV PATH="/home/sunshine/miniconda3/bin:${PATH}"
ARG PATH="/home/sunshine/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
            git \
            ssh \
            build-essential \
            locales \
            ca-certificates \
            curl \
            unzip \
            vim \
            wget \
            tmux \
            screen \
            pciutils

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir -p /home/sunshine/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/sunshine/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Default to utf-8 encodings in python
# Can verify in container with:
# python -c 'import locale; print(locale.getpreferredencoding(False))'
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN conda install python=3.7.0
RUN conda install -y -c rdkit rdkit
RUN conda install -y -c pytorch pytorch=1.7.0
RUN conda install -y numpy networkx scikit-learn
RUN conda install -c pytorch faiss-gpu
RUN pip install e3fp==1.2.1 msgpack==1.0.0 ipykernel==5.3.0 einops
RUN pip install dgl==0.4.2
RUN pip install tensorboard

ENV TORCH=1.7.0
ENV CUDA=cu102
RUN pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-geometric
RUN pip install --upgrade ase
RUN pip install wandb
RUN pip install ogb

RUN apt-get install -y libxrender1
RUN pip install opencv-python

ARG UNAME=sunshine
ARG UID=13416
ARG GID=1014
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME

WORKDIR /home/sunshine/pc2d

CMD bash