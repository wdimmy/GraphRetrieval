FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN mkdir /home/sunshine
RUN chmod -R 777 /home/sunshine

WORKDIR /home/qiliu


RUN conda install pyg -c pyg
RUN pip install rdkit
RUN pip install ogb
RUN pip install faiss-gpu
RUN pip install PyTDC

ARG UNAME=sunshine
ARG UID=13416
ARG GID=1014
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME

WORKDIR /home/sunshine/graphretrieval

CMD bash
