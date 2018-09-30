FROM jupyter/base-notebook
USER root

RUN apt-get update && \
    apt-get -y install sudo \
    git \
    gcc \
    g++ \
    make \
    curl \
    xz-utils \
    liblzma-dev \
    file \
    mecab-ipadic \
    mecab-ipadic-utf8 \
    bzip2 \
    graphviz \
    libgl1-mesa-glx \
    libhdf5-dev \
    openmpi-bin \
    wget && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/downloads && \
    cd /opt/downloads && \
    git clone https://github.com/taku910/mecab.git && \
    git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git

RUN cd /opt/downloads/mecab/mecab && \
    ./configure  --enable-utf8-only && \
    make && \
    make check && \
    make install

RUN apt-get -y install
RUN cd /opt/downloads/mecab-ipadic-neologd && \
    ./bin/install-mecab-ipadic-neologd -n -y

USER jovyan
ENV project_dir /home/jovyan/work
#RUN sudo mkdir -p $project_dir
ADD requirements.txt $project_dir
WORKDIR $project_dir
RUN pip install -r requirements.txt
RUN pip install https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master \
jupyter_nbextensions_configurator

RUN jupyter contrib nbextension install --user && jupyter nbextensions_configurator enable --user