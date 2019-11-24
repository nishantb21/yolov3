FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ARG usr
ARG grp
ARG usrid
ARG grpid
ARG home_path=/${usrid}

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod 777 /usr/bin/tini

RUN mkdir ${home_path}
RUN groupadd -r -g ${grpid} ${grp} && useradd -d ${home_path} -g ${grp} -u ${usrid} ${usr} && adduser ${usr} sudo
RUN chown ${usr}:${grp} ${home_path} /opt
USER ${usr}

COPY environment.yml /code/
COPY core /code/core
COPY __init__.py /code/
COPY train.py /code/
VOLUME /code/save
VOLUME /code/datasets

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    /opt/conda/bin/conda env create -f /code/environment.yml && \
    /opt/conda/bin/conda clean -a -y && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base && conda activate copnet" >> ~/.bashrc

ENTRYPOINT [ "/usr/bin/tini", "--" ]