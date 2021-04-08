FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo  \
	cmake ninja-build protobuf-compiler libprotobuf-dev libmagickwand-dev \
    curl python3.7 python3.7-dev python3.7-distutils && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sv /usr/bin/python3.7 /usr/local/bin/python3 && \
	ln -s /usr/bin/python3.7 /usr/local/bin/python

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	 python3 get-pip.py && \
	 rm get-pip.py

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"

# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

WORKDIR /csc2626-project
COPY requirements.txt requirements.txt
RUN python -m pip --no-cache-dir install --upgrade -r requirements.txt
