FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y apt-utils \
                       git \
                       python3-dev \
                       python3-pip \
                       python3-setuptools \
                       python3-tk \
                       curl \
                       wget \
                       libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install numpy \
                 matplotlib \
                 tqdm \
                 future \
                 tensorflow-gpu==1.14.0 \
                 torch==1.1.0\
                 torchvision \
                 jupyter \
                 keras \
                 opencv-python \
                 pillow \
                 cython

# nvidia environment
# for 1
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
# for 2
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV PYTHONDONTWRITEBYTECODE 1

WORKDIR /root
