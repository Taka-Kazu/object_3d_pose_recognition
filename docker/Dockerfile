FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python-tk \
                       python-vtk \
                       python-qt4 \
                       python-qt4-gl \
                       python-setuptools \
                       python-configobj

RUN apt-get update && \
    apt-get install -y apt-utils \
                       git \
                       python-pip \
                       python-opengl \
                       curl \
                       wget \
                       libopencv-dev

RUN pip install --upgrade pip

RUN pip install numpy \
                matplotlib \
                tqdm \
                tensorflow-gpu==1.9.0 \
                tensorboardX \
                torch \
                torchvision \
                jupyter \
                keras \
                opencv-python \
                pillow \
                vtk

RUN pip install mayavi

# nvidia environment
# for 1
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
# for 2
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update && \
    apt-get install -y --no-install-recommends mesa-utils x11-apps && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE 1

WORKDIR /root
