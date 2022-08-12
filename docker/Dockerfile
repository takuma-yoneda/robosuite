FROM nvidia/cuda:11.3.1-runtime-ubuntu18.04

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y git vim tree curl unzip xvfb patchelf ffmpeg cmake swig wget
RUN apt-get update && apt-get install -y libssl-dev libcurl4-openssl-dev  # Needed for pyCurl
RUN apt-get update && apt-get install -y \
    libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev \
    cmake libopenmpi-dev python3-dev zlib1g-dev \
    qtbase5-dev qtdeclarative5-dev libglib2.0-0

RUN apt-get install -y python3.8-dev python3-pip
RUN ln -s $(which python3.8) /usr/local/bin/python
RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN python -m pip install --upgrade pip

RUN apt-get install -y libgl1-mesa-glx libosmesa6 libglew-dev libglfw3 libglfw3-dev

# # FROM nvidia/cuda@sha256:4df157f2afde1cb6077a191104ab134ed4b2fd62927f27b69d788e8e79a45fa1

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

# ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /mujoco_py

RUN mkdir -p /usr/local/etc/mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /usr/local/etc/mujoco \
    && rm mujoco.tar.gz
ENV MUJOCO_PY_MUJOCO_PATH="/usr/local/etc/mujoco/mujoco210"
ENV LD_LIBRARY_PATH="/usr/local/etc/mujoco/mujoco210/bin"
# Compile-time library location
ENV LIBRARY_PATH=$LD_LIBRARY_PATH
RUN ln -s /usr/local/etc/mujoco $HOME/.mujoco \
    && mkdir -p /mujoco_py/mujoco_py/binaries \
    && ln -s /usr/local/etc/mujoco /mujoco_py/mujoco_py/binaries/linux

RUN pip install robosuite

# This compiles /usr/local/lib/python3.8/dist-packages/mujoco_py/cymj.pyx
RUN python -c 'import mujoco_py'

RUN pip install \
   tdqm \
   wandb \
   kornia \
   ffmpeg \
   imageio \
   imageio-ffmpeg \
   moviepy

# Required by IK_POSE controller
RUN pip install pybullet-svl>=3.1.6.4

# NOTE: Dockerfile is so counter-intuitive.
# `COPY ./vendor .` or `COPY ./vendor ./` copies the contents of ./vendor to .
# https://stackoverflow.com/a/72376058/7057866
COPY ./vendor ./vendor


ENTRYPOINT ["/mujoco_py/vendor/Xdummy-entrypoint"]
