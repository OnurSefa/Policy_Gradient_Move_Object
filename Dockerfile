FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

LABEL maintainer="Onur Sefa Ozcibik"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        git \
        wget \
        cmake \
        ninja-build \
        build-essential \
        python3.9 \
        python3.9-dev \
        python3.9-pip \
        python3.9-venv \
        python-is-python3.9 \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    python3 -m venv /opt/python3/venv/base

COPY requirements.txt /opt/python3/venv/base/
RUN /opt/python3/venv/base/bin/python3 -m pip install --no-cache-dir -r /opt/python3/venv/base/requirements.txt
RUN /opt/python3/venv/base/bin/python3 -m pip install git+https://github.com/alper111/mujoco-python-viewer.git


COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint to bash
ENTRYPOINT ["/entrypoint.sh"]
