# syntax = docker/dockerfile:1.7
ARG CUDA_VERSION=12.2.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS uv-base

# Copy only copy uv & uvx binaries (multi-arch, very small)
COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /uvx /usr/bin/

# ——————————————————————————————————————————————————————————————
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

# Bring uv into the final image
COPY --from=uv-base /usr/bin/uv /usr/bin/uv
COPY --from=uv-base /usr/bin/uvx /usr/bin/uvx

WORKDIR /workspace/voilab

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    # Tell every tool that knows about uv to use it
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    # Make uv the default python/pip replacement
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/workspace/voilab/.venv


# ——————————————————— SYSTEM DEPENDENCIES ———————————————————
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        git cmake build-essential curl wget gnupg2 lsb-release \
        software-properties-common locales pkg-config ca-certificates \
        # Python 3.11 runtime only (no pip, no venv module from pip)
        python3.11 python3.11-dev python3.11-distutils \
        # All the other libraries you had before
        libboost-all-dev libqhull-dev libassimp-dev liboctomap-dev \
        libconsole-bridge-dev libfcl-dev libeigen3-dev \
        libx11-dev libxaw7-dev libxrandr-dev libgl1-mesa-dev libglu1-mesa-dev \
        libglew-dev libgles2-mesa-dev libopengl-dev libfreetype-dev \
        qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools \
        libyaml-cpp-dev libzzip-dev freeglut3-dev libogre-1.9-dev \
        libpng-dev libjpeg-dev python3-pyqt5.qtwebengine \
        libbullet-dev libasio-dev libtinyxml2-dev \
        libcunit1-dev libacl1-dev libfmt-dev \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set python3 → python3.11 (uv works with any python binary)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 100

# ———————————————— UV SETUP & PYTHON PACKAGES ————————————————
# Create a single virtual environment that everything will use
RUN uv venv --python /usr/bin/python3.11 ${VIRTUAL_ENV} && \
    # Make the venv active for all subsequent RUN commands
    echo "source ${VIRTUAL_ENV}/bin/activate" >> /etc/bash.bashrc

COPY . /workspace/voilab

# Upgrade uv itself inside the venv and install the heavy packages
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --python ${VIRTUAL_ENV}

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VIRTUAL_ENV} \
        "isaacsim[all,extscache]==5.1.0" \
        --extra-index-url https://pypi.nvidia.com

# Symlink so that CMake/find_package(Python3 ...) still works
RUN ln -sf /usr/include/python3.11 /usr/include/python3


# ———————————————— ENTRYPOINT ————————————————
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
