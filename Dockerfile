# Use Isaac Sim 5.0.0 as base image
FROM nvcr.io/nvidia/isaac-sim:5.0.0

# Set non-interactive frontend for package installation
ENV DEBIAN_FRONTEND=noninteractive

# Display environment variables for Isaac Sim GUI
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV QT_X11_NO_MITSHM=1
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Install UV package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Remove third-party sources to avoid conflicts
RUN rm -f /etc/apt/sources.list.d/*.list

# Set locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Setup ROS2 sources
RUN add-apt-repository universe && \
    apt-get update && \
    ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
    UBUNTU_CODENAME=$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}}) && \
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${UBUNTU_CODENAME}_all.deb" && \
    dpkg -i /tmp/ros2-apt-source.deb

# Install ROS2 packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --allow-downgrades \
    libbrotli1=1.0.9-2build6 \
    libbrotli-dev && \
    apt-get install -y \
    ros-humble-desktop \
    ros-humble-ros-base \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    ros-humble-cv-bridge \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libfreetype6-dev \
    libfontconfig1-dev && \
    rm -rf /var/lib/apt/lists/* /tmp/ros2-apt-source.deb

# Initialize rosdep
RUN rosdep init && \
    rosdep update

# Set up environment
ENV ROS_DISTRO=humble
ENV ROS_PYTHON_VERSION=3
ENV PATH="/opt/ros/humble/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/ros/humble/lib:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH}"

# Create and set working directory
RUN mkdir -p /workspace/voilab
WORKDIR /workspace/voilab

# Copy project files
COPY . /workspace/voilab

# Set environment variables for the project
ENV PATH="/workspace/voilab/.venv/bin:${PATH}"

# Set entrypoint
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source /workspace/voilab/.venv/bin/activate && bash"]
