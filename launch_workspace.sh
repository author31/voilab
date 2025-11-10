#!/bin/bash

# Enable X11 forwarding
xhost +local: > /dev/null 2>&1

# Parse command line arguments
FORCE_REBUILD=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get the absolute path of the current directory
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create cache directories for Docker
mkdir -p ~/docker/voilab-workspace/cache/pip
mkdir -p ~/docker/voilab-workspace/cache/uv
mkdir -p ~/docker/voilab-workspace/cache/glcache
mkdir -p ~/docker/voilab-workspace/logs

# Build the Docker image if --force-rebuild flag is set
if [ "$FORCE_REBUILD" = true ]; then
    echo "Building Docker image..."
    docker build -t voilab-workspace:latest . || {
        echo "Error: Docker build failed!"
        exit 1
    }
fi

# Run the Docker container
echo "Launching voilab workspace container..."
docker run \
    --name voilab-workspace \
    --runtime=nvidia \
    --gpus all \
    -d \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -e "DISPLAY=${DISPLAY}" \
    -e "ROS_DISTRO=humble" \
    -e "ROS_PYTHON_VERSION=3" \
    -e "QT_X11_NO_MITSHM=1" \
    --network=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "${WORKSPACE_DIR}:/workspace/voilab" \
    -v ~/docker/voilab-workspace/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/voilab-workspace/cache/uv:/root/.cache/uv:rw \
    -v ~/docker/voilab-workspace/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/voilab-workspace/logs:/root/.nvidia-omniverse/logs:rw \
    -w /workspace/voilab \
    voilab-workspace:latest
