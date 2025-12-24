# Docker Setup Guide

This document provides instructions for running the Docker containers in this project, particularly for Isaac Sim with GUI support.

## Prerequisites

### X11 Display Server Access

For GUI applications (like Isaac Sim) to work inside Docker containers, you need to allow Docker containers to access your X11 display server.

**Run this command on your host machine before starting the containers:**

```bash
xhost +local:
```

This grants local Docker containers permission to connect to your X11 display server.

> **Note**: This command needs to be run after each reboot. For a more permanent solution, you can add it to your shell's startup script (e.g., `~/.bashrc` or `~/.zshrc`).

> **Security Note**: `xhost +local:` allows all local connections to your X display. For more restrictive access, you can use `xhost +local:docker` or specific container IDs.

### NVIDIA GPU Support

Ensure you have:
- NVIDIA GPU drivers installed on your host
- nvidia-docker2 runtime installed
- Docker Compose with GPU support

## Services

This project provides two Docker services:

### 1. isaac-sim

The Isaac Sim service runs NVIDIA Isaac Sim with ROS 2 Humble support.

**Make sure you have docker-compose-plugin installed:**

```bash
sudo apt-get update
sudo apt-get install docker-compose-plugin
```

**Start the service:**

```bash
# Make sure X11 forwarding is enabled first
xhost +local:

# Start Isaac Sim container
docker compose up isaac-sim
```

The container will automatically launch the Isaac Sim GUI on your display.

**Alternatively, run in detached mode and attach manually:**

```bash
docker compose up -d isaac-sim
docker exec -it <container-name> bash
# Then run commands inside the container
```

### 2. voilab-workspace

The voilab-workspace service provides a development environment with ROS 2 Humble and visualization tools.

**Start the service:**

```bash
xhost +local:
docker compose up -d voilab-workspace
docker exec -it <container-name> bash
```

This container runs in the background and provides an interactive shell for development.

## Configuration Details

### X11 Forwarding Setup

The containers are configured with:
- **Display**: Uses `$DISPLAY` environment variable (defaults to `:1`)
- **X11 Socket**: Mounted from `/tmp/.X11-unix` on the host
- **IPC Mode**: Set to `host` for shared memory (improves X11 performance)
- **Network Mode**: Set to `host` for direct network access

### NVIDIA GPU Access

Both containers are configured with:
- **Runtime**: `nvidia`
- **GPU Access**: All available GPUs
- **Driver Capabilities**: Graphics, display, utility, and compute
- **Vulkan Support**: ICD files mounted for hardware acceleration

## Troubleshooting

### GUI doesn't appear

1. **Check X11 permissions:**
   ```bash
   xhost +local:
   ```

2. **Verify DISPLAY variable:**
   ```bash
   echo $DISPLAY
   ```
   It should show something like `:0`, `:1`, or `:0.0`.

3. **Test X11 inside container:**
   ```bash
   docker exec -it <container-name> bash
   # Inside container:
   xeyes
   ```
   If `xeyes` appears, X11 forwarding is working.

### Permission denied errors

If you see permission errors related to X11:

```bash
# On host machine
xhost +local:
chmod 1777 /tmp/.X11-unix
```

### NVIDIA GPU not detected

Check that nvidia-docker2 is properly installed:

```bash
docker run --rm --runtime=nvidia nvidia/cuda:11.0-base nvidia-smi
```

## Building Images

To rebuild the images after making changes:

```bash
# Rebuild all images
docker compose build

# Rebuild specific service
docker compose build isaac-sim
docker compose build voilab-workspace

# Force rebuild without cache
docker compose build --no-cache
```

## Launching containers
```bash
docker compose up -d
```

## Cleaning Up

```bash
# Stop all services
docker compose down

# Stop and remove volumes
docker compose down -v

# Remove images
docker rmi $(docker images -q voilab*)
```

