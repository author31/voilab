#!/bin/bash
set -e

# Entrypoint script for Isaac Sim container with GUI support

# Ensure X11 display is available
if [ -z "$DISPLAY" ]; then
    echo "WARNING: DISPLAY variable is not set. GUI may not work."
    export DISPLAY=:1
fi

# Check if X11 socket is accessible
if [ ! -d "/tmp/.X11-unix" ]; then
    echo "WARNING: /tmp/.X11-unix directory not found. X11 forwarding may not work."
fi

# Set up X11 authentication if XAUTHORITY is set
if [ -n "$XAUTHORITY" ] && [ -f "$XAUTHORITY" ]; then
    echo "Using XAUTHORITY: $XAUTHORITY"
else
    echo "No XAUTHORITY file found, using default X11 authentication"
fi

# Print environment info for debugging
echo "=== Isaac Sim Container Starting ==="
echo "DISPLAY: $DISPLAY"
echo "USER: $(whoami)"
echo "Working directory: $(pwd)"
echo "===================================="

# If arguments are provided, execute them instead of default command
if [ $# -gt 0 ]; then
    echo "Executing custom command: $@"
    exec "$@"
else
    # Launch Isaac Sim GUI
    echo "Launching Isaac Sim..."
    exec /isaac-sim/runapp.sh
fi

