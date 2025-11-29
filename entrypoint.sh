#!/bin/bash
set -e

echo "source /workspace/humble_ws/install/local_setup.bash" >> ~/.bashrc
echo "source /workspace/build_ws/install/local_setup.bash" >> ~/.bashrc

echo "[Entrypoint] Handing off to /bin/bash..."
exec /bin/bash "$@"
