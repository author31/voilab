#!/bin/bash
set -e

# ———————————— Activate voilab venv ————————————
if [ -d "/workspace/voilab/.venv" ]; then
  echo "[entrypoint] Activating voilab venv"
  source /workspace/voilab/.venv/bin/activate
else
  echo "[entrypoint][WARN] .venv not found, using system python"
fi

# ———————————— Set default Isaac Sim args ————————————
# 忽略 Vulkan 驅動版本檢查
export OMNI_KIT_ARGS="--/rtx/verifyDriverVersion/enabled=false"

# Omniverse 也會讀這個（關鍵）
export OMNI_KIT_DISABLE_DRIVER_VERSION_CHECK=1
# ———————————— Execute command ————————————
# 如果 uv run 被傳入，就自動帶上 OMNI_KIT_ARGS
if [[ "$1" == "uv" ]] && [[ "$2" == "run" ]]; then
  shift 2
  exec uv run voilab launch-simulator $OMNI_KIT_ARGS "$@"
else
  exec "$@"
fi
