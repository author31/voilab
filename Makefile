.PHONY install-uv:
install-uv:
	@echo "Checking for uv package manager..."
	if ! command -v uv >/dev/null 2>&1; then \
		echo "uv not found, installing via official installer..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "uv installed successfully"; \
	else \
		echo "uv is already installed"; \
	fi;

.PHONY install:
install: install-uv
	@echo "Installing project dependencies using uv..."
	@uv sync
	@echo "Dependencies installed successfully"

.PHONY install-dev:
install-dev: install-uv
	@echo "Installing project dev dependencies using uv..."
	@uv sync --extra dev
	@echo "Dev dependencies installed successfully"

.PHONY launch-jupyterlab:
launch-jupyterlab: install-dev
	@echo "Launching Jupyter Lab"
	@uv run jupyter lab --ip 0.0.0.0 --port 8888 --no-browser
	@echo "Jupyter Lab launched successfully"

.PHONY: install-exiftool
install-exiftool:
	@MIN_VERSION="12.5"; \
	CURRENT_VERSION="$$(command -v exiftool >/dev/null 2>&1 && exiftool -ver || echo 0)"; \
	ver_ge() { \
		[ "$$(printf '%s\n' "$$1" "$$2" | sort -V | head -n1)" = "$$2" ]; \
	}; \
	if ver_ge "$$CURRENT_VERSION" "$$MIN_VERSION"; then \
		echo "ExifTool $$CURRENT_VERSION found (>= $$MIN_VERSION). Skipping installation."; \
	else \
		echo "Installing ExifTool via apt..."; \
		sudo apt update; \
		sudo apt install -y libimage-exiftool-perl; \
		echo "Installed ExifTool version: $$(exiftool -ver)"; \
	fi

.PHONY: install-cmake
install-cmake:
	sudo apt update
	sudo apt install -y cmake

.PHONY: install-ffmpeg
install-ffmpeg:
	@echo "Checking for ffmpeg..."
	@if command -v ffmpeg >/dev/null 2>&1; then \
		echo "ffmpeg is already installed: $$(ffmpeg -version | head -n1)"; \
	else \
		echo "Installing ffmpeg via apt..."; \
		sudo apt update; \
		sudo apt install -y ffmpeg; \
		echo "ffmpeg installed successfully: $$(ffmpeg -version | head -n1)"; \
	fi

.PHONY launch-workspace:
launch-workspace:
	@echo "Launching Docker workspace for development..."
	@./launch_workspace.sh

.PHONY launch-workspace-force:
launch-workspace-force:
	@echo "Launching Docker workspace for development (force rebuild)..."
	@./launch_workspace.sh --force-rebuild

.PHONY: launch-isaac
launch-isaac:
	@set -e; \
	if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then \
		ICD_JSON=/usr/share/vulkan/icd.d/nvidia_icd.json; \
	elif [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then \
		ICD_JSON=/etc/vulkan/icd.d/nvidia_icd.json; \
	else \
		echo "Error: nvidia_icd.json not found in /usr/share/vulkan/icd.d or /etc/vulkan/icd.d"; \
		exit 1; \
	fi; \
	echo "Launching Isaac Sim in detached mode with $$ICD_JSON"; \
	xhost +local: > /dev/null 2>&1 || true; \
	NVIDIA_ICD_JSON="$$ICD_JSON" docker compose up -d isaac-sim

.PHONY make-init-submodule:
init-submodule:
	@echo "Initializing submodules..."
	@git submodule update --init --recursive
	@echo "Submodules initialized successfully"
