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
	TARGET_VERSION="13.30"; \
	MIN_VERSION="12.5"; \
	CURRENT_VERSION="$$(command -v exiftool >/dev/null 2>&1 && exiftool -ver || echo '0')"; \
	ver_ge() { \
		[ "$$(printf '%s\n' "$$1" "$$2" | sort -V | head -n1)" = "$$2" ]; \
	}; \
	if ver_ge "$$CURRENT_VERSION" "$$MIN_VERSION"; then \
		echo "ExifTool version $$CURRENT_VERSION found (>= $$MIN_VERSION). Skipping installation."; \
	else \
		TEMP_DIR="$$(mktemp -d)"; \
		INSTALL_PREFIX="/usr/local"; \
		echo "Installing ExifTool version $$TARGET_VERSION..."; \
		cd "$$TEMP_DIR"; \
		TARBALL="Image-ExifTool-$${TARGET_VERSION}.tar.gz"; \
		URL="https://sourceforge.net/projects/exiftool/files/$${TARBALL}/download"; \
		echo "Downloading $$TARBALL..."; \
		wget -q "$$URL" -O "$$TARBALL"; \
		echo "Extracting..."; \
		tar xzf "$$TARBALL"; \
		cd "Image-ExifTool-$${TARGET_VERSION}"; \
		echo "Building..."; \
		perl Makefile.PL; \
		make; \
		echo "Installing..."; \
		sudo make install; \
		echo "Cleaning up..."; \
		cd /; \
		rm -rf "$$TEMP_DIR"; \
		echo "Installation complete. Verifying version..."; \
		INSTALLED="$$(exiftool -ver)"; \
		echo "Installed ExifTool version: $$INSTALLED"; \
		if [ "$$INSTALLED" = "$$TARGET_VERSION" ]; then \
			echo "Success: Version properly installed."; \
		else \
			echo "Warning: Expected version $$TARGET_VERSION, but found $$INSTALLED."; \
		fi; \
	fi

.PHONY launch-workspace:
launch-workspace:
	@echo "Launching Docker workspace for development..."
	@./launch_workspace.sh

.PHONY launch-workspace-force:
launch-workspace-force:
	@echo "Launching Docker workspace for development (force rebuild)..."
	@./launch_workspace.sh --force-rebuild

.PHONY make-init-submodule:
init-submodule:
	@echo "Initializing submodules..."
	@git submodule update --init --recursive
	@echo "Submodules initialized successfully"
