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

