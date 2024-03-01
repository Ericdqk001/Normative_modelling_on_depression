.EXPORT_ALL_VARIABLES:
.PHONY: venv install pre-commit clean

GLOBAL_PYTHON = ~/.pyenv/versions/3.10.10/bin/python
LOCAL_PYTHON = ./.venv/bin/python

setup: venv install pre-commit

## Create an empty environment
venv: $(GLOBAL_PYTHON)
	@echo "Creating .venv..."
	- deactivate
	${GLOBAL_PYTHON} -m venv .venv
	@echo "Activating .venv..."

## Install dependencies
install: ${LOCAL_PYTHON}
	@echo "Installing dependencies..."
	${LOCAL_PYTHON} -m pip install --upgrade pip
	${LOCAL_PYTHON} -m pip install pre-commit pip-tools

## Install pre-commit hooks
pre-commit:
	@echo "Setting up pre-commit..."
	pre-commit install
	pre-commit autoupdate

# Compile requirements.txt
refresh:
	@echo "Compiling requirements.txt..."
	pip-compile -o requirements.txt pyproject.toml
	pip-sync

## Running checks
checks: ${LOCAL_PYTHON}
	@echo "Running checks..."
	ruff check --fix .

clean:
	if exist ./.git/hooks ( rm -rf ./.git/hooks )
