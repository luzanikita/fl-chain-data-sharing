install: ## Install requirements
	pip install pip-tools mypy ruff -U
	pip-sync requirements.txt

lock: ## Compile all requirements files
	pip-compile --no-emit-index-url --no-header --verbose --output-file requirements.txt requirements.in

upgrade: ## Upgrade requirements files
	pip-compile --no-emit-index-url --no-header --verbose --upgrade --output-file requirements.txt requirements.in

lint: ## Run code linters
	ruff check src
	mkdir -p .mypy_cache
	mypy --install-types --non-interactive src

fmt format: ## Run code formatters
	isort src
	ruff check --fix src
