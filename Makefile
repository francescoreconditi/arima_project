# Makefile per ARIMA Forecaster
.PHONY: help install test lint format clean docs examples all

# Colori per output
BLUE = \033[0;34m
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m # No Color

help: ## Mostra questo help
	@echo "$(BLUE)ARIMA Forecaster - Comandi Disponibili$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Installa dipendenze con uv
	@echo "$(BLUE)Installazione dipendenze con UV...$(NC)"
	uv sync --all-extras
	@echo "$(GREEN)✓ Installazione completata$(NC)"

install-dev: install ## Installa dipendenze e pre-commit hooks
	@echo "$(BLUE)Configurazione ambiente sviluppo...$(NC)"
	uv run pre-commit install
	@echo "$(GREEN)✓ Ambiente sviluppo configurato$(NC)"

test: ## Esegui tutti i test
	@echo "$(BLUE)Esecuzione test...$(NC)"
	uv run pytest tests/ -v

test-cov: ## Esegui test con coverage
	@echo "$(BLUE)Esecuzione test con coverage...$(NC)"
	uv run pytest tests/ --cov=src/arima_forecaster --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Report coverage generato in htmlcov/$(NC)"

test-fast: ## Esegui test in parallelo
	@echo "$(BLUE)Esecuzione test paralleli...$(NC)"
	uv run pytest tests/ -v -n auto

lint: ## Controlli qualità codice
	@echo "$(BLUE)Controllo qualità codice...$(NC)"
	uv run ruff check src/ tests/ examples/
	uv run mypy src/arima_forecaster/
	@echo "$(GREEN)✓ Controlli completati$(NC)"

format: ## Formatta codice
	@echo "$(BLUE)Formattazione codice...$(NC)"
	uv run black src/ tests/ examples/
	uv run ruff format src/ tests/ examples/
	uv run isort src/ tests/ examples/
	@echo "$(GREEN)✓ Codice formattato$(NC)"

pre-commit: ## Esegui pre-commit hooks
	@echo "$(BLUE)Esecuzione pre-commit hooks...$(NC)"
	uv run pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit completato$(NC)"

examples: ## Esegui esempi
	@echo "$(BLUE)Esecuzione esempi...$(NC)"
	@echo "$(YELLOW)Esempio forecasting base:$(NC)"
	uv run python examples/forecasting_base.py
	@echo ""
	@echo "$(YELLOW)Esempio selezione automatica:$(NC)"
	uv run python examples/selezione_automatica.py
	@echo "$(GREEN)✓ Esempi completati$(NC)"

docs-serve: ## Avvia server documentazione locale
	@echo "$(BLUE)Avvio server documentazione...$(NC)"
	uv run mkdocs serve

docs-build: ## Costruisci documentazione
	@echo "$(BLUE)Costruzione documentazione...$(NC)"
	uv run mkdocs build
	@echo "$(GREEN)✓ Documentazione costruita in site/$(NC)"

clean: ## Pulisci file temporanei
	@echo "$(BLUE)Pulizia file temporanei...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	@echo "$(GREEN)✓ Pulizia completata$(NC)"

build: ## Costruisci package
	@echo "$(BLUE)Costruzione package...$(NC)"
	uv build
	@echo "$(GREEN)✓ Package costruito in dist/$(NC)"

publish-test: build ## Pubblica su TestPyPI
	@echo "$(BLUE)Pubblicazione su TestPyPI...$(NC)"
	uv publish --repository testpypi
	@echo "$(GREEN)✓ Pubblicato su TestPyPI$(NC)"

publish: build ## Pubblica su PyPI
	@echo "$(BLUE)Pubblicazione su PyPI...$(NC)"
	uv publish
	@echo "$(GREEN)✓ Pubblicato su PyPI$(NC)"

update: ## Aggiorna dipendenze
	@echo "$(BLUE)Aggiornamento dipendenze...$(NC)"
	uv sync --upgrade
	@echo "$(GREEN)✓ Dipendenze aggiornate$(NC)"

check-deps: ## Controlla dipendenze obsolete
	@echo "$(BLUE)Controllo dipendenze obsolete...$(NC)"
	uv pip list --outdated

security: ## Controlli sicurezza
	@echo "$(BLUE)Controlli sicurezza...$(NC)"
	uv run bandit -r src/ -f json
	@echo "$(GREEN)✓ Controlli sicurezza completati$(NC)"

all: clean install-dev format lint test-cov ## Esegui tutti i controlli
	@echo "$(GREEN)🎉 Tutti i controlli completati con successo!$(NC)"

# Comandi per CI/CD
ci-test: ## Test per CI/CD
	uv run pytest tests/ -v --cov=src/arima_forecaster --cov-report=xml

ci-lint: ## Lint per CI/CD
	uv run ruff check src/ tests/ examples/ --output-format=github
	uv run mypy src/arima_forecaster/ --junit-xml=mypy-results.xml

# Debug e sviluppo
debug: ## Modalità debug con logging verbose
	@echo "$(YELLOW)Modalità debug attivata$(NC)"
	PYTHONPATH=src uv run python -c "import arima_forecaster; print('✓ Package importato correttamente')"

info: ## Informazioni ambiente
	@echo "$(BLUE)Informazioni Ambiente$(NC)"
	@echo "Python: $$(python --version)"
	@echo "UV: $$(uv --version)"
	@echo "Directory: $$(pwd)"
	@echo "Dipendenze installate: $$(uv pip list | wc -l) pacchetti"