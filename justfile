# Justfile per ARIMA Forecaster
# Alternativa cross-platform a Makefile
# Installazione just: cargo install just
# Uso: just <comando>

# Variabili
python := if os() == "windows" { "python" } else { "python3" }
venv_bin := if os() == "windows" { ".venv/Scripts" } else { ".venv/bin" }

# Comando default
default:
    @just --list

# Installazione e configurazione
install: ## Installa dipendenze con uv
    @echo "🔄 Installazione dipendenze con UV..."
    uv sync --all-extras
    @echo "✅ Installazione completata"

setup: install ## Configurazione completa ambiente
    @echo "🔄 Configurazione ambiente sviluppo..."
    uv run pre-commit install
    mkdir -p logs outputs outputs/models outputs/plots
    @echo "✅ Ambiente configurato"

# Test
test: ## Esegui tutti i test
    @echo "🔄 Esecuzione test..."
    uv run pytest tests/ -v

test-cov: ## Test con coverage
    @echo "🔄 Test con coverage..."
    uv run pytest tests/ --cov=src/arima_forecaster --cov-report=html --cov-report=term-missing

test-fast: ## Test paralleli
    @echo "🔄 Test paralleli..."
    uv run pytest tests/ -v -n auto

# Qualità codice
lint: ## Controlli qualità
    @echo "🔄 Controlli qualità codice..."
    uv run ruff check src/ tests/ examples/
    uv run mypy src/arima_forecaster/

format: ## Formatta codice
    @echo "🔄 Formattazione codice..."
    uv run black src/ tests/ examples/
    uv run ruff format src/ tests/ examples/
    uv run isort src/ tests/ examples/

pre-commit: ## Pre-commit hooks
    @echo "🔄 Pre-commit hooks..."
    uv run pre-commit run --all-files

# Esempi
examples: ## Esegui esempi
    @echo "🔄 Esempio forecasting base..."
    uv run {{python}} examples/forecasting_base.py
    @echo "🔄 Esempio selezione automatica..."
    uv run {{python}} examples/selezione_automatica.py

example-basic: ## Solo esempio base
    uv run {{python}} examples/forecasting_base.py

example-auto: ## Solo esempio selezione automatica
    uv run {{python}} examples/selezione_automatica.py

# Build e publish
build: ## Costruisci package
    @echo "🔄 Build package..."
    uv build
    @echo "✅ Package in dist/"

publish-test: build ## Pubblica su TestPyPI
    @echo "🔄 Publish TestPyPI..."
    uv publish --repository testpypi

publish: build ## Pubblica su PyPI
    @echo "🔄 Publish PyPI..."
    uv publish

# Utilità
clean: ## Pulizia file temporanei
    @echo "🔄 Pulizia..."
    -rm -rf build/ dist/ *.egg-info/
    -rm -rf .coverage htmlcov/ .pytest_cache/
    -rm -rf .mypy_cache/ .ruff_cache/
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    @echo "✅ Pulizia completata"

update: ## Aggiorna dipendenze
    @echo "🔄 Aggiornamento dipendenze..."
    uv sync --upgrade

info: ## Info ambiente
    @echo "=== ARIMA Forecaster - Info Ambiente ==="
    @echo "OS: {{os()}}"
    @echo "Python: $({{python}} --version)"
    @echo "UV: $(uv --version)"
    @echo "Directory: $(pwd)"
    @echo "Venv: {{venv_bin}}"

debug: ## Debug import
    @echo "🔍 Test import package..."
    PYTHONPATH=src uv run {{python}} -c "import arima_forecaster; print('✅ Package importato correttamente')"

# Workflow completi
check: format lint test ## Controlli completi
    @echo "✅ Tutti i controlli OK"

ci: ## Pipeline CI/CD
    uv run pytest tests/ --cov=src/arima_forecaster --cov-report=xml
    uv run ruff check src/ tests/ examples/ --output-format=github
    uv run mypy src/arima_forecaster/

all: clean setup format lint test-cov examples ## Workflow completo
    @echo "🎉 Workflow completo eseguito con successo!"

# Windows specifici
@windows-activate:
    @echo "Per attivare ambiente su Windows:"
    @echo "{{venv_bin}}/activate.bat"

@unix-activate: 
    @echo "Per attivare ambiente su Unix:"
    @echo "source {{venv_bin}}/activate"

activate: ## Mostra comando attivazione venv
    @if [ "{{os()}}" = "windows" ]; then just windows-activate; else just unix-activate; fi

# Development shortcuts
dev: setup format lint test ## Setup sviluppo rapido
    @echo "✅ Ambiente sviluppo pronto"

quick: format lint ## Controlli rapidi
    @echo "✅ Controlli rapidi OK"