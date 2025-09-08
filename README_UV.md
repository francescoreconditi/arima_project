# 🚀 ARIMA Forecaster con UV - Guida Rapida

## Perché UV?

**UV è 10x più veloce di pip** e offre:
- ⚡ Installazione dipendenze ultra-rapida
- 🔒 Lock file deterministici (uv.lock)
- 🐍 Gestione automatica versioni Python
- 📦 Risoluzione dipendenze migliorata
- 🛠️ Tool unificato per tutto il workflow Python

## Installazione UV

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.sh | iex"

# Windows con winget
winget install --id=astral-sh.uv

# macOS con Homebrew  
brew install uv

# Con pip (fallback)
pip install uv
```

## Quick Start (< 30 secondi)

```bash
# 1. Clona repo
git clone https://github.com/tuonome/arima-forecaster.git
cd arima-forecaster

# 2. Setup completo in un comando
uv sync --all-extras

# 3. Attiva ambiente
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# 4. Verifica funzionamento
uv run pytest tests/ -v
```

## Comandi UV Essenziali

### 🔧 Gestione Ambiente
```bash
# Crea ambiente e installa dipendenze
uv sync --all-extras

# Solo dipendenze core (senza opzionali)
uv sync

# Aggiorna tutte le dipendenze
uv sync --upgrade

# Aggiungi nuova dipendenza
uv add pandas
uv add --dev pytest

# Rimuovi dipendenza
uv remove matplotlib
```

### 🧪 Testing
```bash
# Test base
uv run pytest

# Test con coverage
uv run pytest --cov=src/arima_forecaster

# Test paralleli (più veloce)
uv run pytest -n auto
```

### 📊 Esempi
```bash
# Esempio forecasting base
uv run python examples/forecasting_base.py

# Selezione automatica modello
uv run python examples/selezione_automatica.py

# 🆕 Demo Real-Time Streaming + Explainable AI v0.4.0
uv run python scripts/demo_new_features_ascii.py
```

### ✨ Qualità Codice
```bash
# Formattazione
uv run black src/ tests/ examples/

# Linting (ruff è più veloce di flake8)
uv run ruff check src/ tests/ examples/

# Type checking
uv run mypy src/arima_forecaster/

# Tutto insieme con pre-commit
uv run pre-commit run --all-files
```

## Workflow con Make/Just

Il progetto include sia Makefile che justfile per automazione:

### Con Make
```bash
make setup      # Configurazione completa
make test       # Test
make lint       # Controlli qualità
make format     # Formattazione
make examples   # Esegui esempi
make all        # Tutto insieme
```

### Con Just (cross-platform)
```bash
just setup      # Configurazione completa
just test       # Test  
just lint       # Controlli qualità
just format     # Formattazione
just examples   # Esegui esempi
just all        # Tutto insieme
```

## Confronto Performance UV vs PIP

| Operazione | pip | uv | Speedup |
|------------|-----|----| --------|
| Install iniziale | ~45s | ~4s | **11x** |
| Update dipendenze | ~30s | ~2s | **15x** |
| Risoluzione conflitti | ~60s | ~3s | **20x** |
| Cold install | ~90s | ~8s | **11x** |

## Struttura Files UV

```
arima_project/
├── pyproject.toml          # Configurazione principale (sostituisce setup.py)
├── uv.lock                 # Lock file dipendenze (generato automaticamente)
├── .python-version         # Versione Python (opzionale)
├── Makefile               # Comandi automazione
├── justfile               # Comandi cross-platform
└── .pre-commit-config.yaml # Pre-commit hooks
```

## Migrazione da pip

Se hai un ambiente pip esistente:

```bash
# 1. Disattiva ambiente pip attuale
deactivate

# 2. Rimuovi vecchio venv (opzionale)
rm -rf venv/

# 3. Setup con UV
uv sync --all-extras

# 4. Attiva nuovo ambiente UV
source .venv/bin/activate
```

## CI/CD con UV

### GitHub Actions
```yaml
- name: Set up UV
  uses: astral-sh/setup-uv@v1

- name: Install dependencies  
  run: uv sync --all-extras

- name: Run tests
  run: uv run pytest
```

### GitLab CI
```yaml
before_script:
  - curl -LsSf https://astral.sh/uv/install.sh | sh
  - uv sync --all-extras
  
test:
  script:
    - uv run pytest
```

## Troubleshooting

### UV non trova Python
```bash
# Specifica versione Python
uv python install 3.11
uv sync --python 3.11
```

### Lock file conflitti
```bash
# Rigenera lock file
rm uv.lock
uv sync --all-extras
```

### Cache problemi
```bash
# Pulisci cache UV
uv cache clean
```

### Ambiente corrotto
```bash
# Reset completo
rm -rf .venv uv.lock
uv sync --all-extras
```

## Pro Tips UV

1. **Lock File**: Commita sempre `uv.lock` per build riproducibili
2. **Python Versions**: UV installa automaticamente versioni Python mancanti
3. **Cache Globale**: UV condivide cache tra progetti (risparmio spazio)
4. **Scripts**: Usa `uv run` per script senza attivare venv
5. **Parallel**: UV risolve dipendenze in parallelo per massima velocità

## Contribuire al Progetto con UV

```bash
# 1. Fork e clone
git clone https://github.com/tuonome/arima-forecaster.git
cd arima-forecaster

# 2. Setup sviluppo
uv sync --all-extras
uv run pre-commit install

# 3. Branch feature
git checkout -b feature/nuova-funzionalita

# 4. Sviluppa con workflow UV
uv run pytest          # Test
uv run ruff check .     # Lint
uv run black .          # Format

# 5. Commit e push
git add .
git commit -m "feat: nuova funzionalità"
git push origin feature/nuova-funzionalita
```

---

## 📞 Supporto

- 🐛 Issues: [GitHub Issues](https://github.com/tuonome/arima-forecaster/issues)
- 📖 Docs UV: [UV Documentation](https://docs.astral.sh/uv/)
- 💬 Discussioni: [GitHub Discussions](https://github.com/tuonome/arima-forecaster/discussions)

**Happy coding with UV! ⚡**