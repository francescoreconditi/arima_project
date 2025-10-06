# Dockerfile per ARIMA Forecaster FastAPI
FROM python:3.11-slim

# Metadata
LABEL maintainer="ARIMA Forecaster Team <support@arima-forecaster.com>"
LABEL description="FastAPI REST API for Time Series Forecasting"
LABEL version="0.4.0"

# Variabili di ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Crea utente non-root per sicurezza
RUN useradd -m -u 1000 -s /bin/bash apiuser

# Directory di lavoro
WORKDIR /app

# Installa dipendenze di sistema necessarie per statsmodels e scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installa UV (package manager veloce)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv && \
    ln -s /root/.local/bin/uvx /usr/local/bin/uvx

# Copia solo i file necessari per l'installazione delle dipendenze
COPY pyproject.toml ./
COPY README.md ./

# Crea directory src placeholder per installazione
RUN mkdir -p src/arima_forecaster

# Installa dipendenze Python con UV (molto più veloce di pip)
# Installa solo le dipendenze necessarie per la API
RUN uv pip install --system \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    statsmodels>=0.14.0 \
    scikit-learn>=1.3.0 \
    scipy>=1.10.0 \
    fastapi>=0.116.1 \
    uvicorn[standard]>=0.35.0 \
    python-multipart>=0.0.20 \
    pydantic>=2.0.0 \
    scalar-fastapi>=1.0.0 \
    python-dotenv>=1.0.0 \
    matplotlib>=3.6.0 \
    seaborn>=0.12.0 \
    plotly>=5.15.0 \
    openpyxl>=3.1.0 \
    prophet>=1.1.5 \
    pytrends>=4.9.2 \
    kafka-python>=2.0.2 \
    websockets>=11.0.2 \
    shap>=0.42.1 \
    redis>=4.6.0 \
    psutil>=5.9.0 \
    rich>=13.0.0 \
    email-validator>=2.3.0 \
    networkx>=3.2.1

# Copia il codice sorgente dell'applicazione
COPY src/ ./src/
COPY scripts/ ./scripts/

# Crea directory per outputs e modelli
RUN mkdir -p outputs/models outputs/plots outputs/reports logs

# Cambia ownership a utente non-root
RUN chown -R apiuser:apiuser /app

# Cambia a utente non-root
USER apiuser

# Espone la porta 8000 per FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando di default: avvia FastAPI con uvicorn
CMD ["uv", "run", "uvicorn", "arima_forecaster.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
