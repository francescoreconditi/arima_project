# Changelog

Tutte le modifiche significative al progetto ARIMA Forecaster sono documentate in questo file.

Il formato è basato su [Keep a Changelog](https://keepachangelog.com/it/1.0.0/),
e questo progetto aderisce a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Aggiunto - 2025-09-02
- 🔩 **Intermittent Demand Forecasting**: Nuovo modulo completo per spare parts e ricambi
  - Implementazione Croston's Method (1972) originale
  - SBA - Syntetos-Boylan Approximation (2005) con bias correction
  - TSB - Teunter-Syntetos-Babai (2011) probability-based
  - Adaptive Croston con smoothing parameter dinamico
  - Pattern classification automatica (Smooth/Intermittent/Erratic/Lumpy)
  - Calcolo automatico Safety Stock e Reorder Point
  - Metriche specializzate: MASE, Fill Rate, Service Level
  - Esempio POC completo Moretti per ricambi medicali
  - Documentazione API completa in `docs/API_INTERMITTENT_DEMAND.md`
  - Guida pratica in `docs/INTERMITTENT_DEMAND_GUIDE.md`

### Modificato - 2025-09-02
- Aggiornato README.md con esempi Intermittent Demand
- Esteso `core/__init__.py` con export IntermittentForecaster
- Aggiornato `evaluation/__init__.py` con IntermittentEvaluator

## [1.5.0] - 2024-08-26

### Aggiunto
- 🌍 **Sistema Traduzioni Centralizzato**: Supporto multilingue completo (IT, EN, ES, FR, ZH)
  - Directory `assets/locales/` con JSON per ogni lingua
  - Modulo `utils/translations.py` per gestione centralizzata
  - Dashboard Streamlit completamente tradotta
  - Report multilingue automatici
  - Fix encoding UTF-8 per caratteri cinesi

### Migliorato
- Dashboard Moretti con filtro "Tutti" i prodotti
- Reset automatico selezione prodotto su cambio categoria
- Visualizzazione dati aggregati multi-prodotto
- Compatibilità Unicode Windows

## [1.4.0] - 2024-08-20

### Aggiunto
- 🏭 **Inventory Management System**: Ottimizzazione magazzino enterprise
  - Slow/Fast Moving Classification con ABC/XYZ analysis
  - Perishable/FEFO Management per prodotti deperibili
  - Multi-Echelon Optimization con risk pooling
  - Capacity Constraints Management (volume, peso, budget)
  - Kitting/Bundle Optimization

## [1.3.0] - 2024-08-15

### Aggiunto
- 📈 **Facebook Prophet Integration**: Modelli per serie con stagionalità complessa
  - ProphetForecaster con festività automatiche
  - ProphetModelSelector con grid search
  - Supporto changepoint detection
  - Esempi prophet_auto_selection_demo.py

### Migliorato
- Cold Start Problem con transfer learning
- GPU acceleration fino a 15x speedup
- Configuration management con .env files

## [1.2.0] - 2024-08-10

### Aggiunto
- 🌊 **SARIMA Models**: Gestione completa stagionalità
  - SARIMAForecaster con parametri (P,D,Q,s)
  - SARIMAModelSelector per ottimizzazione automatica
  - Decomposizione stagionale automatica

### Aggiunto
- 🌐 **SARIMAX Models**: Variabili esogene avanzate
  - SARIMAXForecaster con feature selection
  - Advanced exog handling e diagnostica
  - Preprocessing intelligente variabili esterne

## [1.1.0] - 2024-08-05

### Aggiunto
- 📊 **VAR Models**: Forecasting multivariato
  - VARForecaster per serie multivariate
  - Test causalità Granger
  - Impulse response analysis

### Migliorato
- API REST con FastAPI multilingue
- Dashboard Streamlit interattiva
- Report Quarto dinamici

## [1.0.0] - 2024-08-01

### Rilascio Iniziale
- Core ARIMA forecasting
- Selezione automatica modello
- Preprocessing pipeline
- Metriche valutazione
- Visualizzazioni base
- Testing suite completa

## Legenda

- 🔩 Spare Parts & Inventory
- 📈 Forecasting Models
- 🌍 Internationalization
- 🏭 Enterprise Features
- 🚀 Performance
- 📊 Analytics
- 🌐 Integration
- 🔧 Maintenance