# Changelog

Tutte le modifiche significative al progetto ARIMA Forecaster sono documentate in questo file.

Il formato è basato su [Keep a Changelog](https://keepachangelog.com/it/1.0.0/),
e questo progetto aderisce a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Aggiunto - 2025-10-07
- 🌍 **Supporto Multilingue Esteso**: Aggiunti Tedesco, Portoghese e Giapponese
  - Nuovi file traduzione: `de.json` (Deutsch), `pt.json` (Português), `ja.json` (日本語)
  - Aggiornato TranslationManager con 8 lingue totali: IT, EN, ES, FR, ZH, DE, PT, JA
  - Dashboard Streamlit aggiornata con selettore lingue esteso (8 opzioni)
  - API REST configurabile per tutte le 8 lingue
  - Sistema grafici Matplotlib/Plotly completamente localizzato
  - Report Quarto multilingue con supporto caratteri Unicode
  - Documentazione completa aggiornata in CLAUDE.md e README.md

## [0.4.0] - 2025-01-01

### Aggiunto - 🌊 Real-Time Streaming & 🤖 Explainable AI
- 📡 **Apache Kafka Integration**: Real-time streaming di forecast per sistemi enterprise
  - KafkaForecastProducer con fallback automatico a storage locale
  - StreamingConfig flessibile per connessioni Kafka cluster
  - Batch processing e error handling robusto
  - Esempi completi in `scripts/demo_new_features_ascii.py`

- 🔌 **WebSocket Server**: Dashboard real-time con aggiornamenti live
  - WebSocketServer per push notifications istantanee
  - Sistema subscriptions per modelli specifici
  - Heartbeat monitoring e reconnection automatica
  - Scalabilità Redis per deployment distribuiti

- ⚡ **Real-time Forecaster Service**: Forecasting continuo orchestrato
  - RealtimeForecastService con registry modelli centralizzato
  - Scheduling intelligente con detection anomalie integrate
  - Model versioning e hot-swap per zero-downtime updates
  - Health checks e monitoring metriche performance

- 🎯 **Event Processing Engine**: Sistema eventi enterprise con priorità
  - EventProcessor con priority queues e worker threads configurabili
  - 7+ regole predefinite: logging, alerts, metrics, notifications
  - Action system estensibile per integrazioni custom
  - Async processing per high-throughput scenarios

- 🔍 **SHAP Explainable AI**: Spiegazioni model-agnostic per forecast
  - SHAPExplainer con confidence scoring e feature ranking
  - Spiegazioni locali (singolo forecast) e globali (pattern dataset)
  - Visualizzazioni interpretabili con waterfall e force plots
  - Integration-ready per dashboard e report automatici

- 📊 **Feature Importance Analysis**: Analisi variabili con metodi multipli
  - FeatureImportanceAnalyzer con 5+ tecniche statistiche
  - Ranking automatico feature con stability scoring
  - Comparative analysis tra modelli diversi
  - Export results per business intelligence tools

- 🚨 **Anomaly Explainer**: Spiegazione automatica anomalie con AI
  - AnomalyExplainer con severity classification (LOW/MEDIUM/HIGH/CRITICAL)
  - Raccomandazioni automatiche per azioni correttive
  - Historical context analysis per pattern recognition
  - Integration alerts system per notifiche real-time

- 🏢 **Business Rules Engine**: 7+ regole predefinite per vincoli operativi
  - BusinessRulesEngine con capacity constraints management
  - Weekend/holiday adjustments automatici
  - Historical validation con statistical boundaries
  - Regole custom configurabili per business logic specifici

### Modificato - 2025-01-01
- Aggiornato pyproject.toml da v0.3.0 a v0.4.0
- Aggiunte dependencies: kafka-python, websockets, shap, redis
- Aggiornato `__init__.py` principale con export nuovi moduli
- Enhanced API documentation con streaming/explainability examples
- Demo script ASCII-safe per compatibilità Windows console

### Aggiunto - 2025-09-02
- 🧠 **AutoML Engine**: Sistema di selezione automatica modello ottimale (GAME-CHANGER!)
  - AutoForecastSelector con pattern detection intelligente
  - 6 tipologie pattern auto-detectate: Regular, Seasonal, Trending, Intermittent, Volatile, Short
  - Explanation engine con spiegazioni business e confidence scoring
  - Integrazione automatica con tutti i modelli esistenti (ARIMA, SARIMA, Prophet, Intermittent)
  - One-click forecasting: `automl.fit(data)` → modello ottimale + spiegazioni
  - Esempio showcase completo e quickstart per business users

- ⚡ **Batch Processing Engine**: Portfolio analysis automatica per grandi volumi
  - BatchForecastProcessor per elaborazione parallela centinaia di serie
  - Progress tracking real-time con ETA intelligente e callback personalizzabili
  - Parallelizzazione automatica 4-8x speedup con ThreadPoolExecutor/ProcessPoolExecutor
  - Export multi-formato: CSV/Excel/JSON per integrazione ERP immediata
  - Error handling robusto con retry automatici e fallback strategies
  - Cache system per resume job interrotti e ottimizzazione memory usage

- 🌐 **Web UI Dashboard**: Interfaccia Streamlit enterprise per business users
  - Drag-and-drop CSV upload con supporto file multipli e validazione automatica
  - Real-time progress monitoring con visualizzazioni interactive Plotly
  - Dashboard results con drill-down dettagliato e filtri avanzati
  - Export options integrate: CSV, Excel, HTML Report, JSON API-ready
  - Configurazione parametri user-friendly per AutoML e parallelizzazione
  - Launcher script dedicato: `scripts/run_batch_dashboard.py` porta 8502
  
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
- 🌊 Real-Time Streaming
- 🤖 Explainable AI
- 📡 Apache Kafka
- 🔌 WebSocket
- 🎯 Event Processing
- 🔍 SHAP Analysis
- 🚨 Anomaly Detection
- 🏢 Business Rules