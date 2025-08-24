# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ PROCEDURE DI SICUREZZA OBBLIGATORIE ⚠️

### 🔴 Backup Automatico Pre-Modifica
**PRIMA di qualsiasi modifica a qualsiasi file:**
1. **SEMPRE** copiare il file originale in `C:\Backup\Code\{nome_progetto}\`
2. Mantenere la struttura delle directory originale
3. Rinominare il file aggiungendo in fondo al file la data e ora (es. `file.py.backup_20240427_1530`)
4. **SOLO DOPO** il backup completato, procedere con le modifiche
5. Esempio: `cp src/file.py C:\Backup\Code\arima_project\src\file.py.backup`

### 🔴 Gestione File Grandi (>50KB o errore "token limit")
**Quando appare "superato il numero di token" o file >50KB:**
1. **MAI** modifiche massive su file grandi
2. **SEMPRE** leggere tutto il file a sezioni PRIMA (con Read offset/limit)  
3. **SOLO** modifiche incrementali piccole e verificate (<50 righe)
4. **Test immediato** dopo ogni singola modifica
5. **MAI** sostituire blocchi >50 righe senza mappare completamente il file

### 🔴 Regola d'Oro Inviolabile
```
Un errore = danni ai clienti = soldi persi
SEMPRE: Backup PRIMA → Modifiche piccole → Test subito
```

### 🔴 Procedura di Emergenza
Se qualcosa va storto:
1. **STOP** immediatamente
2. **NON** tentare riparazioni
3. **Avvisare** l'utente del problema
4. Attendere istruzioni per ripristino da backup

### 🔴 Commenti nei Sorgenti
Quando possibile SEMPRE scrivere commenti nei sorgenti in Italiano.

## Panoramica del Progetto

Libreria Python avanzata per forecasting serie temporali con modelli ARIMA, SARIMA e VAR. Include funzionalità enterprise-grade come Auto-ML, API REST, dashboard interattiva e reporting dinamico con Quarto.

## Comandi di Sviluppo Essenziali

### Setup Rapido
```bash
# Installa UV (10x più veloce di pip)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS
# winget install --id=astral-sh.uv  # Windows

# Setup completo con Just
just setup  # Installa dipendenze e configura ambiente

# Oppure manualmente
uv sync --all-extras
```

### Testing
```bash
# Tutti i test
uv run pytest tests/ -v

# Con coverage
uv run pytest tests/ --cov=src/arima_forecaster --cov-report=html

# Test specifici
uv run pytest tests/test_arima_model.py -v
uv run pytest tests/test_sarima_model.py -v  
uv run pytest tests/test_reporting.py -v

# Test paralleli (veloce)
uv run pytest tests/ -v -n auto
```

### Qualità del Codice
```bash
# Formattazione
uv run black src/ tests/ examples/
uv run ruff format src/ tests/ examples/

# Linting
uv run ruff check src/ tests/ examples/
uv run mypy src/arima_forecaster/

# Tutti i controlli
just check  # O: uv run pre-commit run --all-files
```

### Servizi Production
```bash
# API REST (FastAPI)
uv run python scripts/run_api.py
# Swagger UI: http://localhost:8000/docs

# Dashboard Web (Streamlit)  
uv run python scripts/run_dashboard.py
# URL: http://localhost:8501

# Script training
uv run python scripts/train.py --data path/to/data.csv --model sarima

# Script forecasting
uv run python scripts/forecast.py --model path/to/model.pkl --steps 30
```

## Architettura del Codice

### Moduli Principali

#### Modelli Core (`src/arima_forecaster/core/`)
- **ARIMAForecaster** (`arima_model.py`): Modello ARIMA base con parametri (p,d,q)
- **SARIMAForecaster** (`sarima_model.py`): ARIMA stagionale con (P,D,Q,s) 
- **VARForecaster** (`var_model.py`): Vector Autoregression per serie multivariate
- **ARIMAModelSelector** (`model_selection.py`): Grid search automatico per ARIMA
- **SARIMAModelSelector** (`sarima_selection.py`): Selezione automatica SARIMA

#### Data Processing (`src/arima_forecaster/data/`)
- **DataLoader**: Caricamento CSV con validazione automatica
- **TimeSeriesPreprocessor**: Pipeline preprocessing configurabile
  - Gestione valori mancanti: interpolate, drop, forward_fill, backward_fill
  - Rilevamento outlier: IQR, z-score, modified z-score
  - Stazionarietà: difference, log_difference, test ADF/KPSS

#### Valutazione (`src/arima_forecaster/evaluation/`)
- **ModelEvaluator**: 15+ metriche (MAE, RMSE, MAPE, sMAPE, MASE, etc.)
- Diagnostica residui completa (Ljung-Box, Jarque-Bera, ACF/PACF)
- Test statistici e analisi performance

#### Visualizzazione (`src/arima_forecaster/visualization/`)
- **ForecastPlotter**: Grafici forecast con intervalli confidenza
- Dashboard interattivi con decomposizione stagionale
- Analisi residui multi-pannello

#### Reporting (`src/arima_forecaster/reporting/`)
- **QuartoReportGenerator**: Report dinamici HTML/PDF/DOCX
- Template personalizzabili con analisi automatiche
- Comparazione modelli side-by-side

#### Auto-ML (`src/arima_forecaster/automl/`)
- **HyperparameterOptimizer**: Ottimizzazione con Optuna/Hyperopt
- **ModelTuner**: Tuning avanzato multi-obiettivo
- Ensemble methods e stacking

#### API & Dashboard
- **FastAPI REST API** (`src/arima_forecaster/api/`): Endpoints production-ready
- **Streamlit Dashboard** (`src/arima_forecaster/dashboard/`): UI web interattiva

### Pipeline Dati Tipica

1. **Caricamento**: `DataLoader.load_data()` con validazione
2. **Preprocessing**: `TimeSeriesPreprocessor.preprocess_pipeline()`
3. **Selezione Modello**: `ARIMAModelSelector.search()` o manuale
4. **Training**: `model.fit(series)` con metadata
5. **Valutazione**: `ModelEvaluator.evaluate()`
6. **Visualizzazione**: `ForecastPlotter.create_dashboard()`
7. **Reporting**: `QuartoReportGenerator.generate_report()`
8. **Deployment**: API REST o dashboard web

### Pattern Import Consigliati

```python
# Import base
from arima_forecaster import (
    ARIMAForecaster, 
    SARIMAForecaster,
    TimeSeriesPreprocessor,
    ForecastPlotter
)

# Import avanzati
from arima_forecaster.core import ARIMAModelSelector, VARForecaster
from arima_forecaster.evaluation import ModelEvaluator
from arima_forecaster.reporting import QuartoReportGenerator  # Richiede [reports]
from arima_forecaster.automl import HyperparameterOptimizer  # Richiede [automl]
```

### Gestione Errori

```python
from arima_forecaster.utils.exceptions import (
    ModelTrainingError,
    ForecastError, 
    DataProcessingError
)

try:
    model = SARIMAForecaster(order=(1,1,1), seasonal_order=(1,1,1,12))
    model.fit(series)
except ModelTrainingError as e:
    logger.error(f"Training failed: {e}")
```

## Considerazioni Performance

- Usa `validate_input=False` per batch processing dopo validazione iniziale
- Cache preprocessing per training ripetuto (`preprocessor.cache_results=True`)
- Elaborazione parallela: `selector.search(n_jobs=-1)`
- Limita spazio ricerca: `selector.search(max_models=100)`

## Workflow con Just

```bash
just setup       # Setup iniziale ambiente
just test-cov    # Test con coverage
just format      # Formatta codice
just lint        # Controlli qualità
just check       # Tutti i controlli
just examples    # Esegui esempi
just clean       # Pulizia file temporanei
just build       # Build package per distribuzione
```

## Note Implementative

### Modelli Disponibili
- **ARIMA**: Serie univariate non stagionali
- **SARIMA**: Serie con componente stagionale (nuovo)
- **VAR**: Serie temporali multivariate (nuovo)
- **Auto-ARIMA**: Selezione automatica parametri ottimali

### Features Avanzate Verificate
- ✅ SARIMA con decomposizione stagionale automatica
- ✅ VAR con test causalità Granger e impulse response
- ✅ Auto-ML con Optuna, Hyperopt, Scikit-Optimize
- ✅ API REST production-ready con FastAPI
- ✅ Dashboard Streamlit interattiva
- ✅ Reporting Quarto con export multi-formato
- ✅ Ensemble methods e model stacking

### Directory Output
- `outputs/models/`: Modelli serializzati (.pkl)
- `outputs/plots/`: Visualizzazioni (.png, .html)  
- `outputs/reports/`: Report Quarto (.html, .pdf, .docx)
- `logs/`: File di log applicazione

## Dipendenze Chiave

- **statsmodels**: Implementazione modelli ARIMA/SARIMA/VAR
- **pandas/numpy**: Manipolazione dati e calcoli
- **matplotlib/seaborn/plotly**: Visualizzazioni
- **fastapi/uvicorn**: API REST
- **streamlit**: Dashboard web
- **quarto**: Report dinamici (opzionale)
- **optuna/hyperopt**: Auto-ML (opzionale)

---

## 📋 CASE HISTORY - Progetti Completati

### 🏥 **CASO MORETTI S.p.A.** - Sistema Intelligente Gestione Scorte AI

#### **📌 OVERVIEW PROGETTO**
**Client**: Moretti S.p.A. (https://www.morettispa.com)  
**Settore**: Dispositivi Medicali - Home Care, Carrozzine, Materassi Antidecubito, Elettromedicali  
**Obiettivo**: Implementazione sistema AI per ottimizzazione scorte e forecasting domanda  
**Status**: ✅ **PROOF-OF-CONCEPT COMPLETATO** - Pronto per implementazione pilota  
**Periodo**: Agosto 2024  

#### **🎯 BUSINESS CASE**
- **Problema**: Stockout 15-20%, scorte eccessive, gestione manuale riordini, fornitori non ottimizzati
- **Soluzione**: Sistema AI ARIMA per forecasting + ottimizzazione multi-supplier automatica
- **ROI Target**: 15-25% riduzione costi scorte, payback 4 mesi
- **Investimento**: €15,000 per implementazione completa
- **Business Value Anno 1**: €325,000+ (stockout recovery + cash flow + automazione)

#### **💻 IMPLEMENTAZIONE TECNICA**

##### **Architettura Sistema Creata:**
```
examples/moretti/
├── moretti_inventory_fast.py         # ✅ DEMO 30-secondi production-ready
├── moretti_simple_example.py         # ✅ Esempio singolo prodotto educativo
├── moretti_inventory_management.py   # ✅ Sistema enterprise completo
├── moretti_advanced_forecasting.py   # ✅ VAR multi-prodotto avanzato
├── moretti_dashboard.py             # ✅ Dashboard Streamlit interattiva
├── moretti_dashboard_demo.py        # ✅ Dashboard HTML standalone
├── run_demo.py                      # ✅ Launcher interattivo
├── data/                            # ✅ Dati demo realistici
│   ├── vendite_storiche.csv
│   ├── prodotti_config.csv
│   └── fornitori.csv
├── README.md                        # ✅ Documentazione completa
├── PRESENTATION_EXECUTIVE_SUMMARY.md # ✅ Materiale client C-level
├── PRESENTATION_TECHNICAL_GUIDE.md   # ✅ Specs tecniche implementazione
├── PRESENTATION_SLIDES.md           # ✅ 13 slide presentation deck
└── DEMO_SCRIPT.md                   # ✅ Script demo live 5-minuti
```

##### **Modelli AI Implementati:**
1. **ARIMA(1,1,1)**: Modello base affidabile per tutti i prodotti
2. **SARIMA Auto-Selection**: Con fallback multipli per robustezza
3. **VAR Multivariato**: Analisi interdipendenze tra prodotti
4. **Economic Order Quantity (EOQ)**: Ottimizzazione quantità riordino
5. **Multi-Supplier Optimization**: Selezione fornitori con pricing tiers

##### **Performance Raggiunte:**
- **MAPE**: 15.39% (target <20%) - Eccellente per inventory forecasting
- **Velocità**: Demo completo in 30 secondi, analisi production <2 minuti
- **Affidabilità**: Fallback multipli, gestione errori robusta
- **Accuratezza Business**: 84.2% forecast accuracy su 3 prodotti critici

#### **📊 RISULTATI DEMO CONCRETI**

##### **3 Prodotti Critici Analizzati:**
| Prodotto | Codice | Domanda/Giorno | Investimento | Fornitore Ottimale |
|----------|--------|---------------|--------------|-------------------|
| Carrozzina Standard | CRZ001 | 27.2 unità | €33,915 | MedSupply Italia |
| Materasso Antidecubito | MAT001 | 26.7 unità | €40,934 | AntiDecubito Pro |
| Saturimetro | ELT001 | 19.2 unità | €19,890 | DiagnosticPro |

**Totale Investimento Ottimizzato**: €94,739

##### **Output CSV Generati:**
- `moretti_riordini_veloce.csv`: Riordini ottimali per ERP import
- `moretti_previsioni_veloce.csv`: Previsioni 30-giorni dettagliate
- `moretti_piano_scorte.csv`: Piano scorte mensile completo

#### **🎨 MATERIALI PRESENTAZIONE CREATI**

##### **1. Executive Materials:**
- **PRESENTATION_EXECUTIVE_SUMMARY.md**: Business case, ROI, competitive advantage
- **PRESENTATION_SLIDES.md**: 13 slide deck per C-level presentation
- **DEMO_SCRIPT.md**: Script parola-per-parola per demo live 5-minuti

##### **2. Technical Materials:**
- **PRESENTATION_TECHNICAL_GUIDE.md**: Specs implementazione, architettura, deployment
- **Dashboard HTML Interattiva**: Visualizzazione KPI real-time con grafici Plotly

##### **3. Demo Materials:**
- **moretti_dashboard_demo.html**: Dashboard autonoma per presentazioni offline
- **Run scripts**: Launcher per diversi scenari demo (veloce/completo/educativo)

#### **🛠️ SFIDE TECNICHE RISOLTE**

##### **API Compatibility Issues:**
```python
# PRIMA (non funzionava)
forecast = model.forecast(steps=30, return_confidence_intervals=True)

# DOPO (fix implementato)  
forecast = model.forecast(steps=30, confidence_intervals=True)
```

##### **SARIMA NaN Predictions:**
```python
# Implementato sistema fallback robusto
try:
    model = SARIMAForecaster(seasonal_order=(1,1,1,12))
    predictions = model.predict(30)
    if pd.isna(predictions).any():
        raise ValueError("NaN predictions")
except Exception:
    # Fallback ARIMA semplice
    model = ARIMAForecaster(order=(1,1,1))
    predictions = model.predict(30)
```

##### **Unicode Windows Console:**
```python
# PRIMA (errore Windows)
print("🏥 Sistema Moretti")

# DOPO (fix Unicode)  
print("[MEDICAL] Sistema Moretti")
```

##### **Path Resolution Cross-Platform:**
```python
# PRIMA (non funzionava Windows)
output_file = "../../outputs/reports/file.csv"

# DOPO (pathlib fix)
output_file = Path(__file__).parent.parent.parent / "outputs" / "reports" / "file.csv"
```

#### **🚀 STATUS ATTUALE**

##### **✅ COMPLETATO:**
1. **Proof-of-Concept**: Sistema funzionante con 3 prodotti
2. **Demo Materials**: Presentation deck completo per client
3. **Technical Validation**: MAPE <20% raggiunto, performance verificate
4. **Business Case**: ROI quantificato, competitive analysis completata
5. **Integration Ready**: CSV input/output per ERP esistenti

##### **🔄 IN PROGRESS:**
- Nessuna attività in corso - progetto in attesa decisioni client

##### **📋 PROSSIMI STEP PREVISTI:**

###### **Phase 1: Pilot Implementation** (4 settimane)
```bash
# Setup produzione
cd examples/moretti
uv run python moretti_inventory_fast.py  # 30-sec demo ready

# Scalabilità pilota
- Estensione da 3 a 15 prodotti high-volume
- Automazione daily refresh con cron jobs
- Dashboard web operations team
- Integrazione ERP bidireccionale
```

###### **Phase 2: Full Deployment** (8 settimane)  
```bash
# Production scale
- Rollout 50+ prodotti completo
- API REST integration e-commerce
- Alert system email/SMS real-time
- Advanced analytics & reporting
- Multi-warehouse support
```

###### **Phase 3: Advanced Features** (12 settimane)
```bash
# Enterprise features
- Seasonal pattern detection automatica
- Supplier performance scoring
- Demand sensing (external factors)
- Mobile app for procurement team
- BI integration (Power BI/Tableau)
```

#### **📈 METRICHE SUCCESSO DEFINITE**

##### **KPI Target Anno 1:**
| Metrica | Baseline | Target | Valore Business |
|---------|----------|--------|-----------------|
| Stockout Rate | 18% | <8% | +€50k vendite |
| Inventory Turns | 4.2x | 5.5x | +€200k cash flow |
| Forecast Accuracy | 65% | 85% | -€75k safety stock |
| Automazione | 0% | 90% | -1 FTE procurement |

**Total Business Value**: €325k+ Anno 1

#### **🔧 SETUP COMMANDS**

##### **Demo Veloce (30 secondi):**
```bash
cd examples/moretti
uv run python moretti_inventory_fast.py
```

##### **Dashboard Interattiva:**
```bash
uv run python moretti_dashboard_demo.py
# Apre automaticamente browser con dashboard HTML
```

##### **Demo Completo Educativo:**
```bash
uv run python run_demo.py
# Menu interattivo per scegliere tipo demo
```

#### **🎯 LESSONS LEARNED**

##### **Technical:**
1. **ARIMA(1,1,1) più affidabile di SARIMA** per inventory forecasting
2. **Fallback multipli essenziali** per robustezza production
3. **CSV integration più semplice** di API complex per ERP legacy
4. **Windows Unicode issues** richiedono ASCII fallback
5. **Pathlib cruciale** per cross-platform compatibility

##### **Business:**
1. **Demo 30-secondi più impattante** di analisi complesse
2. **ROI quantificato immediato** convince C-level executives
3. **Materiali presentation multipli** coprono diversi stakeholder
4. **Competitive analysis vs SAP/Oracle** evidenzia value proposition
5. **Start small, scale gradually** riduce implementation risk

#### **📞 STAKEHOLDER & NEXT ACTIONS**

##### **Decision Makers:**
- **C-Level**: Presentazione executive summary completata
- **IT Director**: Technical guide & integration specs ready  
- **Operations Manager**: Demo materials & ROI analysis available
- **Procurement Team**: CSV samples & workflow integration defined

##### **Immediate Next Steps (quando client ready):**
1. **Budget approval meeting**: €15k investment presentation
2. **Data access setup**: ERP CSV export configuration
3. **IT kickoff session**: Integration requirements definition
4. **Pilot planning**: 15 prodotti selection & timeline

##### **Success Criteria for Go/No-Go:**
- Budget €15k approved
- Historical data 12+ mesi accessible  
- IT team allocated for integration support
- Operations team committed to daily usage

---

**💡 NOTA STRATEGICA**: Il caso Moretti dimostra perfettamente come la libreria ARIMA possa risolvere problemi business reali con ROI quantificabile immediato. La combinazione di semplicità tecnica (ARIMA classici) con business intelligence (supplier optimization) crea value proposition vincente nel settore B2B medicale.

**🔄 REPLICABILITÀ**: Questo framework è replicabile per qualsiasi azienda con inventory management challenges. Pattern identificati:
- Forecast accuracy >80% 
- Multi-supplier optimization
- ERP integration via CSV
- Demo materials standardizzati
- ROI calculation methodology