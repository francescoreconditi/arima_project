# 🔧 Sistema Scorte Moretti S.p.A. - Technical Guide
## Implementazione & Integration Specs

---

## 🏗️ **Architettura Sistema**

### **Core Components**
```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   ERP Database  │────│  ARIMA Core  │────│  Business Logic │
│  (CSV Export)   │    │ (Forecasting)│    │ (EOQ, Suppliers)│
└─────────────────┘    └──────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Data Pipeline  │    │   ML Models  │    │   CSV Outputs   │
│ (Preprocessing) │    │ (ARIMA/SARIMA)│   │ (Integration)   │
└─────────────────┘    └──────────────┘    └─────────────────┘
```

### **Technology Stack**
- **Language**: Python 3.9+
- **ML Framework**: statsmodels, scikit-learn
- **Data**: pandas, numpy
- **Deployment**: Docker, uv package manager
- **Integration**: CSV/REST API
- **Monitoring**: Structured logging

---

## 📊 **Data Requirements & Formats**

### **Input Data Specification**

#### **Vendite Storiche** (`vendite_storiche.csv`)
```csv
data,prodotto_codice,quantita_venduta,prezzo_unitario
2024-01-01,CRZ001,15,280.00
2024-01-02,CRZ001,18,280.00
2024-01-03,MAT001,12,450.00
```

**Minimum Requirements:**
- **Storico**: 12+ mesi dati giornalieri
- **Coverage**: 90%+ giorni con dati
- **Products**: 3+ prodotti per training iniziale
- **Quality**: <5% valori mancanti

#### **Configurazione Prodotti** (`prodotti_config.csv`)
```csv
codice,nome,categoria,prezzo_medio,lead_time,scorta_minima,criticita
CRZ001,Carrozzina Standard,Carrozzine,280.00,15,20,5
MAT001,Materasso Antidecubito,Antidecubito,450.00,10,15,5
ELT001,Saturimetro,Elettromedicali,120.00,7,25,4
```

#### **Fornitori Database** (`fornitori.csv`)
```csv
prodotto_codice,fornitore_nome,prezzo_base,sconto_50plus,lead_time,affidabilita
CRZ001,MedSupply Italia,300.00,0.15,15,0.95
CRZ001,EuroMedical,310.00,0.12,12,0.92
MAT001,AntiDecubito Pro,480.00,0.18,10,0.98
```

### **Output Data Specification**

#### **Riordini Suggeriti** (`moretti_riordini_YYYYMMDD.csv`)
```csv
prodotto,codice,urgenza,quantita,fornitore_ottimale,costo_totale,lead_time
Carrozzina Standard,CRZ001,MEDIA,133,MedSupply Italia,33915.0,15
```

#### **Previsioni Dettagliate** (`moretti_previsioni_YYYYMMDD.csv`)
```csv
Data,Previsione,Prodotto,Codice
2025-08-25,27.17,Carrozzina Standard,CRZ001
2025-08-26,27.21,Carrozzina Standard,CRZ001
```

---

## 🚀 **Deployment Guide**

### **System Requirements**
```yaml
Environment:
  OS: Windows 10/11, Linux, macOS
  RAM: 4GB+ (8GB recommended)
  Storage: 2GB free space
  Python: 3.9+ with uv package manager

Dependencies:
  - pandas>=2.0
  - numpy>=1.24
  - statsmodels>=0.14
  - scikit-learn>=1.3
  - plotly>=5.0 (optional dashboards)
```

### **Installation Steps**
```bash
# 1. Clone repository
git clone https://github.com/yourcompany/arima_forecaster
cd arima_forecaster

# 2. Setup environment
uv venv
uv sync --all-extras

# 3. Run Moretti demo
cd examples/moretti
uv run python moretti_inventory_fast.py

# 4. Verify outputs
ls -la ../../outputs/reports/moretti_*.csv
```

### **Configuration Files**

#### **Environment Settings** (`.env`)
```bash
# Database connections
ERP_DATABASE_URL=your_erp_connection_string
CSV_INPUT_PATH=./data/input/
CSV_OUTPUT_PATH=./outputs/reports/

# Model parameters
FORECAST_HORIZON_DAYS=30
MIN_HISTORICAL_DAYS=365
CONFIDENCE_LEVEL=0.95

# Alerts
STOCKOUT_THRESHOLD_DAYS=7
EMAIL_ALERTS=procurement@morettispa.com
```

#### **Model Configuration** (`moretti_config.yaml`)
```yaml
products:
  high_volume:
    model: ARIMA
    order: [1, 1, 1]
    seasonal: false
  
  seasonal_products:
    model: SARIMA
    seasonal_period: 7
    auto_select: true

suppliers:
  evaluation_criteria:
    price_weight: 0.6
    reliability_weight: 0.3
    lead_time_weight: 0.1

forecasting:
  validation_split: 0.2
  min_accuracy_threshold: 0.8  # 80% accuracy
  max_mape_threshold: 20       # 20% MAPE
```

---

## 🔄 **Integration Workflows**

### **Daily Batch Process**
```python
# Pseudocode per processo giornaliero
def daily_inventory_update():
    # 1. Extract data from ERP
    sales_data = extract_sales_from_erp()
    
    # 2. Update models if needed
    if should_retrain_models():
        retrain_all_models()
    
    # 3. Generate forecasts
    forecasts = generate_30day_forecasts()
    
    # 4. Check reorder points
    reorders = check_reorder_requirements()
    
    # 5. Export CSV for procurement
    export_reorder_csv(reorders)
    
    # 6. Send alerts if critical
    send_critical_stockout_alerts()
```

### **ERP Integration Points**

#### **Inbound Data Flow**
1. **Sales Transactions** → CSV export ogni notte
2. **Current Inventory** → API call o CSV sync
3. **Supplier Catalogs** → Weekly CSV update
4. **Product Master** → On-demand sync

#### **Outbound Data Flow**
1. **Purchase Requisitions** → CSV import to ERP
2. **Forecast Reports** → Dashboard/email
3. **KPI Metrics** → Business Intelligence tools
4. **Alert Notifications** → Email/SMS integration

---

## ⚡ **Performance Optimization**

### **Model Training Performance**
```python
# Configurazioni performance per diversi scenari

# Small deployment (< 20 products)
CONFIG_SMALL = {
    'model_type': 'ARIMA',
    'max_models_tested': 10,
    'training_frequency': 'weekly',
    'expected_runtime': '< 2 minutes'
}

# Medium deployment (20-50 products)  
CONFIG_MEDIUM = {
    'model_type': 'ARIMA + SARIMA',
    'max_models_tested': 20,
    'training_frequency': 'weekly',
    'expected_runtime': '< 10 minutes'
}

# Large deployment (50+ products)
CONFIG_LARGE = {
    'model_type': 'Auto-ML pipeline',
    'parallel_training': True,
    'training_frequency': 'bi-weekly',
    'expected_runtime': '< 30 minutes'
}
```

### **Memory & Storage Requirements**
- **Small (< 20 products)**: 1GB RAM, 500MB storage
- **Medium (20-50 products)**: 2GB RAM, 1GB storage  
- **Large (50+ products)**: 4GB RAM, 2GB storage
- **Historical data**: ~10MB per product per year

---

## 🔍 **Monitoring & Maintenance**

### **Key Performance Indicators**
```python
# KPI Tracking automatico
kpis = {
    'forecast_accuracy': 'MAPE < 20%',
    'model_drift': 'Accuracy drop < 10% vs baseline',
    'processing_time': 'Full pipeline < 30 minutes',
    'data_quality': 'Missing values < 5%',
    'system_uptime': '99.5% availability'
}
```

### **Alert Conditions**
- **Critical**: MAPE > 25% per 3 giorni consecutivi
- **High**: Stockout previsto < 5 giorni
- **Medium**: Supplier reliability < 90%
- **Low**: Forecast processing time > 60 minuti

### **Log Monitoring**
```bash
# Structured logging per troubleshooting
tail -f logs/moretti_inventory.log | grep ERROR
tail -f logs/moretti_inventory.log | grep "MAPE"
tail -f logs/moretti_inventory.log | grep "STOCKOUT_ALERT"
```

---

## 🧪 **Testing & Validation**

### **Unit Tests**
```bash
# Run test suite completo
uv run pytest tests/ -v

# Test specifici per Moretti
uv run pytest tests/test_moretti_integration.py -v

# Performance tests
uv run pytest tests/test_performance.py --benchmark
```

### **Data Validation Pipeline**
```python
def validate_input_data(df):
    checks = [
        'no_future_dates',
        'positive_quantities', 
        'valid_product_codes',
        'reasonable_prices',
        'minimum_history_length'
    ]
    return run_validation_checks(df, checks)
```

### **Model Validation**
- **Backtesting**: 6 mesi historical validation
- **Cross-validation**: Time series split validation
- **A/B Testing**: Parallel run con sistema attuale
- **Business Logic**: EOQ calculations verification

---

## 🔐 **Security & Compliance**

### **Data Protection**
- **Encryption**: AES-256 for data at rest
- **Transport**: TLS 1.3 for data in transit  
- **Access Control**: Role-based permissions
- **Audit Trail**: All data access logged

### **GDPR Compliance**
- **Data Minimization**: Solo dati necessari per forecasting
- **Retention**: Auto-delete dati > 3 anni
- **Anonymization**: PII removal da logs
- **Right to be Forgotten**: Data deletion procedures

---

## 📞 **Support & Maintenance**

### **Support Tiers**
- **Tier 1**: Email support, 48h response time
- **Tier 2**: Phone support, 24h response time  
- **Tier 3**: On-site support, 8h response time
- **Critical**: Emergency hotline, 2h response time

### **Maintenance Schedule**
- **Daily**: Automated health checks
- **Weekly**: Model performance review
- **Monthly**: System optimization  
- **Quarterly**: Full system audit & upgrade

---

*Sistema progettato per alta affidabilità e facile integrazione con infrastruttura esistente Moretti S.p.A.*