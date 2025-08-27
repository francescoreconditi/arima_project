# Cold Start Problem - Guida Completa

## 📋 Overview

Il **Cold Start Problem** è una sfida comune nel forecasting di serie temporali quando si vuole prevedere la domanda di un nuovo prodotto che non ha dati storici. Questa guida spiega come utilizzare le soluzioni implementate nella libreria ARIMA Forecaster.

## 🎯 Problema Business

### Scenario Tipico
- **Situazione**: Lancio di un nuovo prodotto senza storico vendite
- **Obiettivo**: Stimare la domanda per pianificare scorte e investimenti iniziali  
- **Sfida**: Nessun dato storico su cui basare i modelli tradizionali

### Esempi Pratici
- Nuovo modello di carrozzina con caratteristiche innovative
- Variante premium di un prodotto esistente
- Estensione di gamma in nuova categoria merceologica

## 🧠 Metodologie Implementate

### 1. Pattern Transfer
**Concetto**: Trasferisce pattern stagionali e trend da prodotti simili.

**Come funziona**:
- Estrae pattern di stagionalità, trend e volatilità da prodotto sorgente
- Applica scaling factor basato su caratteristiche del nuovo prodotto
- Genera forecast mantenendo la struttura temporale del prodotto simile

**Migliore per**: Prodotti con forte similarità stagionale

### 2. Analogical Forecasting  
**Concetto**: Combina forecast di più prodotti simili con pesi basati su similarità.

**Come funziona**:
- Identifica top-N prodotti più simili
- Calcola scaling factors per ciascun prodotto simile
- Combina forecast con media pesata basata su similarity scores

**Migliore per**: Quando ci sono multipli prodotti comparabili

### 3. Hybrid Method (Raccomandato)
**Concetto**: Combina Pattern Transfer e Analogical Forecasting.

**Come funziona**:
- Genera forecast con entrambi i metodi
- Combina risultati con pesi ottimizzati (60% Pattern, 40% Analogical)  
- Fornisce maggiore robustezza e accuracy

**Migliore per**: Uso generale, bilanciamento accuracy/robustezza

## 🔧 Utilizzo della Libreria

### Import e Setup

```python
from arima_forecaster.core.cold_start import ColdStartForecaster

# Inizializza forecaster
cold_start_forecaster = ColdStartForecaster(
    similarity_threshold=0.7,  # Soglia minima similarità
    min_history_days=30        # Giorni minimi storia prodotto sorgente
)
```

### Preparazione Dati

```python
# 1. Database prodotti esistenti
products_database = {}

for product_code, sales_data in existing_products.items():
    product_info = get_product_info(product_code)
    
    # Estrai features per matching
    features = cold_start_forecaster.extract_product_features(
        sales_data, product_info
    )
    
    products_database[product_code] = {
        'vendite': sales_data,
        'info': product_info, 
        'features': features
    }

# 2. Nuovo prodotto
target_product_info = {
    'codice': 'NUOVO001',
    'nome': 'Nuovo Prodotto',
    'categoria': 'Categoria',
    'prezzo': 299.0,
    'peso': 5.0,
    'features': {
        'price': 299.0,
        'category_encoded': hash('Categoria') % 1000,
        'weight': 5.0
    }
}
```

### Generazione Forecast

```python
# Genera forecast
forecast_series, metadata = cold_start_forecaster.cold_start_forecast(
    target_product_info=target_product_info,
    products_database=products_database,
    forecast_days=30,
    method='hybrid'  # 'pattern', 'analogical', 'hybrid'
)

# Risultati
print(f"Domanda media: {forecast_series.mean():.1f} unità/giorno")
print(f"Domanda totale: {forecast_series.sum():.0f} unità")
print(f"Affidabilità: {metadata['confidence']}")
```

## 📊 Features per Similarità

### Features Automatiche Estratte

1. **Da Serie Temporale**:
   - `mean_demand`: Domanda media
   - `std_demand`: Deviazione standard
   - `cv_demand`: Coefficiente variazione
   - `trend_slope`: Pendenza trend
   - `weekly_seasonality`: Intensità stagionalità settimanale
   - `monthly_seasonality`: Intensità stagionalità mensile

2. **Da Caratteristiche Prodotto**:
   - `price`: Prezzo unitario
   - `category_encoded`: Categoria codificata
   - `weight`: Peso fisico
   - `volume`: Volume/ingombro

### Calcolo Similarità

Supporta diversi metodi di calcolo:
- **Cosine Similarity** (default): Angolo tra vettori features
- **Correlation**: Correlazione di Pearson
- **Euclidean**: Distanza euclidea normalizzata

## 🎛️ Parametri di Tuning

### Similarity Threshold
```python
similarity_threshold = 0.7  # Default: 0.7
```
- **0.9**: Molto restrittivo, solo prodotti quasi identici
- **0.7**: Bilanciato (raccomandato)
- **0.5**: Permissivo, include prodotti moderatamente simili

### Scaling Factors

Il sistema applica scaling automatico basato su:

```python
# Elasticità prezzo (inversa)
if 'prezzo' in features:
    price_ratio = target_price / source_price
    scaling_factor *= (1.0 / price_ratio) ** 0.3

# Caratteristiche fisiche
if 'peso' in features:
    weight_ratio = target_weight / source_weight  
    scaling_factor *= weight_ratio ** 0.1

# Categoria diversa (penalità)
if source_category != target_category:
    scaling_factor *= 0.8
```

## 📈 Dashboard Moretti Integration

### Accesso

1. Avvia dashboard: `streamlit run examples/moretti/moretti_dashboard.py`
2. Naviga al tab **"🚀 Cold Start"**
3. Configura nuovo prodotto
4. Genera forecast

### Configurazione UI

```
Caratteristiche Prodotto:
├── Codice Prodotto: NUOVO001
├── Nome: Nuovo Dispositivo
├── Categoria: Carrozzine
├── Prezzo: €150.0
└── Caratteristiche Aggiuntive
    ├── Peso: 2.0 kg
    ├── Volume: 10.0 L  
    └── Domanda Attesa: 5.0 unità/giorno

Parametri Forecasting:
├── Metodo: hybrid/pattern/analogical
├── Giorni previsione: 7-90
└── Soglia Similarità: 0.1-0.9
```

### Output Dashboard

- **KPI Cards**: Domanda media, totale, picco, affidabilità
- **Grafico Forecast**: Serie temporale previsioni 30 giorni  
- **Prodotti Simili**: Tabella con similarity scores
- **Export CSV**: Download risultati per ERP integration

## 🔍 Esempio Completo - Moretti

### Caso d'Uso: Lancio Carrozzina Ultra-Light

```python
# Scenario business
new_product = {
    'codice': 'CRZ-ULTRA-001',
    'nome': 'Carrozzina Ultra-Light Premium',
    'categoria': 'Carrozzine', 
    'prezzo': 890.0,  # Premium vs CRZ001 (€450)
    'peso': 8.5,      # Più leggera (vs 12kg)
    'volume': 120.0   # Più compatta
}

# Sistema identifica CRZ001 come più simile (similarity: 0.95)
# Applica scaling: -20% domanda per prezzo premium
# Pattern stagionale mantenuto da CRZ001

# Risultati attesi
forecast_results = {
    'domanda_media': 18.2,      # vs 25 di CRZ001 standard  
    'totale_30_giorni': 546,
    'investimento_scorte': 292_000,  # EUR
    'break_even': 42,           # giorni
    'affidabilita': 'high'
}
```

## ⚠️ Limitazioni e Best Practices

### Limitazioni

1. **Qualità Dipende da Prodotti Simili**: Senza prodotti comparabili, accuracy limitata
2. **Seasonality Transfer**: Pattern stagionali potrebbero non applicarsi a nuovi prodotti
3. **Market Conditions**: Non considera cambiamenti macroeconomici

### Best Practices

1. **Soglia Similarità**: Inizia con 0.7, abbassa se pochi match
2. **Validation**: Confronta con expert judgment e market research
3. **Monitoring**: Tracka performance post-lancio per migliorare modello
4. **Features Enrichment**: Aggiungi features specifiche del dominio business

### Troubleshooting

```python
# Problema: Nessun prodotto simile
# Soluzione: Abbassa similarity_threshold o arricchisci features

# Problema: Forecast troppo conservativo  
# Soluzione: Considera fattori di mercato esterni

# Problema: Alta volatilità previsioni
# Soluzione: Aumenta min_history_days per prodotti sorgente
```

## 🚀 Advanced Usage

### Custom Features

```python
def custom_feature_extractor(product_data, product_info):
    features = cold_start_forecaster.extract_product_features(
        product_data, product_info
    )
    
    # Aggiungi features personalizzate
    features['market_size'] = get_market_size(product_info['categoria'])
    features['competition_level'] = analyze_competition(product_info)
    features['brand_strength'] = get_brand_score(product_info['brand'])
    
    return features
```

### Multi-Product Launch

```python
# Per lanci multipli simultanei
new_products = ['PROD001', 'PROD002', 'PROD003']
forecasts = {}

for product in new_products:
    forecast, meta = cold_start_forecaster.cold_start_forecast(
        target_product_info=product_configs[product],
        products_database=products_database,
        method='hybrid'
    )
    forecasts[product] = forecast

# Analizza cannibalization potential
cross_impact_analysis = analyze_cross_impact(forecasts)
```

## 📚 References

- **Paper**: "Transfer Learning for Time Series Forecasting" 
- **Implementation**: `src/arima_forecaster/core/cold_start.py`
- **Examples**: `examples/moretti/moretti_cold_start_simple.py`
- **Dashboard**: Tab "🚀 Cold Start" in Moretti dashboard

---

*Guida aggiornata: Agosto 2024 - ARIMA Forecaster v2.0*