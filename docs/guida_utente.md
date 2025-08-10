# Guida Utente ARIMA Forecaster

Questa guida completa ti aiuterà a iniziare con la libreria ARIMA Forecaster e dimostrerà le sue caratteristiche principali attraverso esempi pratici.

## Indice
1. [Installazione](#installazione)
2. [Avvio Rapido](#avvio-rapido)
3. [Caricamento e Preprocessing Dati](#caricamento-e-preprocessing-dati)
4. [Addestramento Modello](#addestramento-modello)
5. [Selezione Automatica Modello](#selezione-automatica-modello)
6. [Forecasting](#forecasting)
7. [Valutazione Modello](#valutazione-modello)
8. [Visualizzazione](#visualizzazione)
9. [Funzionalità Avanzate](#funzionalità-avanzate)
10. [Best Practices](#best-practices)

## Installazione

### Con UV (Raccomandato - 10x più veloce) ⚡

```bash
# Installa uv se non ce l'hai già
curl -LsSf https://astral.sh/uv/install.sh | sh
# oppure: winget install --id=astral-sh.uv  (Windows)
# oppure: brew install uv  (macOS)

# Clona il repository
git clone https://github.com/tuonome/arima-forecaster.git
cd arima-forecaster

# Crea ambiente virtuale e installa tutte le dipendenze
uv sync --all-extras

# Attiva ambiente virtuale
source .venv/bin/activate  # Linux/macOS
# oppure: .venv\Scripts\activate  # Windows
```

### Dal Codice Sorgente (pip tradizionale)
```bash
# Clona il repository
git clone https://github.com/tuonome/arima-forecaster.git
cd arima-forecaster

# Installa in modalità sviluppo
pip install -e .

# Oppure installa con tutte le dipendenze opzionali
pip install -e ".[all]"
```

### Da PyPI (quando pubblicato)
```bash
# Con UV
uv add arima-forecaster[all]

# Con pip
pip install "arima-forecaster[all]"
```

## Avvio Rapido

Ecco un semplice esempio per iniziare:

```python
import pandas as pd
from arima_forecaster import ARIMAForecaster, TimeSeriesPreprocessor, ForecastPlotter

# Carica i tuoi dati di serie temporali
dati = pd.read_csv('tuoi_dati.csv', index_col=0, parse_dates=True)
serie = dati['valore']  # Assumendo che la colonna target si chiami 'valore'

# Preprocessa i dati
preprocessore = TimeSeriesPreprocessor()
serie_processata, metadata = preprocessore.preprocess_pipeline(serie)

# Addestra il modello ARIMA
modello = ARIMAForecaster(order=(1, 1, 1))
modello.fit(serie_processata)

# Genera previsioni
previsioni = modello.forecast(steps=12)

# Visualizza i risultati
plotter = ForecastPlotter()
fig = plotter.plot_forecast(serie_processata, previsioni)
fig.show()
```

## Caricamento e Preprocessing Dati

### Caricamento Dati

La classe `DataLoader` fornisce robuste capacità di caricamento dati:

```python
from arima_forecaster.data import DataLoader

loader = DataLoader()

# Carica CSV con parsing automatico delle date
df = loader.load_csv(
    'data/dati_vendite.csv',
    date_column='data',    # Specifica colonna data
    value_column='vendite' # Specifica colonna target
)

# Valida i dati
validazione = loader.validate_time_series(df, 'vendite')
print("I dati sono validi:", validazione['is_valid'])
print("Problemi trovati:", validazione['issues'])
```

### Pipeline di Preprocessing

Il `TimeSeriesPreprocessor` gestisce compiti comuni di preprocessing:

```python
from arima_forecaster.data import TimeSeriesPreprocessor

preprocessore = TimeSeriesPreprocessor()

# Gestisci valori mancanti
serie_pulita = preprocessore.handle_missing_values(
    serie, 
    method='interpolate'  # Opzioni: 'interpolate', 'drop', 'forward_fill', 'backward_fill'
)

# Rimuovi outlier
serie_pulita = preprocessore.remove_outliers(
    serie_pulita,
    method='iqr',         # Opzioni: 'iqr', 'zscore', 'modified_zscore'
    threshold=3.0
)

# Controlla stazionarietà
stazionarieta = preprocessore.check_stationarity(serie_pulita)
print("È stazionaria:", stazionarieta['is_stationary'])
print("P-value:", stazionarieta['p_value'])

# Rendi stazionaria se necessario
if not stazionarieta['is_stationary']:
    serie_stazionaria, n_diff = preprocessore.make_stationary(serie_pulita)
    print(f"Applicate {n_diff} differenze per raggiungere la stazionarietà")

# Oppure usa la pipeline completa
serie_processata, metadata = preprocessore.preprocess_pipeline(
    serie,
    handle_missing=True,
    missing_method='interpolate',
    remove_outliers_flag=True,
    outlier_method='iqr',
    make_stationary_flag=True,
    stationarity_method='difference'
)

print("Passi preprocessing:", metadata['preprocessing_steps'])
```

## Addestramento Modello

### Addestramento Modello Base

```python
from arima_forecaster import ARIMAForecaster

# Crea e addestra modello
modello = ARIMAForecaster(order=(2, 1, 1))
modello.fit(serie_processata)

# Ottieni informazioni modello
info = modello.get_model_info()
print(f"AIC: {info['aic']:.2f}")
print(f"BIC: {info['bic']:.2f}")
print("Parametri:", info['params'])
```

### Addestramento con Validazione

```python
# Valida dati input prima dell'adattamento
modello = ARIMAForecaster(order=(1, 1, 1))
try:
    modello.fit(serie, validate_input=True)
    print("Modello adattato con successo")
except Exception as e:
    print(f"Adattamento modello fallito: {e}")
```

### Salva e Carica Modelli

```python
# Salva modello addestrato
modello.save('models/mio_modello_arima.pkl')

# Carica modello
modello_caricato = ARIMAForecaster.load('models/mio_modello_arima.pkl')
print("Modello caricato con successo")

# Ottieni informazioni sul modello caricato
info = modello_caricato.get_model_info()
print("Ordine modello:", info['order'])
```

## Selezione Automatica Modello

Usa `ARIMAModelSelector` per trovare automaticamente i migliori parametri del modello:

```python
from arima_forecaster.core import ARIMAModelSelector

# Crea selettore modello
selettore = ARIMAModelSelector(
    p_range=(0, 3),
    d_range=(0, 2), 
    q_range=(0, 3),
    information_criterion='aic'  # Opzioni: 'aic', 'bic', 'hqic'
)

# Cerca il modello migliore
ordine_migliore = selettore.search(serie_processata, verbose=True)
print(f"Modello migliore: ARIMA{ordine_migliore}")

# Ottieni risultati dettagliati
risultati_df = selettore.get_results_summary(top_n=5)
print("Top 5 modelli:")
print(risultati_df)

# Ottieni informazioni modello migliore
info_migliore = selettore.get_best_model_info()
print(f"Migliore {info_migliore['criterion_used'].upper()}: {info_migliore[info_migliore['criterion_used']]:.2f}")

# Addestra modello finale con i migliori parametri
modello_finale = ARIMAForecaster(order=ordine_migliore)
modello_finale.fit(serie_processata)
```

## Forecasting

### Forecasting Base

```python
# Genera previsioni puntuali
previsioni = modello.forecast(steps=12)
print("Prossime 12 previsioni:")
print(previsioni)

# Genera previsioni con intervalli di confidenza
previsioni, conf_int = modello.forecast(
    steps=12,
    confidence_intervals=True,
    alpha=0.05,  # Intervalli confidenza 95%
    return_conf_int=True
)

print("Previsioni con intervalli confidenza 95%:")
for i, (f, lower, upper) in enumerate(zip(previsioni, conf_int.iloc[:, 0], conf_int.iloc[:, 1])):
    print(f"Passo {i+1}: {f:.2f} [{lower:.2f}, {upper:.2f}]")
```

### Predizioni In-Sample e Out-of-Sample

```python
# Predizioni in-sample (valori adattati)
valori_adattati = modello.predict(start=0, end=len(serie_processata)-1)

# Predizioni out-of-sample
predizioni_future = modello.predict(
    start=len(serie_processata),
    end=len(serie_processata) + 11
)

# Predizioni dinamiche (usa predizioni precedenti per passo successivo)
predizioni_dinamiche = modello.predict(
    start=len(serie_processata)-24,
    end=len(serie_processata) + 11,
    dynamic=True
)
```

## Valutazione Modello

### Metriche Accuratezza Previsioni

```python
from arima_forecaster.evaluation import ModelEvaluator

valutatore = ModelEvaluator()

# Dividi dati per valutazione
dimensione_train = int(len(serie_processata) * 0.8)
dati_train = serie_processata[:dimensione_train]
dati_test = serie_processata[dimensione_train:]

# Adatta modello sui dati di addestramento
modello = ARIMAForecaster(order=(1, 1, 1))
modello.fit(dati_train)

# Genera previsioni per periodo test
previsioni_test = modello.forecast(steps=len(dati_test))

# Calcola metriche
metriche = valutatore.calculate_forecast_metrics(dati_test, previsioni_test)
print("Metriche Accuratezza Previsioni:")
for metrica, valore in metriche.items():
    print(f"{metrica.upper()}: {valore:.4f}")
```

### Analisi Residui

```python
# Ottieni residui modello
residui = modello.fitted_model.resid

# Analisi completa residui
diagnostica_residui = valutatore.evaluate_residuals(residui)

print("Diagnostica Residui:")
print(f"Media: {diagnostica_residui['mean']:.4f}")
print(f"Deviazione Standard: {diagnostica_residui['std']:.4f}")
print(f"Asimmetria: {diagnostica_residui['skewness']:.4f}")
print(f"Curtosi: {diagnostica_residui['kurtosis']:.4f}")

# Test statistici
test_jb = diagnostica_residui['jarque_bera_test']
print(f"Test Jarque-Bera: statistica={test_jb['statistic']:.4f}, p-value={test_jb['p_value']:.4f}")
print(f"I residui sono normali: {test_jb['is_normal']}")

test_lb = diagnostica_residui['ljung_box_test']
print(f"Test Ljung-Box: statistica={test_lb['statistic']:.4f}, p-value={test_lb['p_value']:.4f}")
print(f"I residui sono rumore bianco: {test_lb['is_white_noise']}")
```

### Report Valutazione Completo

```python
# Genera report valutazione completo
report = valutatore.generate_evaluation_report(
    actual=dati_test,
    predicted=previsioni_test,
    residuals=residui,
    model_info=modello.get_model_info()
)

print("Report Valutazione Generato")
print("Interpretazione:")
for metrica, interpretazione in report['interpretation'].items():
    print(f"{metrica}: {interpretazione}")
```

## Visualizzazione

### Grafico Previsioni Base

```python
from arima_forecaster.visualization import ForecastPlotter

plotter = ForecastPlotter()

# Grafico previsioni semplice
fig = plotter.plot_forecast(
    actual=dati_train,
    forecast=previsioni_test,
    title="Previsioni Vendite",
    save_path="plots/previsioni.png"
)
fig.show()
```

### Previsioni con Intervalli Confidenza

```python
# Previsioni con intervalli confidenza
previsioni_con_ci, conf_int = modello.forecast(
    steps=12, 
    confidence_intervals=True,
    return_conf_int=True
)

fig = plotter.plot_forecast(
    actual=serie_processata,
    forecast=previsioni_con_ci,
    confidence_intervals=conf_int,
    title="Previsioni con Intervalli Confidenza 95%"
)
fig.show()
```

### Grafici Analisi Residui

```python
# Grafici residui completi
fig = plotter.plot_residuals(
    residuals=residui,
    fitted_values=valori_adattati,
    title="Analisi Residui Modello",
    save_path="plots/residui.png"
)
fig.show()
```

### Decomposizione Serie Temporale

```python
# Grafico decomposizione stagionale
fig = plotter.plot_decomposition(
    series=serie_processata,
    model='additive',  # o 'multiplicative'
    period=12,         # per stagionalità mensile
    title="Decomposizione Serie Temporale"
)
fig.show()
```

### Grafici ACF/PACF

```python
# Grafico funzioni autocorrelazione
fig = plotter.plot_acf_pacf(
    series=serie_processata,
    lags=24,
    title="Analisi Autocorrelazione"
)
fig.show()
```

### Confronto Modelli

```python
# Confronta multipli modelli
risultati_modelli = {
    'ARIMA(1,1,1)': {'aic': 1234.5, 'bic': 1245.6, 'rmse': 12.3},
    'ARIMA(2,1,1)': {'aic': 1230.2, 'bic': 1248.9, 'rmse': 11.8}, 
    'ARIMA(1,1,2)': {'aic': 1235.8, 'bic': 1250.1, 'rmse': 12.7}
}

fig = plotter.plot_model_comparison(
    results=risultati_modelli,
    metric='aic',
    title='Confronto Modelli per AIC'
)
fig.show()
```

### Dashboard Completo

```python
# Crea dashboard forecasting completo
fig = plotter.create_dashboard(
    actual=dati_train,
    forecast=previsioni_test,
    residuals=residui,
    confidence_intervals=conf_int,
    metrics=metriche,
    title="Dashboard Forecasting"
)
fig.show()
```

## Funzionalità Avanzate

### Logging Personalizzato

```python
from arima_forecaster.utils import setup_logger

# Configura logging personalizzato
logger = setup_logger(
    name='mio_forecaster',
    level='DEBUG',
    log_file='logs/forecasting.log'
)

# Usa nel tuo codice
logger.info("Avvio processo forecasting")
```

### Gestione Errori

```python
from arima_forecaster.utils.exceptions import (
    ModelTrainingError, 
    ForecastError,
    DataProcessingError
)

try:
    modello = ARIMAForecaster(order=(5, 2, 5))  # Modello complesso
    modello.fit(serie)
except ModelTrainingError as e:
    print(f"Addestramento fallito: {e}")
    # Fallback a modello più semplice
    modello = ARIMAForecaster(order=(1, 1, 1))
    modello.fit(serie)
```

### Lavoro con Configurazione

```python
# Crea file configurazione
config = {
    'data': {
        'file_path': 'data/vendite.csv',
        'date_column': 'data',
        'value_column': 'vendite'
    },
    'preprocessing': {
        'handle_missing': True,
        'missing_method': 'interpolate',
        'remove_outliers': True,
        'outlier_method': 'iqr'
    },
    'model': {
        'auto_select': True,
        'p_range': [0, 3],
        'd_range': [0, 2],
        'q_range': [0, 3],
        'criterion': 'aic'
    },
    'forecast': {
        'steps': 12,
        'confidence_level': 0.95
    }
}

# Salva configurazione
import json
with open('config/forecast_config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

## Best Practices

### 1. Qualità Dati

- Valida sempre i tuoi dati prima della modellazione
- Gestisci appropriatamente i valori mancanti per il tuo caso d'uso
- Controlla outlier e decidi sulla strategia di trattamento
- Assicura lunghezza dati adeguata (almeno 50-100 osservazioni)

```python
# Checklist qualità dati
def controlla_qualita_dati(serie):
    print(f"Lunghezza dati: {len(serie)}")
    print(f"Valori mancanti: {serie.isnull().sum()}")
    print(f"Valori unici: {serie.nunique()}")
    print(f"Range dati: {serie.min()} a {serie.max()}")
    
    # Controlla valori costanti
    if serie.nunique() == 1:
        print("ATTENZIONE: La serie ha valori costanti")
    
    # Controlla variazione sufficiente
    cv = serie.std() / serie.mean()
    print(f"Coefficiente di variazione: {cv:.4f}")
```

### 2. Selezione Modello

- Inizia con modelli semplici (ARIMA(1,1,1))
- Usa selezione automatica modello come punto di partenza
- Valida sempre su dati fuori-campione
- Considera metodi ensemble per robustezza migliorata

```python
# Best practice selezione modello
def seleziona_modello_migliore(dati_train, dati_test):
    selettore = ARIMAModelSelector()
    
    # Cerca modello migliore
    ordine_migliore = selettore.search(dati_train)
    
    # Addestra e valuta
    modello = ARIMAForecaster(order=ordine_migliore)
    modello.fit(dati_train)
    
    # Valida su dati test
    previsioni_test = modello.forecast(steps=len(dati_test))
    valutatore = ModelEvaluator()
    metriche = valutatore.calculate_forecast_metrics(dati_test, previsioni_test)
    
    return modello, metriche
```

### 3. Validazione Modello

- Controlla sempre diagnostica residui
- Usa validazione walk-forward per valutazione performance realistica
- Monitora performance modello nel tempo
- Ri-stima parametri periodicamente

```python
# Validazione walk-forward
def validazione_walk_forward(serie, finestra_iniziale=100, passo=1):
    errori = []
    
    for i in range(finestra_iniziale, len(serie), passo):
        # Addestra su dati storici
        dati_train = serie[:i]
        valore_reale = serie[i]
        
        # Adatta modello e prevedi
        modello = ARIMAForecaster(order=(1, 1, 1))
        modello.fit(dati_train)
        previsione = modello.forecast(steps=1)
        
        # Calcola errore
        errore = valore_reale - previsione.iloc[0]
        errori.append(errore)
    
    return np.array(errori)
```

### 4. Considerazioni Produzione

- Implementa gestione errori e logging appropriati
- Salva metadata modello e passi preprocessing
- Monitora performance previsioni continuamente
- Configura avvisi per degrado modello

```python
# Funzione forecasting pronta per produzione
def forecasting_produzione(percorso_dati, percorso_modello, config):
    logger = setup_logger('forecaster_produzione')
    
    try:
        # Carica dati
        loader = DataLoader()
        df = loader.load_csv(percorso_dati)
        
        # Preprocessa
        preprocessore = TimeSeriesPreprocessor()
        serie, metadata = preprocessore.preprocess_pipeline(
            df[config['target_column']], **config['preprocessing']
        )
        
        # Carica modello o addestra nuovo
        try:
            modello = ARIMAForecaster.load(percorso_modello)
            logger.info("Caricato modello esistente")
        except:
            logger.info("Addestramento nuovo modello")
            modello = ARIMAForecaster(order=config['model_order'])
            modello.fit(serie)
            modello.save(percorso_modello)
        
        # Genera previsioni
        previsioni = modello.forecast(steps=config['forecast_steps'])
        
        # Logga risultati
        logger.info(f"Previsioni generate: {previsioni.iloc[0]:.2f} a {previsioni.iloc[-1]:.2f}")
        
        return previsioni
        
    except Exception as e:
        logger.error(f"Forecasting fallito: {e}")
        raise
```

### 5. Ottimizzazione Performance

- Usa operazioni vettorizzate dove possibile
- Cache risultati preprocessing
- Considera elaborazione parallela per selezione modello
- Profila il tuo codice per identificare colli di bottiglia

```python
# Monitoraggio performance
import time
from functools import wraps

def tempo_funzione(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tempo_inizio = time.time()
        risultato = func(*args, **kwargs)
        tempo_fine = time.time()
        print(f"{func.__name__} ha impiegato {tempo_fine - tempo_inizio:.2f} secondi")
        return risultato
    return wrapper

@tempo_funzione
def adatta_modello(serie, ordine):
    modello = ARIMAForecaster(order=ordine)
    return modello.fit(serie)
```

Questa guida copre le caratteristiche principali e le best practices per usare la libreria ARIMA Forecaster. Per utilizzo più avanzato e opzioni personalizzazione, consulta la documentazione API e gli esempi nel repository.