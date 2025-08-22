# Teoria SARIMAX - Modelli con Variabili Esogene

## Indice
1. [Introduzione](#introduzione)
2. [Definizione Matematica](#definizione-matematica)
3. [Componenti del Modello](#componenti-del-modello)
4. [Variabili Esogene](#variabili-esogene)
5. [Stima dei Parametri](#stima-dei-parametri)
6. [Diagnostica e Validazione](#diagnostica-e-validazione)
7. [Forecasting con SARIMAX](#forecasting-con-sarimax)
8. [Vantaggi e Limitazioni](#vantaggi-e-limitazioni)
9. [Esempi Pratici](#esempi-pratici)

---

## Introduzione

Il modello **SARIMAX** (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables) rappresenta l'estensione più completa della famiglia di modelli ARIMA. Mentre SARIMA considera solo la struttura interna della serie temporale (trend, stagionalità, rumore), SARIMAX incorpora anche **variabili esogene** (esterne) che possono influenzare la serie target.

### Evoluzione dei Modelli
- **ARIMA**: Solo serie temporale interna
- **SARIMA**: Serie temporale + stagionalità
- **SARIMAX**: Serie temporale + stagionalità + **variabili esterne**

Il SARIMAX è particolarmente utile quando si dispone di informazioni aggiuntive che possono migliorare le previsioni, come indicatori economici, variabili meteorologiche, eventi di marketing, o altre serie temporali correlate.

---

## Definizione Matematica

Il modello SARIMAX generale si esprime come:

### Forma Completa
```
φ(B) Φ(B^s) (1-B)^d (1-B^s)^D y_t = θ(B) Θ(B^s) ε_t + β'X_t
```

Dove:
- **y_t**: Serie temporale target al tempo t
- **X_t**: Vettore delle variabili esogene al tempo t (dimensione k×1)
- **β**: Vettore dei coefficienti delle variabili esogene (dimensione k×1)
- **ε_t**: Termine di errore (rumore bianco) ~ N(0, σ²)

### Operatori Polinomiali
- **φ(B)**: Polinomio autoregressivo non stagionale di grado p
- **θ(B)**: Polinomio media mobile non stagionale di grado q
- **Φ(B^s)**: Polinomio autoregressivo stagionale di grado P
- **Θ(B^s)**: Polinomiale media mobile stagionale di grado Q
- **B**: Operatore di ritardo (backward shift)
- **(1-B)^d**: Operatore di differenziazione non stagionale
- **(1-B^s)^D**: Operatore di differenziazione stagionale

### Notazione Compatta
**SARIMAX(p,d,q)(P,D,Q)_s** con k variabili esogene

---

## Componenti del Modello

### 1. Componente SARIMA
La parte SARIMA del modello gestisce la struttura interna della serie:

- **AR(p)**: Dipendenza dai valori passati
- **I(d)**: Integrazione per stazionarietà
- **MA(q)**: Dipendenza dagli errori passati
- **Stagionalità (P,D,Q,s)**: Pattern ricorrenti

### 2. Componente Esogena
La componente esogena modella l'influenza di fattori esterni:

```
β'X_t = β₁X₁,t + β₂X₂,t + ... + βₖXₖ,t
```

Dove:
- **βᵢ**: Coefficiente della i-esima variabile esogena
- **Xᵢ,t**: Valore della i-esima variabile esogena al tempo t
- **k**: Numero totale di variabili esogene

### Interpretazione dei Coefficienti
- **βᵢ > 0**: La variabile esogena ha effetto positivo sulla serie target
- **βᵢ < 0**: La variabile esogena ha effetto negativo sulla serie target
- **βᵢ = 0**: La variabile esogena non ha effetto significativo

---

## Variabili Esogene

### Tipologie Comuni
1. **Economiche**: PIL, inflazione, tassi di interesse, indici di borsa
2. **Meteorologiche**: Temperatura, precipitazioni, umidità, pressione
3. **Marketing**: Spesa pubblicitaria, promozioni, eventi speciali
4. **Demografiche**: Popolazione, reddito medio, età media
5. **Competitive**: Prezzi concorrenti, quote di mercato
6. **Temporali**: Giorni feriali, festività, eventi stagionali
7. **Tecnologiche**: Adozione di nuove tecnologie, innovazioni

### Caratteristiche Richieste
Le variabili esogene devono soddisfare alcuni requisiti:

#### Stazionarietà
- Devono essere stazionarie o rese tali
- Stesso grado di integrazione della serie target

#### Correlazione
- Correlazione significativa con la serie target
- Non perfetta multicollinearità tra loro

#### Disponibilità Temporale
- **Training**: Valori storici allineati con la serie target
- **Forecasting**: Valori futuri noti o prevedibili

#### Qualità dei Dati
- Assenza di valori mancanti o gestione appropriata
- Coerenza temporale e metodologica

### Selezione delle Variabili
La selezione delle variabili esogene può seguire diversi approcci:

#### Approccio Teorico
Basato sulla conoscenza del dominio:
```python
# Esempio: Vendite di gelati
variabili_teoriche = [
    'temperatura_media',      # Correlazione fisica ovvia
    'giorni_weekend',         # Comportamento consumatori
    'vacanze_scolastiche',    # Stagionalità aggiuntiva
    'spesa_pubblicitaria'     # Effetto marketing
]
```

#### Approccio Statistico
Basato su correlazione e significatività:
```python
# Test di correlazione
correlazioni = serie_target.corr(variabili_candidate)
significative = correlazioni[abs(correlazioni) > soglia]

# Test di causalità di Granger
from statsmodels.tsa.stattools import grangercausalitytests
risultati_granger = grangercausalitytests(dati_combinati, max_lag)
```

#### Approccio Misto
Combinazione di teoria e statistica per selezione ottimale.

---

## Stima dei Parametri

### Metodo di Massima Verosimiglianza
Il SARIMAX utilizza tipicamente la **Maximum Likelihood Estimation (MLE)**:

#### Funzione di Verosimiglianza
```
L(φ, θ, Φ, Θ, β, σ²) = ∏ᵗ₌₁ⁿ (1/√(2πσ²)) exp(-ε²ₜ/(2σ²))
```

#### Procedura Iterativa
1. **Inizializzazione**: Valori iniziali per i parametri
2. **Ottimizzazione**: Minimizzazione della log-verosimiglianza negativa
3. **Convergenza**: Criteri di arresto basati su tolleranze

### Algoritmi di Ottimizzazione
- **Quasi-Newton**: BFGS, L-BFGS
- **Nelder-Mead**: Per problemi non differenziabili
- **Powell**: Ottimizzazione multivariata

### Trattamento delle Variabili Esogene
Le variabili esogene possono essere trattate in diversi modi:

#### Contemporanee
```
y_t = ... + β₁X₁,t + β₂X₂,t + ... + εₑ
```

#### Ritardate
```
y_t = ... + β₁X₁,t₋₁ + β₂X₂,t₋₂ + ... + εₑ
```

#### Distribuite nel Tempo
```
y_t = ... + ∑ᵢ₌₀ʰ βᵢXₜ₋ᵢ + εₑ
```

---

## Diagnostica e Validazione

### Test sui Residui
I residui del SARIMAX devono soddisfare le assunzioni standard:

#### Normalità
```python
# Test di Jarque-Bera
from scipy.stats import jarque_bera
stat, p_value = jarque_bera(residui)
```

#### Incorrelazione
```python
# Test di Ljung-Box
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_stat, lb_p = acorr_ljungbox(residui, lags=10)
```

#### Omoschedasticità
```python
# Test di Breusch-Pagan
from statsmodels.stats.diagnostic import het_breuschpagan
lm_stat, lm_p, f_stat, f_p = het_breuschpagan(residui, exog)
```

### Test sulle Variabili Esogene

#### Significatività Individuale
Test t per ogni coefficiente βᵢ:
```
H₀: βᵢ = 0
H₁: βᵢ ≠ 0

t = βᵢ / se(βᵢ) ~ t(n-k-p-q-P-Q)
```

#### Significatività Congiunta
Test F per tutti i coefficienti esogeni:
```
H₀: β₁ = β₂ = ... = βₖ = 0
H₁: Almeno un βᵢ ≠ 0

F = (RSS₀ - RSS₁)/k / (RSS₁/(n-k-p-q-P-Q))
```

#### Stabilità dei Parametri
Test di stabilità strutturale (Chow test, CUSUM).

### Criteri di Selezione Modello

#### Information Criteria
- **AIC**: Akaike Information Criterion
- **BIC**: Bayesian Information Criterion  
- **HQIC**: Hannan-Quinn Information Criterion

Con correzione per variabili esogene:
```
AIC = 2k - 2ln(L)
BIC = k*ln(n) - 2ln(L)
```
Dove k include anche i parametri β delle variabili esogene.

#### Cross-Validation
Validazione temporale per series temporali:
```python
# Time Series Split
for train_idx, test_idx in time_series_split:
    model.fit(train_data, train_exog)
    predictions = model.forecast(test_data, test_exog)
    errors.append(calculate_metrics(test_data, predictions))
```

---

## Forecasting con SARIMAX

### Requisiti per il Forecasting
Il forecasting SARIMAX richiede:

1. **Modello Stimato**: Parametri φ, θ, Φ, Θ, β, σ² stimati
2. **Valori Futuri**: Valori delle variabili esogene per i periodi da prevedere
3. **Condizioni Iniziali**: Valori passati necessari per l'autoreggressione

### Previsione Puntuale
La previsione h-step-ahead è:
```
ŷₜ₊ₕ|ₜ = E[yₜ₊ₕ | Iₜ, Xₜ₊ₕ]
```

Dove:
- **Iₜ**: Insieme informativo al tempo t
- **Xₜ₊ₕ**: Vettore variabili esogene al tempo t+h

### Previsione per Intervalli
Per h = 1, 2, ..., H:
```
ŷₜ₊ₕ|ₜ = φ₁ŷₜ₊ₕ₋₁|ₜ + ... + φₚŷₜ₊ₕ₋ₚ|ₜ + 
         β'Xₜ₊ₕ + 
         θ₁ε̂ₜ₊ₕ₋₁|ₜ + ... + θₑε̂ₜ₊ₕ₋ₑ|ₜ +
         Φ₁ŷₜ₊ₕ₋ₛ|ₜ + ... + Θ₁ε̂ₜ₊ₕ₋ₛ|ₜ
```

### Intervalli di Confidenza
La varianza dell'errore di previsione h-step-ahead:
```
Var(eₜ₊ₕ|ₜ) = σ² * ψₕ²
```

Intervallo di confidenza al livello (1-α):
```
ŷₜ₊ₕ|ₜ ± z_{α/2} * σ * ψₕ
```

### Scenari per Variabili Esogene
Quando i valori futuri delle variabili esogene sono incerti:

#### Scenario Deterministic
Valori futuri noti con certezza (raro).

#### Scenario Forecasted
```python
# Previsioni delle variabili esogene
exog_forecast = forecast_exogenous_variables(exog_history, steps=h)
sarimax_forecast = sarimax_model.forecast(steps=h, exog=exog_forecast)
```

#### Scenario Analysis
```python
scenarios = {
    'optimistic': exog_high_values,
    'baseline': exog_expected_values,
    'pessimistic': exog_low_values
}

forecasts = {}
for scenario, exog_values in scenarios.items():
    forecasts[scenario] = sarimax_model.forecast(steps=h, exog=exog_values)
```

---

## Vantaggi e Limitazioni

### Vantaggi

#### Maggiore Accuratezza
- Incorpora informazioni esterne rilevanti
- Riduce errori di previsione quando variabili esogene sono informative
- Migliora previsioni a lungo termine

#### Interpretabilità
- Coefficienti β forniscono insight sui fattori di influenza
- Quantifica l'impatto di variabili specifiche
- Supporta decision making basato sui dati

#### Flessibilità
- Può incorporare diversi tipi di variabili
- Adattabile a vari domini applicativi
- Gestisce sia effetti contemporanei che ritardati

#### Robustezza
- Meno sensibile a shock specifici della serie
- Migliore performance in presenza di cambiamenti strutturali
- Validazione tramite variabili indipendenti

### Limitazioni

#### Complessità Modello
- Maggior numero di parametri da stimare
- Rischio di overfitting con troppe variabili
- Difficoltà di interpretazione con molte variabili

#### Requisiti Dati
- Necessita variabili esogene di qualità
- Richiede allineamento temporale perfetto
- Valori futuri delle esogene devono essere noti/prevedibili

#### Problemi di Stima
- Convergenza più difficile con molti parametri
- Multicollinearità tra variabili esogene
- Instabilità numerica con matrici mal condizionate

#### Assunzioni Restrittive
- Linearità delle relazioni
- Stazionarietà delle variabili esogene
- Assenza di endogeneità

---

## Esempi Pratici

### Esempio 1: Vendite Retail con Fattori Esterni

#### Contesto
Previsione vendite mensili di un retailer considerando:
- Temperatura media mensile
- Spesa pubblicitaria
- Indice di fiducia dei consumatori
- Giorni festivi

#### Modello
```
SARIMAX(1,1,1)(1,1,1)₁₂ con 4 variabili esogene
```

#### Implementazione
```python
import pandas as pd
from arima_forecaster import SARIMAXForecaster

# Dati
vendite = pd.read_csv('vendite_mensili.csv', index_col='data', parse_dates=True)
variabili_esogene = pd.DataFrame({
    'temperatura': temperatura_data,
    'advertising': advertising_spend,
    'consumer_confidence': confidence_index,
    'holidays': holiday_indicator
})

# Modello
model = SARIMAXForecaster(
    order=(1,1,1),
    seasonal_order=(1,1,1,12),
    exog_names=['temperatura', 'advertising', 'consumer_confidence', 'holidays']
)

# Training
model.fit(vendite['vendite'], exog=variabili_esogene)

# Risultati
print("Coefficienti variabili esogene:")
importance = model.get_exog_importance()
for _, row in importance.iterrows():
    print(f"  {row['variable']}: {row['coefficient']:.4f} (p={row['pvalue']:.4f})")
```

#### Interpretazione
- **Temperatura**: +0.0234 (p=0.001) → +1°C aumenta vendite di 23.4 unità
- **Advertising**: +0.0015 (p=0.012) → +1000€ spesa aumenta vendite di 1.5 unità
- **Consumer Confidence**: +0.0567 (p=0.000) → +1 punto indice aumenta vendite di 56.7 unità
- **Holidays**: +234.5 (p=0.000) → Giorni festivi aumentano vendite di 234.5 unità

### Esempio 2: Domanda Energetica con Variabili Meteorologiche

#### Contesto
Previsione domanda elettrica oraria considerando:
- Temperatura
- Umidità relativa
- Velocità del vento
- Radiazione solare

#### Modello
```
SARIMAX(2,1,2)(1,1,1)₂₄ con 4 variabili esogene
```

#### Considerazioni Speciali
- Stagionalità giornaliera (s=24)
- Variabili meteorologiche altamente correlate
- Effetti non lineari (es: temperatura²)

```python
# Preprocessing con effetti non lineari
variabili_esogene['temperatura_sq'] = variabili_esogene['temperatura']**2
variabili_esogene['temp_humidity'] = (variabili_esogene['temperatura'] * 
                                     variabili_esogene['umidita'])

# Modello con interazioni
model = SARIMAXForecaster(
    order=(2,1,2),
    seasonal_order=(1,1,1,24),
    exog_names=['temperatura', 'temperatura_sq', 'umidita', 'vento', 
               'radiazione', 'temp_humidity']
)
```

### Esempio 3: Serie Finanziaria con Indicatori Economici

#### Contesto
Previsione prezzo azioni considerando:
- Indice di mercato generale
- Tasso di interesse
- Tasso di cambio USD/EUR
- Volatilità VIX

#### Modello
```
SARIMAX(1,1,1)(0,0,0)₀ con 4 variabili esogene
```

#### Particolarità
- Nessuna stagionalità (s=0)  
- Tutte le variabili potrebbero essere non-stazionarie
- Possibili relazioni di cointegrazione

```python
# Test di stazionarietà
from statsmodels.tsa.stattools import adfuller

for col in variabili_esogene.columns:
    adf_stat, p_val = adfuller(variabili_esogene[col])
    if p_val > 0.05:
        # Rendi stazionaria
        variabili_esogene[col] = variabili_esogene[col].diff().dropna()
```

---

## Conclusioni

Il modello SARIMAX rappresenta uno strumento potente e flessibile per il forecasting di serie temporali quando si dispone di informazioni aggiuntive sotto forma di variabili esogene. La sua capacità di incorporare fattori esterni lo rende particolarmente utile in applicazioni business dove decisioni e eventi esterni influenzano significativamente la serie target.

### Punti Chiave per il Successo
1. **Selezione accurata** delle variabili esogene basata su teoria e statistica
2. **Validazione rigorosa** del modello e dei suoi componenti
3. **Gestione appropriata** della disponibilità futura delle variabili esogene
4. **Monitoraggio continuo** della stabilità dei parametri nel tempo

Il SARIMAX trova applicazione ottimale in contesti dove:
- Esistono chiari fattori esterni di influenza
- Le variabili esogene sono di buona qualità e disponibili
- Si richiede interpretabilità dei risultati
- Le previsioni devono incorporare scenari alternativi

Tuttavia, la sua complessità richiede competenze tecniche adeguate e un approccio metodologico rigoroso per ottenere risultati affidabili e interpretabili.