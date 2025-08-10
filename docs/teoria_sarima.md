# Teoria dei Modelli SARIMA

## Introduzione

I modelli SARIMA (Seasonal AutoRegressive Integrated Moving Average) rappresentano un'estensione naturale dei modelli ARIMA per gestire la stagionalità presente nelle serie temporali. Mentre i modelli ARIMA sono efficaci per catturare le dipendenze temporali a breve termine, i modelli SARIMA aggiungono la capacità di modellare pattern stagionali ricorrenti.

## Definizione Formale

Un modello SARIMA è denotato come SARIMA(p,d,q)(P,D,Q)_s, dove:

- **(p,d,q)**: parametri non stagionali (identici ad ARIMA)
  - **p**: ordine autoregressivo
  - **d**: grado di differenziazione
  - **q**: ordine della media mobile

- **(P,D,Q)_s**: parametri stagionali
  - **P**: ordine autoregressivo stagionale
  - **D**: grado di differenziazione stagionale
  - **Q**: ordine della media mobile stagionale
  - **s**: periodo stagionale (es. 12 per dati mensili, 4 per dati trimestrali)

## Struttura Matematica

### Forma Generale del Modello SARIMA

L'equazione generale di un modello SARIMA può essere scritta come:

```
Φ(B^s) φ(B) (1-B)^d (1-B^s)^D X_t = Θ(B^s) θ(B) ε_t
```

Dove:
- **B**: operatore di ritardo (backshift operator)
- **φ(B)**: polinomio autoregressivo non stagionale di ordine p
- **Φ(B^s)**: polinomio autoregressivo stagionale di ordine P
- **θ(B)**: polinomio della media mobile non stagionale di ordine q  
- **Θ(B^s)**: polinomio della media mobile stagionale di ordine Q
- **ε_t**: termine di errore (rumore bianco)

### Componenti Dettagliate

#### 1. Polinomi Autorevessivi

**Non stagionale:**
```
φ(B) = 1 - φ₁B - φ₂B² - ... - φₚBᵖ
```

**Stagionale:**
```
Φ(B^s) = 1 - Φ₁B^s - Φ₂B^{2s} - ... - ΦₚB^{Ps}
```

#### 2. Polinomi della Media Mobile

**Non stagionale:**
```
θ(B) = 1 + θ₁B + θ₂B² + ... + θₚBᵈ
```

**Stagionale:**
```
Θ(B^s) = 1 + Θ₁B^s + Θ₂B^{2s} + ... + ΘₚB^{Qs}
```

#### 3. Operatori di Differenziazione

**Non stagionale:** (1-B)^d
**Stagionale:** (1-B^s)^D

## Esempi di Modelli SARIMA Comuni

### SARIMA(1,1,1)(1,1,1)₁₂

Questo è un modello molto utilizzato per dati mensili:

```
(1-Φ₁B¹²)(1-φ₁B)(1-B)(1-B¹²)X_t = (1+Θ₁B¹²)(1+θ₁B)ε_t
```

Espandendo:
```
X_t = (1+φ₁)X_{t-1} - φ₁X_{t-2} + (1+Φ₁)X_{t-12} - (1+φ₁)Φ₁X_{t-13} + φ₁Φ₁X_{t-14} - Φ₁X_{t-24} + φ₁Φ₁X_{t-25} - φ₁Φ₁X_{t-26} + ε_t + θ₁ε_{t-1} + Θ₁ε_{t-12} + θ₁Θ₁ε_{t-13}
```

### SARIMA(2,1,0)(0,1,1)₄

Modello per dati trimestrali con trend e stagionalità:

```
(1-B)(1-B⁴)X_t = (1-φ₁B-φ₂B²)(1+Θ₁B⁴)ε_t
```

## Interpretazione dei Parametri

### Parametri Autorevessivi

- **φᵢ**: catturano la dipendenza da valori passati recenti
- **Φᵢ**: catturano la dipendenza da valori dello stesso periodo stagionale degli anni precedenti

### Parametri della Media Mobile  

- **θᵢ**: modellano l'impatto di shock passati recenti
- **Θᵢ**: modellano l'impatto di shock stagionali passati

### Condizioni di Stazionarietà e Invertibilità

Per un modello SARIMA stabile:

1. **Stazionarietà**: Le radici dei polinomi φ(B) e Φ(B^s) devono essere esterne al cerchio unitario
2. **Invertibilità**: Le radici dei polinomi θ(B) e Θ(B^s) devono essere esterne al cerchio unitario

## Identificazione del Modello

### 1. Analisi della Stagionalità

Prima di stimare un modello SARIMA, è cruciale identificare:

- **Periodo stagionale (s)**: numero di osservazioni in un ciclo completo
- **Intensità della stagionalità**: quanto forte è il pattern stagionale
- **Tipo di stagionalità**: additiva o moltiplicativa

### 2. Test di Stazionarietà

**Test di Dickey-Fuller Aumentato (ADF):**
- H₀: la serie ha una radice unitaria (non stazionaria)
- H₁: la serie è stazionaria

**Test KPSS:**
- H₀: la serie è stazionaria
- H₁: la serie ha una radice unitaria

### 3. Funzioni di Autocorrelazione

#### ACF (Autocorrelation Function)
```
ρ(k) = Corr(X_t, X_{t-k}) = Cov(X_t, X_{t-k})/Var(X_t)
```

#### PACF (Partial Autocorrelation Function)
```
α(k) = Corr(X_t, X_{t-k} | X_{t-1}, X_{t-2}, ..., X_{t-k+1})
```

### 4. Identificazione dei Parametri

| Comportamento ACF/PACF | Modello Suggerito |
|------------------------|-------------------|
| ACF taglia al lag q, PACF decade | MA(q) |
| PACF taglia al lag p, ACF decade | AR(p) |
| Entrambe decadono | ARMA(p,q) |
| Pattern ripetuti ogni s osservazioni | Componente stagionale |

## Stima dei Parametri

### Metodo della Massima Verosimiglianza

Per un modello SARIMA, la funzione di log-verosimiglianza è:

```
L(θ) = -n/2 * log(2π) - 1/2 * log|Σ| - 1/2 * ε'Σ⁻¹ε
```

Dove:
- **θ**: vettore dei parametri da stimare
- **Σ**: matrice di covarianza degli errori
- **ε**: vettore degli residui

### Algoritmi di Ottimizzazione

1. **Newton-Raphson**: utilizza derivate prime e seconde
2. **Quasi-Newton (BFGS)**: approssima la matrice Hessiana
3. **Levenberg-Marquardt**: combinazione gradient descent e Gauss-Newton

## Diagnostica del Modello

### 1. Analisi dei Residui

I residui di un buon modello SARIMA dovrebbero essere:

- **Incorrelati**: Ljung-Box test
- **Omoschedastici**: test ARCH
- **Normalmente distribuiti**: test Jarque-Bera

### 2. Test di Ljung-Box

```
Q_LB = n(n+2) * Σᵏᵢ₌₁ [ρ²(i)/(n-i)]
```

Sotto H₀ (residui incorrelati): Q_LB ~ χ²(h-p-q)

### 3. Criteri di Selezione

**AIC (Akaike Information Criterion):**
```
AIC = -2*log(L) + 2*k
```

**BIC (Bayesian Information Criterion):**  
```
BIC = -2*log(L) + k*log(n)
```

**HQIC (Hannan-Quinn Information Criterion):**
```
HQIC = -2*log(L) + 2*k*log(log(n))
```

Dove:
- **L**: likelihood del modello
- **k**: numero di parametri
- **n**: numero di osservazioni

## Forecasting con Modelli SARIMA

### Previsioni Puntuali

Per un modello SARIMA(p,d,q)(P,D,Q)_s, la previsione ad h periodi è:

```
X̂_{T+h|T} = E[X_{T+h} | X_T, X_{T-1}, ...]
```

### Intervalli di Confidenza

L'errore di previsione ad h periodi ha varianza:

```
Var[e_{T+h|T}] = σ² * Σʰ⁻¹ⱼ₌₀ ψ²ⱼ
```

Dove ψⱼ sono i coefficienti della rappresentazione MA infinita.

L'intervallo di confidenza al (1-α)% è:

```
X̂_{T+h|T} ± z_{α/2} * √Var[e_{T+h|T}]
```

## Applicazioni Pratiche

### Dati Economici e Finanziari

- **PIL trimestrale**: SARIMA con s=4
- **Vendite mensili al dettaglio**: SARIMA con s=12  
- **Turismo**: stagionalità annuale e possibilmente settimanale

### Dati Ambientali

- **Temperature**: stagionalità annuale (s=12 per dati mensili)
- **Precipitazioni**: pattern stagionali complessi
- **Qualità dell'aria**: stagionalità settimanale e annuale

### Controllo della Produzione

- **Domanda di energia**: stagionalità giornaliera, settimanale, annuale
- **Inventari**: cicli di produzione stagionali

## Limitazioni e Considerazioni

### 1. Complessità Computazionale

La stima di modelli SARIMA è più complessa rispetto ad ARIMA:
- Maggior numero di parametri da stimare
- Superfici di likelihood più complesse
- Rischio di minimi locali nell'ottimizzazione

### 2. Overfitting

Con molti parametri, c'è rischio di sovradattamento:
- Usare criteri di informazione penalizzanti (BIC)
- Validazione incrociata temporale
- Test out-of-sample

### 3. Assunzioni del Modello

- **Linearità**: relazioni lineari tra variabili
- **Stazionarietà**: dopo differenziazione
- **Stagionalità stabile**: pattern stagionali costanti nel tempo

## Estensioni Avanzate

### SARIMA con Variabili Esogene (SARIMAX)

Include variabili esplicative esterne:

```
Φ(B^s) φ(B) (1-B)^d (1-B^s)^D X_t = β'Z_t + Θ(B^s) θ(B) ε_t
```

### SARIMA con Varianza Tempo-Variante

Combinazione con modelli GARCH per eteroschedasticità:

```
X_t = μ_t + ε_t
ε_t = σ_t * z_t
σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
```

### Modelli SARIMA Multivariati (VARMA)

Estensione per sistemi di serie temporali:

```
Φ(B) (1-B)^d X_t = Θ(B) ε_t + μ
```

Dove X_t e ε_t sono vettori.

## Software e Implementazione

### Librerie Python

1. **statsmodels.tsa.statespace.sarimax**: implementazione completa
2. **pmdarima**: selezione automatica di parametri
3. **arima-forecaster**: questa libreria con funzionalità avanzate

### Esempio di Codice

```python
from arima_forecaster import SARIMAForecaster, SARIMAModelSelector
import pandas as pd

# Caricamento dati
data = pd.read_csv('vendite_mensili.csv', index_col='data', parse_dates=True)

# Selezione automatica del modello
selector = SARIMAModelSelector(
    p_range=(0, 3),
    d_range=(0, 2), 
    q_range=(0, 3),
    P_range=(0, 2),
    D_range=(0, 1),
    Q_range=(0, 2),
    seasonal_periods=[12]
)

selector.search(data['vendite'])
best_model = selector.get_best_model()

# Previsioni
forecast = best_model.forecast(steps=12, confidence_intervals=True)
```

## Conclusioni

I modelli SARIMA rappresentano uno strumento potente e versatile per l'analisi e la previsione di serie temporali con componenti stagionali. La loro flessibilità permette di catturare pattern complessi, ma richiede un'attenta fase di identificazione e validazione del modello.

Le considerazioni chiave per un uso efficace includono:

1. **Identificazione accurata** della stagionalità
2. **Bilanciamento** tra complessità del modello e capacità predittiva
3. **Validazione rigorosa** attraverso analisi dei residui
4. **Interpretazione economica** dei parametri stimati

Con l'implementazione di algoritmi avanzati di ottimizzazione automatica, l'uso pratico dei modelli SARIMA diventa più accessibile mantenendo il rigore metodologico necessario per applicazioni professionali.