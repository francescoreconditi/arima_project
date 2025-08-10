# Modelli ARIMA: Guida Teorica Completa

## Indice
1. [Introduzione](#introduzione)
2. [Fondamenti Matematici](#fondamenti-matematici)
3. [Componenti ARIMA](#componenti-arima)
4. [Specifica del Modello](#specifica-del-modello)
5. [Stima dei Parametri](#stima-dei-parametri)
6. [Diagnostica del Modello](#diagnostica-del-modello)
7. [Forecasting](#forecasting)
8. [Selezione del Modello](#selezione-del-modello)
9. [Vantaggi e Limitazioni](#vantaggi-e-limitazioni)
10. [Argomenti Avanzati](#argomenti-avanzati)

## Introduzione

I modelli ARIMA (Autoregressive Integrated Moving Average) sono una classe di modelli statistici per l'analisi e il forecasting di dati di serie temporali. I modelli ARIMA sono particolarmente efficaci per serie temporali univariate che mostrano dipendenza temporale e possono essere rese stazionarie attraverso la differenziazione.

### Contesto Storico

I modelli ARIMA sono stati resi popolari da Box e Jenkins nel loro libro seminale del 1970 "Time Series Analysis: Forecasting and Control". La metodologia Box-Jenkins fornisce un approccio sistematico alla costruzione di modelli ARIMA, consistente nell'identificazione del modello, stima dei parametri e controllo diagnostico.

## Fondamenti Matematici

### Notazione Base

Sia $\{X_t\}$ una serie temporale dove $t = 1, 2, ..., n$. Il modello ARIMA è costruito su tre concetti fondamentali:

1. **Autoregressione (AR)**: I valori attuali dipendono dai valori passati
2. **Integrazione (I)**: La serie deve essere resa stazionaria attraverso la differenziazione
3. **Media Mobile (MA)**: I valori attuali dipendono dagli errori di previsione passati

### Operatore di Ritardo

L'operatore di ritardo (o operatore backshift) $L$ è definito come:
$$L X_t = X_{t-1}$$

Ritardi di ordine superiore: $L^k X_t = X_{t-k}$

L'operatore differenza: $\nabla = (1 - L)$
- Prima differenza: $\nabla X_t = X_t - X_{t-1}$
- Seconda differenza: $\nabla^2 X_t = \nabla(\nabla X_t) = X_t - 2X_{t-1} + X_{t-2}$

## Componenti ARIMA

### Componente Autoregressivo (AR)

Un processo AR(p) è definito come:
$$X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + ... + \phi_p X_{t-p} + \varepsilon_t$$

Dove:
- $\phi_i$ sono i parametri autoregressivi
- $c$ è una costante
- $\varepsilon_t$ è rumore bianco con $E[\varepsilon_t] = 0$ e $Var[\varepsilon_t] = \sigma^2$

**Condizioni di Stazionarietà**: Tutte le radici dell'equazione caratteristica devono giacere fuori dal cerchio unitario:
$$1 - \phi_1 z - \phi_2 z^2 - ... - \phi_p z^p = 0$$

### Componente Media Mobile (MA)

Un processo MA(q) è definito come:
$$X_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + ... + \theta_q \varepsilon_{t-q}$$

Dove:
- $\theta_i$ sono i parametri della media mobile
- $\mu$ è la media del processo

**Condizioni di Invertibilità**: Tutte le radici dell'equazione caratteristica devono giacere fuori dal cerchio unitario:
$$1 + \theta_1 z + \theta_2 z^2 + ... + \theta_q z^q = 0$$

### Componente di Integrazione (I)

L'integrazione di ordine $d$ significa che la serie deve essere differenziata $d$ volte per raggiungere la stazionarietà:
$$Y_t = \nabla^d X_t$$

Dove $Y_t$ è la serie stazionaria ottenuta dopo $d$ differenze.

## Specifica del Modello

### Modello ARIMA(p,d,q)

Il modello ARIMA(p,d,q) generale può essere scritto come:
$$\phi(L)(1-L)^d X_t = \theta(L)\varepsilon_t$$

Dove:
- $\phi(L) = 1 - \phi_1 L - \phi_2 L^2 - ... - \phi_p L^p$ (polinomio AR)
- $\theta(L) = 1 + \theta_1 L + \theta_2 L^2 + ... + \theta_q L^q$ (polinomio MA)
- $(1-L)^d$ è l'operatore di differenziazione di ordine $d$

### Forma Espansa

Per ARIMA(p,d,q), l'equazione espansa è:
$$\nabla^d X_t = c + \phi_1 \nabla^d X_{t-1} + ... + \phi_p \nabla^d X_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + ... + \theta_q \varepsilon_{t-q}$$

### Casi Speciali

- **ARIMA(p,0,0) = AR(p)**: Modello autoregressivo puro
- **ARIMA(0,0,q) = MA(q)**: Modello media mobile puro
- **ARIMA(0,1,0)**: Modello random walk
- **ARIMA(0,1,1)**: Smoothing esponenziale semplice
- **ARIMA(1,1,0)**: Modello autoregressivo del primo ordine su prime differenze

## Stima dei Parametri

### Stima di Massima Verosimiglianza (MLE)

Date le osservazioni $X_1, X_2, ..., X_n$, la funzione di verosimiglianza è:
$$L(\phi, \theta, \sigma^2) = \prod_{t=1}^n f(x_t | x_{t-1}, ..., x_1; \phi, \theta, \sigma^2)$$

La log-verosimiglianza viene massimizzata per ottenere le stime dei parametri.

### Somma dei Quadrati Incondizionata

Per efficienza computazionale, spesso vengono utilizzati metodi approssimati come la somma dei quadrati incondizionata:
$$S(\phi, \theta) = \sum_{t=1}^n \hat{\varepsilon}_t^2$$

Dove $\hat{\varepsilon}_t$ sono i residui stimati.

### Metodo dei Momenti

Per modelli AR, le equazioni di Yule-Walker forniscono stime basate sui momenti:
$$\gamma(k) = \phi_1 \gamma(k-1) + \phi_2 \gamma(k-2) + ... + \phi_p \gamma(k-p)$$

Dove $\gamma(k)$ è l'autocovarianza al ritardo $k$.

## Diagnostica del Modello

### Analisi dei Residui

Dopo aver adattato un modello ARIMA, i residui dovrebbero essere controllati per:

1. **Proprietà di Rumore Bianco**
   - Media: $E[\hat{\varepsilon}_t] = 0$
   - Varianza costante: $Var[\hat{\varepsilon}_t] = \sigma^2$
   - Assenza di autocorrelazione: $Cov[\hat{\varepsilon}_t, \hat{\varepsilon}_{t-k}] = 0$ per $k \neq 0$

2. **Normalità**: I residui dovrebbero essere approssimativamente distribuiti normalmente

### Test Statistici

#### Test di Ljung-Box
Testa per l'autocorrelazione nei residui:
$$Q_{LB} = n(n+2)\sum_{k=1}^h \frac{\hat{\rho}_k^2}{n-k} \sim \chi^2_{h-p-q}$$

Dove $\hat{\rho}_k$ è l'autocorrelazione campionaria dei residui al ritardo $k$.

#### Test di Jarque-Bera
Testa per la normalità dei residui:
$$JB = \frac{n}{6}(S^2 + \frac{(K-3)^2}{4}) \sim \chi^2_2$$

Dove $S$ è la skewness e $K$ è la curtosi.

#### Test ARCH
Testa per l'eteroschedasticità (varianza variabile):
$$LM = n \cdot R^2 \sim \chi^2_q$$

Dove $R^2$ è dalla regressione dei residui quadrati sui residui quadrati ritardati.

## Forecasting

### Previsioni Puntuali

Per ARIMA(p,d,q), la previsione h-passi avanti è:
$$\hat{X}_{n+h|n} = E[X_{n+h} | X_n, X_{n-1}, ...]$$

### Errore di Previsione

L'errore di previsione è:
$$e_{n+h} = X_{n+h} - \hat{X}_{n+h|n}$$

### Intervalli di Predizione

L'intervallo di predizione $(1-\alpha)\%$ per la previsione h-passi avanti:
$$\hat{X}_{n+h|n} \pm z_{\alpha/2} \sqrt{Var[e_{n+h}]}$$

Dove $z_{\alpha/2}$ è il valore critico dalla distribuzione normale standard.

### Varianza di Previsione

Per modelli ARIMA, la varianza di previsione aumenta con l'orizzonte di previsione:
$$Var[e_{n+h}] = \sigma^2 \sum_{j=0}^{h-1} \psi_j^2$$

Dove $\psi_j$ sono i coefficienti dalla rappresentazione MA infinita.

## Selezione del Modello

### Criteri Informativi

#### Criterio di Informazione di Akaike (AIC)
$$AIC = -2 \log L + 2k$$

Dove $L$ è la verosimiglianza e $k$ è il numero di parametri.

#### Criterio di Informazione Bayesiano (BIC)
$$BIC = -2 \log L + k \log n$$

BIC penalizza la complessità del modello più pesantemente di AIC.

#### Criterio di Informazione di Hannan-Quinn (HQIC)
$$HQIC = -2 \log L + 2k \log(\log n)$$

### Validazione Incrociata

La validazione incrociata per serie temporali rispetta l'ordine temporale:
- **Finestra scorrevole**: Dimensione finestra fissa, si muove nel tempo
- **Finestra espandente**: Cresce dal punto iniziale
- **Divisione serie temporale**: Addestra su dati precoci, testa su successivi

### Grid Search

Valutazione sistematica di ARIMA(p,d,q) per:
- $p \in \{0, 1, 2, ..., P_{max}\}$
- $d \in \{0, 1, 2\}$
- $q \in \{0, 1, 2, ..., Q_{max}\}$

## Vantaggi e Limitazioni

### Vantaggi

1. **Fondamento Teorico**: Teoria statistica ben stabilita
2. **Interpretabilità**: I parametri hanno significato chiaro
3. **Flessibilità**: Può gestire vari pattern nelle serie temporali
4. **Intervalli di Predizione**: Fornisce quantificazione dell'incertezza
5. **Parsimonia**: Spesso raggiunge buon adattamento con pochi parametri

### Limitazioni

1. **Requisito di Stazionarietà**: La serie deve essere stazionaria o resa tale
2. **Relazioni Lineari**: Non può catturare pattern non lineari
3. **Univariato**: ARIMA standard gestisce solo singole serie
4. **Selezione Parametri**: Richiede esperienza per specifica modello
5. **Pattern Stagionali**: ARIMA di base fatica con stagionalità complessa

## Argomenti Avanzati

### ARIMA Stagionale (SARIMA)

SARIMA(p,d,q)(P,D,Q)s estende ARIMA per dati stagionali:
$$\phi(L)\Phi(L^s)(1-L)^d(1-L^s)^D X_t = \theta(L)\Theta(L^s)\varepsilon_t$$

Dove:
- $(P,D,Q)$ sono ordini AR, differenziazione e MA stagionali
- $s$ è il periodo stagionale
- $\Phi(L^s)$ e $\Theta(L^s)$ sono polinomi stagionali

### ARIMA-X (ARIMAX)

Include variabili esogene:
$$\phi(L)(1-L)^d X_t = \beta'Z_t + \theta(L)\varepsilon_t$$

Dove $Z_t$ sono variabili esogene e $\beta$ sono i loro coefficienti.

### ARIMA Vettoriale (VARIMA)

Estensione multivariata per sistemi di serie temporali:
$$\Phi(L)(1-L)^d \mathbf{X}_t = \Theta(L)\boldsymbol{\varepsilon}_t$$

Dove $\mathbf{X}_t$ è un vettore di serie temporali.

### Rappresentazione Spazio-Stato

I modelli ARIMA possono essere rappresentati in forma spazio-stato:
$$\mathbf{x}_{t+1} = \mathbf{F}\mathbf{x}_t + \mathbf{G}\varepsilon_{t+1}$$
$$y_t = \mathbf{H}'\mathbf{x}_t$$

Questo abilita il filtro di Kalman per stima parametri e forecasting.

### Media dei Modelli

Invece di selezionare un singolo modello, combina previsioni da multiple specifiche ARIMA:
$$\hat{X}_{n+h} = \sum_{i=1}^M w_i \hat{X}_{n+h}^{(i)}$$

Dove $w_i$ sono i pesi dei modelli e $\hat{X}_{n+h}^{(i)}$ sono le previsioni dei singoli modelli.

## Considerazioni Pratiche

### Preprocessing dei Dati

1. **Valori Mancanti**: Gestire gap nei dati delle serie temporali
2. **Outlier**: Rilevare e trattare osservazioni anomale
3. **Trasformazioni**: Applicare trasformazioni log, Box-Cox o altre
4. **Aggregazione**: Scegliere frequenza temporale appropriata

### Processo di Costruzione del Modello

1. **Esplorazione**: Tracciare dati, identificare pattern
2. **Test Stazionarietà**: Usare test ADF, KPSS
3. **Differenziazione**: Applicare differenze necessarie
4. **Identificazione**: Usare grafici ACF, PACF
5. **Stima**: Adattare modelli candidati
6. **Diagnosi**: Controllare residui
7. **Selezione**: Confrontare modelli usando criteri
8. **Forecasting**: Generare predizioni

### Suggerimenti per l'Implementazione

- Inizia con modelli semplici (ARIMA(1,1,1))
- Usa selezione automatica modello come punto di partenza
- Valida sempre con test fuori-campione
- Considera metodi ensemble per robustezza
- Monitora performance del modello nel tempo
- Ri-stima parametri periodicamente

## Ottimizzazione delle Performance

### Monitoraggio Performance

```python
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
    model = ARIMAForecaster(order=ordine)
    return model.fit(serie)
```

### Suggerimenti per l'Efficienza

- Usa operazioni vettorizzate dove possibile
- Cache risultati preprocessing
- Considera elaborazione parallela per selezione modello
- Profila il tuo codice per identificare colli di bottiglia

## Conclusione

I modelli ARIMA rimangono una pietra miliare dell'analisi delle serie temporali grazie al loro solido fondamento teorico, interpretabilità ed efficacia per molti compiti di forecasting. Mentre metodi più recenti come reti neurali e modelli ensemble possono superare ARIMA in contesti specifici, comprendere i principi ARIMA è essenziale per qualsiasi analista di serie temporali.

La chiave per una modellazione ARIMA di successo risiede nel preprocessing accurato dei dati, costruzione sistematica del modello seguendo la metodologia Box-Jenkins, controllo diagnostico approfondito e validazione su dati fuori-campione. Le implementazioni moderne forniscono strumenti automatizzati per la selezione del modello, ma l'esperienza del dominio e la comprensione statistica rimangono cruciali per ottenere previsioni affidabili.