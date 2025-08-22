# SARIMA vs SARIMAX - Guida Completa alla Scelta del Modello

## Indice
1. [Introduzione](#introduzione)
2. [Confronto Teorico](#confronto-teorico)
3. [Differenze Pratiche](#differenze-pratiche)
4. [Vantaggi e Svantaggi](#vantaggi-e-svantaggi)
5. [Criteri di Scelta](#criteri-di-scelta)
6. [Esempi Comparativi](#esempi-comparativi)
7. [Implementazione in Codice](#implementazione-in-codice)
8. [Linee Guida Pratiche](#linee-guida-pratiche)

---

## Introduzione

La scelta tra **SARIMA** e **SARIMAX** è una decisione fondamentale nella modellazione di serie temporali. Entrambi i modelli appartengono alla famiglia ARIMA ma differiscono significativamente nell'approccio e nelle capacità predittive.

### Definizioni Rapide
- **SARIMA**: Seasonal AutoRegressive Integrated Moving Average
- **SARIMAX**: SARIMA con variabili eXogene (esterne)

La differenza principale sta nella **fonte di informazione**:
- **SARIMA** usa solo la storia della serie temporale
- **SARIMAX** incorpora anche variabili esterne

---

## Confronto Teorico

### Formulazione Matematica

#### SARIMA(p,d,q)(P,D,Q)_s
```
φ(B) Φ(B^s) (1-B)^d (1-B^s)^D y_t = θ(B) Θ(B^s) ε_t
```

#### SARIMAX(p,d,q)(P,D,Q)_s + k variabili esogene
```
φ(B) Φ(B^s) (1-B)^d (1-B^s)^D y_t = θ(B) Θ(B^s) ε_t + β'X_t
```

### Componenti del Modello

| Componente | SARIMA | SARIMAX |
|------------|--------|---------|
| **Autoregressivo (AR)** | ✅ φ(B) | ✅ φ(B) |
| **Integrato (I)** | ✅ (1-B)^d | ✅ (1-B)^d |
| **Media Mobile (MA)** | ✅ θ(B) | ✅ θ(B) |
| **AR Stagionale** | ✅ Φ(B^s) | ✅ Φ(B^s) |
| **I Stagionale** | ✅ (1-B^s)^D | ✅ (1-B^s)^D |
| **MA Stagionale** | ✅ Θ(B^s) | ✅ Θ(B^s) |
| **Variabili Esogene** | ❌ | ✅ β'X_t |

### Parametri da Stimare

#### SARIMA
- **p + q** parametri non stagionali
- **P + Q** parametri stagionali  
- **1** parametro di varianza (σ²)
- **Totale**: p + q + P + Q + 1

#### SARIMAX
- Tutti i parametri SARIMA +
- **k** coefficienti per variabili esogene (β₁, β₂, ..., βₖ)
- **Totale**: p + q + P + Q + k + 1

---

## Differenze Pratiche

### Requisiti Dati

#### SARIMA
```python
# Solo serie temporale target
dati_sarima = {
    'serie_target': pd.Series([100, 105, 98, 112, ...])
}
```

#### SARIMAX
```python
# Serie target + variabili esogene
dati_sarimax = {
    'serie_target': pd.Series([100, 105, 98, 112, ...]),
    'variabili_esogene': pd.DataFrame({
        'temperatura': [20.5, 22.1, 19.8, 21.4, ...],
        'marketing': [1000, 1200, 800, 1500, ...],
        'concorrenti': [85, 87, 82, 89, ...]
    })
}
```

### Processo di Modellazione

| Fase | SARIMA | SARIMAX |
|------|--------|---------|
| **Data Collection** | Solo serie target | Serie target + variabili esogene |
| **Preprocessing** | Stazionarietà serie | Stazionarietà serie + validazione esogene |
| **Model Selection** | Parametri (p,d,q)(P,D,Q,s) | Parametri SARIMA + selezione variabili esogene |
| **Parameter Estimation** | MLE su parametri SARIMA | MLE su parametri SARIMA + coefficienti β |
| **Diagnostics** | Residui + parametri | Residui + parametri + significatività esogene |
| **Forecasting** | Solo valori passati | Valori passati + valori futuri esogene |

### Complessità Computazionale

#### SARIMA
- **Tempo di stima**: O(n × p_max × q_max × P_max × Q_max)
- **Memoria**: Lineare con lunghezza serie
- **Convergenza**: Generalmente robusta

#### SARIMAX  
- **Tempo di stima**: O(n × p_max × q_max × P_max × Q_max × k)
- **Memoria**: Lineare con serie + k variabili
- **Convergenza**: Più sensibile a multicollinearità

---

## Vantaggi e Svantaggi

### SARIMA

#### ✅ Vantaggi
- **Semplicità**: Meno dati richiesti, implementazione più semplice
- **Robustezza**: Meno assunzioni, minor rischio overfitting
- **Interpretabilità**: Focus sui pattern interni della serie
- **Disponibilità Dati**: Non richiede variabili esterne
- **Stabilità**: Parametri più stabili nel tempo
- **Velocità**: Training e forecasting più veloci

#### ❌ Svantaggi
- **Informazione Limitata**: Ignora fattori esterni rilevanti
- **Previsioni a Lungo Termine**: Tendono alla media storica
- **Eventi Esterni**: Non gestisce shock o cambiamenti del contesto
- **Accuratezza**: Potenzialmente inferiore quando esistono variabili esogene informative

### SARIMAX

#### ✅ Vantaggi
- **Maggiore Informazione**: Incorpora fattori esterni rilevanti
- **Accuratezza**: Potenzialmente superiore con variabili informative
- **Interpretabilità**: Quantifica effetto di fattori specifici
- **Scenario Analysis**: Supporta analisi di scenari alternativi
- **Business Insight**: Fornisce insight sui driver del business
- **Flessibilità**: Adattabile a diversi contesti applicativi

#### ❌ Svantaggi
- **Complessità**: Più dati, più assunzioni, più parametri
- **Disponibilità Futura**: Richiede conoscenza valori futuri esogene
- **Overfitting**: Rischio maggiore con molte variabili
- **Multicollinearità**: Problemi con variabili correlate
- **Instabilità**: Parametri meno stabili, model decay più veloce

---

## Criteri di Scelta

### Matrice Decisionale

| Criterio | Favorisce SARIMA | Favorisce SARIMAX |
|----------|------------------|-------------------|
| **Disponibilità Variabili Esogene** | Scarse o non disponibili | Abbondanti e di qualità |
| **Conoscenza del Dominio** | Limitata | Approfondita |
| **Orizzonte Previsionale** | Breve termine | Medio-lungo termine |
| **Complessità Accettabile** | Bassa | Alta |
| **Risorse Computazionali** | Limitate | Abbondanti |
| **Frequenza Update Modello** | Alta | Bassa |
| **Tolleranza al Rischio** | Bassa | Alta |
| **Interpretabilità Richiesta** | Pattern temporali | Relazioni causali |

### Decision Tree Guidata

```
START
│
├─ Hai variabili esogene di qualità?
│  ├─ NO → SARIMA
│  └─ SÌ ↓
│
├─ Puoi ottenere valori futuri delle variabili esogene?
│  ├─ NO → SARIMA  
│  └─ SÌ ↓
│
├─ Le variabili esogene sono correlate con la serie target?
│  ├─ NO → SARIMA
│  └─ SÌ ↓
│
├─ Hai competenze per gestire la complessità aggiuntiva?
│  ├─ NO → SARIMA
│  └─ SÌ ↓
│
└─ SARIMAX (con validazione comparativa)
```

### Test di Validazione

#### 1. Information Criteria Comparison
```python
# Confronta AIC/BIC tra modelli
sarima_aic = sarima_model.get_model_info()['aic']
sarimax_aic = sarimax_model.get_model_info()['aic']

if sarimax_aic < sarima_aic - 2:  # Differenza significativa
    print("SARIMAX preferibile per AIC")
else:
    print("SARIMA competitivo")
```

#### 2. Out-of-Sample Performance
```python
# Confronto performance su test set
sarima_mae = evaluate_model(sarima_model, test_data)['mae']
sarimax_mae = evaluate_model(sarimax_model, test_data, test_exog)['mae']

improvement = (sarima_mae - sarimax_mae) / sarima_mae * 100
print(f"Miglioramento SARIMAX: {improvement:.1f}%")
```

#### 3. Significatività Variabili Esogene
```python
# Test significatività congiunta
exog_importance = sarimax_model.get_exog_importance()
significant_vars = exog_importance[exog_importance['significant']]['variable'].tolist()

if len(significant_vars) == 0:
    print("Nessuna variabile esogena significativa → SARIMA")
else:
    print(f"Variabili significative: {significant_vars} → SARIMAX")
```

---

## Esempi Comparativi

### Caso 1: Vendite Retail

#### Contesto
- **Serie target**: Vendite mensili
- **Variabili disponibili**: Temperatura, spesa pubblicitaria, indice economico
- **Orizzonte**: 12 mesi

#### Analisi SARIMA
```python
# Modello SARIMA puro
sarima = SARIMAForecaster(order=(1,1,1), seasonal_order=(1,1,1,12))
sarima.fit(vendite_serie)

# Risultati
print(f"SARIMA AIC: {sarima.get_model_info()['aic']:.2f}")
forecast_sarima = sarima.forecast(steps=12)
```

#### Analisi SARIMAX
```python
# Modello con variabili esogene
sarimax = SARIMAXForecaster(
    order=(1,1,1), 
    seasonal_order=(1,1,1,12),
    exog_names=['temperatura', 'advertising', 'economic_index']
)
sarimax.fit(vendite_serie, exog=variabili_esogene)

# Risultati
print(f"SARIMAX AIC: {sarimax.get_model_info()['aic']:.2f}")
print("\nSignificatività variabili:")
for _, row in sarimax.get_exog_importance().iterrows():
    print(f"  {row['variable']}: p-value = {row['pvalue']:.4f}")
```

#### Confronto Risultati
```
SARIMA AIC: 1247.83
SARIMAX AIC: 1198.45  ← Miglioramento significativo

Significatività variabili:
  temperatura: p-value = 0.0001  ← Molto significativa  
  advertising: p-value = 0.0234  ← Significativa
  economic_index: p-value = 0.1456  ← Non significativa

Conclusione: SARIMAX preferibile, rimuovere economic_index
```

### Caso 2: Domanda Energetica

#### Contesto
- **Serie target**: Consumo elettrico orario
- **Variabili disponibili**: Temperatura, umidità, giorno settimana
- **Problema**: Variabili molto correlate (multicollinearità)

#### Analisi
```python
# Test multicollinearità
correlation_matrix = variabili_esogene.corr()
print("Correlazioni:")
print(correlation_matrix)

"""
Risultato:
           temp  humidity  weekday
temp       1.00     -0.87     0.12
humidity  -0.87      1.00    -0.08  
weekday    0.12     -0.08     1.00

Problema: temp e humidity molto correlate (-0.87)
"""

# Soluzione: Rimuovere una delle variabili correlate
variabili_ridotte = variabili_esogene[['temperatura', 'weekday']]
sarimax_fixed = SARIMAXForecaster(exog_names=['temperatura', 'weekday'])
```

#### Conclusione
In presenza di multicollinearità forte, SARIMA potrebbe essere più robusta di SARIMAX.

### Caso 3: Serie Finanziaria

#### Contesto
- **Serie target**: Prezzo azioni daily
- **Variabili disponibili**: Indici di mercato, volatilità, tassi
- **Problema**: Variabili esogene difficili da prevedere

#### Analisi
```python
# SARIMA
sarima_fin = SARIMAForecaster(order=(1,1,1), seasonal_order=(0,0,0,0))
sarima_forecast = sarima_fin.forecast(steps=5)

# SARIMAX  
# Problema: Come prevedere VIX, tassi interesse per prossimi 5 giorni?
# Soluzione: Scenario analysis

scenarios = {
    'low_vol': {'vix': [15, 15, 16, 15, 14], 'rates': [2.1, 2.1, 2.1, 2.1, 2.1]},
    'high_vol': {'vix': [25, 26, 28, 30, 25], 'rates': [2.1, 2.1, 2.1, 2.1, 2.1]}
}

sarimax_forecasts = {}
for scenario, exog_values in scenarios.items():
    exog_future = pd.DataFrame(exog_values)
    sarimax_forecasts[scenario] = sarimax_fin.forecast(steps=5, exog_future=exog_future)
```

#### Conclusione
Per serie finanziarie ad alta frequenza, SARIMA spesso preferibile per semplicità. SARIMAX utile per scenario analysis.

---

## Implementazione in Codice

### Template Comparativo Completo

```python
from arima_forecaster import SARIMAForecaster, SARIMAXForecaster, ModelEvaluator
import pandas as pd
import numpy as np

class SARIMAvsSARIMAXComparison:
    """
    Classe per confronto sistematico tra SARIMA e SARIMAX.
    """
    
    def __init__(self, series, exog=None, test_size=0.2):
        self.series = series
        self.exog = exog
        self.test_size = test_size
        
        # Split temporale
        split_idx = int(len(series) * (1 - test_size))
        self.train_series = series[:split_idx]
        self.test_series = series[split_idx:]
        
        if exog is not None:
            self.train_exog = exog[:split_idx]
            self.test_exog = exog[split_idx:]
        
        self.results = {}
    
    def fit_sarima(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        """Addestra modello SARIMA."""
        print("🔄 Training SARIMA...")
        
        self.sarima_model = SARIMAForecaster(
            order=order,
            seasonal_order=seasonal_order
        )
        self.sarima_model.fit(self.train_series)
        
        # Metriche in-sample
        info = self.sarima_model.get_model_info()
        self.results['sarima'] = {
            'aic': info['aic'],
            'bic': info['bic'],
            'parameters': len([p for p in info['params']]),
            'model_type': 'SARIMA'
        }
        
        print(f"✅ SARIMA trained - AIC: {info['aic']:.2f}")
    
    def fit_sarimax(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        """Addestra modello SARIMAX."""
        if self.exog is None:
            raise ValueError("Variabili esogene richieste per SARIMAX")
        
        print("🔄 Training SARIMAX...")
        
        self.sarimax_model = SARIMAXForecaster(
            order=order,
            seasonal_order=seasonal_order,
            exog_names=list(self.exog.columns)
        )
        self.sarimax_model.fit(self.train_series, exog=self.train_exog)
        
        # Metriche in-sample
        info = self.sarimax_model.get_model_info()
        importance = self.sarimax_model.get_exog_importance()
        
        self.results['sarimax'] = {
            'aic': info['aic'],
            'bic': info['bic'],
            'parameters': len([p for p in info['params']]),
            'model_type': 'SARIMAX',
            'exog_significant': int(importance['significant'].sum()),
            'exog_total': len(importance)
        }
        
        print(f"✅ SARIMAX trained - AIC: {info['aic']:.2f}")
        print(f"   Variabili significative: {self.results['sarimax']['exog_significant']}/{self.results['sarimax']['exog_total']}")
    
    def compare_forecasts(self, steps=None):
        """Confronta performance di forecasting."""
        if steps is None:
            steps = len(self.test_series)
        
        print(f"\n📊 Confronto forecasting su {steps} periodi...")
        
        evaluator = ModelEvaluator()
        
        # SARIMA forecast
        sarima_forecast = self.sarima_model.forecast(steps=steps)
        sarima_metrics = evaluator.calculate_forecast_metrics(
            actual=self.test_series[:steps],
            predicted=sarima_forecast[:steps]
        )
        
        # SARIMAX forecast
        sarimax_forecast = self.sarimax_model.forecast(
            steps=steps,
            exog_future=self.test_exog[:steps]
        )
        sarimax_metrics = evaluator.calculate_forecast_metrics(
            actual=self.test_series[:steps],
            predicted=sarimax_forecast[:steps]
        )
        
        # Aggiorna risultati
        self.results['sarima'].update({
            'mae': sarima_metrics['mae'],
            'rmse': sarima_metrics['rmse'],
            'mape': sarima_metrics['mape']
        })
        
        self.results['sarimax'].update({
            'mae': sarimax_metrics['mae'],
            'rmse': sarimax_metrics['rmse'],
            'mape': sarimax_metrics['mape']
        })
        
        return sarima_forecast, sarimax_forecast
    
    def print_comparison_report(self):
        """Stampa report comparativo dettagliato."""
        print("\n" + "="*60)
        print("📋 REPORT COMPARATIVO SARIMA vs SARIMAX")
        print("="*60)
        
        # Metriche in-sample
        print("\n🔍 METRICHE IN-SAMPLE:")
        print(f"{'Metrica':<15} {'SARIMA':<12} {'SARIMAX':<12} {'Winner':<10}")
        print("-" * 50)
        
        sarima_aic = self.results['sarima']['aic']
        sarimax_aic = self.results['sarimax']['aic']
        aic_winner = "SARIMAX" if sarimax_aic < sarima_aic else "SARIMA"
        print(f"{'AIC':<15} {sarima_aic:<12.2f} {sarimax_aic:<12.2f} {aic_winner:<10}")
        
        sarima_bic = self.results['sarima']['bic']
        sarimax_bic = self.results['sarimax']['bic']
        bic_winner = "SARIMAX" if sarimax_bic < sarima_bic else "SARIMA"
        print(f"{'BIC':<15} {sarima_bic:<12.2f} {sarimax_bic:<12.2f} {bic_winner:<10}")
        
        print(f"{'Parameters':<15} {self.results['sarima']['parameters']:<12} {self.results['sarimax']['parameters']:<12}")
        
        # Metriche out-of-sample (se disponibili)
        if 'mae' in self.results['sarima']:
            print("\n📈 METRICHE OUT-OF-SAMPLE:")
            print(f"{'Metrica':<15} {'SARIMA':<12} {'SARIMAX':<12} {'Winner':<10}")
            print("-" * 50)
            
            for metric in ['mae', 'rmse', 'mape']:
                sarima_val = self.results['sarima'][metric]
                sarimax_val = self.results['sarimax'][metric]
                winner = "SARIMAX" if sarimax_val < sarima_val else "SARIMA"
                print(f"{metric.upper():<15} {sarima_val:<12.4f} {sarimax_val:<12.4f} {winner:<10}")
        
        # Analisi variabili esogene
        if 'exog_significant' in self.results['sarimax']:
            print(f"\n🌐 ANALISI VARIABILI ESOGENE:")
            sig_count = self.results['sarimax']['exog_significant']
            total_count = self.results['sarimax']['exog_total']
            print(f"   Variabili significative: {sig_count}/{total_count}")
            
            if sig_count == 0:
                print("   ⚠️  Nessuna variabile esogena significativa - considera SARIMA")
            elif sig_count == total_count:
                print("   ✅ Tutte le variabili esogene sono significative")
            else:
                print("   🔄 Alcune variabili non significative - considera rimozione")
        
        # Raccomandazione finale
        print(f"\n🎯 RACCOMANDAZIONE:")
        
        # Logica decisionale semplificata
        if 'exog_significant' in self.results['sarimax']:
            if self.results['sarimax']['exog_significant'] == 0:
                recommendation = "SARIMA - Nessuna variabile esogena significativa"
            elif sarimax_aic < sarima_aic - 2:  # Differenza sostanziale
                recommendation = "SARIMAX - Miglioramento significativo in AIC"
            elif 'mae' in self.results['sarima']:
                mae_improvement = (self.results['sarima']['mae'] - self.results['sarimax']['mae']) / self.results['sarima']['mae']
                if mae_improvement > 0.05:  # 5% miglioramento
                    recommendation = f"SARIMAX - Miglioramento forecasting del {mae_improvement:.1%}"
                else:
                    recommendation = "SARIMA - Miglioramento SARIMAX non sostanziale"
            else:
                recommendation = "Analisi aggiuntiva necessaria"
        else:
            recommendation = "SARIMA - Dati insufficienti per SARIMAX"
        
        print(f"   {recommendation}")
        print("="*60)

# Esempio di utilizzo
def example_comparison():
    """Esempio completo di confronto."""
    
    # Genera dati di esempio
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Serie target con pattern
    trend = np.arange(200) * 0.1
    seasonal = 10 * np.sin(2 * np.pi * np.arange(200) / 7)  # Pattern settimanale
    noise = np.random.normal(0, 3, 200)
    
    # Variabili esogene
    temp = 20 + 5 * np.sin(2 * np.pi * np.arange(200) / 365) + np.random.normal(0, 2, 200)
    marketing = 1000 + 100 * (np.arange(200) % 7 == 0) + np.random.normal(0, 50, 200)  # Spike nei lunedì
    
    # Serie target influenzata dalle esogene
    target = trend + seasonal + 0.2 * temp + 0.001 * marketing + noise
    
    # Dataframes
    series = pd.Series(target, index=dates, name='sales')
    exog = pd.DataFrame({
        'temperature': temp,
        'marketing_spend': marketing
    }, index=dates)
    
    # Confronto
    comparison = SARIMAvsSARIMAXComparison(series, exog, test_size=0.2)
    
    # Addestra entrambi i modelli
    comparison.fit_sarima(order=(1,1,1), seasonal_order=(1,0,1,7))
    comparison.fit_sarimax(order=(1,1,1), seasonal_order=(1,0,1,7))
    
    # Confronta forecasting
    sarima_forecast, sarimax_forecast = comparison.compare_forecasts(steps=30)
    
    # Report finale
    comparison.print_comparison_report()
    
    return comparison

# Esegui esempio
if __name__ == "__main__":
    comparison = example_comparison()
```

### Output Esempio
```
🔄 Training SARIMA...
✅ SARIMA trained - AIC: 1245.67

🔄 Training SARIMAX...
✅ SARIMAX trained - AIC: 1198.23
   Variabili significative: 2/2

📊 Confronto forecasting su 30 periodi...

============================================================
📋 REPORT COMPARATIVO SARIMA vs SARIMAX
============================================================

🔍 METRICHE IN-SAMPLE:
Metrica         SARIMA       SARIMAX      Winner    
--------------------------------------------------
AIC             1245.67      1198.23      SARIMAX   
BIC             1264.12      1225.45      SARIMAX   
Parameters      4            6            

📈 METRICHE OUT-OF-SAMPLE:
Metrica         SARIMA       SARIMAX      Winner    
--------------------------------------------------
MAE             4.2341       3.1245       SARIMAX   
RMSE            5.6732       4.2134       SARIMAX   
MAPE            8.9234       6.7891       SARIMAX   

🌐 ANALISI VARIABILI ESOGENE:
   Variabili significative: 2/2
   ✅ Tutte le variabili esogene sono significative

🎯 RACCOMANDAZIONE:
   SARIMAX - Miglioramento forecasting del 26.2%
============================================================
```

---

## Linee Guida Pratiche

### Workflow Decisionale Consigliato

#### Fase 1: Analisi Preliminare
1. **Ispeziona i dati**: Lunghezza serie, frequenza, pattern visibili
2. **Identifica variabili esogene**: Disponibilità, qualità, correlazione teorica
3. **Valuta risorse**: Tempo, competenze, infrastruttura

#### Fase 2: Modello Base SARIMA  
1. **Stima SARIMA**: Trova parametri ottimali con grid search
2. **Valida modello**: Diagnostica residui, metriche performance
3. **Stabilisci baseline**: Performance da battere per SARIMAX

#### Fase 3: Analisi Variabili Esogene
1. **Test correlazione**: Correlazione con serie target
2. **Test stazionarietà**: Stesso ordine integrazione della serie
3. **Test multicollinearità**: VIF < 5 per ogni variabile
4. **Disponibilità futura**: Verifica possibilità ottenere valori futuri

#### Fase 4: Modello SARIMAX
1. **Stima SARIMAX**: Con tutte le variabili candidate
2. **Selezione variabili**: Rimuovi non significative step-by-step
3. **Valida modello**: Diagnostica estesa + test stabilità

#### Fase 5: Confronto e Decisione
1. **Information criteria**: AIC, BIC per confronto in-sample
2. **Cross-validation**: Performance out-of-sample
3. **Significatività economica**: Miglioramento giustifica complessità?
4. **Robustezza**: Test su diversi periodi e scenari

### Checklist Finale

#### Usa SARIMA se:
- [ ] Non hai variabili esogene di qualità
- [ ] Serie temporale ha pattern interni forti e stabili  
- [ ] Serve semplicità e robustezza
- [ ] Orizzonte previsionale molto breve
- [ ] Risorse computazionali limitate
- [ ] Frequenza aggiornamento modello alta

#### Usa SARIMAX se:
- [ ] Hai variabili esogene significative e di qualità
- [ ] Puoi ottenere valori futuri delle variabili esogene
- [ ] Miglioramento performance > 5-10%
- [ ] Interpretabilità delle relazioni causali è importante
- [ ] Hai competenze per gestire la complessità
- [ ] Scenario analysis è richiesta

#### Considera Approccio Ibrido se:
- [ ] Alcune variabili esogene sono significative ma non tutte
- [ ] Disponibilità variabili esogene è intermittente
- [ ] Serve bilanciare semplicità e performance

### Raccomandazioni per Dominio

#### Business/Retail
- **Preferenza**: SARIMAX per incorporare fattori marketing, economici
- **Variabili tipiche**: Prezzi, promozioni, stagionalità, concorrenza
- **Horizon**: Medio termine (1-12 mesi)

#### Finance/Trading  
- **Preferenza**: SARIMA per robustezza, SARIMAX per scenario analysis
- **Variabili tipiche**: Volatilità, tassi, indici di mercato
- **Horizon**: Breve termine (1-30 giorni)

#### Energy/Utilities
- **Preferenza**: SARIMAX per fattori meteorologici
- **Variabili tipiche**: Temperatura, umidità, vento, radiazione solare
- **Horizon**: Breve-medio termine (1 ora - 1 settimana)

#### Manufacturing/IoT
- **Preferenza**: SARIMAX per sensori e indicatori di processo
- **Variabili tipiche**: Sensori, manutenzioni, materie prime
- **Horizon**: Operativo (1 ora - 1 giorno)

---

## Conclusioni

La scelta tra SARIMA e SARIMAX non è mai banale e deve essere basata su un'analisi sistematica che considera:

### Fattori Tecnici
- **Disponibilità e qualità** delle variabili esogene
- **Significatività statistica** delle relazioni
- **Miglioramento performance** quantificabile  
- **Robustezza** e stabilità del modello

### Fattori Pratici
- **Competenze** del team
- **Risorse computazionali** disponibili
- **Frequenza aggiornamento** richiesta
- **Interpretabilità** necessaria

### Fattori di Business
- **Orizzonte previsionale** richiesto
- **Accuratezza critica** per decisioni
- **Costo errori** vs costo complessità
- **Scenario planning** necessario

**La regola d'oro**: Inizia sempre con SARIMA per stabilire una baseline robusta, poi considera SARIMAX solo se le variabili esogene portano un miglioramento sostanziale e sostenibile.

Ricorda che **il modello migliore è quello che bilancia accuratezza, interpretabilità e praticità** per il tuo specifico caso d'uso.