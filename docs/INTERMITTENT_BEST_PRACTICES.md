# Best Practices e Use Cases - Intermittent Demand

## 📋 Panoramica

Questa guida fornisce best practices, use cases e linee guida pratiche per l'utilizzo ottimale del modulo Intermittent Demand.

## 🎯 Quando Utilizzare Intermittent Demand

### ✅ Use Cases Ideali

1. **Spare Parts Management**
   - Ricambi automotive (filtri, pastiglie freni, ammortizzatori)
   - Componenti aerospace (O-rings, sensori, valvole)
   - Ricambi industriali (cuscinetti, guarnizioni, motorini)
   - Ricambi medicali (elettrodi, batterie, cinghie)

2. **Slow Moving Products**  
   - Farmaci specialistici
   - Prodotti luxury a bassa rotazione
   - Componentistica elettronica B2B
   - Attrezzature professionali

3. **Emergency Stock**
   - Kit di emergenza
   - Prodotti sicurezza
   - Materiali critici per continuità business

4. **Season-End Products**
   - Articoli stagionali in off-season
   - Prodotti promozionali sporadici
   - Inventario a fine vita

### ❌ Quando NON Utilizzare

1. **Domanda Regolare**: Se intermittenza < 30%, usa ARIMA/SARIMA
2. **Trend Marcati**: Se presenza trend significativo, usa Prophet/SARIMA
3. **Stagionalità Forte**: Se pattern stagionale chiaro, usa SARIMA
4. **Domanda Continua**: Fast movers con vendite giornaliere regolari

## 🔍 Identificazione Pattern

### Analisi Preliminare

```python
from arima_forecaster import IntermittentForecaster

# Analizza pattern prima del modelling
forecaster = IntermittentForecaster()
pattern = forecaster.analyze_pattern(demand_history)

print(f"ADI: {pattern.adi:.1f} giorni")
print(f"CV²: {pattern.cv2:.2f}")
print(f"Intermittenza: {pattern.intermittence:.1%}")
print(f"Classificazione: {pattern.classification}")
```

### Decision Tree per Pattern

```python
def select_model_by_pattern(pattern):
    """Logica di selezione modello basata su pattern"""
    
    if pattern.intermittence < 0.3:
        return "ARIMA"  # Domanda troppo regolare
    
    elif pattern.classification == "Smooth":
        return "ARIMA"  # Usa modelli tradizionali
    
    elif pattern.classification == "Intermittent":
        return "SBA"    # SBA ottimale per intermittent puro
    
    elif pattern.classification == "Lumpy":
        return "TSB"    # TSB gestisce meglio la lumpiness
    
    elif pattern.classification == "Erratic":
        return "ADAPTIVE_CROSTON"  # Alpha dinamico per erratic
    
    else:
        return "CROSTON"  # Fallback default

# Esempio utilizzo
recommended_method = select_model_by_pattern(pattern)
print(f"Metodo consigliato: {recommended_method}")
```

## ⚙️ Selezione e Configurazione Metodi

### 1. Croston's Method

**Quando usare:**
- Pattern Intermittent classico
- Prima implementazione (baseline)
- ADI > 1.32 e CV² < 0.49

**Configurazione ottimale:**
```python
config = IntermittentConfig(
    method=IntermittentMethod.CROSTON,
    alpha=0.1,               # Conservative per spare parts
    optimize_alpha=False     # Alpha fisso per stabilità
)
```

**Vantaggi:**
- Semplice e affidabile
- Interpretabile
- Basso rischio

**Svantaggi:**
- Bias positivo (sovrastima)
- Non ottimale per pattern Lumpy

### 2. SBA (Syntetos-Boylan Approximation)

**Quando usare:**
- Pattern Intermittent puro (ADI ≥ 1.32, CV² < 0.49)
- Necessità accuratezza elevata
- Ricambi critici

**Configurazione ottimale:**
```python
config = IntermittentConfig(
    method=IntermittentMethod.SBA,
    alpha=0.1,
    bias_correction=True,    # Fondamentale per SBA
    optimize_alpha=True      # Consigliato per SBA
)
```

**Vantaggi:**
- Corregge bias di Croston
- Migliore accuracy per intermittent
- Standard industry per spare parts

**Svantaggi:**
- Leggermente più complesso
- Richiede ottimizzazione alpha

### 3. TSB (Teunter-Syntetos-Babai)

**Quando usare:**
- Pattern Lumpy (ADI ≥ 1.32, CV² ≥ 0.49)
- Domanda molto variabile in size
- Ordini sporadici ma grandi

**Configurazione ottimale:**
```python
config = IntermittentConfig(
    method=IntermittentMethod.TSB,
    alpha=0.15,              # Leggermente più alto per TSB
    optimize_alpha=True
)
```

**Vantaggi:**
- Ottimale per Lumpy demand
- Probability-based approach
- Gestisce meglio variabilità size

**Svantaggi:**
- Meno stabile per Intermittent puro
- Richiede più dati per convergere

### 4. Adaptive Croston

**Quando usare:**
- Pattern Erratic o cambio frequente
- Domanda con shift strutturali
- Ricambi nuovi senza storico stabile

**Configurazione ottimale:**
```python
config = IntermittentConfig(
    method=IntermittentMethod.ADAPTIVE_CROSTON,
    alpha=0.1,               # Valore iniziale
    optimize_alpha=False     # Alpha si adatta automaticamente
)
```

**Vantaggi:**
- Si adatta a cambiamenti
- Gestisce pattern instabili
- Reattivo a nuovi trend

**Svantaggi:**
- Più complesso
- Può essere instabile
- Richiede tuning parameters

## 📊 Ottimizzazione Parametri

### Alpha Tuning

```python
# Test range alpha per diversi pattern
alpha_ranges = {
    'Smooth': (0.05, 0.10),      # Conservative
    'Intermittent': (0.08, 0.15), # Balanced
    'Erratic': (0.10, 0.20),     # More reactive
    'Lumpy': (0.12, 0.25)        # Most reactive
}

# Grid search personalizzato
def optimize_alpha_custom(data, pattern_type):
    best_alpha = 0.1
    best_mase = float('inf')
    
    alpha_min, alpha_max = alpha_ranges[pattern_type]
    
    for alpha in np.arange(alpha_min, alpha_max + 0.01, 0.01):
        config = IntermittentConfig(
            method=IntermittentMethod.SBA,
            alpha=alpha
        )
        
        # Cross-validation
        mase = validate_model(data, config)
        
        if mase < best_mase:
            best_mase = mase
            best_alpha = alpha
    
    return best_alpha, best_mase
```

### Walk-Forward Validation

```python
def validate_intermittent_model(data, config, window_size=100, step_size=10):
    """
    Validazione robusta per intermittent demand
    """
    errors = []
    
    for i in range(window_size, len(data) - step_size, step_size):
        # Split
        train = data[i-window_size:i]
        test = data[i:i+step_size]
        
        # Verifica dati sufficienti
        if np.sum(train > 0) < 2:
            continue
            
        # Train e predict
        model = IntermittentForecaster(config)
        model.fit(train)
        forecast = model.forecast(step_size)
        
        # Calcola MASE per questo fold
        mase = calculate_mase(test, forecast)
        errors.append(mase)
    
    return np.mean(errors) if errors else float('inf')
```

## 🎯 Inventory Optimization

### Safety Stock Calculation

```python
def calculate_optimal_safety_stock(forecaster, lead_time, target_service_levels):
    """
    Calcola safety stock per diversi service level
    """
    results = {}
    
    for sl in target_service_levels:
        ss = forecaster.calculate_safety_stock(lead_time, sl)
        rop = forecaster.calculate_reorder_point(lead_time, sl)
        
        results[f"{sl:.0%}"] = {
            'safety_stock': ss,
            'reorder_point': rop,
            'investment': rop * unit_cost
        }
    
    return results

# Esempio per ricambio critico
service_levels = [0.85, 0.90, 0.95, 0.99]
optimal_levels = calculate_optimal_safety_stock(
    forecaster, 
    lead_time=15, 
    target_service_levels=service_levels
)

# Visualizza trade-off
for level, params in optimal_levels.items():
    print(f"Service Level {level}: ROP={params['reorder_point']:.0f}, Investment=€{params['investment']:,.2f}")
```

### Economic Order Quantity Integration

```python
def calculate_eoq_intermittent(forecaster, ordering_cost, holding_cost_rate, unit_cost):
    """
    EOQ modificato per intermittent demand
    """
    # Annual demand estimate
    annual_forecast = forecaster.forecast_[0] * 365
    
    # Holding cost
    holding_cost = unit_cost * holding_cost_rate
    
    # Standard EOQ formula
    if annual_forecast > 0:
        eoq = np.sqrt(2 * annual_forecast * ordering_cost / holding_cost)
        orders_per_year = annual_forecast / eoq
        
        return {
            'eoq': eoq,
            'orders_per_year': orders_per_year,
            'cycle_stock': eoq / 2,
            'annual_ordering_cost': orders_per_year * ordering_cost,
            'annual_holding_cost': (eoq / 2) * holding_cost
        }
    
    return None
```

## 🔍 Monitoring e Controllo Performance

### KPI Dashboard

```python
def monitor_intermittent_performance(actual_demand, forecasts, reorder_points):
    """
    Monitoraggio performance continuo
    """
    metrics = {}
    
    # Forecast accuracy
    evaluator = IntermittentEvaluator()
    results = evaluator.evaluate(actual_demand, forecasts)
    
    metrics['forecast_mase'] = results.mase
    metrics['fill_rate'] = results.fill_rate
    metrics['service_level'] = results.achieved_service_level
    
    # Inventory turnover
    avg_stock = np.mean(reorder_points)
    annual_demand = np.sum(actual_demand) * 365 / len(actual_demand)
    metrics['inventory_turnover'] = annual_demand / avg_stock if avg_stock > 0 else 0
    
    # Stockout frequency  
    stockouts = np.sum(np.array(actual_demand) > np.array(forecasts))
    metrics['stockout_frequency'] = stockouts / len(actual_demand)
    
    return metrics
```

### Alert System

```python
def setup_performance_alerts(metrics, thresholds):
    """
    Sistema di alert per performance degradation
    """
    alerts = []
    
    if metrics['forecast_mase'] > thresholds['max_mase']:
        alerts.append(f"⚠️  MASE troppo alto: {metrics['forecast_mase']:.2f}")
    
    if metrics['fill_rate'] < thresholds['min_fill_rate']:
        alerts.append(f"⚠️  Fill Rate basso: {metrics['fill_rate']:.1f}%")
    
    if metrics['inventory_turnover'] < thresholds['min_turnover']:
        alerts.append(f"⚠️  Turnover basso: {metrics['inventory_turnover']:.1f}x")
    
    if metrics['stockout_frequency'] > thresholds['max_stockout_freq']:
        alerts.append(f"⚠️  Troppi stockout: {metrics['stockout_frequency']:.1%}")
    
    return alerts

# Thresholds esempio
thresholds = {
    'max_mase': 1.5,
    'min_fill_rate': 85,
    'min_turnover': 2.0,
    'max_stockout_freq': 0.15
}
```

## 🏭 Implementation Patterns

### Pattern 1: Single SKU Analysis

```python
class SparePartManager:
    def __init__(self, sku_code, unit_cost, lead_time):
        self.sku_code = sku_code
        self.unit_cost = unit_cost
        self.lead_time = lead_time
        self.forecaster = None
        
    def analyze_and_optimize(self, demand_history, service_level=0.95):
        """Pipeline completo per singolo SKU"""
        
        # 1. Pattern analysis
        temp_forecaster = IntermittentForecaster()
        pattern = temp_forecaster.analyze_pattern(demand_history)
        
        # 2. Method selection
        if pattern.classification == 'Intermittent':
            method = IntermittentMethod.SBA
        elif pattern.classification == 'Lumpy':
            method = IntermittentMethod.TSB
        else:
            method = IntermittentMethod.CROSTON
            
        # 3. Model training
        config = IntermittentConfig(method=method, optimize_alpha=True)
        self.forecaster = IntermittentForecaster(config)
        self.forecaster.fit(demand_history)
        
        # 4. Inventory optimization
        self.reorder_point = self.forecaster.calculate_reorder_point(
            self.lead_time, service_level
        )
        
        return {
            'pattern': pattern.classification,
            'method': method.value,
            'reorder_point': self.reorder_point,
            'investment': self.reorder_point * self.unit_cost
        }
```

### Pattern 2: Portfolio Management

```python
class SparePartsPortfolio:
    def __init__(self):
        self.parts = {}
        
    def add_part(self, sku, demand_history, unit_cost, lead_time):
        """Aggiungi spare part al portfolio"""
        manager = SparePartManager(sku, unit_cost, lead_time)
        result = manager.analyze_and_optimize(demand_history)
        self.parts[sku] = {
            'manager': manager,
            'result': result
        }
        
    def get_portfolio_summary(self):
        """Summary investimenti e performance"""
        summary = {
            'total_skus': len(self.parts),
            'total_investment': 0,
            'by_pattern': {},
            'by_method': {}
        }
        
        for sku, data in self.parts.items():
            result = data['result']
            
            # Total investment
            summary['total_investment'] += result['investment']
            
            # By pattern
            pattern = result['pattern']
            if pattern not in summary['by_pattern']:
                summary['by_pattern'][pattern] = {'count': 0, 'investment': 0}
            summary['by_pattern'][pattern]['count'] += 1
            summary['by_pattern'][pattern]['investment'] += result['investment']
            
            # By method
            method = result['method']
            if method not in summary['by_method']:
                summary['by_method'][method] = {'count': 0, 'investment': 0}
            summary['by_method'][method]['count'] += 1
            summary['by_method'][method]['investment'] += result['investment']
            
        return summary
```

## 📋 Checklist Implementazione

### Pre-Implementation

- [ ] **Data Quality Check**
  - [ ] Storico almeno 12 mesi
  - [ ] Almeno 3 periodi con domanda > 0
  - [ ] Data cleaning completato
  - [ ] Zero values preservati (non filtrati)

- [ ] **Pattern Analysis**
  - [ ] Intermittenza > 30%
  - [ ] Classificazione pattern identificata
  - [ ] Trend/stagionalità verificati (devono essere minimi)

- [ ] **Business Requirements**
  - [ ] Service level target definito
  - [ ] Lead time fornitore noto
  - [ ] Costi (holding, stockout) quantificati
  - [ ] Budget disponibile per inventory

### Implementation

- [ ] **Model Selection**
  - [ ] Metodo scelto basandosi su pattern
  - [ ] Alpha ottimizzato se necessario
  - [ ] Validation cross-fold eseguita

- [ ] **Inventory Parameters**
  - [ ] Safety stock calcolato
  - [ ] Reorder point determinato
  - [ ] EOQ calcolato se applicabile

- [ ] **Integration**
  - [ ] Export per ERP configurato
  - [ ] Workflow riordini definito
  - [ ] Alert system configurato

### Post-Implementation

- [ ] **Monitoring**
  - [ ] KPI dashboard attivo
  - [ ] Performance tracking regolare
  - [ ] Model retraining schedulato

- [ ] **Optimization**
  - [ ] Review trimestrale performance
  - [ ] Aggiustamenti parametri se necessario
  - [ ] Expansion ad altri SKU

## 🚫 Errori Comuni da Evitare

### 1. Errori nei Dati

❌ **Rimuovere gli zeri**: Gli zeri sono informazione cruciale  
✅ **Preservare tutti i valori**, inclusi zeri e outlier

❌ **Aggregare per settimana/mese**: Perde informazione su intermittenza  
✅ **Mantenere granularità giornaliera** quando possibile

### 2. Errori nella Configurazione

❌ **Alpha troppo alto (>0.3)**: Modello instabile  
✅ **Alpha 0.05-0.15** per la maggior parte degli spare parts

❌ **Non ottimizzare alpha**: Performance subottimali  
✅ **Sempre usare optimize_alpha=True** tranne per Adaptive Croston

### 3. Errori nella Valutazione

❌ **Usare RMSE/MAE**: Non appropriate per intermittent  
✅ **Usare MASE** come metrica principale

❌ **Ignorare service level**: Focus solo su forecast accuracy  
✅ **Monitorare Fill Rate e Service Level** per business value

### 4. Errori nell'Implementazione

❌ **Reorder point fisso**: Non considera variazioni stagionali  
✅ **Review periodico** parametri inventory

❌ **Un metodo per tutti**: Approccio one-size-fits-all  
✅ **Pattern-specific method selection** per ogni SKU

## 📈 Metriche di Successo

### KPI Primari

1. **MASE < 1.2**: Accuracy superiore a naive forecast
2. **Fill Rate > 90%**: Soddisfazione cliente
3. **Service Level > 95%**: Disponibilità prodotto
4. **Inventory Turnover 3-6x**: Efficienza capitale

### KPI Secondari

- **Stockout Frequency < 10%**: Continuità business
- **Overstock Periods < 5%**: Controllo giacenze eccessive
- **Forecast Bias ± 15%**: Accuracy bilanciata
- **Cost Optimization**: Total cost (holding + stockout) minimizzato

### ROI Measurement

```python
def calculate_intermittent_roi(baseline_costs, optimized_costs):
    """
    Calcola ROI implementazione Intermittent Demand
    """
    savings = baseline_costs - optimized_costs
    roi_percentage = (savings / baseline_costs) * 100
    
    return {
        'annual_savings': savings,
        'roi_percentage': roi_percentage,
        'payback_months': 12 * (optimized_costs['implementation'] / savings) if savings > 0 else float('inf')
    }
```

ROI tipici:
- **Riduzione stock 20-40%** mantenendo service level
- **Miglioramento fill rate 10-15%** 
- **Payback 3-6 mesi** per implementazione completa
- **Savings annuali 15-25%** dei costi inventory

Questa guida fornisce le fondamenta per implementazione di successo dell'Intermittent Demand forecasting nel tuo portfolio di spare parts.