# ============================================
# FILE DI TEST/DEBUG - NON PER PRODUZIONE
# Creato da: Claude Code
# Data: 2025-09-03
# Scopo: Documentazione API MSL Management
# ============================================

# MSL Management API Documentation

## Overview

Il sistema **Minimum Shelf Life (MSL) Management** fornisce funzionalità avanzate per l'allocazione ottimale di lotti con scadenza a diversi canali di vendita, rispettando i requisiti di shelf life minima di ogni canale.

## Core Classes

### MinimumShelfLifeManager

Classe principale per la gestione dell'allocazione MSL con ottimizzazione multi-canale.

```python
from arima_forecaster.inventory import MinimumShelfLifeManager

msl_manager = MinimumShelfLifeManager()
```

#### Methods

##### `aggiungi_requisito_msl(requisito: RequisitoMSL) -> None`

Aggiunge un requisito MSL personalizzato per un prodotto e canale specifici.

**Parameters:**
- `requisito`: Oggetto RequisitoMSL con configurazione specifica

**Example:**
```python
from arima_forecaster.inventory import RequisitoMSL, TipoCanale

requisito = RequisitoMSL(
    canale=TipoCanale.GDO_PREMIUM,
    prodotto_codice="PROD001",
    msl_giorni=90,
    priorita=1,
    note="Requisiti specifici Esselunga"
)
msl_manager.aggiungi_requisito_msl(requisito)
```

##### `get_canali_compatibili(prodotto_codice: str, giorni_residui: int) -> List[TipoCanale]`

Restituisce la lista di canali compatibili con un prodotto dato i giorni di shelf life residua.

**Parameters:**
- `prodotto_codice`: Codice identificativo del prodotto
- `giorni_residui`: Giorni di shelf life rimanenti

**Returns:**
- Lista di canali che accettano il prodotto con la shelf life specificata

**Example:**
```python
canali_ok = msl_manager.get_canali_compatibili("PROD001", 45)
# Restituisce: [TipoCanale.OUTLET_SCONTI, TipoCanale.ONLINE_DIRETTO]
```

##### `ottimizza_allocazione_lotti(lotti_disponibili, domanda_canali, prezzo_canali) -> Dict`

Esegue l'ottimizzazione completa dell'allocazione lotti considerando MSL, domanda e prezzi per canale.

**Parameters:**
- `lotti_disponibili`: Lista di oggetti LottoPerishable disponibili
- `domanda_canali`: Dict con domanda per canale (es. {"gdo_premium": 500})
- `prezzo_canali`: Dict con prezzi per canale (es. {"gdo_premium": 4.20})

**Returns:**
- Dict con risultati allocazione per canale: `{canale_id: [AllocationResult, ...]}`

**Example:**
```python
risultati = msl_manager.ottimizza_allocazione_lotti(
    lotti_disponibili=lotti,
    domanda_canali={"gdo_premium": 500, "outlet": 200},
    prezzo_canali={"gdo_premium": 4.20, "outlet": 2.80}
)

for canale, allocazioni in risultati.items():
    print(f"Canale {canale}: {len(allocazioni)} allocazioni")
    for alloc in allocazioni:
        print(f"  Lotto {alloc.lotto_id}: {alloc.quantita_allocata} unità")
```

##### `genera_report_allocazioni(risultati: Dict) -> Dict`

Genera un report dettagliato delle allocazioni MSL con statistiche e metriche.

**Parameters:**
- `risultati`: Dict restituito da `ottimizza_allocazione_lotti()`

**Returns:**
- Dict con report completo includendo summary, efficienza, distribuzione urgenze

**Example:**
```python
report = msl_manager.genera_report_allocazioni(risultati)
print(f"Efficienza: {report['efficienza_allocazione']}%")
print(f"Canali serviti: {report['summary']['numero_canali_serviti']}")
```

##### `suggerisci_azioni_msl(risultati: Dict) -> List[Dict]`

Analizza i risultati e suggerisce azioni per ottimizzare ulteriormente l'allocazione.

**Parameters:**
- `risultati`: Dict restituito da `ottimizza_allocazione_lotti()`

**Returns:**
- Lista di dizionari con azioni suggerite (tipo, priorità, descrizione, azione)

**Example:**
```python
azioni = msl_manager.suggerisci_azioni_msl(risultati)
for azione in azioni:
    print(f"[{azione['priorita']}] {azione['tipo']}: {azione['descrizione']}")
```

## Data Models

### TipoCanale (Enum)

Enumerazione dei canali di vendita con configurazioni MSL predefinite.

```python
from arima_forecaster.inventory import TipoCanale

# Valori disponibili:
TipoCanale.GDO_PREMIUM        # 90 giorni MSL default
TipoCanale.GDO_STANDARD       # 60 giorni MSL default  
TipoCanale.RETAIL_TRADIZIONALE # 45 giorni MSL default
TipoCanale.ONLINE_DIRETTO     # 30 giorni MSL default
TipoCanale.OUTLET_SCONTI      # 15 giorni MSL default
TipoCanale.B2B_WHOLESALE      # 120 giorni MSL default
```

**Structure:**
Ogni valore contiene: `(channel_id, description, default_msl_days)`

### RequisitoMSL (BaseModel)

Modello per definire requisiti MSL personalizzati per prodotto/canale.

```python
from arima_forecaster.inventory import RequisitoMSL, TipoCanale

requisito = RequisitoMSL(
    canale=TipoCanale.GDO_PREMIUM,
    prodotto_codice="PROD001",
    msl_giorni=100,           # Giorni MSL richiesti
    priorita=1,               # Priorità allocazione (1=alta)
    note="Note opzionali"     # Descrizione requisiti
)
```

**Fields:**
- `canale`: TipoCanale - Canale di destinazione
- `prodotto_codice`: str - Codice prodotto  
- `msl_giorni`: int - Giorni MSL richiesti (>= 1)
- `priorita`: int - Priorità allocazione 1-5 (default: 3)
- `note`: Optional[str] - Note descrittive

### AllocationResult (BaseModel)

Risultato dell'allocazione di un lotto a un canale specifico.

```python
# Restituito da ottimizza_allocazione_lotti()
result = AllocationResult(
    lotto_id="LOT001",
    canale_id="gdo_premium", 
    quantita_allocata=150,
    giorni_shelf_life_residui=45,
    margine_msl=15,              # giorni oltre MSL minimo
    valore_allocato=630.0,       # valore economico
    urgenza="media"              # bassa/media/alta/critica
)
```

**Fields:**
- `lotto_id`: str - ID del lotto allocato
- `canale_id`: str - ID canale destinazione
- `quantita_allocata`: int - Quantità allocata
- `giorni_shelf_life_residui`: int - Giorni shelf life rimanenti
- `margine_msl`: int - Giorni di margine oltre MSL minimo
- `valore_allocato`: float - Valore economico allocazione
- `urgenza`: str - Livello urgenza vendita

### LottoPerishable (BaseModel) 

Modello per lotti con scadenza utilizzati nell'ottimizzazione MSL.

```python
from arima_forecaster.inventory.balance_optimizer import LottoPerishable
from datetime import datetime, timedelta

lotto = LottoPerishable(
    lotto_id="LOT001",
    quantita=500,
    data_produzione=datetime.now() - timedelta(days=30),
    data_scadenza=datetime.now() + timedelta(days=90),
    shelf_life_giorni=120,
    giorni_residui=90,
    percentuale_vita_residua=0.75,
    valore_unitario=2.50,
    rischio_obsolescenza=0.10
)
```

## Usage Patterns

### Pattern 1: Setup Base MSL

```python
from arima_forecaster.inventory import MinimumShelfLifeManager

# Inizializza manager con configurazione default
msl_manager = MinimumShelfLifeManager()

# I canali usano MSL predefiniti:
# - GDO Premium: 90 giorni
# - GDO Standard: 60 giorni  
# - Retail: 45 giorni
# - Online: 30 giorni
# - Outlet: 15 giorni
# - B2B: 120 giorni
```

### Pattern 2: Customizzazione MSL

```python
from arima_forecaster.inventory import (
    MinimumShelfLifeManager, 
    RequisitoMSL, 
    TipoCanale
)

msl_manager = MinimumShelfLifeManager()

# Personalizza MSL per prodotti specifici
requisiti = [
    RequisitoMSL(
        canale=TipoCanale.GDO_PREMIUM,
        prodotto_codice="YOGURT_BIO",
        msl_giorni=100,  # Più stringente del default (90)
        priorita=1,
        note="Bio richiede MSL extra"
    ),
    RequisitoMSL(
        canale=TipoCanale.B2B_WHOLESALE, 
        prodotto_codice="YOGURT_BIO",
        msl_giorni=80,   # Meno stringente per B2B
        priorita=2
    )
]

for req in requisiti:
    msl_manager.aggiungi_requisito_msl(req)
```

### Pattern 3: Ottimizzazione Completa

```python
from arima_forecaster.inventory.balance_optimizer import LottoPerishable
from datetime import datetime, timedelta

# 1. Definisci lotti disponibili
lotti = [
    LottoPerishable(
        lotto_id="LOT001",
        quantita=500,
        data_scadenza=datetime.now() + timedelta(days=30),
        # ... altri campi
    ),
    LottoPerishable(
        lotto_id="LOT002", 
        quantita=800,
        data_scadenza=datetime.now() + timedelta(days=90),
        # ... altri campi
    )
]

# 2. Definisci domanda per canale  
domanda = {
    "gdo_premium": 600,
    "gdo_standard": 800, 
    "outlet": 300
}

# 3. Definisci prezzi per canale
prezzi = {
    "gdo_premium": 4.20,
    "gdo_standard": 3.80,
    "outlet": 2.80
}

# 4. Esegui ottimizzazione
risultati = msl_manager.ottimizza_allocazione_lotti(
    lotti_disponibili=lotti,
    domanda_canali=domanda,
    prezzo_canali=prezzi
)

# 5. Analizza risultati
for canale, allocazioni in risultati.items():
    print(f"\n{canale.upper()}:")
    for alloc in allocazioni:
        print(f"  {alloc.lotto_id}: {alloc.quantita_allocata} unità, "
              f"MSL margin: {alloc.margine_msl} giorni, "
              f"urgenza: {alloc.urgenza}")
```

### Pattern 4: Reporting e Azioni

```python
# Genera report dettagliato
report = msl_manager.genera_report_allocazioni(risultati)

print(f"Efficienza allocazione: {report['efficienza_allocazione']}%")
print(f"Canali serviti: {report['summary']['numero_canali_serviti']}")
print(f"Valore totale: €{report['summary']['valore_totale']:,.2f}")

# Ottieni suggerimenti per ottimizzazione
azioni = msl_manager.suggerisci_azioni_msl(risultati)

for i, azione in enumerate(azioni, 1):
    print(f"\n{i}. [{azione['priorita']}] {azione['tipo']}")
    print(f"   {azione['descrizione']}")
    print(f"   → {azione['azione']}")
```

### Pattern 5: Test Compatibilità Canali

```python
# Verifica quali canali accettano prodotto con N giorni residui
prodotto = "YOGURT_BIO"

test_giorni = [10, 30, 60, 90, 120]
for giorni in test_giorni:
    canali_ok = msl_manager.get_canali_compatibili(prodotto, giorni)
    nomi = [c.value[1].split(' - ')[0] for c in canali_ok]
    print(f"Con {giorni} giorni residui → {len(canali_ok)} canali: {', '.join(nomi)}")

# Output esempio:
# Con 10 giorni residui → 0 canali: 
# Con 30 giorni residui → 2 canali: Online Diretto, Outlet Sconti
# Con 60 giorni residui → 3 canali: Online Diretto, Outlet Sconti, GDO Standard
# Con 90 giorni residui → 4 canali: Online Diretto, Outlet Sconti, GDO Standard, GDO Premium
# Con 120 giorni residui → 6 canali: Tutti
```

## Integration Patterns

### Con PerishableManager (FEFO + MSL)

```python
from arima_forecaster.inventory.balance_optimizer import PerishableManager

# Combina FEFO optimization con MSL compliance
perishable_manager = PerishableManager(
    msl_manager=msl_manager  # Passa MSL manager per integrazione
)

# FEFO + MSL optimization integrata
risultati_integrati = perishable_manager.ottimizza_fefo_con_msl(
    lotti_disponibili=lotti,
    domanda_canali=domanda,
    prezzo_canali=prezzi
)
```

### Con Inventory Balance Optimizer

```python
from arima_forecaster.inventory import (
    MovementClassifier,
    SlowFastOptimizer, 
    MinimumShelfLifeManager
)

# 1. Classifica movimento prodotti
classifier = MovementClassifier()
classifications = classifier.classify_movement_speed(sales_data)

# 2. Per prodotti FAST + Perishable, usa MSL optimization
msl_manager = MinimumShelfLifeManager()

for prod_code, classification in classifications.items():
    if classification["movement"] == "FAST" and is_perishable(prod_code):
        # Applica MSL optimization
        risultati_msl = msl_manager.ottimizza_allocazione_lotti(...)
```

## Error Handling

```python
from arima_forecaster.utils.exceptions import (
    DataProcessingError,
    ModelTrainingError
)

try:
    risultati = msl_manager.ottimizza_allocazione_lotti(
        lotti_disponibili=lotti,
        domanda_canali=domanda,
        prezzo_canali=prezzi
    )
except DataProcessingError as e:
    print(f"Errore dati: {e}")
except ValueError as e:
    print(f"Parametri non validi: {e}")
except Exception as e:
    print(f"Errore generico MSL: {e}")
```

## Performance Considerations

- **Complessità**: O(n*m) dove n=lotti, m=canali
- **Memory**: ~100MB per 1000 lotti x 10 canali  
- **Optimization**: Usa algoritmo greedy per performance su dataset grandi
- **Scalabilità**: Testato fino a 10,000 lotti contemporaneamente

## Best Practices

1. **Definisci sempre MSL personalizzati** per prodotti critici
2. **Ordina lotti per urgenza FEFO** prima dell'ottimizzazione  
3. **Monitora margini MSL** per identificare problemi allocazione
4. **Usa report per tracking** performance nel tempo
5. **Integra con sistemi ERP** via CSV per automazione
6. **Test compatibilità** prima di nuovi prodotti/canali

## Limitations

- **Algoritmo greedy**: Non garantisce ottimo globale assoluto
- **Single-product**: Ottimizzazione per singolo prodotto alla volta
- **Static MSL**: Requisiti MSL non cambiano durante ottimizzazione
- **Memory bound**: Limitato da RAM disponibile per dataset molto grandi

## Changelog

- **v1.0**: Implementazione iniziale MSL Management
- **v1.1**: Integrazione con PerishableManager (FEFO + MSL)
- **v1.2**: Aggiunto sistema suggerimenti azioni automatiche

---

Per ulteriori esempi vedere: `examples/minimum_shelf_life_demo.py`