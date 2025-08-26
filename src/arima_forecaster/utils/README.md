# Utils Module - ARIMA Forecaster

Modulo utilità per funzionalità trasversali: logging, gestione errori, preprocessing e sistema traduzioni centralizzato.

## 📋 Moduli Disponibili

### 🌍 translations.py - Sistema Traduzioni Multilingue **⭐ NUOVO**

Sistema centralizzato per gestire traduzioni in 5 lingue across tutto il progetto.

#### Lingue Supportate
- **Italiano** (`it`) - Lingua default  
- **English** (`en`) - Mercato internazionale
- **Español** (`es`) - Spagna e America Latina
- **Français** (`fr`) - Mercato francofono  
- **中文** (`zh`) - Asia-Pacific e Cina

#### Quick Start

```python
# Import base
from arima_forecaster.utils.translations import translate as _

# Traduzione rapida
title = _('title', 'en')  # "Inventory Management Report - Moretti S.p.A."
title_zh = _('title', 'zh')  # "库存管理报告 - Moretti S.p.A."

# Sistema avanzato
from arima_forecaster.utils.translations import TranslationManager

translator = TranslationManager()
all_translations = translator.get_all('es')
available_langs = translator.get_available_languages()
```

#### Architettura File
```
src/arima_forecaster/assets/locales/
├── it.json    # Italiano (default)
├── en.json    # English
├── es.json    # Español  
├── fr.json    # Français
└── zh.json    # 中文 (Cinese)
```

#### API Reference

| Funzione | Scopo | Esempio |
|----------|-------|---------|
| `translate(key, lang)` | Traduzione rapida | `_('title', 'en')` |
| `get_all_translations(lang)` | Tutte le traduzioni | `get_all_translations('zh')` |
| `get_translator()` | Istanza singleton | `translator = get_translator()` |
| `TranslationManager()` | Classe completa | `tm = TranslationManager()` |

#### Compatibilità
```python
# Per dashboard esistenti (compatibilità Moretti)
from arima_forecaster.utils.translations import get_translations_dict

translations = get_translations_dict('Italiano')  # Accetta nomi user-friendly
# Automaticamente convertito in: get_all_translations('it')
```

#### Gestione Errori & Fallback
- **File mancante**: Automatico fallback a lingua default (italiano)
- **Chiave mancante**: Ritorna la chiave stessa come fallback
- **Encoding errato**: Gestione UTF-8 robusta per caratteri cinesi
- **Thread-safe**: Cache LRU per performance in ambienti concorrenti

---

### 📝 logger.py - Sistema Logging

```python
from arima_forecaster.utils.logger import setup_logger, get_logger

# Setup logging
setup_logger()

# Usage
logger = get_logger(__name__)
logger.info("ARIMA model training completed")
logger.error("Forecast generation failed", exc_info=True)
```

**Features:**
- Configurazione automatica console + file
- Rotazione log automatica
- Formattazione structured per parsing
- Support per diversi livelli (DEBUG, INFO, WARNING, ERROR)

---

### ⚠️ exceptions.py - Eccezioni Specializzate

```python
from arima_forecaster.utils.exceptions import (
    ARIMAForecasterError,
    ModelTrainingError,
    ForecastError,
    DataProcessingError
)

# Usage in modelli
try:
    model.fit(data)
except ModelTrainingError as e:
    logger.error(f"Training failed: {e}")
    raise
```

**Gerarchia Eccezioni:**
```
ARIMAForecasterError
├── DataProcessingError
├── ModelTrainingError  
└── ForecastError
```

---

### 🔧 preprocessing.py - Utilità Preprocessing

```python
from arima_forecaster.utils.preprocessing import (
    ExogenousPreprocessor,
    validate_exog_data,
    suggest_preprocessing_method
)

# Preprocessing variabili esogene per SARIMAX
preprocessor = ExogenousPreprocessor()
exog_clean = preprocessor.fit_transform(exog_data)

# Validazione dati
is_valid = validate_exog_data(exog_data, target_series)
```

**Funzionalità:**
- Preprocessing automatico variabili esogene
- Validazione consistenza temporale
- Gestione valori mancanti
- Feature scaling e normalization

---

## 🚀 Utilizzo nei Moduli del Progetto

### Dashboard Streamlit
```python
# examples/moretti/moretti_dashboard.py
from arima_forecaster.utils.translations import get_all_translations

def render_ui(language='Italiano'):
    translations = get_all_translations(language)
    st.title(translations['title'])
    st.metric(translations['warehouse_value'], value)
```

### API REST
```python  
# src/arima_forecaster/api/main.py
from arima_forecaster.utils.translations import translate as _

@app.get("/forecast")
async def get_forecast(language: str = "en"):
    return {
        "message": _('forecast_completed', language),
        "data": forecast_data
    }
```

### Report Generation
```python
# src/arima_forecaster/reporting/generator.py
from arima_forecaster.utils.translations import get_all_translations

def generate_report(language='it'):
    translations = get_all_translations(language)
    template_vars = {
        'title': translations['title'],
        'summary': translations['executive_summary']
    }
```

---

## 📈 Performance & Optimizations

### Caching System
- **LRU Cache**: `@lru_cache(maxsize=10)` per traduzioni caricate
- **Lazy Loading**: Caricamento file solo quando necessario
- **Singleton Pattern**: Una sola istanza TranslationManager in memoria

### Memory Usage
- **File JSON**: ~5-10KB per lingua (49 keys x 5 lingue = ~50KB totale)  
- **Cache Memory**: ~100KB massimo per tutte le traduzioni in memoria
- **Overhead**: <1ms per traduzione con cache hit

### Threading
- **Thread-safe**: Operazioni lettura sicure in ambienti multi-thread
- **Immutable**: Traduzioni read-only dopo caricamento
- **No contention**: Nessun lock necessario per performance

---

## 🔧 Development & Maintenance

### Aggiungere Nuova Lingua
1. **Creare file JSON**: `src/arima_forecaster/assets/locales/de.json` 
2. **Aggiornare mapping**: Aggiungere `'German': 'de'` in `language_mapping`
3. **Aggiornare supported**: Aggiungere `'de'` in `supported_languages`
4. **Testare**: Verificare caricamento con `get_available_languages()`

### Aggiungere Nuova Traduzione
1. **Identificare key**: Scegliere nome key descrittivo (`new_feature_title`)
2. **Aggiornare tutti JSON**: Aggiungere key in tutti i 5 file locales
3. **Verificare Unicode**: Testare caratteri speciali per cinese/francese
4. **Usare nel codice**: `_('new_feature_title', language)`

### Testing
```bash
# Test sistema traduzioni
uv run pytest tests/utils/test_translations.py -v

# Test encoding UTF-8
uv run python -c "
from arima_forecaster.utils.translations import translate as _
print(_('title', 'zh'))  # Deve stampare caratteri cinesi
"
```

---

## 🆕 Changelog

### v2.0.0 - Agosto 2024
- ✅ **Sistema traduzioni centralizzato**: 5 lingue supportate
- ✅ **Fix encoding UTF-8**: Risolti problemi caratteri cinesi  
- ✅ **Thread-safe caching**: Performance ottimizzate
- ✅ **Compatibilità**: Supporto dashboard e API esistenti
- ✅ **File in Git**: Traduzioni incluse nel repository

### v1.x.x - Versioni precedenti
- ✅ Logging configurabile
- ✅ Eccezioni specializzate  
- ✅ Preprocessing utilità
- ✅ Gestione errori robusta

---

*Modulo utils progettato per fornire funzionalità core trasversali robuste e performanti per tutta la libreria ARIMA Forecaster.*