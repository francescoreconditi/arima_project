# ARIMA Forecaster 🚀

## Libreria Completa per Forecasting Serie Temporali con Modelli ARIMA

Una libreria Python professionale e modulare per l'analisi, modellazione e previsione di serie temporali utilizzando modelli ARIMA (Autoregressive Integrated Moving Average). Progettata con best practices di sviluppo software e documentazione completa sia teorica che pratica.

---

### ✨ **Caratteristiche Principali**

- **🎯 Selezione Automatica Modello**: Grid search intelligente per trovare parametri ARIMA ottimali
- **🔧 Preprocessing Avanzato**: Gestione valori mancanti, rimozione outlier, test stazionarietà
- **📊 Valutazione Completa**: 15+ metriche accuratezza, diagnostica residui, test statistici
- **📈 Visualizzazioni Professionali**: Dashboard interattivi, grafici con intervalli confidenza
- **⚡ Gestione Errori Robusta**: Eccezioni personalizzate e logging configurabile
- **🧪 Testing Estensivo**: Suite test completa con alta coverage
- **📚 Documentazione Completa**: Guide teoriche e pratiche in italiano

---

### 🏗️ **Architettura Modulare**

```
├── src/arima_forecaster/           # Package principale
│   ├── core/                       # Modelli ARIMA e selezione automatica  
│   ├── data/                       # Caricamento dati e preprocessing
│   ├── evaluation/                 # Metriche valutazione e diagnostica
│   ├── visualization/              # Grafici e dashboard avanzati
│   └── utils/                      # Logging ed eccezioni personalizzate
├── docs/                           # Documentazione completa
├── examples/                       # Script esempio pratici
├── tests/                          # Suite test completa
└── outputs/                        # Modelli salvati e visualizzazioni
```

---

### 🚀 **Installazione Rapida**

#### Con UV (Raccomandato - 10x più veloce) ⚡

```bash
# Installa uv se non ce l'hai già
curl -LsSf https://astral.sh/uv/install.sh | sh
# oppure: winget install --id=astral-sh.uv

# Clona il repository
git clone https://github.com/tuonome/arima-forecaster.git
cd arima-forecaster

# Crea ambiente virtuale e installa dipendenze
uv sync --all-extras

# Attiva ambiente virtuale
source .venv/bin/activate  # Linux/macOS
# oppure: .venv\Scripts\activate  # Windows

# Verifica installazione
uv run pytest tests/ -v
```

#### Con pip (Alternativa tradizionale)

```bash
git clone https://github.com/tuonome/arima-forecaster.git
cd arima-forecaster
pip install -e ".[all]"
python -m pytest tests/ -v
```

---

### 💡 **Esempio di Utilizzo**

```python
from arima_forecaster import ARIMAForecaster, TimeSeriesPreprocessor, ForecastPlotter
from arima_forecaster.core import ARIMAModelSelector
import pandas as pd

# 1. Carica e preprocessa dati
dati = pd.read_csv('vendite.csv', index_col='data', parse_dates=True)
preprocessore = TimeSeriesPreprocessor()
serie_pulita, metadata = preprocessore.preprocess_pipeline(dati['vendite'])

# 2. Selezione automatica modello ottimale
selettore = ARIMAModelSelector(p_range=(0,3), d_range=(0,2), q_range=(0,3))
ordine_migliore = selettore.search(serie_pulita)
print(f"Modello ottimale: ARIMA{ordine_migliore}")

# 3. Addestramento e previsioni
modello = ARIMAForecaster(order=ordine_migliore)
modello.fit(serie_pulita)
previsioni, intervalli = modello.forecast(steps=12, return_conf_int=True)

# 4. Valutazione e visualizzazione
plotter = ForecastPlotter()
dashboard = plotter.create_dashboard(
    actual=serie_pulita,
    forecast=previsioni, 
    confidence_intervals=intervalli,
    title="Dashboard Forecasting Vendite"
)
dashboard.show()
```

---

### 📊 **Capacità Avanzate**

#### Preprocessing Intelligente
- **Valori Mancanti**: 4 strategie (interpolazione, drop, forward/backward fill)
- **Outlier**: 3 metodi rilevamento (IQR, Z-score, Z-score modificato)
- **Stazionarietà**: Test ADF automatici e differenziazione adattiva

#### Selezione Modelli Automatica
- **Grid Search**: Esplorazione sistematica spazio parametri
- **Criteri Informativi**: AIC, BIC, HQIC per selezione ottimale
- **Visualizzazioni**: Grafici processo selezione e confronto modelli

#### Valutazione Completa
- **Metriche Forecast**: MAE, RMSE, MAPE, SMAPE, R², MASE, Theil's U
- **Diagnostica Residui**: Jarque-Bera, Ljung-Box, Durbin-Watson, Breusch-Pagan
- **Report Automatici**: Interpretazione risultati e raccomandazioni

---

### 📚 **Documentazione**

| Documento | Descrizione |
|-----------|-------------|
| **[Teoria ARIMA](docs/teoria_arima.md)** | Fondamenti matematici, componenti AR/I/MA, diagnostica |
| **[Guida Utente](docs/guida_utente.md)** | Esempi pratici, API, best practices |
| **[CLAUDE.md](CLAUDE.md)** | Guida per sviluppatori e architettura |

---

### 🧪 **Testing e Qualità**

#### Con UV

```bash
# Esegui tutti i test
uv run pytest tests/ -v

# Test con coverage
uv run pytest tests/ --cov=src/arima_forecaster --cov-report=html

# Controllo qualità codice (tutto in parallelo)
uv run black src/ tests/ examples/
uv run ruff check src/ tests/ examples/
uv run mypy src/arima_forecaster/

# Oppure tutto insieme con pre-commit
uv run pre-commit run --all-files
```

#### Comandi Tradizionali

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src/arima_forecaster --cov-report=html
black src/ tests/ examples/
ruff check src/ tests/ examples/
mypy src/arima_forecaster/
```

---

### 🎨 **Esempi Pratici**

Esplora gli script nella cartella `examples/`:

```bash
# Con UV (raccomandato)
uv run python examples/forecasting_base.py
uv run python examples/selezione_automatica.py

# Oppure tradizionale
python examples/forecasting_base.py
python examples/selezione_automatica.py
```

---

### 🛠️ **Dipendenze Principali**

| Libreria | Scopo | Versione |
|----------|-------|----------|
| **statsmodels** | Implementazione ARIMA | >=0.13.0 |
| **pandas** | Manipolazione dati | >=1.3.0 |
| **numpy** | Calcoli numerici | >=1.21.0 |
| **matplotlib** | Visualizzazione | >=3.4.0 |
| **scipy** | Test statistici | >=1.7.0 |

---

### 🎯 **Roadmap**

- [ ] **Modelli SARIMA**: Supporto stagionalità avanzata
- [ ] **ARIMA Multivariato**: Estensione a serie multiple
- [ ] **API Web**: Interfaccia REST per forecasting
- [ ] **Dashboard Interattiva**: Interfaccia web con Streamlit
- [ ] **Auto-ML**: Ottimizzazione iperparametri automatica

---

### 🤝 **Contributi**

Contributi benvenuti! Per favore:

1. Fork il repository
2. Crea branch feature (`git checkout -b feature/nuova-funzionalita`)
3. Commit modifiche (`git commit -am 'Aggiunge nuova funzionalità'`)
4. Push al branch (`git push origin feature/nuova-funzionalita`)
5. Apri Pull Request

---

### 📄 **Licenza**

Questo progetto è rilasciato sotto Licenza MIT. Vedi file [LICENSE](LICENSE) per dettagli.

---

### 👥 **Autori**

- **Il Tuo Nome** - Sviluppo iniziale - [@tuonome](https://github.com/tuonome)

---

### 🙏 **Ringraziamenti**

- Box & Jenkins per la metodologia ARIMA fondamentale
- Comunità statsmodels per l'eccellente implementazione
- Tutti i contributori open source che rendono possibili progetti come questo

---

### 📞 **Supporto**

- 📖 **Documentazione**: Consulta `docs/` per guide complete
- 🐛 **Bug Reports**: Apri issue su GitHub
- 💡 **Feature Requests**: Discuti nelle GitHub Discussions  
- 📧 **Contatto**: [tuo.email@example.com](mailto:tuo.email@example.com)

---

<div align="center">

**⭐ Se questo progetto ti è utile, lascia una stella! ⭐**

*Sviluppato con ❤️ per la comunità data science italiana*

</div>