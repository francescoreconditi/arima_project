# 🏥 Sistema Gestione Scorte - Moretti S.p.A.

Caso pratico completo per l'applicazione della libreria ARIMA Forecaster alla gestione intelligente delle scorte per prodotti medicali critici.

## 📁 Struttura Files

### 📊 Moduli Principali

#### `moretti_simple_example.py`
**Esempio base e demo funzionante**
- Sistema semplificato ARIMA(1,1,1) 
- Gestione scorte carrozzine standard
- Calcolo punto riordino e EOQ
- Output: CSV con previsioni e metriche
- **Status**: ✅ Funzionante e testato

#### `moretti_inventory_management.py`  
**Sistema enterprise completo**
- Modelli SARIMA con selezione automatica parametri
- Ottimizzazione fornitori multi-tier pricing
- Integrazione dati demografici ISTAT
- Piano approvvigionamento trimestrale
- Dashboard interattivi con Plotly
- **Status**: ⚠️ Avanzato (richiede debugging)

#### `moretti_advanced_forecasting.py`
**Modulo multi-variato avanzato**
- Modelli VAR per correlazioni tra prodotti
- Test causalità di Granger
- Analisi Impulse Response
- Integrazione fattori esterni (demografici, economici, epidemiologici)
- Sistema alert automatici
- **Status**: ⚠️ Sperimentale

#### `moretti_dashboard.py`
**Dashboard web interattivo**
- Interfaccia Streamlit per monitoraggio real-time
- KPI e metriche chiave
- Visualizzazioni scorte e previsioni
- Suggerimenti riordino automatici
- **Status**: 🚧 Demo (dati simulati)

#### `test_moretti_minimal.py`
**Test diagnostico libreria**
- Verifica funzionamento base ARIMA
- Debug problemi previsioni
- **Status**: ✅ Utilità per sviluppo

## 🚀 Come Iniziare

### 🎯 Demo Launcher (Raccomandato)
```bash
# Dalla directory root del progetto
cd examples/moretti
uv run python run_demo.py
```
**Seleziona automaticamente la demo più adatta:**
- **Demo Veloce** (30 sec): 3 prodotti con ARIMA ✅ **Testata**
- **Demo Semplice** (45 sec): Singolo prodotto dettagliato ✅ **Testata**  
- **Demo Enterprise** (5+ min): Sistema SARIMA completo ⚠️ **Avanzata**

### 🎮 Demo Specifiche

#### 1. Test Veloce e Affidabile
```bash
uv run python examples/moretti/moretti_inventory_fast.py
```

#### 2. Esempio Base Dettagliato
```bash
uv run python examples/moretti/moretti_simple_example.py
```

#### 3. Dashboard Interattivo  
```bash
uv run streamlit run examples/moretti/moretti_dashboard.py
```

#### 4. Sistema Enterprise Completo
```bash
# ⚠️ Richiede tempo - Solo per utenti avanzati
uv run python examples/moretti/moretti_inventory_management.py
```

## 📈 Risultati Demo

### 🚀 Demo Veloce (moretti_inventory_fast.py)
**3 Prodotti Critici Analizzati:**
- **Carrozzina Standard**: MAPE 16.5%, Domanda 25.5/giorno
- **Materasso Antidecubito**: MAPE 15.4%, Domanda 26.7/giorno  
- **Saturimetro**: MAPE 16.5%, Domanda 18.7/giorno

**Decisioni Business**: 1 riordino necessario (€40,934)

### 📊 Demo Semplice (moretti_simple_example.py)
**Prodotto**: Carrozzina Standard
- **Performance**: MAPE 15.39%, RMSE 4.13
- **Previsioni**: ~20 unità/giorno (30 giorni)
- **Gestione Scorte**: Punto riordino 294, EOQ 218 unità
- **Alert**: Rischio stockout rilevato

### 🏢 Demo Enterprise (moretti_inventory_management.py)
**Sistema SARIMA Multi-Prodotto:**
- Selezione automatica parametri
- Ottimizzazione fornitori multi-tier
- Piano approvvigionamento trimestrale
- Dashboard interattivi avanzati

**Output Generati:**
- `outputs/reports/moretti_previsioni_*.csv` 
- `outputs/reports/moretti_riordini_*.csv`
- `outputs/reports/moretti_piano_scorte.xlsx`
- `outputs/plots/moretti_*_dashboard.html`

## 💡 Applicazione Reale

### Prodotti Moretti S.p.A. Analizzati:
1. **Carrozzine e Mobilità** (CRZ001, CRZ002)
2. **Materassi Antidecubito** (MAT001, MAT002)  
3. **Ausili Riabilitazione** (RIA001, RIA002)
4. **Elettromedicali** (ELT001, ELT002)

### Fattori Considerati:
- ✅ Trend demografico invecchiamento popolazione
- ✅ Stagionalità domanda (inverno vs estate)
- ✅ Eventi eccezionali (es. COVID-19)
- ✅ Lead time fornitori variabili
- ✅ Pricing multi-tier per quantità
- ✅ Correlazioni cross-selling tra prodotti

## 🔧 Personalizzazione

### Per Adattare ai Dati Reali:
1. **Sostituire dati simulati** con connessione database ERP
2. **Configurare parametri prodotto** in `GeneratoreDatiMoretti`
3. **Aggiornare fornitori** e pricing reali
4. **Integrare API ISTAT** per dati demografici veri
5. **Setup alert email/SMS** per riordini critici

### Estensioni Suggerite:
- **Multi-location**: Gestione magazzini multipli
- **Promotions impact**: Effetto sconti/campagne
- **Supply chain risk**: Gestione disruption fornitori
- **Seasonal adjustment**: Calibrazione stagionale automatica

## 📞 Supporto Tecnico

Per domande sull'implementazione o personalizzazione:
- Vedere `CLAUDE.md` per istruzioni sviluppo
- Test con `test_moretti_minimal.py` per debugging
- Log dettagliati in `logs/` directory

---

*Sistema sviluppato per dimostrare capabilities enterprise della libreria ARIMA Forecaster nel settore medicale.*