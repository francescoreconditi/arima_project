# ============================================
# FILE DI DOCUMENTAZIONE - SISTEMA PRODUZIONE
# Creato da: Claude Code
# Data: 2025-01-29 | Aggiornato: 2025-09-15
# Scopo: Documentazione struttura CSV per dashboard Moretti
# ============================================

# Struttura File CSV - Dashboard Moretti S.p.A.

## 📋 Panoramica Sistema

Il dashboard Moretti utilizza un sistema di configurazione completamente esternalizzato basato su file CSV. Tutti i dati operativi, parametri di calcolo e configurazioni sono gestiti tramite file CSV per permettere personalizzazione cliente per cliente senza modifiche al codice.

## 📁 File Principali (Richiesti dalla Dashboard)

### 🎯 File Dati Core (Richiesti per operatività base)

### 1. **prodotti_dettaglio.csv**
File master con anagrafica completa prodotti.

**Colonne:**
- `codice` (string): Codice univoco prodotto (es. CRZ001)
- `nome` (string): Descrizione estesa prodotto
- `categoria` (string): Categoria merceologica (Carrozzine, Antidecubito, etc.)
- `scorte_attuali` (int): Quantità attualmente in magazzino
- `scorta_minima` (int): Livello minimo prima del riordino
- `scorta_sicurezza` (int): Buffer di sicurezza
- `prezzo_medio` (float): Prezzo medio di acquisto in EUR
- `lead_time` (int): Giorni medi di consegna dal fornitore
- `criticita` (int): Livello criticità 1-5 (5=massima)

### 2. **vendite_storiche_dettagliate.csv**
Serie storica vendite giornaliere per prodotto (ultimi 90 giorni).

**Colonne:**
- `data` (date): Data nel formato YYYY-MM-DD
- `[codice_prodotto]` (int): Una colonna per ogni prodotto con quantità venduta
  - Es: CRZ001, CRZ002, MAT001, etc.

**Note:**
- Minimo 90 righe (3 mesi di storico)
- Valori interi rappresentano unità vendute
- Zeri indicano nessuna vendita
- Pattern stagionali e trend inclusi nei dati

### 3. **ordini_attivi.csv**
Ordini di acquisto in corso non ancora ricevuti.

**Colonne:**
- `id_ordine` (string): Identificativo univoco ordine
- `prodotto_codice` (string): Codice prodotto ordinato
- `quantita` (int): Quantità ordinata
- `fornitore` (string): Nome fornitore
- `data_ordine` (date): Data emissione ordine
- `data_consegna_prevista` (date): Data consegna stimata
- `stato` (string): Stato ordine (In elaborazione/Confermato/In transito/In consegna)
- `costo_totale` (float): Valore totale ordine in EUR

## 🔧 File Configurazione Sistema (Nuovi - Obbligatori dal 2025-09-15)

### 4. **depositi_config.csv** 🆕
Configurazione depositi con stock correnti e parametri operativi. **Utilizzato dalla sezione "Depositi" del dashboard.**

**Colonne:**
- `deposito_nome` (string): Nome deposito mostrato nella UI (es. "Centrale Milano")
- `deposito_id` (string): Codice univoco interno (es. "DEP_MI_01")
- `tipo` (string): Tipologia deposito (Hub Principale/Filiale/Magazzino/Centro Distribuzione)
- `regione` (string): Regione geografica (Lombardia/Lazio/etc.)
- `capacita_max` (int): Capacità massima in unità
- `stock_[CODICE]` (int): Stock correnti per prodotto specifico
  - `stock_CRZ001`: Stock correnti carrozzine
  - `stock_MAT001`: Stock correnti materassi
  - `stock_ELT001`: Stock correnti elettromedicali
- `lead_time_interno` (int): Lead time interno in giorni (1-3)
- `costo_stoccaggio_m3` (float): Costo stoccaggio per m³ in EUR

**Esempio riga:**
```csv
Centrale Milano,DEP_MI_01,Hub Principale,Lombardia,10000,302,242,94,1,12.50
```

**Utilizzo nel Dashboard:**
- Dropdown "Deposito" popolato dinamicamente da questa tabella
- Stock levels mostrati in tempo reale per deposito selezionato
- Calcoli inventory personalizzati per deposito

### 5. **parametri_bilanciamento.csv** 🆕
Parametri operativi per calcoli inventory e gestione rischio. **Utilizzato dalla sezione "Analisi Bilanciamento Scorte".**

**Colonne:**
- `parametro` (string): Nome parametro di configurazione
- `valore` (numeric/string): Valore del parametro (auto-convertito in int/float/string)
- `unita` (string): Unità di misura (giorni/percentuale/volte, etc.)
- `descrizione` (string): Descrizione business del parametro

**Parametri Obbligatori:**
```csv
service_level_default,0.95,,Livello di servizio target predefinito
lead_time_default,15,giorni,Lead time predefinito per calcoli
criticality_factor_default,1.2,,Fattore di criticità per prodotti importanti
inventory_turnover_target,9.5,volte/anno,Target di rotazione inventario
days_of_supply_target,50,giorni,Target giorni di copertura
overstock_threshold,150,percentuale,Soglia per identificare overstock
stockout_risk_threshold,20,percentuale,Soglia rischio stockout
```

**Utilizzo nel Dashboard:**
- Service level default caricato automaticamente
- Lead time e criticality factor utilizzati nei calcoli safety stock
- Soglie utilizzate per identificazione alert automatici

## 📊 File Supplementari (Opzionali ma Utili)

### 6. **fornitori.csv** / **fornitori_dettaglio.csv**
Anagrafica fornitori con metriche performance.

**Colonne:**
- `codice_fornitore` (string): Codice univoco
- `nome` (string): Ragione sociale
- `categoria_prodotti` (string): Specializzazione
- `lead_time_medio` (int): Giorni medi consegna
- `affidabilita_percentuale` (float): % consegne puntuali
- `sconto_volume_percentuale` (float): Sconto per grandi ordini
- `pagamento_giorni` (int): Termini di pagamento
- `valutazione` (float): Rating 1-5
- `paese` (string): Nazione sede
- `certificazioni` (string): Certificazioni possedute (separate da ;)

### 5. **storico_prezzi.csv**
Evoluzione prezzi per prodotto/fornitore.

**Colonne:**
- `data` (date): Data rilevazione prezzo
- `prodotto_codice` (string): Codice prodotto
- `fornitore` (string): Nome fornitore
- `prezzo_unitario` (float): Prezzo per unità
- `quantita_minima_ordine` (int): MOQ (Minimum Order Quantity)
- `sconto_applicato` (float): Percentuale sconto

### 6. **alert_configurazione.csv**
Configurazione sistema di allerta automatico.

**Colonne:**
- `prodotto_codice` (string): Prodotto monitorato
- `tipo_alert` (string): Tipo allerta (Scorte Basse/Domanda Anomala/Lead Time Lungo)
- `soglia_minima` (int): Soglia warning
- `soglia_critica` (int): Soglia critica
- `giorni_previsione` (int): Orizzonte temporale analisi
- `email_notifica` (string): Email destinatario alert
- `priorita` (string): Priorità (Critica/Alta/Media/Bassa)

### 9. **budget_mensile.csv**
Budget acquisti per categoria.

**Colonne:**
- `anno_mese` (string): Periodo (formato YYYY-MM)
- `categoria` (string): Categoria prodotti
- `budget_allocato` (float): Budget stanziato EUR
- `speso_attuale` (float): Speso ad oggi EUR
- `rimanente` (float): Budget residuo EUR

### 10. **scenari_whatif.csv** (Opzionale)
Scenari predefiniti per analisi what-if e simulazioni.

### 11. **categorie_config.csv** (Opzionale)
Configurazione categorie prodotti per raggruppamenti e filtri.

## 🎯 File Monitorati nella Sezione "Dati CSV" Dashboard

Il dashboard include una sezione dedicata **"🗃️ Dati CSV"** che monitora automaticamente tutti i file CSV e ne mostra:
- ✅ Status di caricamento
- 📊 Dimensioni (righe × colonne)
- 📁 Peso file in bytes
- 🕐 Data ultima modifica
- 👁️ Anteprima primi 10 record

### File Attualmente Monitorati:
```python
data_files = {
    'prodotti_dettaglio.csv': 'Catalogo prodotti con scorte e parametri',
    'vendite_storiche_dettagliate.csv': 'Storico vendite ultimi 120 giorni',
    'ordini_attivi.csv': 'Ordini in corso con fornitori',
    'fornitori_dettaglio.csv': 'Database fornitori e condizioni',
    'scenari_whatif.csv': 'Scenari predefiniti per analisi',
    'categorie_config.csv': 'Configurazione categorie prodotti',
    'depositi_config.csv': 'Configurazione depositi e stock correnti per bilanciamento',  # 🆕
    'parametri_bilanciamento.csv': 'Parametri operativi per calcoli inventory e analisi rischio'  # 🆕
}
```

## 💻 Esempi di Utilizzo

### Caricamento Base Dashboard
```python
# La dashboard carica automaticamente tutti i CSV necessari
def main():
    # File core sempre caricati
    prodotti, vendite, ordini = carica_dati_da_csv()

    # File configurazione (nuovi dal 2025-09-15)
    parametri = carica_parametri_bilanciamento()
    depositi_df = carica_configurazione_depositi()

    # Logica dashboard con dati esterni
    deposito_options = depositi_df['deposito_nome'].tolist()
    service_level = parametri.get('service_level_default', 0.95)
```

### Configurazione Multi-Cliente
```bash
# Struttura directory raccomandata per multi-cliente
clients/
├── moretti/data/
│   ├── parametri_bilanciamento.csv    # Service level 95%
│   └── depositi_config.csv           # 6 depositi Italia
├── farmacia_rossi/data/
│   ├── parametri_bilanciamento.csv    # Service level 98% (farmacia)
│   └── depositi_config.csv           # 2 depositi Roma
└── clinica_verde/data/
    ├── parametri_bilanciamento.csv    # Service level 99% (clinica)
    └── depositi_config.csv           # 1 deposito centrale
```

### Modificatori Scenario Dashboard
```python
# La funzione carica_dati_da_csv() supporta simulazioni
carica_dati_da_csv(
    lead_time_mod=150,    # +50% lead time (simulazione crisi)
    domanda_mod=80,       # -20% domanda (simulazione recessione)
    language='English'    # Dashboard multilingue
)
```

### Validazione Dati Estesa
```python
# Controlli minimi richiesti (aggiornati 2025-09-15)
assert len(vendite) >= 90  # Almeno 90 giorni di storico
assert all(col in prodotti.columns for col in ['codice', 'nome', 'scorte_attuali'])
assert 'data' in vendite.columns

# Nuovi controlli per configurazione esterna
assert len(depositi_df) > 0, "depositi_config.csv vuoto o mancante"
assert 'service_level_default' in parametri, "service_level_default mancante in parametri"
assert all(f'stock_{code}' in depositi_df.columns for code in ['CRZ001', 'MAT001', 'ELT001'])
```

## 🔧 Note Implementative

### Sistema di Configurazione Esterna
1. **Auto-conversione tipi**: I parametri CSV vengono convertiti automaticamente in int/float/string
2. **Fallback robusto**: Se i CSV di configurazione mancano, il sistema usa valori di default
3. **Validazione real-time**: La sezione "Dati CSV" mostra errori di caricamento in tempo reale
4. **Cache intelligente**: Le configurazioni vengono ricaricate solo quando i file cambiano

### Requisiti Tecnici
1. **Encoding**: Tutti i file devono essere UTF-8 con BOM opzionale
2. **Date**: Formato ISO (YYYY-MM-DD)
3. **Decimali**: Punto come separatore (es. 123.45)
4. **Valori Mancanti**: Usare celle vuote, non "N/A" o "NULL"
5. **Case Sensitive**: I codici prodotto sono case-sensitive
6. **Performance**: File >10MB potrebbero causare rallentamenti UI

## 🚀 Estensibilità Sistema

### Nuovi File CSV Supportati
Il sistema è progettato per essere esteso facilmente con:
- **Dati esterni**: Meteo, eventi, competitors
- **Configurazioni avanzate**: Zone geografiche, canali vendita
- **Analytics**: Sentiment analysis, forecast accuracies
- **Integration**: ERP exports, API imports

### Aggiunta Nuovo File CSV
```python
# 1. Aggiungi alla sezione "Dati CSV" del dashboard
data_files['nuovo_file.csv'] = 'Descrizione funzionale'

# 2. Crea funzione caricamento
def carica_nuovo_file():
    return pd.read_csv('data/nuovo_file.csv')

# 3. Integra nella logica dashboard
nuovo_data = carica_nuovo_file()
```

## 🧪 Testing e Validazione

### Test Rapido Compatibilità
```bash
cd examples/moretti

# Test dashboard completo
streamlit run moretti_dashboard.py

# Test solo caricamento dati
python -c "
from moretti_dashboard import carica_dati_da_csv, carica_parametri_bilanciamento
prodotti, vendite, ordini = carica_dati_da_csv()
parametri = carica_parametri_bilanciamento()
print(f'Test OK: {len(prodotti)} prodotti, {len(parametri)} parametri')
"
```

### Benchmark Performance
```python
# File CSV raccomandati per performance ottimale
prodotti_dettaglio.csv: < 1,000 righe
vendite_storiche.csv: < 10,000 righe (90-365 giorni)
ordini_attivi.csv: < 500 righe
parametri_bilanciamento.csv: 15-50 parametri
depositi_config.csv: < 50 depositi
```

## 📈 Roadmap e Futuro

### Prossimi Sviluppi Pianificati
- **2025 Q4**: Integration con database SQL per grandi volumi
- **2026 Q1**: API REST per aggiornamenti real-time
- **2026 Q2**: Machine learning per parametri auto-tuning
- **2026 Q3**: Multi-tenancy con isolamento clienti automatico

---

## 🗺️ Mappa Completa File CSV Dashboard Moretti

### Stato Utilizzo File (Aggiornato 2025-09-15)

| File CSV | Status | Sezione Dashboard | Funzione | Obbligatorio |
|----------|--------|-------------------|----------|--------------|
| `prodotti_dettaglio.csv` | ✅ Attivo | Tutte le sezioni | Anagrafica prodotti | ✅ Sì |
| `vendite_storiche_dettagliate.csv` | ✅ Attivo | Trend, Previsioni, Analisi | Storico vendite | ✅ Sì |
| `ordini_attivi.csv` | ✅ Attivo | Ordini, Dashboard principale | Ordini in corso | ✅ Sì |
| `depositi_config.csv` | 🆕 Nuovo | **🏪 Depositi** | Config depositi e stock | ✅ Sì |
| `parametri_bilanciamento.csv` | 🆕 Nuovo | **🏪 Depositi** | Parametri calcoli | ✅ Sì |
| `fornitori_dettaglio.csv` | ✅ Attivo | Dati CSV, Suggerimenti | Anagrafica fornitori | ⚠️ Opzionale |
| `scenari_whatif.csv` | ⚠️ Opzionale | Advanced Exog, Cold Start | Scenari simulazione | ❌ No |
| `categorie_config.csv` | ⚠️ Opzionale | Filtri, Raggruppamenti | Config categorie | ❌ No |
| `alert_configurazione.csv` | ⚠️ Opzionale | Sistema alerts | Config notifiche | ❌ No |
| `budget_mensile.csv` | ⚠️ Opzionale | Reports, Analytics | Budget tracking | ❌ No |
| `storico_prezzi.csv` | ⚠️ Opzionale | Analytics fornitori | Trend pricing | ❌ No |

### Legenda Status:
- ✅ **Attivo**: File caricato e utilizzato attivamente
- 🆕 **Nuovo**: File aggiunto nell'ultimo update (2025-09-15)
- ⚠️ **Opzionale**: File supportato ma non obbligatorio
- ❌ **Non richiesto**: File documentato ma non necessario per operatività

### File Minimi per Operatività Dashboard:
1. `prodotti_dettaglio.csv` - Anagrafica base
2. `vendite_storiche_dettagliate.csv` - Dati storici per forecasting
3. `ordini_attivi.csv` - Situazione ordini correnti
4. `depositi_config.csv` - 🆕 Configurazione depositi
5. `parametri_bilanciamento.csv` - 🆕 Parametri operativi

**Total**: 5 file CSV obbligatori (era 3 prima del 2025-09-15)

### Impatto Aggiornamento 2025-09-15:
- ➕ **2 nuovi file CSV obbligatori** per sezione Depositi
- 🔄 **Sezione "Dati CSV"** aggiornata per includere i nuovi file
- 📊 **Sistema monitoraggio** esteso a 8 file totali
- 🎯 **Dashboard completamente configurabile** per depositi e parametri

---

## ❓ FAQ - Domande Frequenti

### Q: Cosa succede se un file CSV obbligatorio manca?
A: Il sistema ha fallback robusti. Per depositi_config.csv e parametri_bilanciamento.csv vengono utilizzati valori di default hardcoded.

### Q: Posso aggiungere nuove colonne ai CSV esistenti?
A: Sì, il sistema ignora colonne non utilizzate. Attenzione a non rimuovere colonne esistenti.

### Q: Come posso personalizzare per un nuovo cliente?
A: Duplica la cartella `data/` e personalizza i CSV `depositi_config.csv` e `parametri_bilanciamento.csv`.

### Q: Il sistema supporta CSV con separatori diversi da virgola?
A: No, attualmente sono supportati solo CSV standard con separatore virgola.

### Q: Quanto spesso vengono ricaricati i CSV?
A: I file vengono ricaricati ad ogni avvio del dashboard. Per reload dinamico riavviare Streamlit.

---

## 📞 Supporto e Contatti

Per domande tecniche su struttura CSV o problemi di caricamento:
- 📧 Email: support@moretti-dashboard.com
- 📱 WhatsApp: +39 XXX XXX XXXX
- 💻 Issues: GitHub repository del progetto

**Ultimo aggiornamento**: 2025-09-15 by Claude Code