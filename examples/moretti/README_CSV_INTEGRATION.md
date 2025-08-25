# 🗃️ Integrazione CSV Dashboard Moretti

## 📋 Overview

La dashboard Moretti è stata aggiornata per **caricare tutti i dati da file CSV esterni**, rendendo la demo molto più flessibile e personalizzabile per diversi clienti.

## ✨ Nuove Funzionalità

### 🔄 Caricamento Dinamico Dati
- ✅ **Prodotti**: `prodotti_dettaglio.csv` (12 prodotti con dettagli completi)
- ✅ **Vendite Storiche**: `vendite_storiche_dettagliate.csv` (120 giorni di dati realistici)
- ✅ **Ordini Attivi**: `ordini_attivi.csv` (8 ordini con diversi stati)
- ✅ **Fornitori**: `fornitori_dettaglio.csv` (10 fornitori con condizioni dettagliate)
- ✅ **Scenari What-If**: `scenari_whatif.csv` (10 scenari predefiniti)
- ✅ **Categorie**: `categorie_config.csv` (6 categorie con parametri)

### 🎭 Scenari What-If Predefiniti
La sidebar ora include scenari business realistici:

| Scenario | Descrizione | Lead Time | Domanda |
|----------|-------------|-----------|---------|
| **Scenario_Base** | Situazione Attuale | 100% | 100% |
| **Crisi_Fornitori** | Problemi Supply Chain | 150% | 100% |
| **Boom_Domanda** | Crescita Post-Pandemia | 100% | 180% |
| **Efficienza_Digitale** | Ottimizzazione Processi | 75% | 100% |
| **Recessione_Economica** | Contrazione Mercato | 100% | 70% |
| **Partnership_Premium** | Accordi Fornitori VIP | 60% | 100% |
| **Espansione_Sud** | Apertura Nuovi Mercati | 120% | 140% |
| **Automazione_AI** | Sistema Predittivo Completo | 100% | 100% |

### 🗃️ Tab Gestione CSV
Nuova tab "Dati CSV" che permette di:
- 📊 Visualizzare status e statistiche di tutti i file CSV
- 👁️ Anteprima contenuto con statistiche dettagliate
- 🔄 Ricarica dati in tempo reale
- 📥 Download template per nuovi file
- 💡 Suggerimenti per personalizzazione client-specific

## 📁 Struttura File CSV

```
examples/moretti/data/
├── prodotti_dettaglio.csv          # Catalogo prodotti completo
├── vendite_storiche_dettagliate.csv # 120 giorni vendite realistiche
├── ordini_attivi.csv               # Ordini in corso con fornitori
├── fornitori_dettaglio.csv         # Database fornitori dettagliato
├── scenari_whatif.csv              # Scenari business predefiniti
└── categorie_config.csv            # Configurazione categorie
```

## 🚀 Vantaggi per Demo Clienti

### 🎯 Personalizzazione Immediata
1. **Sostituisci prodotti_dettaglio.csv** con prodotti del cliente
2. **Modifica fornitori_dettaglio.csv** con fornitori reali del territorio  
3. **Crea scenari_whatif.csv** specifici per le sfide del cliente
4. **Aggiorna vendite_storiche_dettagliate.csv** con pattern settoriali

### 📈 Demo Più Credibili
- Dati realistici generati con pattern statistici accurati
- Nomi prodotti specifici settore medicale
- Fornitori con condizioni commerciali reali
- Scenari what-if basati su situazioni business concrete

### ⚡ Setup Veloce
- Nessuna modifica al codice richiesta
- Basta sostituire i file CSV
- Dashboard si adatta automaticamente
- Fallback ai dati simulati se CSV mancanti

## 📊 Dati Generati Automaticamente

### 🏥 Prodotti Medicali (12 totali)
- **Carrozzine**: Standard, Elettrica, Sportiva Titanio
- **Antidecubito**: Materassi Aria/Memory, Cuscini 
- **Riabilitazione**: Deambulatori, Bastoni Canadesi
- **Elettromedicali**: Saturimetri, Misuratori Pressione, Termometri
- **Assistenza**: Sollevatori Pazienti

### 📈 Vendite Storiche Realistiche
- Pattern stagionali per prodotto
- Trend crescita/decrescita
- Variabilità settimanale (meno vendite weekend)
- Eventi casuali (picchi e cali)
- 120 giorni di dati con media 1.7-8.2 unità/giorno per prodotto

### 🏭 Fornitori Completi
- **MedSupply Italia**: Leader carrozzine (Rating 5⭐)
- **AntiDecubito Pro**: Specialista materassi (Rating 5⭐)
- **DiagnosticPro**: Elettromedicali (Rating 5⭐)
- **RehaMed Solutions**: Riabilitazione (Rating 5⭐)
- **LiftCare Pro**: Equipaggiamenti pesanti (Rating 3⭐)

## 🔧 Utilizzo Tecnico

### Avvio Dashboard
```bash
cd examples/moretti
uv run streamlit run moretti_dashboard.py
```

### Rigenerazione Vendite
```bash
cd examples/moretti  
uv run python generate_vendite_storiche.py
```

### Personalizzazione File
1. Modifica i CSV nella directory `data/`
2. La dashboard ricarica automaticamente
3. Usa la tab "Dati CSV" per verificare il caricamento
4. Fallback automatico ai dati simulati se errori

## 💡 Tips Demo Clienti

### 🎨 Brandizzazione
- **Sostituisci logo aziendale** nel CSS
- **Modifica colori** per match brand cliente
- **Personalizza terminologia** prodotti/categorie
- **Aggiorna scenari** per sfide specifiche settore

### 📊 Dati Credibili  
- **Usa nomi prodotti reali** del cliente
- **Importa storico vendite** se disponibile
- **Configura fornitori locali** del territorio
- **Crea scenari business** realistici per il settore

### ⚡ Performance Demo
- **Pre-carica scenari interessanti** (Boom Domanda, Crisi Fornitori)
- **Prepara story telling** per ogni scenario
- **Mostra tab CSV** per dimostrare flessibilità
- **Evidenzia personalizzazione** senza programmazione

## 🔄 Compatibilità

- ✅ **Backward Compatible**: Codice esistente funziona senza modifiche
- ✅ **Fallback Robusto**: Dati simulati se CSV non disponibili  
- ✅ **Error Handling**: Gestione errori elegante con warning
- ✅ **Performance**: Caricamento ottimizzato con caching

## 🎯 Risultato Finale

**Dashboard completamente data-driven che può essere personalizzata per qualsiasi cliente del settore medicale semplicemente sostituendo i file CSV!**

🏆 **Perfect per demo clienti con dati realistici e scenari business credibili!**