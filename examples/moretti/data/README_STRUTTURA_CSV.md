# ============================================
# FILE DI TEST/DEBUG - NON PER PRODUZIONE
# Creato da: Claude Code
# Data: 2025-01-29
# Scopo: Documentazione struttura CSV per dashboard Moretti
# ============================================

# Struttura File CSV - Dashboard Moretti S.p.A.

## File Principali (Richiesti dalla Dashboard)

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

## File Supplementari (Opzionali ma Utili)

### 4. **fornitori.csv**
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

### 7. **budget_mensile.csv**
Budget acquisti per categoria.

**Colonne:**
- `anno_mese` (string): Periodo (formato YYYY-MM)
- `categoria` (string): Categoria prodotti
- `budget_allocato` (float): Budget stanziato EUR
- `speso_attuale` (float): Speso ad oggi EUR
- `rimanente` (float): Budget residuo EUR

## Esempi di Utilizzo

### Caricamento Base
```python
# La dashboard cerca automaticamente i file nella cartella data/
prodotti = pd.read_csv('data/prodotti_dettaglio.csv')
vendite = pd.read_csv('data/vendite_storiche_dettagliate.csv', parse_dates=['data'])
ordini = pd.read_csv('data/ordini_attivi.csv', parse_dates=['data_ordine', 'data_consegna_prevista'])
```

### Modificatori Dashboard
La funzione `carica_dati_da_csv()` accetta parametri per simulare scenari:
- `lead_time_mod`: Modifica lead time (100 = normale, 150 = +50%)
- `domanda_mod`: Modifica domanda (100 = normale, 120 = +20%)

### Validazione Dati
```python
# Controlli minimi richiesti
assert len(vendite) >= 90  # Almeno 90 giorni di storico
assert all(col in prodotti.columns for col in ['codice', 'nome', 'scorte_attuali'])
assert 'data' in vendite.columns
```

## Note Implementative

1. **Encoding**: Tutti i file devono essere UTF-8
2. **Date**: Formato ISO (YYYY-MM-DD)
3. **Decimali**: Punto come separatore (es. 123.45)
4. **Valori Mancanti**: Usare celle vuote, non "N/A" o "NULL"
5. **Case Sensitive**: I codici prodotto sono case-sensitive

## Estensibilità

Il sistema è progettato per essere esteso con:
- Dati meteo per correlazioni stagionali
- Eventi promozionali e campagne marketing
- Dati competitors per analisi di mercato
- Feedback qualità fornitori
- Resi e reclami clienti

## Testing

Per verificare la compatibilità dei CSV:
```bash
cd examples/moretti
python moretti_dashboard.py
```

Se i CSV sono corretti, la dashboard si avvierà senza errori di caricamento dati.