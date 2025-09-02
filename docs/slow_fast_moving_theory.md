# Teoria e Pratica della Gestione Slow/Fast Moving

## 📚 Indice
1. [Introduzione](#introduzione)
2. [Definizioni e Classificazioni](#definizioni-e-classificazioni)
3. [Metriche Chiave](#metriche-chiave)
4. [Strategie di Gestione](#strategie-di-gestione)
5. [Casi Pratici Reali](#casi-pratici-reali)
6. [Formule e Calcoli](#formule-e-calcoli)
7. [Best Practices](#best-practices)
8. [Errori Comuni da Evitare](#errori-comuni-da-evitare)

---

## 🎯 Introduzione

La gestione differenziata di prodotti **Slow Moving** e **Fast Moving** è una delle strategie più efficaci per ottimizzare il capitale circolante e migliorare la redditività aziendale. Questa distinzione permette di applicare politiche di inventory management specifiche, riducendo simultaneamente il rischio di stockout per prodotti critici e l'obsolescenza per articoli a bassa rotazione.

### Impatto Economico
- **Capitale immobilizzato**: Il 20% dei prodotti (slow moving) può rappresentare l'80% del capitale immobilizzato
- **Costi di gestione**: I slow moving hanno costi di mantenimento 3-5x superiori rispetto ai fast moving
- **Rischio obsolescenza**: 60-80% del rischio obsolescenza deriva da prodotti slow moving

---

## 📊 Definizioni e Classificazioni

### Fast Moving Items
**Definizione**: Prodotti con alta frequenza di vendita e rotazione di magazzino elevata.

**Caratteristiche:**
- **Turnover Rate**: > 12 volte/anno
- **Frequenza vendita**: Giornaliera o settimanale
- **Domanda**: Stabile e prevedibile
- **Lead time accettabile**: Breve (1-7 giorni)
- **Rischio obsolescenza**: Basso

### Slow Moving Items
**Definizione**: Prodotti con bassa frequenza di vendita e lunga permanenza in magazzino.

**Caratteristiche:**
- **Turnover Rate**: < 6 volte/anno
- **Frequenza vendita**: Mensile o trimestrale
- **Domanda**: Irregolare e difficile da prevedere
- **Lead time accettabile**: Lungo (15-45 giorni)
- **Rischio obsolescenza**: Alto

### Classificazione Dettagliata

| Categoria | Turnover/Anno | Giorni Giacenza | % Vendite | Strategia Base |
|-----------|---------------|-----------------|-----------|----------------|
| **Super Fast** | > 24x | < 15 gg | Giornaliere | JIT, Consignment stock |
| **Fast Moving** | 12-24x | 15-30 gg | Settimanali | EOQ ottimizzato, Safety stock minimo |
| **Medium Moving** | 6-12x | 30-60 gg | Bisettimanali | Bilanciamento standard |
| **Slow Moving** | 3-6x | 60-120 gg | Mensili | Min-Max, Lotti piccoli |
| **Very Slow** | 1-3x | 120-365 gg | Trimestrali | Make-to-order |
| **Dead Stock** | < 1x | > 365 gg | Rare/Nulle | Liquidazione |

---

## 📈 Metriche Chiave

### 1. Inventory Turnover Rate (ITR)
```
ITR = Costo del Venduto Annuo / Valore Medio Giacenza
```
- **Fast Moving**: ITR > 12
- **Slow Moving**: ITR < 6

### 2. Days of Supply (DOS)
```
DOS = Giacenza Attuale / Consumo Medio Giornaliero
```
- **Fast Moving**: DOS < 30 giorni
- **Slow Moving**: DOS > 60 giorni

### 3. Coefficiente di Variazione (CV)
```
CV = Deviazione Standard Domanda / Domanda Media
```
- **Stabile (X)**: CV < 0.5
- **Variabile (Y)**: 0.5 ≤ CV ≤ 1.0
- **Erratico (Z)**: CV > 1.0

### 4. Stock Cover Ratio
```
SCR = Mesi di Copertura = Giacenza / Vendite Mensili Medie
```
- **Fast Moving**: SCR < 1 mese
- **Slow Moving**: SCR > 3 mesi

### 5. Obsolescence Risk Index (ORI)
```
ORI = (Età Media Stock × CV) / ITR
```
- **Basso rischio**: ORI < 1
- **Alto rischio**: ORI > 3

---

## 🎨 Strategie di Gestione

### Strategie per Fast Moving

#### 1. **Continuous Review (s,Q)**
- Monitoraggio continuo del livello scorte
- Ordine di quantità fissa Q quando si raggiunge il punto s
- Ideale per prodotti A-Fast con domanda stabile

**Parametri:**
```
s = Lead Time Demand + Safety Stock
Q = EOQ = √(2DS/H)
```

#### 2. **Periodic Review (R,S)**
- Revisione a intervalli fissi R
- Riordino fino al livello obiettivo S
- Adatto per consolidamento ordini

**Parametri:**
```
R = Periodo revisione (es. settimanale)
S = μ(R+L) + z×σ×√(R+L)
```

#### 3. **Just-In-Time (JIT)**
- Minimizzazione scorte
- Consegne frequenti in piccoli lotti
- Richiede fornitori affidabili

### Strategie per Slow Moving

#### 1. **Min-Max Policy**
- Stock minimo = Safety stock ridotto
- Stock massimo = Min + Lotto economico
- Riordino quando si raggiunge il minimo

**Parametri:**
```
Min = SS × 0.5 (ridotto del 50%)
Max = Min + EOQ × 0.5 (lotti piccoli)
```

#### 2. **Make-to-Order (MTO)**
- Zero stock per articoli very slow
- Produzione/acquisto su ordine cliente
- Elimina rischio obsolescenza

#### 3. **Pooling Strategy**
- Centralizzazione stock in hub principale
- Distribuzione on-demand a filiali
- Riduce stock totale del 30-40%

---

## 💼 Casi Pratici Reali

### Caso 1: Azienda Farmaceutica - Gestione Farmaci

**Contesto**: Distributore farmaceutico con 10,000 SKU

**Problema Iniziale:**
- 70% prodotti slow moving
- €5M capitale immobilizzato in dead stock
- Scadenze frequenti per farmaci specialistici

**Classificazione Implementata:**

| Categoria | Prodotti | Esempi | Strategia Applicata |
|-----------|----------|---------|---------------------|
| **Fast A** | Farmaci da banco | Aspirina, Tachipirina | Consignment stock, riordino automatico |
| **Fast B** | Antibiotici comuni | Amoxicillina | EOQ settimanale, SS 98% |
| **Slow A** | Farmaci oncologici | Chemioterapici | Stock minimo garantito, MTO per eccedenze |
| **Slow C** | Farmaci rari | Malattie rare | 100% MTO, accordi con ospedali |

**Risultati:**
- -45% capitale immobilizzato (€2.25M liberati)
- -80% scaduti (-€400k/anno)
- +5% disponibilità farmaci critici

### Caso 2: E-commerce Fashion - Gestione Stagionalità

**Contesto**: Retailer online abbigliamento, 50,000 SKU

**Problema Iniziale:**
- Fine stagione con 40% invenduto
- Svalutazioni del 60% su collezioni precedenti
- Magazzino saturo

**Analisi Movimento per Categoria:**

```
FAST MOVING (20% SKU, 75% fatturato):
- T-shirt basiche
- Jeans classici
- Sneakers popolari
→ Strategia: Replenishment continuo, forecast ML

MEDIUM MOVING (30% SKU, 20% fatturato):
- Camicie business
- Accessori standard
→ Strategia: Ordini bisettimanali, SS moderato

SLOW MOVING (50% SKU, 5% fatturato):
- Taglie estreme (XS, XXXL)
- Colori particolari
- Edizioni limitate
→ Strategia: Dropshipping, virtual inventory
```

**Risultati:**
- -35% giacenza media
- -50% svalutazioni end-of-season
- +15% margine lordo

### Caso 3: Ricambi Auto - Aftermarket

**Contesto**: Distributore ricambi con 200,000 codici

**Classificazione ABC-XYZ implementata:**

| | X (Stabile) | Y (Variabile) | Z (Erratico) |
|---|------------|---------------|--------------|
| **A (Alto valore)** | Filtri olio comuni<br>*JIT, VMI* | Freni premium<br>*Buffer 95%* | Motori rigenerati<br>*MTO* |
| **B (Medio valore)** | Candele standard<br>*EOQ mensile* | Ammortizzatori<br>*Min-Max* | Kit frizione<br>*Cross-docking* |
| **C (Basso valore)** | Viti, bulloni<br>*Kanban* | Guarnizioni<br>*Consolidato* | Ricambi vintage<br>*Marketplace* |

**Innovazioni Applicate:**
1. **Virtual Pooling**: Database condiviso tra 5 magazzini regionali
2. **Dynamic Classification**: Riclassificazione automatica mensile
3. **Predictive Ordering**: ML per anticipare picchi domanda

**Risultati:**
- Fill rate: 92% → 97%
- Inventory turns: 4.2 → 7.8
- Working capital: -€3.5M (-28%)

### Caso 4: Dispositivi Medici Ospedalieri

**Contesto**: Fornitore ospedaliero, 5,000 SKU

**Segmentazione Critica:**

```python
# Matrice Criticità vs Movimento
CRITICO_FAST:
- Siringhe, guanti
- Strategy: Buffer 99.9%, consignment

CRITICO_SLOW:  
- Defibrillatori, ventilatori
- Strategy: Stock minimo garantito

NON_CRITICO_FAST:
- Materiale ufficio medico
- Strategy: EOQ standard

NON_CRITICO_SLOW:
- Accessori specialistici
- Strategy: MTO o pooling regionale
```

**KPI Raggiunti:**
- Stockout critici: 5% → 0.1%
- Obsolescenza: -€500k/anno
- Emergency orders: -75%

---

## 🔢 Formule e Calcoli

### Safety Stock Differenziato

**Fast Moving (distribuzione normale):**
```
SS_fast = z × σ_d × √LT
dove:
- z = z-score per service level desiderato
- σ_d = deviazione standard domanda
- LT = lead time
```

**Slow Moving (metodo Croston):**
```
SS_slow = k × MAD × √(LT/p)
dove:
- k = fattore sicurezza (1.25 per 90% SL)
- MAD = Mean Absolute Deviation
- p = probabilità di domanda in periodo
```

### Economic Order Quantity Adattato

**Fast Moving Standard:**
```
EOQ = √(2 × D × S / H)
```

**Slow Moving Modificato:**
```
EOQ_slow = EOQ_standard × √(p) × RF
dove:
- p = probabilità domanda
- RF = Risk Factor (0.5-0.7)
```

### Calcolo Punto di Riordino

**Fast Moving:**
```
ROP = (d_avg × LT) + SS
```

**Slow Moving (Poisson):**
```
ROP = POISSON.INV(SL, λ × LT)
dove λ = tasso medio arrivi
```

### Esempio Calcolo Completo

**Prodotto Fast Moving:**
```
Dati:
- Domanda media (d): 100 unità/giorno
- Deviazione standard (σ): 20 unità
- Lead time (LT): 5 giorni
- Service Level: 95% (z = 1.65)
- Costo ordine (S): €50
- Costo mantenimento (H): 20% × €10 = €2

Calcoli:
SS = 1.65 × 20 × √5 = 74 unità
EOQ = √(2 × 36,500 × 50 / 2) = 1,350 unità
ROP = (100 × 5) + 74 = 574 unità
```

**Prodotto Slow Moving:**
```
Dati:
- Domanda media: 2 unità/settimana
- Pattern: 70% settimane zero domanda
- Lead time: 4 settimane
- Service Level target: 90%

Calcoli (Croston):
- Domanda quando presente: 2/0.3 = 6.67 unità
- Intervallo medio: 1/0.3 = 3.33 settimane
- SS = 1.25 × 2.5 × √(4/0.3) = 11 unità
- ROP = 8 + 11 = 19 unità
```

---

## ✅ Best Practices

### 1. Segmentazione Dinamica
- **Revisione mensile** delle classificazioni
- **Alert automatici** per cambio categoria
- **Stagionalità** considerata nella classificazione

### 2. Politiche Differenziate

| Aspetto | Fast Moving | Slow Moving |
|---------|------------|-------------|
| **Forecast** | Time series, ML | Croston, Bootstrap |
| **Review** | Continuo/Giornaliero | Periodico/Mensile |
| **Supplier** | Partnership strategiche | Ordini spot |
| **Storage** | Prime locations | Aree remote/verticali |
| **Handling** | Automazione | Picking manuale |
| **Disposal** | Raro | Policy obsolescenza |

### 3. Technology Stack Consigliato

**Fast Moving:**
- WMS con RF/Voice picking
- Sistemi di replenishment automatico
- Integration VMI (Vendor Managed Inventory)
- Forecast engine ML/AI

**Slow Moving:**
- Inventory optimization software
- Marketplace integration per liquidazione
- Pooling network systems
- Make-to-order workflow automation

### 4. Metriche di Monitoraggio

**Dashboard Fast Moving:**
```
- Fill Rate (target >98%)
- Stockout events/mese
- Inventory turns
- Order cycle time
```

**Dashboard Slow Moving:**
```
- Aging analysis
- Obsolescence provision
- Dead stock %
- Storage cost/unit
```

### 5. Gestione del Cambiamento

**Da Slow a Fast (crescita domanda):**
1. Alert precoce su trend crescente
2. Aumento graduale safety stock
3. Negoziazione nuovi termini fornitore
4. Cambio location magazzino

**Da Fast a Slow (declino domanda):**
1. Riduzione immediata ordini
2. Promozioni per smaltimento
3. Valutazione delisting
4. Accordi resi con fornitore

---

## ⚠️ Errori Comuni da Evitare

### 1. One-Size-Fits-All Approach
❌ **Errore**: Applicare stesse politiche a tutti i prodotti
✅ **Soluzione**: Segmentazione minimo 4-6 categorie

### 2. Classificazione Statica
❌ **Errore**: Classificare una volta l'anno
✅ **Soluzione**: Review dinamico mensile/trimestrale

### 3. Ignorare il Ciclo di Vita
❌ **Errore**: Non considerare fase del prodotto (lancio/maturità/declino)
✅ **Soluzione**: Fattori correttivi per lifecycle stage

### 4. Focus Solo su Turnover
❌ **Errore**: Classificare solo per velocità rotazione
✅ **Soluzione**: Matrice multidimensionale (ABC-XYZ-Criticità)

### 5. Sottovalutare Slow Moving
❌ **Errore**: Trascurare gestione slow movers
✅ **Soluzione**: Slow moving spesso = margini alti o strategici

### 6. Safety Stock Uniforme
❌ **Errore**: Stesso service level per tutto
✅ **Soluzione**: Service level differenziato per categoria

### 7. Ignorare i Costi Totali
❌ **Errore**: Guardare solo costo prodotto
✅ **Soluzione**: TCO incluso storage, handling, obsolescenza

### 8. Mancanza di Exit Strategy
❌ **Errore**: Non avere piano per dead stock
✅ **Soluzione**: Policy chiara liquidazione/write-off

---

## 📊 Template Analisi

### Quick Assessment Checklist

```markdown
□ Turnover rate calcolato per SKU
□ Classificazione ABC completata
□ Analisi XYZ variabilità
□ Matrice combinata ABC-XYZ
□ Costi di mantenimento per categoria
□ Lead time per fornitore/categoria
□ Service level target definiti
□ Politiche riordino differenziate
□ KPI monitoraggio implementati
□ Review period stabilito
```

### Matrice Decisionale Rapida

| Turnover | Valore | Variabilità | → Azione |
|----------|--------|-------------|----------|
| Alto | Alto | Bassa | → JIT/VMI |
| Alto | Alto | Alta | → Buffer++ |
| Alto | Basso | Bassa | → Kanban |
| Basso | Alto | Bassa | → Min stock |
| Basso | Alto | Alta | → MTO |
| Basso | Basso | Alta | → Delist |

---

## 🚀 Implementazione Pratica

### Fase 1: Analisi (2-4 settimane)
1. Estrazione dati 12-24 mesi
2. Calcolo metriche base
3. Classificazione iniziale
4. Identificazione quick wins

### Fase 2: Pilot (4-8 settimane)
1. Selezione categoria test (es. Fast A)
2. Implementazione nuove policy
3. Monitoraggio giornaliero KPI
4. Adjustments real-time

### Fase 3: Rollout (3-6 mesi)
1. Estensione graduale categorie
2. Automazione classificazione
3. Training team
4. Integration sistemi

### Fase 4: Ottimizzazione (ongoing)
1. Machine learning per forecast
2. Dynamic classification
3. Supplier collaboration
4. Continuous improvement

---

## 📚 Riferimenti e Approfondimenti

### Letteratura Accademica
- Silver, E.A., Pyke, D.F., Peterson, R. (1998). *Inventory Management and Production Planning and Scheduling*
- Syntetos, A.A., Boylan, J.E. (2005). *The accuracy of intermittent demand estimates*
- Teunter, R.H., Syntetos, A.A., Babai, M.Z. (2011). *Intermittent demand: Linking forecasting to inventory obsolescence*

### Standard e Framework
- **APICS/ASCM**: Classification guidelines
- **ISO 9001:2015**: Inventory management requirements
- **SCOR Model**: Inventory classification metrics

### Tools e Software
- **SAP IBP**: Integrated Business Planning
- **Oracle Inventory Optimization**
- **Manhattan Associates**: Slotting Optimization
- **Llamasoft/Coupa**: Network inventory optimization

---

## 🎯 Conclusioni

La gestione differenziata Slow/Fast Moving non è solo una tecnica di inventory management, ma una **strategia competitiva** che impatta:

- **Financial Performance**: ROI, Working Capital, Cash Flow
- **Customer Service**: Fill Rate, Lead Time, Availability  
- **Operational Efficiency**: Storage costs, Handling, Obsolescence
- **Strategic Positioning**: Product range, Market responsiveness

Il successo dipende da:
1. **Classificazione accurata e dinamica**
2. **Politiche differenziate e coerenti**
3. **Monitoraggio continuo e adjustment**
4. **Integrazione cross-funzionale** (Sales, Finance, Operations)
5. **Technology enablement** appropriato

L'obiettivo finale è trovare il **bilanciamento ottimale** tra:
- **Disponibilità** per fast movers critici
- **Efficienza** di capitale per slow movers
- **Flessibilità** per rispondere ai cambiamenti di mercato

---

*Documento aggiornato al: 02/09/2025*
*Versione: 1.0*
*Autore: Sistema ARIMA Forecaster - Inventory Optimization Module*