# ============================================
# FILE DI TEST/DEBUG - NON PER PRODUZIONE
# Creato da: Claude Code
# Data: 2025-09-03
# Scopo: README per dashboard Moretti Angular
# ============================================

# Dashboard Moretti S.p.A. - Angular Version (Drag & Drop Edition)

Una versione Angular avanzata della dashboard Streamlit esistente per la gestione intelligente delle scorte di Moretti S.p.A., con **funzionalità drag & drop complete**.

## 📋 Panoramica

Questa è una replica funzionalmente equivalente e **migliorata** della dashboard Streamlit Moretti, implementata utilizzando Angular 18+ con **angular-gridster2**. La dashboard fornisce:

### 🎛️ **Funzionalità Core**
- **Visualizzazione KPI aziendali** con metriche in tempo reale
- **Gestione prodotti per categoria** con filtri dinamici
- **Grafici di forecasting** con dati storici e previsioni future
- **Sistema di alert** per monitorare situazioni critiche
- **Interfaccia multilingue** (Italiano, English, Español, Français, 中文)
- **Design responsive** ottimizzato per desktop e mobile

### ✨ **Funzionalità Drag & Drop Avanzate**
- **🖱️ Widget trascinabili** - Sposta liberamente tutti gli elementi dashboard
- **📏 Ridimensionamento dinamico** - Resize con handle agli angoli/bordi
- **💾 Persistenza automatica** - Layout salvato automaticamente in localStorage
- **🎨 Modalità editing** - Toggle edit mode con controlli visuali
- **📋 Template system** - Salva/carica layout personalizzati
- **🔄 Reset layout** - Ripristino rapido configurazione default
- **📱 Responsive drag** - Funziona anche su mobile/tablet

## 🏗️ Architettura

### Struttura del Progetto

```
fe_angular/moretti-dashboard/
├── src/
│   ├── app/
│   │   ├── components/              # Componenti riutilizzabili
│   │   │   ├── metric-card/         # Card per visualizzazione metriche
│   │   │   ├── alert-box/           # Box per avvisi sistema
│   │   │   ├── forecast-chart/      # Grafici Chart.js
│   │   │   └── dashboard-widget/    # 🆕 Wrapper widget drag & drop
│   │   ├── services/               # Servizi per dati e API
│   │   │   ├── data.service.ts      # Servizio dati mock
│   │   │   ├── api.service.ts       # Servizio API FastAPI
│   │   │   └── dashboard-layout.service.ts  # 🆕 Gestione layout drag & drop
│   │   ├── models/                 # Modelli TypeScript
│   │   │   ├── product.model.ts     # Interfacce dati prodotti
│   │   │   └── dashboard-widget.model.ts  # 🆕 Modelli widget drag & drop
│   │   ├── app.ts                  # Componente principale (con drag & drop)
│   │   ├── app.html                # Template principale (con gridster)
│   │   └── app.css                 # Stili principali (con drag & drop)
│   └── main.ts                     # Entry point applicazione
├── package.json                    # Dipendenze npm (+ angular-gridster2)
└── README.md                      # Questo file
```

### Componenti Principali

#### 🏠 App Component (`app.ts`)
- **Responsabilità**: Orchestrazione generale dashboard + gestione drag & drop
- **Funzionalità Base**:
  - Gestione stato applicazione (prodotti, categorie, metriche)
  - Coordinamento tra filtri sidebar e contenuto principale
  - Integrazione servizi dati e API
  - Gestione eventi utente (selezioni, slider scenario)
- **Funzionalità Drag & Drop**:
  - Configurazione **angular-gridster2** con opzioni complete
  - Gestione modalità editing (enable/disable drag & resize)
  - Event handling spostamento/ridimensionamento widget
  - Persistenza automatica layout via **DashboardLayoutService**
  - Controlli header (Toggle Edit, Reset, Save Template)

#### 🧩 DashboardWidgetComponent (NUOVO)
- **Scopo**: Wrapper universale per rendere qualsiasi componente draggable
- **Features**: 
  - Header widget configurabile (show/hide)
  - Controlli modifica/rimozione widget
  - Supporto tutti i tipi: metric, chart, alert, product-info
  - Template switching automatico basato su widget.type
  - Gestione dati dinamica tramite Input properties

#### 📊 MetricCardComponent
- **Scopo**: Visualizzazione KPI aziendali con stile card
- **Features**: Valori numerici, delta trends, formattazione automatica (€, %, numeri)

#### 🚨 AlertBoxComponent  
- **Scopo**: Notifiche sistema per situazioni critiche
- **Livelli**: Critica (rosso), Alta (arancione), Media (giallo)

#### 📈 ForecastChartComponent
- **Scopo**: Grafici interattivi con Chart.js
- **Visualizzazioni**: Dati storici, previsioni future, intervalli confidenza

### Servizi

#### 📡 ApiService
- **Integrazione FastAPI**: Chiamate REST endpoint per forecasting, modelli, inventory
- **Fallback Strategy**: Se API non disponibile, degrada gracefully a dati mock
- **Metodi principali**:
  - `forecast(modelId, steps)` - Previsioni ARIMA/SARIMA
  - `getModels()` - Lista modelli disponibili
  - `trainModel(data)` - Training nuovo modello
  - `getInventoryAnalysis(productCode)` - Analisi inventory management

#### 💾 DataService
- **Dati Mock**: Simulazione realistica per sviluppo/demo
- **Prodotti Sample**: 3 prodotti Moretti (Carrozzina, Materasso, Saturimetro)
- **Generazione Dinamica**: Dati storici e KPI calcolati algoritmicamente

#### 🎛️ DashboardLayoutService (NUOVO)
- **Gestione Layout**: CRUD completo per layout dashboard personalizzati
- **Persistenza**: Salvataggio automatico in localStorage del browser
- **Template System**: Salva/carica configurazioni multiple con nome
- **Layout Default**: Configurazione ottimizzata iniziale (4 metriche + info + chart + alert)
- **Metodi principali**:
  - `updateLayout(widgets)` - Aggiorna posizione/dimensioni widget
  - `saveLayoutAsTemplate(name)` - Salva configurazione corrente
  - `loadTemplate(name)` - Carica configurazione salvata
  - `resetToDefault()` - Ripristina layout iniziale
- **Observable Pattern**: `layout$` stream per aggiornamenti real-time

## 🚀 Setup e Avvio

### Prerequisiti

- **Node.js** 18+ ([Download](https://nodejs.org/))
- **Angular CLI** (`npm install -g @angular/cli`)
- **FastAPI Backend** (opzionale - fallback automatico a mock data)

### Installazione

```bash
# Clona il repository e naviga nella directory
cd C:\ZCS_PRG\arima_project\fe_angular\moretti-dashboard

# Installa dipendenze
npm install

# Avvia server sviluppo
ng serve

# Apri browser a http://localhost:4200
```

### Build Produzione

```bash
# Build ottimizzato
ng build --prod

# File generati in dist/moretti-dashboard/
```

## 🔗 Integrazione API FastAPI

### Configurazione Endpoint

Il servizio Angular è pre-configurato per comunicare con l'API FastAPI su `http://localhost:8000`.

Per modificare l'URL base:

```typescript
// src/app/services/api.service.ts
private baseUrl = 'http://your-api-domain:port';
```

### API Endpoints Supportati

#### Forecasting
- `POST /models/{model_id}/forecast` - Genera previsioni
- `GET /models` - Lista modelli disponibili  
- `POST /models/train/arima` - Training modello ARIMA

#### Inventory Management  
- `GET /inventory/analysis/{product_code}` - Analisi inventory

### Estrategia Fallback

Se l'API FastAPI non è disponibile:
1. **Tentativi di connessione** con timeout configurabile
2. **Degradazione automatica** a dati mock locali
3. **Notifica console** per debugging (non visibile all'utente)
4. **Funzionalità complete** mantenute con dati simulati

Questo garantisce che la dashboard sia sempre funzionante per demo/sviluppo.

## 🎨 Features Implementate

### ✅ Layout & UI
- [x] **Header gradiente** con branding Moretti
- [x] **Sidebar filtri** con controlli interattivi
- [x] **Grid layout responsive** per contenuto principale
- [x] **Card design system** coerente con Streamlit originale

### ✅ Funzionalità Core
- [x] **Filtri categoria** con aggiornamento dinamico prodotti
- [x] **Selezione prodotto** con dettagli completi
- [x] **Slider scenario** per Lead Time e Domanda (% modificatori)
- [x] **Controllo giorni previsione** (7-90 giorni, step 7)

### ✅ Visualizzazioni Dati  
- [x] **Metriche KPI** con delta trends e formattazione
- [x] **Grafici Chart.js** storici + previsioni + intervalli confidenza
- [x] **Sistema alert** con codifica colori criticità
- [x] **Dettagli prodotto** in layout card strutturato

### ✅ Integrazione Backend
- [x] **ApiService** per chiamate FastAPI
- [x] **DataService** per dati mock/fallback  
- [x] **Gestione errori** trasparente per utente
- [x] **Mock data realistici** per 3 prodotti Moretti

### 🆕 **Sistema Drag & Drop Completo**
- [x] **🖱️ Drag & Drop Universale** - Trascina qualsiasi widget liberamente
- [x] **📏 Resize Handles** - Ridimensiona con 8 handle (angoli + lati)
- [x] **🎛️ Modalità Editing** - Toggle "✏️ Modifica" / "🔒 Blocca" in header
- [x] **💾 Persistenza Automatica** - Layout salvato in localStorage
- [x] **📋 Template System** - Salva configurazioni con nome personalizzato
- [x] **🔄 Reset Layout** - Ripristino rapido configurazione default
- [x] **🎨 Visual Feedback** - Handle colorati, preview ghost, drag handlers
- [x] **📱 Touch Support** - Funziona su dispositivi touch/mobile
- [x] **⚡ Performance** - Grid virtualization per dashboard complesse
- [x] **🔧 API Integration** - Drag & drop completamente integrato con dati live

## 🌐 Supporto Multilingue

Attualmente implementato il **framework multilingue**:
- **Selector lingua** in header (5 lingue disponibili)
- **Event handler** `onLanguageChange()` per switch  
- **Preparazione** per integrazione sistema traduzioni centralizzato

**Note**: Per traduzioni complete, integrare con:
- Sistema traduzioni progetto principale (`src/arima_forecaster/utils/translations.py`)
- File JSON localizzazioni (`src/arima_forecaster/assets/locales/`)

## 📱 Responsive Design

### Breakpoints
- **Desktop** (1200px+): Layout full sidebar + main content
- **Tablet** (768px-1199px): Sidebar ridotta, layout ottimizzato
- **Mobile** (< 768px): Stack verticale header/sidebar/main

### Adattamenti Mobile
- **Grid metriche**: Da 4 colonne a 1 colonna
- **Sidebar**: Da pannello laterale a sezione superiore
- **Dettagli prodotto**: Layout single-column
- **Chart container**: Ridimensionamento automatico

## 🔄 Confronto con Versione Streamlit

| Feature | Streamlit | Angular | Status |
|---------|-----------|---------|---------|
| Layout Dashboard | ✅ | ✅ | ✅ Equivalente |
| Filtri Categoria/Prodotto | ✅ | ✅ | ✅ Equivalente |
| Metriche KPI | ✅ | ✅ | ✅ Equivalente |
| Grafici Forecasting | ✅ | ✅ | ✅ Equivalente |
| Sistema Alert | ✅ | ✅ | ✅ Equivalente |
| Slider Scenario | ✅ | ✅ | ✅ Equivalente |
| Multilingue | ✅ | 🔄 | 🔄 In sviluppo |
| API Integration | ✅ | ✅ | ✅ Con fallback |
| Mobile Responsive | ❌ | ✅ | ✅ Miglioramento |

## 🛠️ Sviluppo e Debug

### Server Sviluppo
```bash
ng serve --open          # Avvia + apri browser
ng serve --port 4201     # Porta personalizzata
ng serve --host 0.0.0.0  # Accesso rete locale
```

### Debug Tools
- **Angular DevTools**: [Installazione Chrome Extension](https://angular.io/guide/devtools)
- **Console Logging**: Attivato per API calls e data loading
- **Hot Reload**: Modifiche codice riflesse automaticamente

### Testing
```bash
ng test              # Unit tests con Jasmine/Karma
ng e2e              # End-to-end tests (se configurati)
ng lint             # Code quality checks
```

## 📋 TODO Future

### 🔄 Prossimi Sviluppi
- [ ] **Traduzioni complete** integrazione sistema centralizzato
- [ ] **Testing suite** unit tests per componenti e servizi  
- [ ] **Ottimizzazioni performance** lazy loading e caching
- [ ] **PWA features** service worker per offline functionality
- [ ] **Authentication** integrazione sistema login
- [ ] **Real-time updates** WebSocket per dati live

### 🎯 Enhancement Potenziali
- [ ] **Dark mode** toggle tema scuro/chiaro
- [ ] **Export funzioni** CSV/PDF per grafici e report
- [ ] **Notifiche push** per alert critici
- [ ] **Dashboard customization** drag&drop widget layout
- [ ] **Advanced filtering** filtri multipli e saved views

## 🐛 Known Issues

1. **Chart.js rendering**: Potenziali problemi ridimensionamento mobile
2. **API timeout**: Default 2 minuti, configurabile se necessario  
3. **Mock data**: Dati simulati, potrebbero non riflettere esattamente business logic reale

## 📞 Supporto

Per problemi tecnici o richieste:
- **Issues**: Utilizzare sistema ticketing progetto principale
- **Documentation**: Riferimento CLAUDE.md del progetto principale
- **API Documentation**: FastAPI `/docs` endpoint per reference completa

---

## 🎛️ Come Utilizzare il Sistema Drag & Drop

### 🚀 Avvio Rapido
1. **Apri la dashboard**: `ng serve` → `http://localhost:4200`  
2. **Entra in modalità editing**: Clicca "✏️ Modifica" nell'header  
3. **Trascina i widget**: Afferra dalla barra blu in cima e sposta  
4. **Ridimensiona**: Usa gli handle agli angoli per cambiare dimensioni  
5. **Blocca le modifiche**: Clicca "🔒 Blocca" per fissare il layout

### 🎨 Controlli Dashboard
| Bottone | Funzione | Descrizione |
|---------|----------|-------------|
| **✏️ Modifica** | Abilita editing | Mostra drag handles e resize controls |
| **🔒 Blocca** | Disabilita editing | Nasconde controlli, layout fisso |
| **🔄 Reset** | Ripristina default | Torna al layout iniziale ottimizzato |
| **💾 Salva** | Salva template | Crea template riutilizzabile con nome |

### 📐 Widget Grid System
- **Griglia 12 colonne**: Layout responsive basato su CSS Grid  
- **Altezza flessibile**: Ridimensiona verticalmente senza limiti  
- **Snap to Grid**: Allineamento automatico durante trascinamento  
- **Anti-collision**: I widget si spostano per evitare sovrapposizioni  
- **Mobile Friendly**: Su mobile (<768px) griglia si adatta automaticamente

### 💾 Sistema Persistenza
```typescript
// Layout salvato automaticamente a ogni modifica
localStorage: 'moretti_dashboard_layout'

// Template salvati dall'utente
localStorage: 'moretti_dashboard_layout_templates'

// Struttura dati:
{
  widgets: DashboardWidget[],  // Posizioni e dimensioni
  lastModified: Date,          // Timestamp ultima modifica  
  name: string                 // Nome template
}
```

### 🔧 Personalizzazione Avanzata
Per personalizzare ulteriormente il comportamento:

```typescript
// In app.ts, modifica gridsterOptions:
this.gridsterOptions = {
  margin: 10,              // Spazio tra widget (px)
  minCols: 12,            // Colonne minime griglia
  maxCols: 12,            // Colonne massime griglia  
  defaultItemCols: 3,     // Larghezza default nuovo widget
  defaultItemRows: 2,     // Altezza default nuovo widget
  pushItems: true,        // Sposta widget esistenti se necessario
  // ... altre opzioni
};
```

---

**🏥 Moretti S.p.A. - Sistema Gestione Scorte Intelligente v2.0**  
*Versione Angular Dashboard con Drag & Drop - Settembre 2024*