# Auto-Training Component

## Panoramica

Il componente `AutoTraining` permette agli utenti di utilizzare la funzionalità di selezione automatica dei parametri ottimali per modelli ARIMA, SARIMA e SARIMAX. Il componente esegue una ricerca grid search automatica e presenta all'utente una lista di modelli candidati ordinati per AIC/BIC, permettendo di selezionare e addestrare il modello desiderato.

## Funzionalità

### 1. Selezione Modello Automatica
- **Auto-ARIMA**: Ricerca automatica parametri (p,d,q) per modelli ARIMA
- **Auto-SARIMA**: Ricerca automatica parametri stagionali (P,D,Q,s) in aggiunta ai parametri base
- **Auto-SARIMAX**: Come SARIMA ma con supporto variabili esogene

### 2. Parametri Configurabili
- **max_p**: Valore massimo per parametro AR (default: 3)
- **max_d**: Valore massimo per differencing (default: 2)
- **max_q**: Valore massimo per parametro MA (default: 3)
- **seasonal_period**: Periodo stagionale per SARIMA/SARIMAX (default: 12)
- **criterion**: Criterio selezione AIC o BIC (default: AIC)
- **max_models**: Numero massimo modelli candidati da visualizzare (default: 20)

### 3. Workflow
1. **Configurazione**: L'utente seleziona tipo modello e parametri ricerca
2. **Input Dati**: L'utente inserisce dati CSV (formato: data,valore) o carica dati esempio
3. **Ricerca Automatica**: Click su "Avvia Ricerca" esegue grid search sul backend
4. **Visualizzazione Candidati**: Lista modelli ordinati per AIC/BIC con metriche
5. **Selezione**: Click su card modello per selezionarlo
6. **Training**: Click su "Addestra Selezionato" per trainare il modello scelto
7. **Risultati**: Visualizzazione model_id, status e metriche del modello addestrato

## Architettura

### Component Structure
```
auto-training/
├── auto-training.ts      # Logica TypeScript
├── auto-training.html    # Template UI
├── auto-training.scss    # Stili CSS
└── README.md            # Documentazione
```

### TypeScript Logic

#### Interface ModelCandidate
```typescript
interface ModelCandidate {
  order: number[];           // [p, d, q]
  seasonal_order?: number[]; // [P, D, Q, s] (opzionale)
  aic?: number;             // Valore AIC
  bic?: number;             // Valore BIC
  selected: boolean;        // Flag selezione UI
}
```

#### Metodi Principali

**onSearchModels()**
- Parse dati CSV input
- Costruisce `AutoSelectionRequest`
- Chiama API `/models/auto-select`
- Popola array `modelCandidates` con risultati
- Seleziona automaticamente primo modello (migliore)

**selectModel(index: number)**
- Aggiorna selezione UI
- Imposta flag `selected` sul modello scelto

**onTrainSelectedModel()**
- Verifica selezione modello valida
- Determina tipo modello da parametri
- Costruisce `ModelTrainingRequest`
- Chiama API `/models/train`
- Avvia polling status training

**pollModelStatus(modelId: string)**
- Verifica periodicamente stato training
- Intervallo: 2 secondi
- Max tentativi: 30 (timeout 60s)
- Aggiorna `trainedModel` con risultati

### API Integration

Il componente utilizza i seguenti endpoint FastAPI:

#### POST /models/auto-select
**Request:**
```typescript
{
  data: TimeSeriesData,
  max_p?: number,
  max_d?: number,
  max_q?: number,
  seasonal?: boolean,
  seasonal_period?: number,
  criterion?: 'aic' | 'bic'
}
```

**Response:**
```typescript
{
  best_model: {
    order: number[],
    seasonal_order?: number[],
    aic?: number,
    bic?: number
  },
  all_results: Array<{...}>,
  models_tested: number,
  search_time_seconds: number
}
```

#### POST /models/train
**Request:**
```typescript
{
  model_type: 'arima' | 'sarima' | 'sarimax',
  data: TimeSeriesData,
  order: ModelOrder,
  seasonal_order?: SeasonalOrder
}
```

**Response:**
```typescript
{
  model_id: string,
  model_type: string,
  status: 'training' | 'completed' | 'failed',
  created_at: string,
  training_observations: number,
  parameters: Record<string, any>,
  metrics: Record<string, number>
}
```

#### GET /models/{model_id}
Verifica stato modello durante training.

## UI Design

### Layout
- **Header**: Titolo e sottotitolo sezione
- **Config Form**: Form configurazione parametri
- **Search Button**: Pulsante avvio ricerca (disabilitato se no dati)
- **Candidates Section**: Lista card modelli candidati (visibile solo dopo ricerca)
- **Training Actions**: Pulsanti "Addestra Selezionato" e "Reset"
- **Results Section**: Card risultati training (visibile solo dopo training)

### Styling
- **Card-based design**: Ogni modello candidato in card cliccabile
- **Selected state**: Bordo verde e sfondo highlight per modello selezionato
- **Hover effects**: Feedback visivo su hover cards
- **Responsive layout**: Grid layout per parametri e candidati
- **Loading states**: Pulsanti disabilitati durante operazioni async

## Formato Dati CSV

Il componente accetta dati CSV nel formato:
```
data,valore
2024-01-01,100
2024-01-02,105
2024-01-03,103
...
```

- **Separatore**: virgola
- **Header**: opzionale (viene skippato automaticamente)
- **Date format**: qualsiasi formato riconosciuto da backend
- **Values**: numeri decimali (punto come separatore)

## Esempio Utilizzo

1. Navigare su `/auto-training`
2. Selezionare "Auto-SARIMA" come tipo modello
3. Impostare max_p=3, max_d=2, max_q=3, seasonal_period=12
4. Click su "Carica Esempio" per popolare dati
5. Click su "Avvia Ricerca"
6. Attendere risultati grid search (~5-30s)
7. Click su card secondo modello per selezionarlo
8. Click su "Addestra Selezionato"
9. Attendere completamento training
10. Visualizzare model_id e metriche nella sezione risultati

## Gestione Errori

Il componente gestisce i seguenti errori:

- **Parse CSV Error**: Visualizza errore se formato CSV invalido
- **Search Error**: Visualizza messaggio se grid search fallisce
- **Training Error**: Visualizza messaggio se training fallisce
- **Timeout Training**: Messaggio dopo 60s senza completamento
- **Status Check Error**: Retry automatico su errori temporanei

Tutti gli errori sono visualizzati in un banner rosso sotto il form.

## Performance

- **Grid Search**: Da pochi secondi a ~1 minuto (dipende da spazio ricerca)
- **Training**: Da 5 a 30 secondi (dipende da complessità modello e dimensione dati)
- **Polling**: Ogni 2 secondi fino a completamento
- **Max Models Display**: Limitato a 20 per non sovraccaricare UI

## Routing

Il componente è registrato nelle route Angular:
```typescript
{ path: 'auto-training', component: AutoTraining }
```

Link di navigazione disponibile nel menu principale.

## Dipendenze

- **Angular**: CommonModule, FormsModule, RouterModule
- **RxJS**: Observable pattern per chiamate API
- **ArimaApiService**: Servizio per chiamate REST API
- **API Models**: Interfacce TypeScript per request/response

## Future Enhancements

Possibili miglioramenti futuri:
- Visualizzazione grafico AIC/BIC per tutti i modelli testati
- Export CSV lista modelli candidati
- Comparazione side-by-side tra modelli
- Save/Load configurazioni ricerca
- Filtri e ordinamento candidati
- Batch training di top-N modelli
- Integration con sezione forecast per modelli trainati
