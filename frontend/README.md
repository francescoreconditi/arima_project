# ARIMA Forecaster - Frontend Angular

Frontend Angular moderno per l'API REST di ARIMA Forecaster, che permette di addestrare modelli di time series forecasting e generare previsioni tramite interfaccia web.

**Generato con**: [Angular CLI](https://github.com/angular/angular-cli) version 20.3.3

## 📋 Funzionalità

### ✅ Implementate

1. **Training Modelli**
   - Supporto ARIMA, SARIMA e SARIMAX
   - Configurazione parametri (p, d, q) e parametri stagionali (P, D, Q, s)
   - Input dati CSV diretto con esempio predefinito
   - Polling automatico stato training
   - Visualizzazione metriche performance

2. **Generazione Forecast**
   - Selezione modello addestrato
   - Configurazione passi forecast e livello confidenza
   - Intervalli di confidenza opzionali
   - Visualizzazione grafica risultati
   - Tabella dati dettagliati
   - Statistiche summary (media, min, max, std dev)
   - Export CSV risultati

3. **Lista Modelli**
   - Visualizzazione tutti i modelli addestrati
   - Filtro per stato (completed, training, failed)

## 🚀 Quick Start

### Prerequisiti

- Node.js (v18+) e npm (v9+)
- Backend FastAPI in esecuzione su http://localhost:8000

### Installazione e Avvio

```bash
# Installa dipendenze
npm install

# Avvia server di sviluppo
ng serve
```

L'applicazione sarà disponibile su `http://localhost:4200/`. Il server ricaricherà automaticamente i cambiamenti.

## Code scaffolding

Angular CLI includes powerful code scaffolding tools. To generate a new component, run:

```bash
ng generate component component-name
```

For a complete list of available schematics (such as `components`, `directives`, or `pipes`), run:

```bash
ng generate --help
```

## Building

To build the project run:

```bash
ng build
```

This will compile your project and store the build artifacts in the `dist/` directory. By default, the production build optimizes your application for performance and speed.

## Running unit tests

To execute unit tests with the [Karma](https://karma-runner.github.io) test runner, use the following command:

```bash
ng test
```

## Running end-to-end tests

For end-to-end (e2e) testing, run:

```bash
ng e2e
```

Angular CLI does not come with an end-to-end testing framework by default. You can choose one that suits your needs.

## Additional Resources

For more information on using the Angular CLI, including detailed command references, visit the [Angular CLI Overview and Command Reference](https://angular.dev/tools/cli) page.
