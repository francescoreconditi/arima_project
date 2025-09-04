# ARIMA Dashboard Angular

Dashboard Angular moderna e modulare per il sistema di forecasting ARIMA, ispirata alla Dashboard Moretti con integrazione FastAPI.

## ✨ Caratteristiche Principali

- **🎨 Design Moderno**: UI Material Design pulita e professionale
- **📊 Grafici Interattivi**: Visualizzazioni dinamiche con Chart.js e Plotly
- **🌍 Multilingua**: Supporto per 5 lingue (IT, EN, ES, FR, ZH)
- **🌙 Temi Multipli**: Sistema di temi SCSS modulare (default + dark mode)
- **📱 Responsive**: Layout completamente responsivo
- **🔌 API Integration**: Comunicazione con FastAPI backend
- **⚡ Performance**: Lazy loading e ottimizzazioni avanzate
- **🧩 Modulare**: Architettura scalabile e mantenibile

## 🏗️ Architettura

```
src/app/
├── core/                   # Servizi core, modelli, interceptors
│   ├── models/            # Interfacce TypeScript
│   ├── services/          # Servizi API e business logic
│   └── interceptors/      # HTTP interceptors
├── features/              # Moduli feature
│   └── dashboard/         # Dashboard principale
│       └── components/    # Componenti specifici
├── shared/               # Componenti e moduli condivisi
└── assets/
    ├── themes/           # Sistema temi SCSS
    │   └── default/      # Tema default
    └── i18n/            # File traduzioni
```

## 🚀 Setup e Avvio

### Prerequisiti
- Node.js 18+
- Angular CLI 17+
- Backend FastAPI in esecuzione (porta 8000)

### Installazione

```bash
# Installa dipendenze
npm install

# Avvia server di sviluppo
npm start
# Oppure
ng serve

# Build per produzione
npm run build
```

L'applicazione sarà disponibile su: `http://localhost:4200`

## 🎨 Sistema di Temi

Il sistema di temi è completamente modulare:

```scss
// assets/themes/default/
├── _variables.scss    # Variabili colori, spacing, typography
├── _mixins.scss      # Mixins riutilizzabili  
└── theme.scss        # Tema completo
```

### Personalizzazione Tema

1. Duplica la cartella `default` con un nuovo nome
2. Modifica le variabili in `_variables.scss`
3. Importa il nuovo tema in `styles.scss`

## 📊 Componenti Dashboard

### KPI Cards
Componenti riutilizzabili per visualizzare metriche chiave:

```typescript
interface KpiData {
  title: string;
  value: number | string;
  unit?: string;
  trend?: 'up' | 'down' | 'stable';
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'danger';
  target?: number;
}
```

### Charts
- **Line Charts**: Andamenti temporali e previsioni
- **Bar Charts**: Confronti inventario 
- **Pie Charts**: Distribuzioni categorie
- **Area Charts**: Trend con intervalli confidenza

### Tabelle Dinamiche
- Ordinamento e filtri
- Paginazione automatica
- Export CSV/Excel
- Azioni bulk

## 🔌 Integrazione API

### Configurazione Endpoints

```typescript
// environments/environment.ts
export const environment = {
  apiUrl: 'http://localhost:8000/api',
  wsUrl: 'ws://localhost:8000/ws'
};
```

### Servizi Principali

- **ForecastService**: Gestione previsioni ARIMA/SARIMA
- **InventoryService**: Gestione inventario e ottimizzazione  
- **ApiService**: HTTP client base con error handling

## 🌍 Sistema Traduzioni

Le traduzioni sono centralizzate in:

```typescript
// Utilizzo
import { TranslateService } from '@ngx-translate/core';

// Cambio lingua
this.translate.use('en');

// Traduzione template
{{ 'DASHBOARD.TITLE' | translate }}
```

## 📱 Responsive Design

Breakpoints ottimizzati:

- **Mobile**: < 576px
- **Tablet**: 576px - 768px  
- **Desktop**: 768px - 992px
- **Large**: 992px - 1200px
- **XL**: > 1200px

## ⚡ Performance

### Ottimizzazioni Implementate

- **Lazy Loading**: Caricamento moduli on-demand
- **OnPush Strategy**: Change detection ottimizzata
- **Service Workers**: Caching intelligente
- **Bundle Splitting**: Code splitting automatico
- **Tree Shaking**: Eliminazione codice non utilizzato

### Metriche Target

- **First Contentful Paint**: < 1.5s
- **Time to Interactive**: < 3s
- **Bundle Size**: < 2MB
- **Lighthouse Score**: > 90

## 🧪 Testing

```bash
# Unit tests
ng test

# E2E tests  
ng e2e

# Coverage report
ng test --code-coverage
```

## 🔧 Configurazione Avanzata

### Environment Variables

```typescript
export const environment = {
  production: false,
  apiUrl: 'http://localhost:8000/api',
  features: {
    darkMode: true,
    multiLanguage: true,
    realTimeUpdates: true,
    exportData: true
  }
};
```

### Build Configuration

```json
{
  "budgets": [
    {
      "type": "initial",
      "maximumWarning": "2mb",
      "maximumError": "5mb"
    }
  ]
}
```

## 📦 Deployment

### Docker

```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
```

### Build Commands

```bash
# Development
ng build

# Production  
ng build --configuration=production

# Analisi bundle
ng build --stats-json
npx webpack-bundle-analyzer dist/stats.json
```

## 🔍 Monitoring e Debug

### Debug Tools

- **Angular DevTools**: Chrome extension
- **Redux DevTools**: State management
- **Network Tab**: API calls monitoring
- **Performance Tab**: Bottleneck analysis

### Logging

```typescript
// Configurazione logging
export const environment = {
  logLevel: 'debug', // 'error' | 'warn' | 'info' | 'debug'
  enableAnalytics: true
};
```

## 🤝 Sviluppo

### Coding Standards

- **ESLint + Prettier**: Code formatting automatico
- **Conventional Commits**: Commit message standard
- **Husky**: Pre-commit hooks
- **TypeScript Strict**: Type safety completo

### Branch Strategy

```
main            # Produzione
develop         # Sviluppo
feature/*       # Nuove funzionalità  
bugfix/*        # Fix bugs
hotfix/*        # Fix urgenti produzione
```

## 📋 Roadmap

### v1.1.0 - Q1 2025
- [ ] PWA Support
- [ ] Real-time WebSocket updates
- [ ] Advanced filtering system
- [ ] Mobile app companion

### v1.2.0 - Q2 2025  
- [ ] Machine Learning predictions
- [ ] Custom dashboard builder
- [ ] API rate limiting
- [ ] Advanced user roles

### v2.0.0 - Q3 2025
- [ ] Micro-frontends architecture
- [ ] GraphQL integration
- [ ] Offline-first support
- [ ] White-label customization

## 🐛 Troubleshooting

### Problemi Comuni

**Build Failures**
```bash
# Pulisci cache node_modules
rm -rf node_modules package-lock.json
npm install
```

**CORS Errors**  
```typescript
// proxy.conf.json
{
  "/api/*": {
    "target": "http://localhost:8000",
    "secure": false,
    "changeOrigin": true
  }
}
```

**Memory Issues**
```bash
# Aumenta memoria Node.js
node --max-old-space-size=8192 node_modules/@angular/cli/bin/ng build
```

## 📞 Supporto

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)

## 📄 License

MIT License - vedi [LICENSE.md](LICENSE.md) per dettagli.

---

**Creato con ❤️ per il progetto ARIMA Forecasting**