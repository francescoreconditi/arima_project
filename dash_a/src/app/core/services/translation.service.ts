// ============================================
// SERVIZIO TRADUZIONI
// Creato da: Claude Code
// Data: 2025-09-04
// Scopo: Gestione traduzioni multilingue
// ============================================

import { Injectable, signal } from '@angular/core';

interface Translations {
  [key: string]: {
    [lang: string]: string;
  };
}

@Injectable({
  providedIn: 'root'
})
export class TranslationService {
  private currentLang = signal<string>('it');
  
  private translations: Translations = {
    // Menu items
    'menu.dashboard': {
      it: 'Dashboard',
      en: 'Dashboard'
    },
    'menu.inventory': {
      it: 'Inventario',
      en: 'Inventory'
    },
    'menu.forecast': {
      it: 'Previsioni',
      en: 'Forecast'
    },
    'menu.orders': {
      it: 'Ordini',
      en: 'Orders'
    },
    'menu.suppliers': {
      it: 'Fornitori',
      en: 'Suppliers'
    },
    'menu.analytics': {
      it: 'Analisi',
      en: 'Analytics'
    },
    'menu.alerts': {
      it: 'Alert',
      en: 'Alerts'
    },
    'menu.settings': {
      it: 'Impostazioni',
      en: 'Settings'
    },
    
    // Dashboard
    'dashboard.title': {
      it: 'Dashboard Inventario',
      en: 'Inventory Dashboard'
    },
    'dashboard.filters.category': {
      it: 'Categoria Prodotto',
      en: 'Product Category'
    },
    'dashboard.filters.product': {
      it: 'Prodotto',
      en: 'Product'
    },
    'dashboard.filters.allCategories': {
      it: 'Tutte le categorie',
      en: 'All categories'
    },
    'dashboard.filters.selectProduct': {
      it: 'Seleziona prodotto',
      en: 'Select product'
    },
    'dashboard.filters.forecastDays': {
      it: 'Giorni Previsione',
      en: 'Forecast Days'
    },
    
    // KPIs
    'kpi.inventoryValue': {
      it: 'Valore Inventario',
      en: 'Inventory Value'
    },
    'kpi.productsInStock': {
      it: 'Prodotti in Stock',
      en: 'Products in Stock'
    },
    'kpi.pendingOrders': {
      it: 'Ordini Pendenti',
      en: 'Pending Orders'
    },
    'kpi.serviceLevel': {
      it: 'Livello Servizio',
      en: 'Service Level'
    },
    
    // Table headers
    'table.code': {
      it: 'Codice',
      en: 'Code'
    },
    'table.name': {
      it: 'Nome',
      en: 'Name'
    },
    'table.currentStock': {
      it: 'Stock Attuale',
      en: 'Current Stock'
    },
    'table.minStock': {
      it: 'Stock Minimo',
      en: 'Min Stock'
    },
    'table.unitPrice': {
      it: 'Prezzo Unitario',
      en: 'Unit Price'
    },
    'table.status': {
      it: 'Stato',
      en: 'Status'
    },
    'table.actions': {
      it: 'Azioni',
      en: 'Actions'
    },
    
    // Status
    'status.outOfStock': {
      it: 'Esaurito',
      en: 'Out of Stock'
    },
    'status.low': {
      it: 'Basso',
      en: 'Low'
    },
    'status.excess': {
      it: 'Eccesso',
      en: 'Excess'
    },
    'status.ok': {
      it: 'OK',
      en: 'OK'
    },
    
    // Actions
    'action.export': {
      it: 'Esporta',
      en: 'Export'
    },
    'action.refresh': {
      it: 'Aggiorna',
      en: 'Refresh'
    },
    'action.generateForecast': {
      it: 'Genera Previsione',
      en: 'Generate Forecast'
    },
    'action.changeLanguage': {
      it: 'Cambia lingua',
      en: 'Change language'
    },
    'action.changeTheme': {
      it: 'Cambia tema',
      en: 'Change theme'
    },
    'action.notifications': {
      it: 'Notifiche',
      en: 'Notifications'
    },
    'action.close': {
      it: 'Chiudi',
      en: 'Close'
    },
    
    // Messages
    'message.noNotifications': {
      it: 'Nessuna notifica',
      en: 'No notifications'
    },
    'message.selectProductForForecast': {
      it: 'Seleziona un prodotto per la previsione',
      en: 'Select a product for forecast'
    },
    'message.forecastGenerated': {
      it: 'Previsione generata con successo',
      en: 'Forecast generated successfully'
    },
    'message.forecastError': {
      it: 'Errore nella generazione della previsione',
      en: 'Error generating forecast'
    },
    'message.dataRefreshed': {
      it: 'Dati aggiornati',
      en: 'Data refreshed'
    },
    'message.reportExported': {
      it: 'Report esportato con successo',
      en: 'Report exported successfully'
    },
    'message.exportError': {
      it: 'Errore esportazione report',
      en: 'Error exporting report'
    },
    'message.loadingError': {
      it: 'Errore caricamento dati',
      en: 'Error loading data'
    },
    
    // Charts
    'chart.sales': {
      it: 'Vendite',
      en: 'Sales'
    },
    'chart.forecasts': {
      it: 'Previsioni',
      en: 'Forecasts'
    },
    'chart.currentStock': {
      it: 'Stock Attuale',
      en: 'Current Stock'
    },
    'chart.minStock': {
      it: 'Stock Minimo',
      en: 'Minimum Stock'
    },
    'chart.confidenceInterval': {
      it: 'Intervallo Confidenza',
      en: 'Confidence Interval'
    },
    'chart.date': {
      it: 'Data',
      en: 'Date'
    },
    'chart.value': {
      it: 'Valore',
      en: 'Value'
    }
  };

  constructor() {}

  setLanguage(lang: string): void {
    this.currentLang.set(lang);
  }

  getLanguage(): string {
    return this.currentLang();
  }

  translate(key: string, lang?: string): string {
    const targetLang = lang || this.currentLang();
    
    if (!this.translations[key]) {
      console.warn(`Translation key not found: ${key}`);
      return key;
    }
    
    return this.translations[key][targetLang] || this.translations[key]['it'] || key;
  }

  // Helper method per ottenere tutte le traduzioni di una categoria
  getTranslations(prefix: string): { [key: string]: string } {
    const result: { [key: string]: string } = {};
    const lang = this.currentLang();
    
    Object.keys(this.translations)
      .filter(key => key.startsWith(prefix))
      .forEach(key => {
        const shortKey = key.replace(prefix + '.', '');
        result[shortKey] = this.translate(key, lang);
      });
    
    return result;
  }
}